"""The VRAM allocator is shipped, not described in prose."""

import pytest

from gpu_quiescence.allocators import torch_cuda_allocator
from gpu_quiescence.core import ProbeStage
from gpu_quiescence.errors import UsageError


class FakeCudaOutOfMemoryError(RuntimeError):
    pass


class FakeCuda:
    OutOfMemoryError = FakeCudaOutOfMemoryError

    def __init__(self):
        self.synchronised = 0
        self.emptied = 0

    def is_available(self):
        return True

    def synchronize(self, device=None):
        self.synchronised += 1

    def empty_cache(self):
        self.emptied += 1


class FakeTensor:
    def __init__(self, nbytes):
        self.nbytes = nbytes
        self.filled = None

    def fill_(self, value):
        self.filled = value


class FakeTorch:
    uint8 = "uint8"

    def __init__(self, fail_with=None):
        self.cuda = FakeCuda()
        self.calls = []
        self._fail_with = fail_with

    def empty(self, nbytes, dtype=None, device=None):
        self.calls.append((nbytes, dtype, device))
        if self._fail_with is not None:
            raise self._fail_with
        return FakeTensor(nbytes)


class ScriptedProbe:
    def __init__(self, values):
        self._values = list(values)

    def free_mib(self):
        if len(self._values) > 1:
            return self._values.pop(0)
        return self._values[0]


def test_the_vram_allocator_allocates_on_cuda_and_names_itself():
    torch = FakeTorch()
    alloc = torch_cuda_allocator(1, _torch=torch)
    assert alloc.label == "cuda:1-uint8"
    buf = alloc.alloc(1024)
    assert torch.calls == [(1024, "uint8", "cuda:1")]
    alloc.touch(buf)
    assert buf.filled == 1  # one kernel, not 65k python-level writes
    assert torch.cuda.synchronised == 1  # a lazy failure surfaces inside the probe
    alloc.release()
    assert torch.cuda.emptied == 1  # so free_after_mib agrees with nvidia-smi


def test_the_vram_allocator_reports_a_cuda_oom_as_refused_through_the_stage():
    torch = FakeTorch(fail_with=FakeCudaOutOfMemoryError("CUDA out of memory."))
    stage = ProbeStage(
        ScriptedProbe([9000]), 1000, allocator=torch_cuda_allocator(0, _torch=torch)
    )
    r = stage.run()
    assert not r.ok
    assert r.observations["outcome"] == "refused"
    assert r.observations["allocator"] == "cuda:0-uint8"


def test_the_vram_allocator_asks_for_torch_as_a_usage_error_not_an_import_crash():
    class NoCuda(FakeCuda):
        def is_available(self):
            return False

    torch = FakeTorch()
    torch.cuda = NoCuda()
    with pytest.raises(UsageError, match="CUDA"):
        torch_cuda_allocator(0, _torch=torch)


def test_the_vram_allocator_runs_against_real_torch_when_present():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    r = ProbeStage(ScriptedProbe([9000, 8900]), 1000, allocator=torch_cuda_allocator(0)).run()
    assert r.observations["outcome"] in {"succeeded", "refused"}
