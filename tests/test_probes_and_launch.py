import subprocess
import sys

import pytest

from gpu_quiescence.launch import launch_low_priority, run_then_restore
from gpu_quiescence.probes import NvidiaSmiProbe, SystemMemoryProbe


def test_nvidia_probe_parses_multi_gpu_output():
    def fake_run(*_a, **_k):
        return subprocess.CompletedProcess(
            [], 0, stdout="0, GPU-aaaa-1111, 10240\n1, GPU-bbbb-2222, 2048\n", stderr=""
        )

    assert NvidiaSmiProbe(0, _run=fake_run).free_mib() == 10240.0
    assert NvidiaSmiProbe(1, _run=fake_run).free_mib() == 2048.0


def test_nvidia_probe_rejects_out_of_range_index():
    def fake_run(*_a, **_k):
        return subprocess.CompletedProcess([], 0, stdout="0, GPU-aaaa-1111, 10240\n", stderr="")

    with pytest.raises(RuntimeError, match="out of range"):
        NvidiaSmiProbe(3, _run=fake_run).free_mib()


def test_system_probe_reads_this_platform():
    free = SystemMemoryProbe().free_mib()
    assert free > 0


class RecordingEvictor:
    def __init__(self):
        self.restored_with = "unset"

    def restore(self, model=None):
        self.restored_with = model


def test_job_runs_and_restoration_happens_even_on_failure():
    ev = RecordingEvictor()
    code = run_then_restore(
        [sys.executable, "-c", "import sys; sys.exit(3)"], ev, restore_model="m"
    )
    assert code == 3
    assert ev.restored_with == "m"


def test_probes_name_the_resource_they_read():
    def fake_run(*_a, **_k):
        return subprocess.CompletedProcess(
            [], 0, stdout="0, GPU-aaaa-1111, 10240\n1, GPU-bbbb-2222, 2048\n", stderr=""
        )

    assert SystemMemoryProbe().label == "system-ram"

    gpu1 = NvidiaSmiProbe(1, _run=fake_run)
    assert gpu1.free_mib() == 2048.0
    assert gpu1.label == "vram:gpu1:GPU-bbbb-2222"


def test_the_job_runs_as_a_reduced_priority_child():
    seen = {}

    class FakeChild:
        def wait(self):
            return 0

    def fake_popen(_cmd, **kwargs):
        seen.update(kwargs)
        return FakeChild()

    assert launch_low_priority(["anything"], _popen=fake_popen) == 0
    if sys.platform == "win32":
        assert seen["creationflags"] == subprocess.BELOW_NORMAL_PRIORITY_CLASS
    else:
        assert seen["preexec_fn"] is not None  # os.nice(10)
