import subprocess
import sys

import pytest

from gpu_quiescence.evictors import OllamaEvictor
from gpu_quiescence.launch import run_then_restore
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


def test_ollama_evictor_settled_polls_until_empty(monkeypatch):
    ev = OllamaEvictor(_sleep=lambda _s: None, _clock=lambda: 0.0)
    responses = [["m1"], ["m1"], []]
    monkeypatch.setattr(ev, "loaded_models", lambda: responses.pop(0))
    assert ev.settled()


def test_ollama_evictor_settled_times_out(monkeypatch):
    t = {"now": 0.0}

    def clock():
        t["now"] += 30.0
        return t["now"]

    ev = OllamaEvictor(timeout_s=20.0, _sleep=lambda _s: None, _clock=clock)
    monkeypatch.setattr(ev, "loaded_models", lambda: ["stuck"])
    assert not ev.settled()


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
