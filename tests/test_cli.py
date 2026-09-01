"""The exit-code contract, end to end through cli.main().

README: 0 ready (and the job succeeded) - 1 not ready, or the job failed -
2 usage error. A usage error is never a "not ready" verdict, and never a
traceback: a cron that branches on the exit code has to be able to tell
"this box is busy" from "you configured me wrong".
"""

import json
import subprocess
import sys
import urllib.error

import pytest

from gpu_quiescence import cli, core, probes
from gpu_quiescence.errors import UsageError

READY = "0, GPU-aaaa-1111, 20480\n"
TIGHT = "0, GPU-aaaa-1111, 100\n"


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.t += s


def smi_returning(stdout):
    def fake_run(*_a, **_k):
        return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")

    return fake_run


def smi_raising(exc):
    def fake_run(*_a, **_k):
        raise exc

    return fake_run


@pytest.fixture
def box(monkeypatch):
    """A box wired to fakes. Only the exit code is under test."""
    state = {"smi": smi_returning(READY)}
    monkeypatch.setattr(
        cli, "NvidiaSmiProbe", lambda index=0: probes.NvidiaSmiProbe(index, _run=state["smi"])
    )
    clock = FakeClock()
    settle = core.SettleStage
    monkeypatch.setattr(
        cli,
        "SettleStage",
        lambda probe, **kw: settle(probe, _sleep=clock.sleep, _clock=clock.monotonic, **kw),
    )
    return state


BASE = ["--require-mib", "512"]


def test_a_missing_nvidia_smi_is_a_usage_error(box):
    box["smi"] = smi_raising(FileNotFoundError("nvidia-smi"))
    assert cli.main(BASE) == 2


def test_a_failing_nvidia_smi_is_a_usage_error_not_a_traceback(box):
    # Driver/library version mismatch: the tool cannot measure, so it has no
    # verdict. Exit 1 here would make a cron retry forever against a dead driver.
    box["smi"] = smi_raising(
        subprocess.CalledProcessError(
            9, ["nvidia-smi"], output="", stderr="NVML: Driver/library version mismatch\n"
        )
    )
    assert cli.main(BASE) == 2


@pytest.mark.parametrize("reading", ["[N/A]", "[Not Supported]", "Insufficient Permissions"])
def test_unreadable_nvidia_smi_output_is_a_usage_error(box, reading):
    # MIG and vGPU report memory.free as [N/A].
    box["smi"] = smi_returning(f"0, GPU-aaaa-1111, {reading}\n")
    assert cli.main(BASE) == 2


def test_an_unreachable_ollama_is_a_usage_error_not_a_not_ready_verdict(box, monkeypatch):
    def refuse(*_a, **_k):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    assert cli.main([*BASE, "--ollama", "http://127.0.0.1:99999"]) == 2


def test_a_missing_job_binary_is_a_usage_error_not_a_traceback(box):
    # The worst version of this: a fully green handshake, then a traceback,
    # which reads as a bug in the preflight rather than a typo in the command.
    assert cli.main([*BASE, "--", "gpu-quiescence-no-such-binary-9d1f"]) == 2


def test_probe_vram_without_torch_is_a_usage_error(box, monkeypatch):
    def missing(*_a, **_k):
        raise UsageError("VRAM probing needs torch")

    monkeypatch.setattr(cli, "torch_cuda_allocator", missing)
    assert cli.main([*BASE, "--probe-vram"]) == 2


def test_probe_vram_contradicts_system_mode(box):
    assert cli.main([*BASE, "--system", "--probe-vram"]) == 2


def test_short_headroom_gates_the_job_with_exit_one(box):
    box["smi"] = smi_returning(TIGHT)
    assert cli.main([*BASE, "--", sys.executable, "-c", "raise SystemExit(0)"]) == 1


def test_a_green_handshake_runs_the_job_and_returns_its_exit_code(box):
    assert cli.main([*BASE, "--", sys.executable, "-c", "import sys; sys.exit(3)"]) == 3


def test_a_green_handshake_with_no_command_is_ready(box):
    assert cli.main(BASE) == 0


def test_json_is_parseable_on_both_the_ready_and_the_gated_path(box, capsys):
    assert cli.main([*BASE, "--json"]) == 0
    ready = json.loads(capsys.readouterr().out)
    assert ready["ok"] is True
    assert ready["tool"] == "gpu-quiescence"
    assert [s["name"] for s in ready["stages"]] == ["settle", "probe", "headroom"]

    box["smi"] = smi_returning(TIGHT)
    assert cli.main([*BASE, "--json"]) == 1
    gated = json.loads(capsys.readouterr().out)
    assert gated["ok"] is False
    assert gated["stages"][-1]["name"] == "headroom"
