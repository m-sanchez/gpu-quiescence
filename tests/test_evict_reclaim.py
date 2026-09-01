"""Eviction is certified in bytes, and the report states the real wait.

"The server says it unloaded" is a claim by the server. "Free VRAM went up by
as much as the server said it was holding" is a measurement. The evict stage
makes the second one, and it reports the time it actually waited rather than
the deadline it was configured with.
"""

import json

import pytest

from gpu_quiescence.core import EvictStage
from gpu_quiescence.evictors import OllamaEvictor

MIB = 1024 * 1024
LOADED = [
    {"name": "llama3:latest", "size": 9 * 1024 * MIB, "size_vram": 8192 * MIB},
    {"name": "nomic-embed:latest", "size": 512 * MIB, "size_vram": 512 * MIB},
]
HELD_MIB = 8704.0


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.t += s


class ScriptedProbe:
    def __init__(self, values, label="vram:gpu0:GPU-test"):
        self._values = list(values)
        self.label = label

    def free_mib(self):
        if len(self._values) > 1:
            return self._values.pop(0)
        return self._values[0]


class FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


class FakeOllama:
    """Serves realistic /api/ps payloads; one entry of the script per call."""

    def __init__(self, ps_script):
        self._script = list(ps_script)
        self.ps_calls = 0
        self.generates = []

    def install(self, monkeypatch):
        monkeypatch.setattr("urllib.request.urlopen", self.urlopen)
        return self

    def urlopen(self, req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        if url.endswith("/api/ps"):
            self.ps_calls += 1
            models = self._script.pop(0) if len(self._script) > 1 else self._script[0]
            return FakeResponse(json.dumps({"models": models}).encode())
        if url.endswith("/api/generate"):
            self.generates.append(json.loads(req.data))
            return FakeResponse(b"{}")
        raise AssertionError(f"unexpected request to {url}")


def evictor():
    return OllamaEvictor("http://127.0.0.1:11434", timeout_s=2.0)


def stage(ev, probe, clock, **kw):
    return EvictStage(
        ev, poll_interval_s=1.0, probe=probe, _sleep=clock.sleep, _clock=clock.monotonic, **kw
    )


def test_loaded_models_keep_the_vram_the_server_says_they_hold(monkeypatch):
    fake = FakeOllama([LOADED]).install(monkeypatch)
    ev = evictor()
    assert [m.name for m in ev.loaded_models()] == ["llama3:latest", "nomic-embed:latest"]
    assert ev.held_vram_mib() == pytest.approx(HELD_MIB)
    assert fake.ps_calls == 2


def test_settled_asks_the_server_once_and_answers(monkeypatch):
    # settled() is a predicate. The poll loop belongs to the stage, which owns
    # the deadline; two components cannot both own the same wait.
    fake = FakeOllama([LOADED]).install(monkeypatch)
    assert evictor().settled() is False
    assert fake.ps_calls == 1

    fake = FakeOllama([[]]).install(monkeypatch)
    assert evictor().settled() is True
    assert fake.ps_calls == 1


def test_the_evict_report_states_the_time_it_actually_waited():
    clock = FakeClock()

    class SlowToAnswer:
        """Answering is not free - the shipped evictor used to block for 20s."""

        def evict(self):
            pass

        def settled(self):
            clock.t += 20.0
            return False

        def restore(self, model=None):
            pass

    r = EvictStage(
        SlowToAnswer(),
        deadline_s=30.0,
        poll_interval_s=1.0,
        _sleep=clock.sleep,
        _clock=clock.monotonic,
    ).run()
    assert not r.ok
    assert f"{r.duration_s:.0f}s" in r.detail  # the wait, not the deadline
    assert r.observations["waited_s"] == pytest.approx(round(r.duration_s, 1))


def test_a_completed_reclaim_is_certified_in_bytes(monkeypatch):
    fake = FakeOllama([LOADED, LOADED, []]).install(monkeypatch)
    clock = FakeClock()
    r = stage(evictor(), ScriptedProbe([2000.0, 2000.0 + HELD_MIB]), clock).run()
    assert r.ok
    assert r.observations["held_vram_mib"] == pytest.approx(HELD_MIB)
    assert r.observations["free_before_mib"] == 2000.0
    assert r.observations["free_after_mib"] == pytest.approx(2000.0 + HELD_MIB)
    assert r.observations["reclaimed_mib"] == pytest.approx(HELD_MIB)
    assert r.observations["source"] == "vram:gpu0:GPU-test"
    assert fake.generates  # it actually asked


def test_a_reclaim_short_of_what_the_server_held_gates_the_job(monkeypatch):
    # /api/ps is empty, but 30% of the VRAM never came back: trusting the
    # server here is how a trainer OOMs a minute after a green preflight.
    FakeOllama([LOADED, LOADED, []]).install(monkeypatch)
    clock = FakeClock()
    r = stage(evictor(), ScriptedProbe([2000.0, 2000.0 + HELD_MIB * 0.7]), clock).run()
    assert not r.ok
    assert "8704" in r.detail and "6093" in r.detail  # held and reclaimed, in the detail


def test_vram_that_never_comes_back_gates_the_job(monkeypatch):
    FakeOllama([LOADED, LOADED, []]).install(monkeypatch)
    clock = FakeClock()
    r = stage(evictor(), ScriptedProbe([2000.0, 2000.0]), clock).run()
    assert not r.ok
    assert r.observations["reclaimed_mib"] == 0.0


def test_a_stuck_server_gates_after_the_stages_own_deadline(monkeypatch):
    fake = FakeOllama([LOADED]).install(monkeypatch)
    clock = FakeClock()
    r = stage(evictor(), ScriptedProbe([2000.0]), clock, deadline_s=5.0).run()
    assert not r.ok
    assert r.duration_s == pytest.approx(5.0)
    # every poll is one question to the server, and the count says so
    assert r.observations["polls"] == fake.ps_calls - 2  # minus held-read and evict-read
    assert "5s" in r.detail


def test_a_reclaim_check_is_skipped_when_the_stage_has_no_probe():
    # Without a probe the stage cannot certify bytes, and says only what it saw.
    class Empty:
        def evict(self):
            pass

        def settled(self):
            return True

        def restore(self, model=None):
            pass

    r = EvictStage(Empty()).run()
    assert r.ok
    assert "held_vram_mib" not in r.observations
