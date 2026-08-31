"""The handshake gates: every claim in the README has a test here."""

from gpu_quiescence.core import (
    EvictStage,
    Handshake,
    HeadroomStage,
    ProbeStage,
    SettleStage,
    StageReport,
)


class ScriptedProbe:
    """free_mib() returns the next scripted value, repeating the last."""

    def __init__(self, values):
        self._values = list(values)

    def free_mib(self):
        if len(self._values) > 1:
            return self._values.pop(0)
        return self._values[0]


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.t += s


def make_settle(probe, clock, **kw):
    return SettleStage(probe, _sleep=clock.sleep, _clock=clock.monotonic, **kw)


def test_settle_passes_when_readings_sit_in_the_band():
    clock = FakeClock()
    stage = make_settle(ScriptedProbe([8000, 8010, 8005, 8002, 8008, 8004]), clock)
    r = stage.run()
    assert r.ok
    assert r.observations["spread_mib"] < 64


def test_settle_keeps_waiting_while_memory_is_still_moving():
    clock = FakeClock()
    # A reclaim in progress: big jumps first, then flat.
    values = [2000, 3000, 4500, 6000, 7500, 8000, 8001, 8002, 8003, 8004, 8005]
    stage = make_settle(ScriptedProbe(values), clock)
    r = stage.run()
    assert r.ok
    assert r.observations["samples"] > 5


def test_settle_times_out_when_memory_never_stops_moving():
    clock = FakeClock()
    drifting = ScriptedProbe(list(range(1000, 100000, 500)))
    stage = make_settle(drifting, clock, timeout_s=10.0)
    r = stage.run()
    assert not r.ok
    assert "never settled" in r.detail


def test_probe_size_is_proportional_and_clamped():
    p = ScriptedProbe([9000])
    assert ProbeStage(p, required_mib=100).size_mib == 128  # floor
    assert ProbeStage(p, required_mib=2000).size_mib == 500  # 25%
    assert ProbeStage(p, required_mib=40000).size_mib == 1024  # ceiling


def test_probe_success_claims_only_its_own_allocation():
    r = ProbeStage(ScriptedProbe([9000, 8900]), required_mib=1000).run()
    assert r.ok
    assert "succeeded" in r.detail
    assert "contigu" not in r.detail.lower()  # the claim stays honest
    assert r.observations["size_mib"] == 250


def test_probe_failure_is_a_gated_report_not_an_exception():
    def failing_alloc(_n):
        raise MemoryError

    r = ProbeStage(ScriptedProbe([9000]), required_mib=1000, _alloc=failing_alloc).run()
    assert not r.ok
    assert "failed" in r.detail


def test_headroom_uses_factor_and_margin():
    # need 1000 * 1.1 + 512 = 1612
    assert not HeadroomStage(ScriptedProbe([1600]), 1000).run().ok
    assert HeadroomStage(ScriptedProbe([1700]), 1000).run().ok


def test_headroom_reports_numbers_on_failure():
    r = HeadroomStage(ScriptedProbe([100]), 1000).run()
    assert r.observations["needed_mib"] == 1612.0
    assert r.observations["free_mib"] == 100.0


class FakeEvictor:
    def __init__(self, settles=True):
        self.evicted = False
        self._settles = settles

    def evict(self):
        self.evicted = True

    def settled(self):
        return self._settles

    def restore(self, model=None):
        self.restored = model


def test_handshake_stops_at_first_failure():
    bad = EvictStage(FakeEvictor(settles=False), deadline_s=0.0, _sleep=lambda s: None)
    never_reached = HeadroomStage(ScriptedProbe([10**6]), 1)
    report = Handshake([bad, never_reached]).run()
    assert not report.ok
    assert [s.name for s in report.stages] == ["evict"]


def test_handshake_captures_stage_exceptions_as_failures():
    class Exploding:
        name = "exploding"

        def run(self):
            raise RuntimeError("boom")

    report = Handshake([Exploding()]).run()
    assert not report.ok
    assert "boom" in report.stages[0].detail


def test_full_green_path_reports_every_stage():
    probe = ScriptedProbe([8000, 8001, 8002, 8003, 8004, 8005])
    clock = FakeClock()
    stages = [
        EvictStage(FakeEvictor()),
        make_settle(probe, clock),
        ProbeStage(probe, 1000),
        HeadroomStage(probe, 1000),
    ]
    report = Handshake(stages).run()
    assert report.ok
    assert [s.name for s in report.stages] == ["evict", "settle", "probe", "headroom"]
    assert report.to_dict()["stages"][2]["observations"]["size_mib"] == 250


def test_empty_handshake_fails_because_nothing_was_tested():
    report = Handshake([]).run()
    assert not report.ok
    assert "nothing was tested" in report.stages[0].detail


def test_settle_rejects_a_timeout_that_can_never_succeed():
    import pytest

    with pytest.raises(ValueError, match="being able to succeed"):
        SettleStage(ScriptedProbe([1]), window=5, interval_s=1.0, timeout_s=3.0)


def test_evict_stage_polls_until_settled_or_deadline():
    class SlowEvictor:
        def __init__(self):
            self.checks = 0

        def evict(self):
            pass

        def settled(self):
            self.checks += 1
            return self.checks >= 3

        def restore(self, model=None):
            pass

    clock = FakeClock()
    ev = SlowEvictor()
    stage = EvictStage(ev, deadline_s=10.0, poll_interval_s=1.0, _sleep=clock.sleep, _clock=clock.monotonic)
    r = stage.run()
    assert r.ok
    assert r.observations["polls"] == 3


def test_probe_distinguishes_refused_from_errored_and_both_gate():
    def refusing(_n):
        raise MemoryError

    def exploding(_n):
        raise RuntimeError("allocator bug")

    refused = ProbeStage(ScriptedProbe([9000]), 1000, _alloc=refusing).run()
    assert not refused.ok and refused.observations["outcome"] == "refused"

    errored = ProbeStage(ScriptedProbe([9000]), 1000, _alloc=exploding).run()
    assert not errored.ok and errored.observations["outcome"] == "errored"
    assert "not a memory refusal" in errored.detail


def test_probe_reports_its_allocator_so_the_record_says_what_was_tested():
    r = ProbeStage(ScriptedProbe([9000, 8900]), 1000).run()
    assert r.observations["allocator"] == "host-bytearray"
