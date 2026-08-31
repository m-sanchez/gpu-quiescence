"""Allocation-readiness handshake: composable stages, honest reports.

The handshake is a gate. It runs its stages in order, stops at the first
failure, captures exceptions as failed stages, and returns a report that
records precisely what was tested. It never claims more than it measured:
a successful probe means one allocation of a known size succeeded at a
known moment - not that the box cannot OOM later.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class StageReport:
    name: str
    ok: bool
    duration_s: float
    observations: dict[str, float | int | str] = field(default_factory=dict)
    detail: str = ""


@dataclass
class ReadinessReport:
    ok: bool
    started_at: float
    stages: list[StageReport] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "started_at": self.started_at,
            "stages": [
                {
                    "name": s.name,
                    "ok": s.ok,
                    "duration_s": round(s.duration_s, 3),
                    "observations": s.observations,
                    "detail": s.detail,
                }
                for s in self.stages
            ],
        }


class MemoryProbe(Protocol):
    """Answers one question: how many MiB are free right now?"""

    def free_mib(self) -> float: ...


class Stage(Protocol):
    name: str

    def run(self) -> StageReport: ...


class Handshake:
    """Run stages in order; first failure gates the rest."""

    def __init__(self, stages: list[Stage]):
        self.stages = stages

    def run(self) -> ReadinessReport:
        report = ReadinessReport(ok=True, started_at=time.time())
        for stage in self.stages:
            t0 = time.monotonic()
            try:
                result = stage.run()
            except Exception as exc:  # a gate must report, never raise past itself
                result = StageReport(
                    name=stage.name,
                    ok=False,
                    duration_s=time.monotonic() - t0,
                    detail=f"unexpected error: {exc}",
                )
            report.stages.append(result)
            if not result.ok:
                report.ok = False
                break
        return report


class EvictStage:
    """Ask the inference server to release what it holds."""

    name = "evict"

    def __init__(self, evictor) -> None:
        self._evictor = evictor

    def run(self) -> StageReport:
        t0 = time.monotonic()
        self._evictor.evict()
        settled = self._evictor.settled()
        return StageReport(
            name=self.name,
            ok=settled,
            duration_s=time.monotonic() - t0,
            detail="server reports nothing loaded" if settled else "server still holds models",
        )


class SettleStage:
    """Free memory must sit inside a variation band before we trust it.

    The reading is stable when the spread (max - min) of the most recent
    `window` samples is below `band_mib`. A reclaim still in progress shows
    up as a wide band and keeps the stage waiting, up to `timeout_s`.
    """

    name = "settle"

    def __init__(
        self,
        probe: MemoryProbe,
        band_mib: float = 64.0,
        window: int = 5,
        interval_s: float = 1.0,
        timeout_s: float = 30.0,
        _sleep=time.sleep,
        _clock=time.monotonic,
    ) -> None:
        self._probe = probe
        self._band = band_mib
        self._window = window
        self._interval = interval_s
        self._timeout = timeout_s
        self._sleep = _sleep
        self._clock = _clock

    def run(self) -> StageReport:
        t0 = self._clock()
        samples: list[float] = [self._probe.free_mib()]
        while self._clock() - t0 < self._timeout:
            self._sleep(self._interval)
            samples.append(self._probe.free_mib())
            recent = samples[-self._window :]
            spread = max(recent) - min(recent)
            if len(recent) >= self._window and spread < self._band:
                return StageReport(
                    name=self.name,
                    ok=True,
                    duration_s=self._clock() - t0,
                    observations={
                        "free_mib": round(recent[-1], 1),
                        "spread_mib": round(spread, 1),
                        "samples": len(samples),
                    },
                    detail=f"free memory settled within a {self._band:.0f} MiB band",
                )
        return StageReport(
            name=self.name,
            ok=False,
            duration_s=self._clock() - t0,
            observations={"samples": len(samples)},
            detail="free memory never settled; something is still reclaiming or leaking",
        )


class ProbeStage:
    """Allocate, touch, and release one representative buffer.

    Success claims exactly what happened: an allocation of `size_mib`
    succeeded at that moment. Free-before and free-after are recorded as
    observations, and no conclusion is drawn from them.
    """

    name = "probe"

    MIN_MIB = 128
    MAX_MIB = 1024
    FRACTION = 0.25

    def __init__(self, probe: MemoryProbe, required_mib: float, _alloc=bytearray) -> None:
        self._probe = probe
        self._required = required_mib
        self._alloc = _alloc
        self.size_mib = int(min(self.MAX_MIB, max(self.MIN_MIB, required_mib * self.FRACTION)))

    def run(self) -> StageReport:
        t0 = time.monotonic()
        before = self._probe.free_mib()
        try:
            buf = self._alloc(self.size_mib * 1024 * 1024)
            step = 4096
            for i in range(0, len(buf), step):  # touch pages so the allocation is real
                buf[i] = 1
            del buf
        except MemoryError:
            return StageReport(
                name=self.name,
                ok=False,
                duration_s=time.monotonic() - t0,
                observations={"size_mib": self.size_mib, "free_before_mib": round(before, 1)},
                detail=f"an allocation of {self.size_mib} MiB failed",
            )
        after = self._probe.free_mib()
        return StageReport(
            name=self.name,
            ok=True,
            duration_s=time.monotonic() - t0,
            observations={
                "size_mib": self.size_mib,
                "free_before_mib": round(before, 1),
                "free_after_mib": round(after, 1),
            },
            detail=f"an allocation of {self.size_mib} MiB succeeded at {time.strftime('%H:%M:%S')}",
        )


class HeadroomStage:
    """free >= required * factor + margin. The numbers go in the report either way."""

    name = "headroom"

    def __init__(
        self,
        probe: MemoryProbe,
        required_mib: float,
        factor: float = 1.10,
        margin_mib: float = 512.0,
    ) -> None:
        self._probe = probe
        self._required = required_mib
        self._factor = factor
        self._margin = margin_mib

    def run(self) -> StageReport:
        t0 = time.monotonic()
        free = self._probe.free_mib()
        needed = self._required * self._factor + self._margin
        ok = free >= needed
        return StageReport(
            name=self.name,
            ok=ok,
            duration_s=time.monotonic() - t0,
            observations={
                "free_mib": round(free, 1),
                "required_mib": round(self._required, 1),
                "needed_mib": round(needed, 1),
            },
            detail="headroom sufficient" if ok else f"need {needed:.0f} MiB free, have {free:.0f}",
        )
