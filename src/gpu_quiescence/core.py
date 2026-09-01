"""Allocation-readiness handshake: composable stages, honest reports.

The handshake is a gate. It runs its stages in order, stops at the first
failure, captures exceptions as failed stages, and returns a report that
records precisely what was tested. It never claims more than it measured:
a successful probe means one allocation of a known size succeeded at a
known moment - not that the box cannot OOM later.
"""

from __future__ import annotations

import datetime
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from .errors import UsageError

#: Allocators that are not `MemoryError`-based still say "no" in a recognisable
#: way. torch raises `torch.cuda.OutOfMemoryError(RuntimeError)`; ROCm and a few
#: vendor runtimes raise plain RuntimeErrors carrying the same sentence. Matching
#: on shape keeps a genuine refusal from being filed as an allocator bug.
_OOM_MESSAGE = re.compile(r"out of memory", re.IGNORECASE)


def looks_like_a_refusal(exc: BaseException) -> bool:
    """True when an exception is an allocator saying no rather than breaking."""
    if type(exc).__name__.endswith("OutOfMemoryError"):
        return True
    return bool(_OOM_MESSAGE.search(str(exc)))


@dataclass
class StageReport:
    name: str
    ok: bool
    duration_s: float
    observations: dict[str, float | int | str] = field(default_factory=dict)
    detail: str = ""


SCHEMA_VERSION = 1


def tool_version() -> str:
    """The installed distribution version, or "unknown" - never a guess."""
    try:
        from importlib.metadata import version

        return version("gpu-quiescence")
    except Exception:
        return "unknown"


@dataclass
class ReadinessReport:
    ok: bool
    started_at: float
    stages: list[StageReport] = field(default_factory=list)

    def to_dict(self) -> dict:
        """A stored report has to say what produced it, when, and against what.

        An archived --json record is evidence only if it is self-describing:
        the envelope names the tool and its version, and every stage stamps
        the resource it measured.
        """
        started = datetime.datetime.fromtimestamp(self.started_at, datetime.timezone.utc)
        return {
            "schema_version": SCHEMA_VERSION,
            "tool": "gpu-quiescence",
            "version": tool_version(),
            "ok": self.ok,
            "started_at": self.started_at,
            "started_at_iso": started.isoformat(),
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
    """Answers one question: how many MiB are free right now - and of what?"""

    label: str

    def free_mib(self) -> float: ...


def probe_label(probe) -> str:
    """The resource a probe measures, for the record. Never raises."""
    try:
        return str(getattr(probe, "label", "") or "unknown")
    except Exception:
        return "unknown"


class Stage(Protocol):
    name: str

    def run(self) -> StageReport: ...


class Handshake:
    """Run stages in order; first failure gates the rest."""

    def __init__(self, stages: list[Stage]):
        self.stages = stages

    def run(self) -> ReadinessReport:
        report = ReadinessReport(ok=True, started_at=time.time())
        if not self.stages:
            # a gate with nothing to test has tested nothing
            report.ok = False
            report.stages.append(
                StageReport(name="handshake", ok=False, duration_s=0.0, detail="no stages configured; nothing was tested")
            )
            return report
        for stage in self.stages:
            t0 = time.monotonic()
            try:
                result = stage.run()
            except UsageError:
                # NOT a verdict. The gate could not run, so it must not answer
                # "not ready"; the caller turns this into exit 2.
                raise
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
    """Ask the inference server to release what it holds, then check the bytes.

    Eviction is asynchronous on real servers, so the STAGE owns the wait:
    settled() is a predicate and this loop is the only deadline. One poll is
    one question to the server, and the report says how long it actually
    waited - not how long it was allowed to.

    Given a probe and an evictor that can say how much VRAM it holds, the
    stage certifies the reclaim in bytes: free memory must rise by at least
    `reclaim_fraction` of what the server said it was holding. "/api/ps is
    empty" is the server's claim; the delta is the measurement. Without a
    probe the stage reports only what it saw.
    """

    name = "evict"

    def __init__(
        self,
        evictor,
        deadline_s: float = 30.0,
        poll_interval_s: float = 1.0,
        probe: MemoryProbe | None = None,
        reclaim_fraction: float = 0.9,
        _sleep=time.sleep,
        _clock=time.monotonic,
    ) -> None:
        self._evictor = evictor
        self._deadline = deadline_s
        self._poll = poll_interval_s
        self._probe = probe
        self._fraction = reclaim_fraction
        self._sleep = _sleep
        self._clock = _clock

    def _held_vram_mib(self) -> float | None:
        if self._probe is None:
            return None
        held = getattr(self._evictor, "held_vram_mib", None)
        return None if held is None else float(held())

    def run(self) -> StageReport:
        t0 = self._clock()
        free_before = None if self._probe is None else self._probe.free_mib()
        held_before = self._held_vram_mib()
        self._evictor.evict()
        polls = 0
        while True:
            polls += 1
            if self._evictor.settled():
                return self._certify(self._clock() - t0, polls, free_before, held_before)
            waited = self._clock() - t0
            if waited >= self._deadline:
                return StageReport(
                    name=self.name,
                    ok=False,
                    duration_s=waited,
                    observations={"polls": polls, "waited_s": round(waited, 1)},
                    detail=f"server still holds models after {waited:.0f}s of polling",
                )
            self._sleep(self._poll)

    def _certify(self, waited, polls, free_before, held_before) -> StageReport:
        observations: dict[str, float | int | str] = {
            "polls": polls,
            "waited_s": round(waited, 1),
        }
        if free_before is None or not held_before:
            return StageReport(
                name=self.name,
                ok=True,
                duration_s=waited,
                observations=observations,
                detail=f"server reports nothing loaded after {waited:.0f}s",
            )
        free_after = self._probe.free_mib()
        reclaimed = free_after - free_before
        observations.update(
            {
                "source": probe_label(self._probe),
                "free_before_mib": round(free_before, 1),
                "free_after_mib": round(free_after, 1),
                "held_vram_mib": round(held_before, 1),
                "reclaimed_mib": round(reclaimed, 1),
            }
        )
        if self._fraction > 0 and reclaimed < held_before * self._fraction:
            return StageReport(
                name=self.name,
                ok=False,
                duration_s=waited,
                observations=observations,
                detail=(
                    f"server reports nothing loaded, but only {reclaimed:.0f} MiB of the "
                    f"{held_before:.0f} MiB it held came back after {waited:.0f}s"
                ),
            )
        return StageReport(
            name=self.name,
            ok=True,
            duration_s=waited,
            observations=observations,
            detail=(
                f"server released {reclaimed:.0f} MiB of the {held_before:.0f} MiB "
                f"it held, after {waited:.0f}s"
            ),
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
        if timeout_s < window * interval_s:
            raise ValueError(
                f"timeout_s={timeout_s} cannot fit {window} samples at {interval_s}s; "
                "this stage would fail forever without ever being able to succeed"
            )
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
                        "source": probe_label(self._probe),
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
            observations={"source": probe_label(self._probe), "samples": len(samples)},
            detail="free memory never settled; something is still reclaiming or leaking",
        )


class Allocator(Protocol):
    """Allocate one buffer, touch it, and clean up once it has been dropped."""

    label: str

    def alloc(self, nbytes: int): ...

    def touch(self, buf) -> None: ...

    def release(self) -> None:
        """Runs AFTER the stage has dropped its reference to the buffer.

        Runtimes with a caching allocator (torch) only hand memory back to the
        driver once nothing references the tensor, so this hook runs after the
        drop - otherwise `free_after_mib` reports a number the driver disagrees
        with.
        """


class _ByteArrayAllocator:
    """The default: host memory via `bytearray`, pages touched one per 4 KiB."""

    def __init__(self, alloc=bytearray, label: str = "host-bytearray") -> None:
        self._alloc = alloc
        self.label = label

    def alloc(self, nbytes: int):
        return self._alloc(nbytes)

    def touch(self, buf) -> None:
        for i in range(0, len(buf), 4096):  # touch pages so the allocation is real
            buf[i] = 1

    def release(self) -> None:
        pass  # dropping the last reference is all host memory needs


class ProbeStage:
    """Allocate, touch, and release one representative buffer.

    HONESTY FIRST: the default allocator is `bytearray`, which exercises
    HOST memory. In GPU mode that means the probe proves the host can
    allocate, while the VRAM readiness signal comes from the settle and
    headroom stages reading real VRAM numbers. To probe VRAM itself, pass
    an allocator that actually touches it (for example a torch CUDA tensor
    factory); the report names the allocator either way, so the record
    says exactly what was tested.

    Outcomes are three-valued and never disguised: `succeeded`, `refused`
    (the allocator said no - the expected failure mode), or `errored` (the
    allocator broke - a programming or environment problem). Both refusals
    and errors gate the launch; the report tells you which one happened.
    """

    name = "probe"

    MIN_MIB = 128
    MAX_MIB = 1024
    FRACTION = 0.25

    def __init__(
        self,
        probe: MemoryProbe,
        required_mib: float,
        _alloc=bytearray,
        allocator_label: str = "host-bytearray",
        refusal_exceptions: tuple[type[BaseException], ...] = (MemoryError,),
        allocator: Allocator | None = None,
    ) -> None:
        self._probe = probe
        self._required = required_mib
        self._allocator = allocator if allocator is not None else _ByteArrayAllocator(_alloc, allocator_label)
        self._alloc = _alloc
        self.allocator_label = getattr(self._allocator, "label", allocator_label)
        self._refusals = tuple(refusal_exceptions) + tuple(
            getattr(self._allocator, "refusal_exceptions", ())
        )
        self.size_mib = int(min(self.MAX_MIB, max(self.MIN_MIB, required_mib * self.FRACTION)))

    def run(self) -> StageReport:
        t0 = time.monotonic()
        before = self._probe.free_mib()
        base = {
            "source": probe_label(self._probe),
            "size_mib": self.size_mib,
            "allocator": self.allocator_label,
            "free_before_mib": round(before, 1),
        }
        refused = None
        try:
            buf = self._allocator.alloc(self.size_mib * 1024 * 1024)
            try:
                self._allocator.touch(buf)
            finally:
                buf = None  # drop the buffer BEFORE the allocator cleans up
                self._allocator.release()
        except self._refusals as exc:
            refused = exc
        except Exception as exc:
            # An allocator that is not MemoryError-based still has to be heard:
            # a CUDA OOM is a refusal, and filing it as a bug sends the operator
            # to debug the allocator instead of freeing memory.
            if not looks_like_a_refusal(exc):
                return StageReport(
                    name=self.name,
                    ok=False,
                    duration_s=time.monotonic() - t0,
                    observations={**base, "outcome": "errored"},
                    detail=f"probe errored (not a memory refusal): {exc}",
                )
            refused = exc
        if refused is not None:
            return StageReport(
                name=self.name,
                ok=False,
                duration_s=time.monotonic() - t0,
                observations={**base, "outcome": "refused", "refusal": type(refused).__name__},
                detail=f"probe refused: an allocation of {self.size_mib} MiB via {self.allocator_label} failed",
            )
        after = self._probe.free_mib()
        return StageReport(
            name=self.name,
            ok=True,
            duration_s=time.monotonic() - t0,
            observations={**base, "outcome": "succeeded", "free_after_mib": round(after, 1)},
            detail=f"an allocation of {self.size_mib} MiB via {self.allocator_label} succeeded at {time.strftime('%H:%M:%S')}",
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
                "source": probe_label(self._probe),
                "free_mib": round(free, 1),
                "required_mib": round(self._required, 1),
                "needed_mib": round(needed, 1),
            },
            detail="headroom sufficient" if ok else f"need {needed:.0f} MiB free, have {free:.0f}",
        )
