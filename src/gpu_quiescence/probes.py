"""Memory probes: system RAM and NVIDIA GPU free memory, zero dependencies."""

from __future__ import annotations

import subprocess
import sys

from .errors import UsageError

__all__ = ["NvidiaSmiProbe", "SystemMemoryProbe", "UsageError"]


class SystemMemoryProbe:
    """Free system RAM in MiB, without third-party packages where possible."""

    label = "system-ram"

    def free_mib(self) -> float:
        if sys.platform.startswith("linux"):
            return self._linux()
        if sys.platform == "win32":
            return self._windows()
        return self._fallback()

    @staticmethod
    def _linux() -> float:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024  # kB -> MiB
        raise RuntimeError("MemAvailable not present in /proc/meminfo")

    @staticmethod
    def _windows() -> float:
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise RuntimeError("GlobalMemoryStatusEx failed")
        return status.ullAvailPhys / (1024 * 1024)

    @staticmethod
    def _fallback() -> float:
        try:
            import psutil  # optional, only on platforms without a native path
        except ImportError as exc:
            raise RuntimeError(
                f"no native free-memory reader for {sys.platform}; install psutil"
            ) from exc
        return psutil.virtual_memory().available / (1024 * 1024)


class NvidiaSmiProbe:
    """Free VRAM in MiB for one GPU, read through nvidia-smi.

    The query carries the device UUID as well as the free bytes, so a stored
    report identifies the physical card and not only a position in nvidia-smi
    PCI enumeration order.
    """

    def __init__(self, gpu_index: int = 0, _run=subprocess.run) -> None:
        self._index = gpu_index
        self._run = _run
        self._uuid = ""

    @property
    def label(self) -> str:
        """vram:gpu<index>:GPU-<uuid> once the device has been read."""
        if not self._uuid:
            try:
                self._query()
            except Exception:
                # Naming the resource must never be the thing that fails a run.
                return f"vram:gpu{self._index}"
        return f"vram:gpu{self._index}:{self._uuid}" if self._uuid else f"vram:gpu{self._index}"

    def _query(self) -> tuple[str, str]:
        try:
            result = self._run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except FileNotFoundError as exc:
            raise UsageError("nvidia-smi is not installed or not on PATH; GPU mode needs it") from exc
        except subprocess.CalledProcessError as exc:
            # A driver that will not answer is a usage problem: the tool could
            # not measure, so it has no opinion on whether the box is ready.
            said = (exc.stderr or exc.output or "").strip() or f"exit status {exc.returncode}"
            raise UsageError(f"nvidia-smi failed: {said}") from exc
        rows: dict[int, tuple[str, str]] = {}
        for line in result.stdout.splitlines():
            parts = [c.strip() for c in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                index = int(parts[0])
            except ValueError:
                continue
            rows[index] = (parts[1], parts[2])
        if self._index not in rows:
            raise RuntimeError(f"gpu index {self._index} out of range; {len(rows)} gpu(s) reported")
        uuid, free = rows[self._index]
        self._uuid = uuid
        return uuid, free

    def free_mib(self) -> float:
        _uuid, free = self._query()
        try:
            return float(free)
        except ValueError as exc:
            # MIG and vGPU devices answer memory.free with [N/A] or
            # [Not Supported]. Guessing a number here would be worse than
            # refusing to run.
            raise UsageError(
                f"nvidia-smi reported memory.free as {free!r} for gpu {self._index}; "
                "this device does not expose per-device free memory (MIG or vGPU)"
            ) from exc
