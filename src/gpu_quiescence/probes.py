"""Memory probes: system RAM and NVIDIA GPU free memory, zero dependencies."""

from __future__ import annotations

import subprocess
import sys


class UsageError(RuntimeError):
    """A missing prerequisite is a usage problem, not a 'not ready' verdict."""


class SystemMemoryProbe:
    """Free system RAM in MiB, without third-party packages where possible."""

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
    """Free VRAM in MiB for one GPU, read through nvidia-smi."""

    def __init__(self, gpu_index: int = 0, _run=subprocess.run) -> None:
        self._index = gpu_index
        self._run = _run

    def free_mib(self) -> float:
        try:
            result = self._run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        except FileNotFoundError as exc:
            raise UsageError("nvidia-smi is not installed or not on PATH; GPU mode needs it") from exc
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if self._index >= len(lines):
            raise RuntimeError(f"gpu index {self._index} out of range; {len(lines)} gpu(s) reported")
        return float(lines[self._index])
