"""Allocation-readiness handshake for boxes that serve and train on one GPU."""

from .allocators import torch_cuda_allocator
from .core import (
    EvictStage,
    Handshake,
    HeadroomStage,
    ProbeStage,
    ReadinessReport,
    SettleStage,
    StageReport,
)
from .errors import UsageError
from .evictors import LoadedModel, OllamaEvictor
from .launch import launch_low_priority, run_then_restore
from .probes import NvidiaSmiProbe, SystemMemoryProbe

__all__ = [
    "EvictStage",
    "Handshake",
    "HeadroomStage",
    "LoadedModel",
    "NvidiaSmiProbe",
    "OllamaEvictor",
    "ProbeStage",
    "ReadinessReport",
    "SettleStage",
    "StageReport",
    "SystemMemoryProbe",
    "UsageError",
    "launch_low_priority",
    "run_then_restore",
    "torch_cuda_allocator",
]
