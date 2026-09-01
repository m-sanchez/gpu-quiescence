"""Allocators for ProbeStage - including the VRAM one, shipped not described.

ProbeStage's default allocator is host memory, on purpose: it is the one
that works everywhere and it says so in the report. To probe the resource
this package is named after you need an allocator that actually touches
VRAM. That allocator lives here rather than in a README snippet, so it
carries a label, a refusal type, and a release that the driver agrees with.

    pip install "gpu-quiescence[torch]"
"""

from __future__ import annotations

from .errors import UsageError


class TorchCudaAllocator:
    """Allocate a uint8 tensor on one CUDA device, fill it, and give it back."""

    def __init__(self, torch, device: int = 0) -> None:
        self._torch = torch
        self.device = device
        self.label = f"cuda:{device}-uint8"
        # torch.cuda.OutOfMemoryError subclasses RuntimeError, not MemoryError,
        # so the stage is told the exact class rather than left to infer it.
        oom = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        self.refusal_exceptions: tuple[type[BaseException], ...] = (
            (oom,) if isinstance(oom, type) else ()
        )

    def alloc(self, nbytes: int):
        return self._torch.empty(nbytes, dtype=self._torch.uint8, device=f"cuda:{self.device}")

    def touch(self, buf) -> None:
        # One kernel over the whole tensor. Indexing per 4 KiB the way the host
        # allocator does would be ~65k-262k python-level CUDA writes to prove
        # the same thing.
        buf.fill_(1)
        # CUDA is asynchronous: without this, an allocation failure could
        # surface in some later, unrelated call instead of inside the probe.
        self._torch.cuda.synchronize(self.device)

    def release(self) -> None:
        # The tensor has already been dropped by the stage. empty_cache hands
        # the block back to the driver, so free_after_mib agrees with what
        # nvidia-smi will report a moment later.
        self._torch.cuda.empty_cache()


def torch_cuda_allocator(device: int = 0, _torch=None) -> TorchCudaAllocator:
    """Build a CUDA allocator, or explain what is missing as a usage error.

    A missing torch or a box with no CUDA device is a usage problem (exit 2),
    not a "not ready" verdict about the GPU.
    """
    torch = _torch
    if torch is None:
        try:
            import torch as imported
        except ImportError as exc:
            raise UsageError(
                "VRAM probing needs torch; install it with "
                '`pip install "gpu-quiescence[torch]"`'
            ) from exc
        torch = imported
    if not torch.cuda.is_available():
        raise UsageError("VRAM probing needs a CUDA device; torch reports none available")
    return TorchCudaAllocator(torch, device)
