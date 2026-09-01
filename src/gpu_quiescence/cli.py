"""gpu-quiescence CLI.

    gpu-quiescence --require-mib 8000 --gpu --ollama http://127.0.0.1:11434 \
        --restore-model llama3 --json -- python train.py

Exit codes: 0 ready (and job succeeded, if given) - 1 not ready or job
failed - 2 usage error.

A usage error is anything that stopped the handshake from running at all:
no nvidia-smi, a driver that will not answer, a device that does not report
free memory, an unreachable inference server, a command that does not exist.
None of those are "not ready" - they are "you configured me wrong" - so they
never share an exit code with a verdict.
"""

from __future__ import annotations

import argparse
import json
import sys

from .allocators import torch_cuda_allocator
from .core import EvictStage, Handshake, HeadroomStage, ProbeStage, SettleStage
from .errors import UsageError
from .evictors import OllamaEvictor
from .launch import run_then_restore
from .probes import NvidiaSmiProbe, SystemMemoryProbe


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gpu-quiescence", description=__doc__)
    p.add_argument("--require-mib", type=float, required=True, help="memory the job needs, in MiB")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--gpu", action="store_true", help="watch GPU memory via nvidia-smi (default)")
    src.add_argument("--system", action="store_true", help="watch system RAM instead")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--ollama", metavar="URL", help="evict models from this Ollama server first")
    p.add_argument("--restore-model", metavar="NAME", help="model to re-warm after the job")
    p.add_argument(
        "--probe-vram",
        action="store_true",
        help="probe real VRAM with a CUDA tensor instead of host memory "
        '(needs the torch extra: pip install "gpu-quiescence[torch]")',
    )
    p.add_argument("--band-mib", type=float, default=64.0, help="settle band width (default 64)")
    p.add_argument(
        "--reclaim-fraction",
        type=float,
        default=0.9,
        help="fraction of the VRAM the server said it held that must actually "
        "come back after eviction (default 0.9; 0 disables the check)",
    )
    p.add_argument("--headroom-factor", type=float, default=1.10)
    p.add_argument("--headroom-margin-mib", type=float, default=512.0)
    p.add_argument("--json", action="store_true", help="emit the readiness report as JSON")
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command to run once ready")
    return p


def _usage(message: object) -> int:
    print(f"gpu-quiescence: {message}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.probe_vram and args.system:
        return _usage("--probe-vram allocates on a CUDA device; it cannot be used with --system")
    probe = SystemMemoryProbe() if args.system else NvidiaSmiProbe(args.gpu_index)
    allocator = None
    # Preflight: a missing prerequisite is a usage error (exit 2), never a
    # "not ready" verdict (exit 1). The two must not share an exit code.
    try:
        probe.free_mib()
        if args.probe_vram:
            # --gpu-index is nvidia-smi PCI order and is used here as the CUDA
            # ordinal; on a box where those disagree, pin CUDA_VISIBLE_DEVICES.
            allocator = torch_cuda_allocator(args.gpu_index)
    except UsageError as exc:
        return _usage(exc)
    evictor = OllamaEvictor(args.ollama) if args.ollama else None

    stages = []
    if evictor:
        # Certify the reclaim in bytes only when the probe reads the same pool
        # the server reports holding; in --system mode it does not.
        stages.append(
            EvictStage(
                evictor,
                probe=None if args.system else probe,
                reclaim_fraction=args.reclaim_fraction,
            )
        )
    stages.append(SettleStage(probe, band_mib=args.band_mib))
    stages.append(ProbeStage(probe, args.require_mib, allocator=allocator))
    stages.append(
        HeadroomStage(
            probe,
            args.require_mib,
            factor=args.headroom_factor,
            margin_mib=args.headroom_margin_mib,
        )
    )

    try:
        report = Handshake(stages).run()
    except UsageError as exc:
        return _usage(exc)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        for s in report.stages:
            mark = "ok " if s.ok else "FAIL"
            print(f"{mark} {s.name:9s} {s.detail}")
    if not report.ok:
        print("not ready: the handshake gates the job", file=sys.stderr)
        return 1

    cmd = [c for c in args.cmd if c != "--"]
    if not cmd:
        return 0
    try:
        return run_then_restore(cmd, evictor, args.restore_model)
    except UsageError as exc:
        return _usage(exc)


if __name__ == "__main__":
    raise SystemExit(main())
