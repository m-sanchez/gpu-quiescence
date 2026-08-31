"""gpu-quiescence CLI.

    gpu-quiescence --require-mib 8000 --gpu --ollama http://127.0.0.1:11434 \
        --restore-model llama3 --json -- python train.py

Exit codes: 0 ready (and job succeeded, if given) - 1 not ready or job
failed - 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import sys

from .core import EvictStage, Handshake, HeadroomStage, ProbeStage, SettleStage
from .evictors import OllamaEvictor
from .launch import run_then_restore
from .probes import NvidiaSmiProbe, SystemMemoryProbe, UsageError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gpu-quiescence", description=__doc__)
    p.add_argument("--require-mib", type=float, required=True, help="memory the job needs, in MiB")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--gpu", action="store_true", help="watch GPU memory via nvidia-smi (default)")
    src.add_argument("--system", action="store_true", help="watch system RAM instead")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--ollama", metavar="URL", help="evict models from this Ollama server first")
    p.add_argument("--restore-model", metavar="NAME", help="model to re-warm after the job")
    p.add_argument("--band-mib", type=float, default=64.0, help="settle band width (default 64)")
    p.add_argument("--headroom-factor", type=float, default=1.10)
    p.add_argument("--headroom-margin-mib", type=float, default=512.0)
    p.add_argument("--json", action="store_true", help="emit the readiness report as JSON")
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command to run once ready")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    probe = SystemMemoryProbe() if args.system else NvidiaSmiProbe(args.gpu_index)
    # Preflight: a missing prerequisite is a usage error (exit 2), never a
    # "not ready" verdict (exit 1). The two must not share an exit code.
    try:
        probe.free_mib()
    except UsageError as exc:
        print(f"gpu-quiescence: {exc}", file=sys.stderr)
        return 2
    evictor = OllamaEvictor(args.ollama) if args.ollama else None

    stages = []
    if evictor:
        stages.append(EvictStage(evictor))
    stages.append(SettleStage(probe, band_mib=args.band_mib))
    stages.append(ProbeStage(probe, args.require_mib))
    stages.append(
        HeadroomStage(
            probe,
            args.require_mib,
            factor=args.headroom_factor,
            margin_mib=args.headroom_margin_mib,
        )
    )

    report = Handshake(stages).run()
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
    return run_then_restore(cmd, evictor, args.restore_model)


if __name__ == "__main__":
    raise SystemExit(main())
