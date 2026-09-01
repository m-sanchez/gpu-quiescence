# gpu-quiescence

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Dependencies](https://img.shields.io/badge/dependencies-0-B45309)
[![CI](https://github.com/m-sanchez/gpu-quiescence/actions/workflows/test.yml/badge.svg)](https://github.com/m-sanchez/gpu-quiescence/actions/workflows/test.yml)
![Ollama](https://img.shields.io/badge/evictor-Ollama-6E6E6E)
![License](https://img.shields.io/badge/license-MIT-6E6E6E)
[![PyPI](https://img.shields.io/pypi/v/gpu-quiescence?color=3775A9&logo=pypi&logoColor=white)](https://pypi.org/project/gpu-quiescence/)

> **In plain English:** this checks a GPU really has enough free memory and has settled before you launch a big job, so it does not crash halfway through.

Train on the box you serve from: a VRAM readiness preflight for machines
that serve and train on one GPU.

[More tools](https://github.com/m-sanchez) · [Working rules](https://miguelsanchez.co.uk/ethics)

*Provenance: this came out of one body of production LLM work, extracted and
generalised into a standalone package. First published 2026-08-31.*

One machine, two jobs: an inference server holding models warm, and a
training run that needs the memory back. Start the trainer cold and it OOMs,
thrashes, or silently degrades serving. gpu-quiescence runs a preflight
first, and records precisely what was tested - including which resource
each stage actually touched.

```
evict → settle → probe → headroom → launch (reduced priority) → restore
```

- **Evict**: ask the inference server to release its models. Ships an
  Ollama evictor; anything with `evict / settled / restore` plugs in. The
  stage owns the wait: `settled()` is polled against a deadline, because
  eviction is asynchronous on real servers.
- **Settle**: free memory (VRAM in GPU mode, via `nvidia-smi`) must sit
  inside a variation band before any reading is trusted. A reclaim still
  in progress shows up as a wide band and keeps the stage waiting. A
  timeout too short to ever succeed is rejected at construction.
- **Probe**: allocate, touch, and release one representative buffer.
  **Honesty first: the default allocator is host memory** (`bytearray`).
  In GPU mode the VRAM readiness signal comes from settle and headroom
  reading real VRAM numbers; the probe proves the host side, and the
  report names its allocator so the record says exactly what was tested.
  To probe VRAM itself, pass an allocator that touches it (for example
  `lambda n: torch.empty(n, dtype=torch.uint8, device='cuda')`) with its
  own label. Probe outcomes are three-valued: `succeeded`, `refused` (the
  allocator said no - the expected failure), or `errored` (the allocator
  broke). Both failures gate the launch; the report never disguises one
  as the other. And success claims only itself: not "memory is
  contiguous", not "training cannot OOM".
- **Headroom**: `free >= required * 1.10 + 512 MiB`, both knobs
  configurable. The numbers go in the report either way.
- **Launch + restore**: the job runs as a reduced-priority child, and the
  inference model is re-warmed after it exits, success or not. The box goes
  back to serving either way.

Every stage produces a report entry with its observations; a failed stage
gates the job instead of raising, and an empty handshake fails because
nothing was tested. `--json` emits the whole readiness report.

## Install

```bash
pip install gpu-quiescence

# gate a fine-tune on 8 GiB of GPU memory, evicting Ollama first
gpu-quiescence --require-mib 8192 --gpu \
  --ollama http://127.0.0.1:11434 --restore-model llama3 \
  -- python train.py
```

Also installable from a pinned git tag: `pip install
"git+https://github.com/m-sanchez/gpu-quiescence@v2.0.0"`. CI proves the
built wheel installs, imports, and exposes the command. No required runtime
dependencies; `psutil` is an optional fallback probe.

Exit codes: `0` ready (and the job succeeded, if one was given) · `1` not
ready, or the job failed · `2` usage error - a missing `nvidia-smi` is a
usage problem, never a "not ready" verdict.

Python 3.10+, zero runtime dependencies. System-RAM mode (`--system`)
reads `/proc/meminfo` on Linux and `GlobalMemoryStatusEx` on Windows;
other platforms need the optional extra
(`pip install "gpu-quiescence[fallback]"`, which brings psutil). GPU mode
parses `nvidia-smi`. Develop with `pip install -e . pytest` and `pytest`.

## Library use

```python
from gpu_quiescence import Handshake, SettleStage, ProbeStage, HeadroomStage, NvidiaSmiProbe

probe = NvidiaSmiProbe()
report = Handshake([
    SettleStage(probe),
    ProbeStage(probe, required_mib=8192),
    HeadroomStage(probe, required_mib=8192),
]).run()

report.ok           # gate decision
report.to_dict()    # what was tested, stage by stage, with observations
```

## The tests are the point

| Test | Claim |
| :-- | :-- |
| settle waits through a moving reclaim, times out on drift | a wide band means "not yet", not "fail fast" |
| probe refused and probe errored are distinct outcomes | the report never disguises a bug as an OOM |
| the probe names its allocator | the record says which resource was actually tested |
| an empty handshake fails | nothing tested is not ready |
| an impossible settle timeout is rejected at construction | a stage that can only fail is a lie |
| the evict stage polls to its own deadline | eviction is asynchronous, and the stage owns the wait |
| probe success text never says "contiguous" | the claim is exactly what the probe established |
| probe failure is a gated report, not an exception | the handshake is a gate; it reports, it does not crash |
| headroom failure carries the numbers | a refusal you can act on |
| restoration runs even when the job fails | the box always goes back to serving |
