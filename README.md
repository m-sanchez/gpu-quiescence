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
each stage actually touched. Every observation carries the resource it
measured, and the envelope names the tool and version that produced it, so an
archived `--json` record identifies itself.

```
evict → settle → probe → headroom → launch (reduced priority) → restore
```

- **Evict**: ask the inference server to release its models, then check
  that the memory actually came back. Ships an Ollama evictor; anything with
  `evict / settled / restore` plugs in. `settled()` is a predicate and the
  stage owns the only deadline, so the report states the time it really
  waited and counts every question it asked. `/api/ps` says how much VRAM
  each model holds, so the reclaim is certified in bytes: free memory must
  rise by at least 90% of what the server said it was holding
  (`--reclaim-fraction`, `0` to switch the check off). "The server says it
  unloaded" is a claim; the delta is a measurement.
- **Settle**: free memory (VRAM in GPU mode, via `nvidia-smi`) must sit
  inside a variation band before any reading is trusted. A reclaim still
  in progress shows up as a wide band and keeps the stage waiting. A
  timeout too short to ever succeed is rejected at construction.
- **Probe**: allocate, touch, and release one representative buffer.
  **Honesty first: the default allocator is host memory** (`bytearray`).
  In GPU mode the VRAM readiness signal comes from settle and headroom
  reading real VRAM numbers; the probe proves the host side, and the
  report names its allocator so the record says exactly what was tested.
  To probe VRAM itself, pass `--probe-vram` (library:
  `ProbeStage(..., allocator=torch_cuda_allocator())`), which allocates and
  fills a uint8 tensor on the CUDA device and needs the `torch` extra.
  Probe outcomes are three-valued: `succeeded`, `refused` (the allocator
  said no - the expected failure), or `errored` (the allocator broke). A
  CUDA OOM counts as a refusal even though `torch.cuda.OutOfMemoryError` is
  not a `MemoryError`; `refusal_exceptions=` teaches the stage about any
  other allocator. Both failures gate the launch; the report never disguises
  one as the other. And success claims only itself: not "memory is
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
"git+https://github.com/m-sanchez/gpu-quiescence@v3.0.0"`. CI proves the
built wheel installs, imports, and exposes the command; releases are built
and published from a tag by GitHub Actions using PyPI trusted publishing,
with PEP 740 attestations, so the wheel can be verified rather than trusted.
No required runtime dependencies; `psutil` and `torch` are optional extras.

Exit codes: `0` ready (and the job succeeded, if one was given) · `1` not
ready, or the job failed · `2` usage error. A usage error is anything that
stopped the handshake from running at all - no `nvidia-smi`, a driver that
will not answer, a MIG or vGPU device that reports `[N/A]` for free memory,
an unreachable `--ollama` URL, a command that does not exist - and it never
shares an exit code with a verdict. Every row of that contract has a test
against `main()`.

Python 3.10+, zero runtime dependencies. System-RAM mode (`--system`) reads
`/proc/meminfo` on Linux and `GlobalMemoryStatusEx` on Windows, and CI runs
the suite on both; other platforms need the optional extra
(`pip install "gpu-quiescence[fallback]"`, which brings psutil). GPU mode
parses `nvidia-smi`; `--probe-vram` needs
`pip install "gpu-quiescence[torch]"`. Develop with `pip install -e . pytest`
and `pytest`.

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
report.to_dict()    # envelope + what was tested, stage by stage
```

`EvictStage(evictor, probe=probe)` certifies the reclaim in bytes;
`ProbeStage(probe, required_mib, allocator=torch_cuda_allocator())` probes
VRAM instead of host memory.

## The tests are the point

Every externally checkable claim on this page is mapped to the test that
enforces it in [CLAIMS.md](CLAIMS.md).

| Test | Claim |
| :-- | :-- |
| settle waits through a moving reclaim, times out on drift | a wide band means "not yet", not "fail fast" |
| a CUDA-shaped OOM is `refused`, an allocator bug is `errored` | the report never disguises a bug as an OOM, on the allocator that touches VRAM |
| the probe names its allocator, every stage names its resource | the record says which resource was actually tested |
| a reclaim short of the bytes the server held gates the job | "it says it unloaded" is not "the memory came back" |
| the evict report states the time it actually waited | a diagnostic with the wrong number is worse than none |
| an empty handshake fails | nothing tested is not ready |
| an impossible settle timeout is rejected at construction | a stage that can only fail is a lie |
| the evict stage owns the only deadline | eviction is asynchronous, and one wait has one owner |
| probe success text never says "contiguous" | the claim is exactly what the probe established |
| probe failure is a gated report, not an exception | the handshake is a gate; it reports, it does not crash |
| a driver error, a MIG `[N/A]`, a bad URL and a missing binary all exit 2 | a usage error is never a "not ready" verdict |
| headroom failure carries the numbers | a refusal you can act on |
| restoration runs even when the job fails | the box always goes back to serving |
