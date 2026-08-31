# gpu-quiescence

Train on the box you serve from: an allocation-readiness handshake for
shared VRAM.

[More tools](https://github.com/m-sanchez) · [Working rules](https://miguelsanchez.co.uk/ethics)

One machine, two jobs: an inference server holding models warm, and a
training run that needs the memory back. Start the trainer cold and it OOMs,
thrashes, or silently degrades serving. gpu-quiescence runs a handshake
first — and records precisely what was tested.

```
evict → settle → probe → headroom → launch (reduced priority) → restore
```

- **Evict** — ask the inference server to release its models. Ships an
  Ollama evictor; anything with `evict / settled / restore` plugs in.
- **Settle** — free memory must sit inside a variation band (spread of the
  last five samples under 64 MiB) before any reading is trusted. A reclaim
  still in progress shows up as a wide band and keeps the stage waiting.
- **Probe** — allocate, touch, and release one representative buffer (25%
  of the requested footprint, clamped to 128 MiB–1 GiB). Success claims
  exactly what happened: *an allocation of N MiB succeeded at that moment*.
  Not "memory is contiguous", not "training cannot OOM" — allocator
  behaviour is backend-specific and a probe cannot promise the future.
- **Headroom** — `free ≥ required × 1.10 + 512 MiB`, both knobs
  configurable. The numbers go in the report either way.
- **Launch + restore** — the job runs as a reduced-priority child, and the
  inference model is re-warmed after it exits, success or not. The box goes
  back to serving either way.

Every stage produces a report entry with its observations; a failed stage
gates the job instead of raising. `--json` emits the whole readiness report.

## Run

```bash
pip install -e . && pytest

# gate a fine-tune on 8 GiB of GPU memory, evicting Ollama first
gpu-quiescence --require-mib 8192 --gpu \
  --ollama http://127.0.0.1:11434 --restore-model llama3 \
  -- python train.py
```

Exit codes: `0` ready (and the job succeeded, if one was given) · `1` not
ready, or the job failed · `2` usage error.

Python 3.10+, zero dependencies. System-RAM mode (`--system`) reads
`/proc/meminfo` on Linux and `GlobalMemoryStatusEx` on Windows; GPU mode
parses `nvidia-smi`.

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
| probe success text never says "contiguous" | the claim is exactly what the probe established |
| probe failure is a gated report, not an exception | the handshake is a gate; it reports, it does not crash |
| headroom failure carries the numbers | a refusal you can act on |
| restoration runs even when the job fails | the box always goes back to serving |
