# CLAIMS

Every externally falsifiable claim in `README.md` and in the package
description, mapped to the executable test that enforces it. A claim with no
test is a claim you are asked to take on trust; this file exists so there are
none of those.

Decorative prose (the plain-English intro line, the provenance note, badge
images, links) is out of scope: those are not checkable behavioural claims.

Run everything with `pytest -q`.

## The handshake

| Claim | Where it is made | Enforced by |
| :-- | :-- | :-- |
| A VRAM readiness preflight runs evict → settle → probe → headroom and reports every stage | description; README pipeline diagram | `tests/test_handshake.py::test_full_green_path_reports_every_stage` |
| The first failing stage gates the rest | "a failed stage gates the job" | `tests/test_handshake.py::test_handshake_stops_at_first_failure` |
| A failed stage gates the job instead of raising | "a failed stage gates the job instead of raising" | `tests/test_handshake.py::test_handshake_captures_stage_exceptions_as_failures`, `tests/test_handshake.py::test_probe_failure_is_a_gated_report_not_an_exception` |
| An empty handshake fails, because nothing was tested | "an empty handshake fails because nothing was tested" | `tests/test_handshake.py::test_empty_handshake_fails_because_nothing_was_tested` |

## Evict

| Claim | Where it is made | Enforced by |
| :-- | :-- | :-- |
| Anything with `evict / settled / restore` plugs in | "anything with `evict / settled / restore` plugs in" | `tests/test_evict_reclaim.py::test_a_reclaim_check_is_skipped_when_the_stage_has_no_probe`, `tests/test_handshake.py::test_evict_stage_polls_until_settled_or_deadline` |
| `settled()` is a predicate; the stage owns the only deadline | "`settled()` is a predicate and the stage owns the only deadline" | `tests/test_evict_reclaim.py::test_settled_asks_the_server_once_and_answers`, `tests/test_evict_reclaim.py::test_a_stuck_server_gates_after_the_stages_own_deadline` |
| The report states the time it really waited, and counts every question asked | "states the time it really waited and counts every question it asked" | `tests/test_evict_reclaim.py::test_the_evict_report_states_the_time_it_actually_waited`, `tests/test_evict_reclaim.py::test_a_stuck_server_gates_after_the_stages_own_deadline` |
| `/api/ps` says how much VRAM each model holds | "`/api/ps` says how much VRAM each model holds" | `tests/test_evict_reclaim.py::test_loaded_models_keep_the_vram_the_server_says_they_hold` |
| Free memory must rise by ≥90% of what the server said it held, or the job is gated | "the reclaim is certified in bytes" | `tests/test_evict_reclaim.py::test_a_completed_reclaim_is_certified_in_bytes`, `tests/test_evict_reclaim.py::test_a_reclaim_short_of_what_the_server_held_gates_the_job`, `tests/test_evict_reclaim.py::test_vram_that_never_comes_back_gates_the_job` |
| `--reclaim-fraction 0` switches the check off; no probe means no byte claim | "`--reclaim-fraction`, `0` to switch the check off" | `tests/test_evict_reclaim.py::test_a_reclaim_check_is_skipped_when_the_stage_has_no_probe` |

## Settle

| Claim | Where it is made | Enforced by |
| :-- | :-- | :-- |
| A reading is trusted only once it sits inside a variation band | "must sit inside a variation band before any reading is trusted" | `tests/test_handshake.py::test_settle_passes_when_readings_sit_in_the_band` |
| A reclaim in progress shows as a wide band and keeps the stage waiting | "shows up as a wide band and keeps the stage waiting" | `tests/test_handshake.py::test_settle_keeps_waiting_while_memory_is_still_moving`, `tests/test_handshake.py::test_settle_times_out_when_memory_never_stops_moving` |
| A timeout too short to ever succeed is rejected at construction | "rejected at construction" | `tests/test_handshake.py::test_settle_rejects_a_timeout_that_can_never_succeed` |

## Probe

| Claim | Where it is made | Enforced by |
| :-- | :-- | :-- |
| The default allocator is host memory, and the report names it | "the default allocator is host memory (`bytearray`)" | `tests/test_handshake.py::test_probe_reports_its_allocator_so_the_record_says_what_was_tested` |
| The probe buffer is a clamped fraction of the requested size | pipeline description, "one representative buffer" | `tests/test_handshake.py::test_probe_size_is_proportional_and_clamped` |
| `--probe-vram` allocates and fills a uint8 tensor on the CUDA device | "`--probe-vram` … allocates and fills a uint8 tensor on the CUDA device" | `tests/test_allocators.py::test_the_vram_allocator_allocates_on_cuda_and_names_itself`, `tests/test_allocators.py::test_the_vram_allocator_runs_against_real_torch_when_present` (runs when torch + CUDA are present) |
| Outcomes are three-valued and both failures gate | "`succeeded`, `refused` … or `errored`"; "Both failures gate the launch" | `tests/test_handshake.py::test_probe_distinguishes_refused_from_errored_and_both_gate` |
| A CUDA OOM is a refusal, not an allocator bug | "A CUDA OOM counts as a refusal even though `torch.cuda.OutOfMemoryError` is not a `MemoryError`" | `tests/test_handshake.py::test_a_cuda_style_oom_is_a_refusal_not_an_allocator_bug`, `tests/test_handshake.py::test_an_oom_message_from_an_unnamed_exception_is_still_a_refusal`, `tests/test_allocators.py::test_the_vram_allocator_reports_a_cuda_oom_as_refused_through_the_stage` |
| A real allocator bug stays `errored` — the report never disguises one as the other | "the report never disguises one as the other" | `tests/test_handshake.py::test_a_real_allocator_bug_is_still_errored` |
| `refusal_exceptions=` teaches the stage about any other allocator | "`refusal_exceptions=` teaches the stage about any other allocator" | `tests/test_handshake.py::test_refusal_exceptions_are_configurable` |
| Success claims only itself — never "contiguous" | "success claims only itself" | `tests/test_handshake.py::test_probe_success_claims_only_its_own_allocation` |

## Headroom

| Claim | Where it is made | Enforced by |
| :-- | :-- | :-- |
| `free >= required * 1.10 + 512 MiB`, both knobs configurable | "**Headroom**: `free >= required * 1.10 + 512 MiB`" | `tests/test_handshake.py::test_headroom_uses_factor_and_margin` |
| The numbers go in the report either way | "The numbers go in the report either way" | `tests/test_handshake.py::test_headroom_reports_numbers_on_failure` |

## Launch and restore

| Claim | Where it is made | Enforced by |
| :-- | :-- | :-- |
| The job runs as a reduced-priority child | "the job runs as a reduced-priority child" | `tests/test_probes_and_launch.py::test_the_job_runs_as_a_reduced_priority_child` |
| The model is re-warmed after the job exits, success or not | "re-warmed after it exits, success or not" | `tests/test_probes_and_launch.py::test_job_runs_and_restoration_happens_even_on_failure` |

## The record

| Claim | Where it is made | Enforced by |
| :-- | :-- | :-- |
| Every observation carries the resource it measured | "records precisely what was tested - including which resource each stage actually touched" | `tests/test_handshake.py::test_every_observation_names_the_resource_it_measured`, `tests/test_probes_and_launch.py::test_probes_name_the_resource_they_read` |
| The envelope names the tool and version that produced the report | "the envelope names the tool and version that produced it" | `tests/test_handshake.py::test_the_report_envelope_identifies_the_tool_that_produced_it` |
| `--json` emits the whole readiness report | "`--json` emits the whole readiness report" | `tests/test_cli.py::test_json_is_parseable_on_both_the_ready_and_the_gated_path` |

## The exit-code contract

| Claim | Where it is made | Enforced by |
| :-- | :-- | :-- |
| `0` = ready, with no command given | "`0` ready" | `tests/test_cli.py::test_a_green_handshake_with_no_command_is_ready` |
| `0` … "and the job succeeded" — the child's code is the exit code | "`0` ready (and the job succeeded, if one was given)" | `tests/test_cli.py::test_a_green_handshake_runs_the_job_and_returns_its_exit_code` |
| `1` = not ready, and the job does not run | "`1` not ready" | `tests/test_cli.py::test_short_headroom_gates_the_job_with_exit_one` |
| `2` = no `nvidia-smi` | "no `nvidia-smi`" | `tests/test_cli.py::test_a_missing_nvidia_smi_is_a_usage_error` |
| `2` = a driver that will not answer | "a driver that will not answer" | `tests/test_cli.py::test_a_failing_nvidia_smi_is_a_usage_error_not_a_traceback` |
| `2` = a MIG or vGPU device reporting `[N/A]` | "a MIG or vGPU device that reports `[N/A]`" | `tests/test_cli.py::test_unreadable_nvidia_smi_output_is_a_usage_error` |
| `2` = an unreachable `--ollama` URL | "an unreachable `--ollama` URL" | `tests/test_cli.py::test_an_unreachable_ollama_is_a_usage_error_not_a_not_ready_verdict` |
| `2` = a command that does not exist | "a command that does not exist" | `tests/test_cli.py::test_a_missing_job_binary_is_a_usage_error_not_a_traceback` |
| `2` = `--probe-vram` without torch, or against `--system` | "`--probe-vram` needs `pip install "gpu-quiescence[torch]"`" | `tests/test_cli.py::test_probe_vram_without_torch_is_a_usage_error`, `tests/test_cli.py::test_probe_vram_contradicts_system_mode`, `tests/test_allocators.py::test_the_vram_allocator_asks_for_torch_as_a_usage_error_not_an_import_crash` |
| A usage error never shares an exit code with a verdict | "it never shares an exit code with a verdict" | the whole of `tests/test_cli.py` — every `2` row above is a case that used to be `1` or a traceback |

## Platform and packaging

| Claim | Where it is made | Enforced by |
| :-- | :-- | :-- |
| Zero required runtime dependencies; psutil and torch are optional extras | "zero runtime dependencies"; "`psutil` and `torch` are optional extras" | `tests/test_packaging.py::test_the_package_declares_no_required_runtime_dependencies` (runs against installed metadata) |
| GPU mode parses `nvidia-smi`, including multi-GPU output | "GPU mode parses `nvidia-smi`" | `tests/test_probes_and_launch.py::test_nvidia_probe_parses_multi_gpu_output`, `tests/test_probes_and_launch.py::test_nvidia_probe_rejects_out_of_range_index` |
| `--system` reads `/proc/meminfo` on Linux and `GlobalMemoryStatusEx` on Windows | "reads `/proc/meminfo` on Linux and `GlobalMemoryStatusEx` on Windows" | `tests/test_probes_and_launch.py::test_system_probe_reads_this_platform`, executed on both by the `.github/workflows/test.yml` OS matrix |
| CI runs the suite on Linux and Windows | "CI runs the suite on both" | `.github/workflows/test.yml` (`os: [ubuntu-latest, windows-latest]`) |
| Python 3.10+ | "Python 3.10+" | `.github/workflows/test.yml` (`python: ['3.10', '3.13']`) |
| CI proves the built wheel installs, imports, and exposes the command | "CI proves the built wheel installs, imports, and exposes the command" | `.github/workflows/test.yml` → the `install proof` step |
| Releases are built and published from a tag, with attestations | "published from a tag by GitHub Actions using PyPI trusted publishing, with PEP 740 attestations" | `.github/workflows/release.yml` — tag/version guard, then `pypa/gh-action-pypi-publish` with `attestations: true` and `id-token: write` |

## Not enforced by a test

| Claim | Why not |
| :-- | :-- |
| `pip install "git+…@v3.0.0"` installs a pinned tag | Requires network and a pushed tag. The closest enforcement is the release workflow's guard that a `v*` tag matches `project.version` in `pyproject.toml`, so a tag can never name a version it does not contain. |
