# PROGRESS.md — comidaGPT

Maintained by the coding agent per `CLAUDE.md §15`. Newest entries first within each section.

## Milestone status

| M | Status | Gate | Result |
|---|---|---|---|
| M0 | in progress | G0 | mini-05: env + upstream tests done (43 pass, 14 skip, 1 fail); throughput, runcpu, spark SDPA, upstream PR pending |
| M1 | not started | G1 | — |
| M2 | not started | G2 | — |
| M3 | not started | G3 | — |
| M4 | not started | G4 | — |
| M5 | not started | G5 | — |
| M6 | not started | sign-off | — |

## Runs ledger

| run_id | machine | git sha | config | data sha | start | end | result |
|---|---|---|---|---|---|---|---|

## Decisions

- 2026-10-19 — **mini-05 (hostname catnode05.local) is the project master.** Cluster = mini-01/02/03, catalina, spark, athena (+ mini-05). mini-04 is on another assignment: excluded. Topology in `config/cluster.yaml` (aliases only); IPs/usernames only in `~/.ssh/config`, never in git. 

- 2026-10-19 — M0 started on `mini-05` (M4 Mini, 24 GB). Scaffolded §3 layout (`comida/`, `ext/`, `bench/`, `export/`, `docs/`, `config/`). `uv.lock` rewrites from newer uv (revision 3→5) are reverted, not committed.
- 2026-10-19 — Upstream sharp edge found: `tests/test_execution.py::test_memory_limit` fails on macOS because `nanochat/execution.py:50` skips RLIMIT on darwin but the test has no darwin skip. Candidate for the G0 upstream PR (branch from `upstream/master`).

- 2026-10-19 — Project named `comidaGPT`; forked from karpathy/nanochat @ `92d63d4` (2026-07-03). Two tracks (N: from-scratch nanochat; Q: continued pretraining of a small open base). Local compute only. MIT.
- 2026-10-19 — Numbers come from tables via a tool call; the model never emits nutrient values from weights (RAFT principle carried over from ADAM).
- 2026-10-19 — Frontier APIs permitted for ComidaBench baselines only, never for training data.
- 2026-10-19 — No participant data (MANASI/SODIET) in this repository, ever.

## Assumptions (verify, then mark)

| # | Assumption | Where | Status |
|---|---|---|---|
| A1 | PyTorch CUDA aarch64 wheels + nanochat SDPA fallback work on spark (GB10) | §8.1 | open |
| A2 | BM25 over names+aliases gives recall@10 ≥ 0.95 for `search` | §5.1 | open |
| A3 | IFCT 14-group names match ODT's | §5.1 | open |
| A4 | IFCT 2017 is redistributable in verbalised form | §6.1 | open |
| A5 | Qwen3-0.6B/1.7B-Base is the Track Q base | §7 | open — decide at M4 |
| A6 | SJRI can release a lab-measured Indian recipe set / dietitian reference subset | §1, §9 | open |

## Cluster readiness (read-only check, 2026-10-19)

Key auth to all six workers works (dedicated key `id_comidagpt`, master mini-05). Nothing installed or changed on them.

| Machine | Load / sessions | Free disk | GPU | uv | Verdict |
|---|---|---|---|---|---|
| mini-01 | 1.3 / 5 users | **10 GB (95% used)** | - | no | too full for model/data work; needs cleanup first |
| mini-02 | 0.9 / 0 | 126 GB | - | yes | ready |
| mini-03 | 0.9 / 3 | 58 GB | - | yes | ready |
| catalina | 1.0 (falling from 4) / 5 | 473 GB | L40S idle (15 MiB used, 0 %) | no | ready for the d12 throughput run |
| spark | 1.1 / 4 | 3.2 TB | GB10 **94 % busy, 67 GB held by another user's process** | no | busy; do not launch here |
| athena | 0.0 / 0 | 1.5 TB | **NVML driver/library mismatch** (library 595.99) | no | GPU unusable until driver reload/reboot (needs sudo: owner) |

System Python is 3.9 on the Minis and 3.12 on the Linux boxes; `uv` manages the interpreter, so this is fine.

## Pinned versions

uv 0.13.1; torch 2.9.1 (cpu extra) on mini-05; Python 3.14.4 system / `.python-version` 3.10 for venv. Full pins: `uv.lock`.

## Open questions

- 2026-10-19 — mini-01 needs its address added to `~/.ssh/config` (not in the FVE doc).
- 2026-10-19 — spark's `gpt-oss:120b` lives in a shared Ollama instance: is using it for teacher traces OK with its owner (rule 0.10)?
- 2026-10-19 — Master's key is not yet installed on workers (`ssh-copy-id` needed, interactive).

See `CLAUDE.md §16`. Answers recorded here with dates.
