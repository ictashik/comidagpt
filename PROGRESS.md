# PROGRESS.md — comidaGPT

Maintained by the coding agent per `CLAUDE.md §15`. Newest entries first within each section.

## Milestone status

| M | Status | Gate | Result |
|---|---|---|---|
| M0 | not started | G0 | — |
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

## Pinned versions

(fill at M0 from `uv.lock`)

## Open questions

See `CLAUDE.md §16`. Answers recorded here with dates.
