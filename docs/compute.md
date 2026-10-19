# Measured throughput (G0)

Command (CLAUDE.md §8.2): `base_train --depth=12 --num-iterations=60 --core-metric-every=-1 --sample-every=-1 --save-every=-1`
with the largest `--device-batch-size` that fits. Measured values only; no estimates.

| machine | GPU | depth | device-batch-size | tok/s | mfu | peak VRAM | git sha | date |
|---|---|---|---|---|---|---|---|---|
| catalina | L40S 46GB | 12 | 32 | ~145k (median of last 40 steps; range 128k-150k) | ~30% (bf16) | 28.8 GB | 154fde5 | 2026-10-19 |

Notes (catalina, 2026-10-19): `--run=dummy`, 60 iterations, 31.5M tokens, 1 GPU; val bpb 1.644 after 60 steps (untrained, not a result).
Host load average was 5 at launch and rose to 21 during the run (other users' CPU jobs), GPU itself was idle before; treat the
throughput as a lower bound. Batch 32 used 28.8 GB, so 64 would not fit in 46 GB (not tried). Training took 2.96 min.
Planning figure (CLAUDE.md §8.2 says d12 ~1.6B tokens): 1.6e9 / 145k tok/s ~ 3.1 h (CLAUDE.md guessed 5-7 h). Replace with the actual
token count from `base_train` at run time. A final checkpoint was written to `~/.cache/nanochat/base_checkpoints` on catalina despite `--save-every=-1`.
| athena | RTX 4080 16GB | 12 | 8 | ~92.7k (median of last 40 steps) | n/a (repo has no peak-flops entry for the 4080) | 9.4 GB | 154fde5 | 2026-10-19 |

Notes (athena, 2026-10-19): batch 16 OOM'd (16 GB card); 8 is the largest power of two that fits. 60 steps took 4.62 min, val bpb 1.646 after 60 steps
(untrained, not a result). Host idle (load ~0.2). Driver 595.99.02 after the owner's reload.
| spark | GB10 121GB unified | 12 | 32 | ~23.9k (median of last 40 steps, steady) | n/a (no peak-flops entry) | 28.8 GB | 154fde5 | 2026-10-19 |

Notes (spark, 2026-10-19): needs torch 2.9.1+**cu130** (cu128 gives 11 TFLOPS bf16 matmul, cu130 gives 62), `TRITON_PTXAS_PATH=/usr/local/cuda-13/bin/ptxas`,
and the FA3 arch guard (SDPA path used). Run with `--eval-tokens=524288`: the default step-0 + final validation passes (21M tokens each) take longer
than the 60 training steps at this speed. 60 steps took 17.94 min; val bpb 1.627 after 60 steps (untrained, not a result). Host idle, nothing else on the GPU.
Planning figure: d12 at ~1.6B tokens would take ~18.6 h here (vs ~3.1 h on catalina): spark is for teacher traces and low-priority runs, not for d12+ pretraining.

## G0 throughput summary (d12, 60 it, 2026-10-19)

| machine | dbs | tok/s | vs catalina |
|---|---|---|---|
| catalina (L40S) | 32 | ~145k | 1.00x |
| athena (RTX 4080) | 8 | ~92.7k | 0.64x |
| spark (GB10) | 32 | ~23.9k | 0.16x |

