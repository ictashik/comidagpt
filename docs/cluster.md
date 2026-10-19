# Cluster

Adapted from the FVE project's cluster doc (state of 2026-09-03/04). Facts here drift: re-verify before planning
a run. `config/cluster.yaml` is the machine-readable twin; keep them in sync.

**No IPs, usernames, passwords or key paths in this repo.** They live in `~/.ssh/config` on the master. Use aliases.

## Topology

| Alias | Role in comidaGPT | Hardware | Free disk (FVE snapshot) |
|---|---|---|---|
| mini-05 | **master** (this machine): orchestration, data building, bench, checkpoint/result sink | M4, 24 GB | 94 GB |
| mini-01, mini-02, mini-03 | workers: data building, teacher traces (MLX), bench baselines | M4, 24 GB | mini-02 37 GB, mini-03 122 GB, mini-01 unknown |
| catalina | Track N/Q primary trainer | L40S 46 GB, 503 GB host RAM | 193 GB |
| spark | teacher traces, long low-priority runs | GB10, 121 GB unified, Ubuntu aarch64 | 3.2 TB |
| athena | SFT/GRPO ≤1.7B, evals, export smoke tests | RTX 4080 16 GB, 62 GB host RAM | 1.5 TB |

**mini-04 is on another assignment and is not part of this project.** Do not connect to it. (It also has an external
drive at `/Volumes/FastStorage` that must never be touched.)

All machines are shared personal machines with other users' work on them (load 1-2 at idle is normal). Never
`sudo`, never touch other users' Ollama instances or model stores, stay inside the work dir.

## Access

- Master reaches workers by key auth through aliases in `~/.ssh/config` (`ssh catalina`, `rsync ... spark:~/comidagpt/`).
- Install the master's public key once per machine with `ssh-copy-id <alias>` (interactive password, never scripted).
- Verify: `for h in mini-01 mini-02 mini-03 catalina spark athena; do ssh -n -o BatchMode=yes -o ConnectTimeout=5 $h 'echo OK $(hostname)'; done`
- Use `ssh -n` for every ssh call inside a shell loop, or it eats the loop's stdin and silently stops after one host.
- Non-interactive SSH on the Minis does not have `brew` on PATH; use full paths (`/opt/homebrew/bin/...`).
- Orchestration is SSH + rsync, one process per machine, no Ray, no queue (CLAUDE.md §8.3). Checkpoints rsync to the master without `--delete`.

## Machine quirks (from FVE; check each before M0 runs)

- **catalina:** no `python3-venv`, no sudo, and the conda env created there had a broken stdlib path. `uv` installs its own
  Python without root, so try `uv sync --extra gpu` there first. Pre-existing Ollama models and a production Ollama belong to others.
- **athena:** NVIDIA driver was broken until `dkms` was installed (fixed by its owner). First CUDA call after a driver
  or service restart can take ~40 s (kernel JIT); do not treat a slow first step as a failure.
- **spark (GB10, sm_121):** verified 2026-10-19. Needs three things, none of which `uv sync --extra gpu` gives you:
  1. **torch cu130, not cu128.** The pinned `torch==2.9.1+cu128` has no native sm_121 kernels: bf16 matmul 11 TFLOPS vs 62 TFLOPS with cu130.
     After `uv sync --extra gpu --group dev`, overlay: `uv pip install --python .venv/bin/python --reinstall-package torch torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130`.
     A later `uv sync` reverts it; redo the overlay.
  2. **`export TRITON_PTXAS_PATH=/usr/local/cuda-13/bin/ptxas`** (system CUDA 13.0, already installed). Triton's bundled ptxas rejects `sm_121a` during `torch.compile` of the training graph.
  3. **The FA3 arch guard** in `nanochat/flash_attention.py` (see `docs/upstream-diff.md`); without it the first attention call aborts the process.
  Also: `nvidia-smi` utilization on this chip is unreliable (showed 96% with nothing running); judge busy-ness by `--query-compute-apps` and `ps`.
  Unified memory shows `[N/A]` for per-GPU used memory. The existing `gpt-oss:120b` sits in a shared Ollama instance: using it for teacher traces needs the owner's OK (see PROGRESS.md open questions).
  `gpt-oss:120b` sits in a shared Ollama instance: using it for teacher traces needs the owner's OK (see PROGRESS.md open questions).
- **Minis:** system `python3` may be 3.9; `uv` manages the interpreter so this should not matter. Install `uv` per machine. If `uv` cannot download its interpreter (mini-03 timed out twice), rsync `~/.local/share/uv/python/cpython-3.10.*` from another Mini.
- **Gotchas:** `pkill -f <pattern>` over SSH matches the SSH command's own shell and kills the session; kill by PID. In zsh, write `"${h}:path"`, never `$h:path` (`:p` is a modifier). macOS `rsync` is openrsync; check with `ls` on the far side. A `base_train` run does a full validation pass at step 0; on slow GPUs pass `--eval-tokens=524288` or it looks hung.
- Shared machines show load from others: a slow worker is not a broken one, and a throughput number measured under load
  is a lower bound; record `uptime` next to every `docs/compute.md` row.
