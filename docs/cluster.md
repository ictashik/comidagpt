# Cluster

Adapted from the FVE project's cluster doc (state of 2026-09-03/04). Facts here drift: re-verify before planning
a run. `config/cluster.yaml` is the machine-readable twin; keep them in sync.

**No IPs, usernames, passwords or key paths in this repo.** They live in `~/.ssh/config` on the master. Use aliases.

## Topology

| Alias | Role in comidaGPT | Hardware | Free disk (FVE snapshot) |
|---|---|---|---|
| mini-05 | **master** (this machine): orchestration, data building, bench, checkpoint/result sink | M4, 24 GB | 94 GB |
| mini-02, mini-03 | workers: data building, teacher traces (MLX), bench baselines | M4, 24 GB | 37 / 122 GB |
| mini-04 | worker, same as above | M4, 24 GB | ~14 GB (tight) |
| catalina | Track N/Q primary trainer | L40S 46 GB, 503 GB host RAM | 193 GB |
| spark | teacher traces, long low-priority runs | GB10, 121 GB unified, Ubuntu aarch64 | 3.2 TB |
| athena | SFT/GRPO ≤1.7B, evals, export smoke tests | RTX 4080 16 GB, 62 GB host RAM | 1.5 TB |

**mini-04 has an external drive at `/Volumes/FastStorage`. Never touch it** (read, write, mount, enumerate `/Volumes`).
Any cluster-wide disk/mount scan must exclude mini-04 explicitly.

All machines are shared personal machines with other users' work on them (load 1-2 at idle is normal). Never
`sudo`, never touch other users' Ollama instances or model stores, stay inside the work dir.

## Access

- Master reaches workers by key auth through aliases in `~/.ssh/config` (`ssh catalina`, `rsync ... spark:~/comidagpt/`).
- Install the master's public key once per machine with `ssh-copy-id <alias>` (interactive password, never scripted).
- Verify: `for h in mini-02 mini-03 mini-04 catalina spark athena; do ssh -n -o BatchMode=yes -o ConnectTimeout=5 $h 'echo OK $(hostname)'; done`
- Use `ssh -n` for every ssh call inside a shell loop, or it eats the loop's stdin and silently stops after one host.
- Non-interactive SSH on the Minis does not have `brew` on PATH; use full paths (`/opt/homebrew/bin/...`).
- Orchestration is SSH + rsync, one process per machine, no Ray, no queue (CLAUDE.md §8.3). Checkpoints rsync to the master without `--delete`.

## Machine quirks (from FVE; check each before M0 runs)

- **catalina:** no `python3-venv`, no sudo, and the conda env created there had a broken stdlib path. `uv` installs its own
  Python without root, so try `uv sync --extra gpu` there first. Pre-existing Ollama models and a production Ollama belong to others.
- **athena:** NVIDIA driver was broken until `dkms` was installed (fixed by its owner). First CUDA call after a driver
  or service restart can take ~40 s (kernel JIT); do not treat a slow first step as a failure.
- **spark:** PyTorch aarch64 CUDA wheels are unverified (Assumption A1). MLX does not exist on this arch. The existing
  `gpt-oss:120b` sits in a shared Ollama instance: using it for teacher traces needs the owner's OK (see PROGRESS.md open questions).
- **Minis:** system `python3` may be 3.9; `uv` manages the interpreter so this should not matter. Install `uv` per machine.
- Shared machines show load from others: a slow worker is not a broken one, and a throughput number measured under load
  is a lower bound; record `uptime` next to every `docs/compute.md` row.
