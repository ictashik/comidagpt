# Upstream diff ledger

Every change under `nanochat/` is marked `# comida:` in source and listed here (CLAUDE.md §3).
Base: karpathy/nanochat @ `92d63d4`.

| File | Change | Upstreamable? | PR |
|---|---|---|---|
| `nanochat/flash_attention.py`, `tests/test_attention_fallback.py` | Only try FA3 when `major in (8, 9)`; regression test with fake hub. Fixes hard abort on GB10 (sm_121) | **Yes** | branch `upstream-fixes` @ `646adcb`, PR not yet opened. Carried on master unmarked so it merges cleanly when upstream lands it (deviation from "merge after PR" in CLAUDE.md rule 0.5: spark can't run without it) |
