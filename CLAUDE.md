# CLAUDE.md — comidaGPT

> **comidaGPT** (comida — "food"; sibling of the Comida Harness idea) is an open-source,
> nanochat-derived research harness for training **small food-domain language models that run on edge devices**
> and compute nutrition by **reasoning over retrieved tables**, never by reciting numbers from weights.
>
> It is a fork of [karpathy/nanochat](https://github.com/karpathy/nanochat) (upstream commit `92d63d4`, 2026-07-03,
> MIT). Everything upstream stays intact and re-mergeable; everything comidaGPT adds lives in clearly marked places (§3).
> Not to be confused with the Comida Harness (ingredient-level cooking simulation) — that is a separate project; comidaGPT may later consume its outputs as ground truth.
>
> Owner: Ashikh (SJRI / Three Piano Innovations). Side project; trains **only on locally available compute** (§8).
> Licence: MIT for code; datasets carry their own licences (§6). Companion docs: `PROGRESS.md` (you maintain it, §15),
> `docs/compute.md` (measured throughput per machine), `docs/data.md` (corpus manifest + licences).

---

## 0. Read this first — non-negotiable rules

1. **Numbers come from tables, reasoning comes from the model.** The model is trained to parse, normalise, estimate
   portions and *call a lookup tool*; the deterministic nutrient engine (`comida/nutrition/`) does the arithmetic from
   IFCT 2017 / USDA FDC rows. A model that emits "protein: 12.4 g" from its own weights is a bug, not a feature.
   (This is the RAFT / retrieval-over-memorisation principle the ADAM project already established.)
2. **Local compute only.** No cloud GPU rentals. Frontier APIs may be used for **evaluation baselines only**
   (ComidaBench), never to generate training data. Teacher traces come from open-weight models we run ourselves
   (`gpt-oss-120b` on spark, Qwen3 on the Minis).
3. **Measure before building.** Every milestone in §12 has a gate with numbers. If a gate fails, stop, write the numbers
   to `PROGRESS.md`, and ask — do not pile on features. Throughput is measured on each machine before any run is planned
   (§8.2); no compute estimate in this file is to be trusted over a measured one.
4. **Keep nanochat's shape.** One dial (`--depth`), no config objects, no factories, no provider abstractions. A new
   feature is a small file or a small diff. Prefer deleting to adding switches. Any change to `nanochat/` core must be
   principled enough to work at every depth, exactly as upstream demands.
5. **Upstream-clean history.** Fixes that belong upstream (MPS/CPU sharp edges, device guards, tests) are made on a
   branch cut from `upstream/master`, PR'd to karpathy/nanochat, and only then merged here. Upstream's AI policy applies:
   **declare any LLM-written part you do not fully understand in the PR.** Core `engine.py`/tool-loop changes are written
   by hand; Claude reviews.
6. **No participant data, ever, in this repo.** MANASI recall sheets, SODIET WhatsApp logs, any clinical images or
   records are out of scope for an open-source project. Public datasets, SJRI data with an explicit open licence, and
   staff-authored test cases only. Recipe *instructions* text is copyrightable — the released corpus stores ingredient
   lists, titles and derived fields unless the source licence permits full text (§6).
7. **Provenance on every artifact.** Every checkpoint, eval result and dataset shard records git sha, config, data
   manifest sha, machine, and wall-clock. `PROGRESS.md › Runs` is the ledger. A number without provenance does not go in
   a table.
8. **Secrets:** API keys for baseline evals live in `.env` (git-ignored). Never in code, logs, exports or wandb.
9. **Flag, don't hardcode, assumptions.** Anything marked `ASSUMPTION` is for Ashikh to verify; implement behind a
   config value and log it in `PROGRESS.md › Assumptions`.
10. **Shared machines.** catalina, spark and athena host other people's work. Never `sudo`, never touch another user's
    Ollama instance or model store, never write outside the project work dir, and on mini-04 never touch
    `/Volumes/FastStorage`. Long runs use `nice`/`CUDA_VISIBLE_DEVICES` as the machine owner allows.

---

## 1. Context that shapes the design (from project history — do not relitigate)

- A from-scratch "foundation model for food" cannot work: the entire public food corpus is ~2–5B tokens, and a
  GPT-2-grade nanochat run is ~40B tokens of mostly general text. A domain-only pretrain learns food words but not
  instruction following. So comidaGPT has **two tracks** (§2): a nanochat-native from-scratch track (the learning track and
  the paper's controlled baseline) and a continued-pretraining track on a small open base (the deployable product).
- Frontier models win on open-ended generation and world knowledge; a small tool-augmented model can **match or beat**
  them on narrow structured tasks (ingredient normalisation to IFCT codes, household-measure → grams, recipe → per-serving
  nutrients via lookup) and on determinism, offline operation and cost. Design the product around the second column.
- Earlier ADAM testing: GPT silver labels carry **5–15 % noise** on regional Indian dishes; a student inherits that
  ceiling. Treat teacher traces as *process* supervision (how to call the tool), and use **table-verified** answers as the
  reward signal, never the teacher's numbers.
- **ADS and IFCT 2017 energy fields are kJ.** Convert with 4.184 kJ/kcal in exactly one place (`comida/nutrition/units.py`).
- No public Indian recipe → weighed-nutrient dataset exists. ComidaBench's Indian split is therefore built from
  IFCT-computable recipes (ground truth = deterministic engine on a human-verified ingredient list), plus a small
  lab-measured set if SJRI can release one under an open licence (`ASSUMPTION`).

---

## 2. What comidaGPT is, in one table

| Track | Base | Where it trains | Purpose | Deliverable |
|---|---|---|---|---|
| **N** (nano) | nanochat `gpt.py` from scratch, d6–d20 | catalina (L40S), spark, athena | Learn the full stack; controlled experiments (food-mix ratio, tool-use SFT/RL, scale) | Miniseries `comidagpt-d{6,12,16,20}` + paper-grade ablations |
| **Q** (quick) | Qwen3-0.6B / 1.7B (Apache-2.0) via HF + TRL in `ext/` | catalina, athena | The model people will run | `comidagpt-0.6b`, `comidagpt-1.7b` (GGUF + MLX exports) |
| **Bench** | — | Minis (MLX), any | ComidaBench: tasks, metrics, frontier + open baselines | `bench/` + leaderboard in README |
| **Tool** | — | everywhere | Deterministic nutrient engine + lookup tool the model calls | `comida/nutrition/` (pure Python, no torch) |

Track N and Track Q share **the same tasks, the same tool, the same bench, the same post-training data**. Only the
base weights and the trainer differ. The tool-call grammar is identical, so a trace generated for one trains the other.

---

## 3. Repository layout

```
comidagpt/
├─ CLAUDE.md · PROGRESS.md · README.md · LICENSE (MIT, upstream notice kept)
├─ nanochat/                   # UPSTREAM — minimal diffs only, each marked "# comida:" and listed in docs/upstream-diff.md
│  ├─ engine.py                #   tool registry replaces the inline calculator (§5.2)
│  ├─ dataset.py               #   + food shard set and mixing ratio (§6.3)
│  └─ ...                      #   everything else untouched
├─ scripts/                    # UPSTREAM scripts untouched; comidaGPT adds:
│  ├─ comida_bench.py           #   run ComidaBench on a checkpoint, an Ollama/MLX model, or an API model
│  └─ comida_build_data.py      #   build ComidaMix shards + task parquet files from docs/data.md manifest
├─ tasks/                      # UPSTREAM Task framework; comidaGPT adds one file per task (§4)
│  ├─ food_parse.py · food_code.py · food_portion.py · nutrition_calc.py · food_mc.py
│  └─ comida_common.py          #   shared rendering of tool calls, tolerance-based evaluate()
├─ comida/                     # NEW — pure Python, no torch
│  ├─ nutrition/               #   tables, units, yield/retention factors, lookup tool, engine (§5.1)
│  └─ data/                    #   corpus filters, licence tagging, dedup, shard writer
├─ ext/                        # NEW — Track Q: HF/TRL continued-pretrain + SFT + GRPO scripts (§7). Own uv extra.
├─ bench/                      # NEW — ComidaBench task specs, baselines, metrics, report
├─ runs/                       # UPSTREAM run scripts; comidaGPT adds comida_d12.sh, comida_mix_ablation.sh, comida_q.sh
├─ export/                     # NEW — GGUF (llama.cpp) and MLX conversion + on-device smoke tests (§10)
├─ docs/                       # compute.md · data.md · upstream-diff.md · cluster.md (copied from FVE, trimmed)
├─ tests/                      # UPSTREAM tests + test_tools.py · test_nutrition.py · test_tasks_comida.py
└─ config/cluster.yaml         # machine topology (host, user, work_dir, role, runtime)
```

Working rule: if a change touches `nanochat/`, ask whether it could go upstream. If yes → rule 0.5. If no → mark it.

---

## 4. Tasks (nanochat `Task` subclasses — the heart of the project)

Each task is one file in `tasks/`, built on `load_hub_dataset` or a local parquet, with `eval_type`, `get_example` →
`Conversation`, and `evaluate(problem, completion)`. All data files are produced by `scripts/comida_build_data.py` from
sources in `docs/data.md`; nothing is hand-edited. Splits are fixed by seed and recorded with a sha.

| Task | Input → output | eval_type | Metric | Source | Role |
|---|---|---|---|---|---|
| **food_parse** | ingredient line ("2 katori toor dal, cooked") → `{qty, unit, name, prep}` JSON | generative | exact-match on fields; F1 on name span | RecipeNLG NER field, TASTEset, Indian lines authored from ADS ingredient rows | SFT |
| **food_code** | ingredient name + a retrieved shortlist of 5–10 candidate table rows → the correct code | generative | top-1 accuracy | IFCT 2017 + USDA FDC names; aliases from Wikipedia/Wiktionary (Indian languages) + ADS alias map | SFT (RAFT-style: the shortlist is in the prompt) |
| **food_portion** | household measure text (+ dish type) → grams with range | generative | MAPE; interval coverage | ICMR-NIN household measures, `household_units_v1.csv` (FVE project), USDA FNDDS portion weights | SFT |
| **nutrition_calc** | recipe (ingredient list + servings) → per-serving energy, P/F/C/fibre **via tool calls** | generative | per-nutrient MAPE, % within ±10 %, severe-failure rate (>50 %) | ground truth = deterministic engine on verified ingredient codes; recipes from RecipeNLG, FNDDS, ADS-open subset | SFT + **RL** (verifiable reward) |
| **food_mc** | multiple-choice food knowledge ("Which has more iron per 100 g?", "Which group does X belong to?") | categorical | accuracy | generated from IFCT/USDA tables, 4 options, `render_mc` | pretraining-knowledge probe; cheap, runs every eval |

Traces for `nutrition_calc` follow nanochat's GSM8K convention: the assistant turn interleaves text with tool spans,
e.g. `<|python_start|>lookup("toor dal, dry", 60)<|python_end|><|output_start|>{"kcal":200.4,...}<|output_end|>`.
`tasks/comida_common.py` renders and parses these; SFT masks tool outputs (forced tokens), exactly as the engine's
`token_masks` already distinguish sampled from forced.

**ComidaBench** = the five held-out test splits + a fixed baseline set (§9). Indian and non-Indian examples are
stratified in every split and reported separately.

---

## 5. The tool and the engine

### 5.1 `comida/nutrition/` — deterministic, pure Python, exhaustively unit-tested
- `tables.py` — loads IFCT 2017 and USDA FDC (SR Legacy + Foundation + FNDDS) into one normalised frame:
  `code, name, aliases[], group14, per_100g{energy_kcal, protein_g, fat_g, carb_g, fiber_g, + micros}`, `source`,
  `energy_basis`. kJ → kcal conversion in `units.py` only. Tables are read from `data/tables/` with sha-pinned files;
  `docs/data.md` records where each came from and its licence.
- `lookup.py` — `lookup(name, grams)` → per-portion nutrients; `search(name, k)` → shortlist (BM25 over names + aliases,
  an optional small embedding index later — `ASSUMPTION`: BM25 is enough for v1; measure recall@10 before adding dense).
- `cook.py` — yield and retention factors (FAO/INFOODS) keyed by `cooking_method`; v1 applies yield only, retention
  for vitamin C / folate / thiamine marked "upper bound" until the bench shows the bias (same discipline as seefood).
- `engine.py` — `recipe → per-serving nutrients` given coded ingredients. This produces **ground truth** for
  `nutrition_calc` and is the thing the model learns to drive, not replace.
- Group taxonomy: the 14-group IFCT-derived list from `food_groups_v1.csv` (FVE project). `ASSUMPTION`: names match ODT.

### 5.2 Tool loop in `nanochat/engine.py`
Upstream has a hard-wired calculator (`use_calculator`) triggered by `<|python_start|>…<|python_end|>`. comidaGPT
generalises it to a **registry** of ≤3 callables — `calc` (upstream behaviour, unchanged), `lookup`, `search` — chosen
by the leading identifier in the span. Diff target: < 40 lines, no new special tokens, no change to `KVCache` or
sampling, bit-identical behaviour when only `calc` is used (test: `tests/test_tools.py` reproduces upstream
`test_engine.py` outputs). Tool results are forced tokens with `mask = 0`, as today. Timeouts and the sandbox stay.

---

## 6. Data

### 6.1 ComidaMix — the pretraining/continued-pretraining corpus (target 2–5B tokens, measured not guessed)

| Source | Licence | What we keep | Est. tokens |
|---|---|---|---|
| RecipeNLG (2.2M recipes) | research use only | titles, ingredient lists, NER; **instructions only in the research shard, excluded from any commercial build** | ~0.8B |
| Open Food Facts | ODbL | product names, ingredient text, nutriments (multilingual) | ~0.5B |
| USDA FDC (all), FNDDS, FAO/INFOODS | public domain | verbalised rows ("X per 100 g contains …") | ~0.1B |
| IFCT 2017 | **confirm redistribution terms** (`ASSUMPTION`) | verbalised rows; code only if not redistributable | ~0.01B |
| Wikipedia + Wikibooks Cookbook: food/cuisine/nutrition categories, en + hi/ta/kn/ml/te | CC BY-SA | full text | ~0.4B |
| Project Gutenberg cookbooks | public domain | full text | ~0.05B |
| PubMed nutrition abstracts | abstracts OK | abstracts | ~0.3B |
| FineWeb-Edu, food-filtered (keyword + small classifier) | ODC-By | full text | 1–3B (the dial) |
| ADS recipe DB (SJRI, ~4,955 recipes / 33.5k ingredient rows) | **decide with SJRI** | only an explicitly released subset | ~0.01B |

Verbalised table rows are deliberately **small**: the model should learn *that* tables exist and how to read them, not
memorise them (rule 0.1). Dedup (MinHash) against all ComidaBench test splits before sharding; the dedup report is part
of the corpus manifest.

### 6.2 Post-training data
- SFT: the five tasks' train splits (§4), plus a slice of SmolTalk so chat ability is not forgotten (ratio from the
  M3 ablation, default 1:1 food:general).
- Teacher traces for `nutrition_calc` (how to decompose a recipe and call `lookup`): generated locally by
  `gpt-oss-120b` on spark and Qwen3-32B on catalina, **filtered by the engine** (a trace is kept only if its tool calls
  reproduce the ground truth within 2 %). No API-generated training data (rule 0.2).
- RL: `nutrition_calc` with reward from the engine (within ±10 % per nutrient → +1; formatting penalties as GSM8K).

### 6.3 Mixing into nanochat pretraining (Track N)
`nanochat/dataset.py` gains a second shard set and a `--food-mix` fraction (0, 0.25, 0.5 in the M2 ablation). Food
shards are built with the same `dev/repackage_data_reference.py` packer so the loader does not change. Tokenizer:
train **our own** BPE on the mixed stream (`scripts/tok_train`) and report compression on Indian-language food text
vs upstream's tokenizer — this is a real, cheap, first result.

---

## 7. Track Q — `ext/` (continued pretraining + post-training on an open base)

- Base: `Qwen/Qwen3-0.6B-Base` first; `Qwen3-1.7B-Base` once 0.6B clears G4. (`ASSUMPTION`: Qwen3 remains the best
  Apache-licensed sub-2B base with Indic tokenisation; re-check against Gemma 3 1B / SmolLM3 at M4 start.)
- Stages: continued pretrain on ComidaMix (full fine-tune, bf16, ~1–3B tokens) → SFT on §6.2 → GRPO on `nutrition_calc`
  (TRL). Each stage evaluated on ComidaBench; each stage's delta recorded.
- Tool calls use the same `<|python_start|>` grammar, added as special tokens to the Qwen tokenizer; the serving side
  (`export/`) implements the same registry so a GGUF/MLX model gets the same tool loop.
- `ext/` has its own `uv` extra (`--extra q`) so nanochat core stays dependency-light. Everything is single-GPU;
  no multi-node training anywhere in this project (heterogeneous GPUs on a shared LAN are not worth DDP).

---

## 8. Compute — local only

### 8.1 Roles (from `docs/cluster.md`; verify before each milestone — shared machines drift)

| Machine | Hardware | Role in comidaGPT |
|---|---|---|
| **catalina** | L40S 46 GB | Primary trainer: Track N d12–d20, Track Q continued pretrain + SFT |
| **spark** | GB10, 121 GB unified | Teacher traces (`gpt-oss-120b` already present), long low-priority runs, Track Q 1.7B if L40S is busy. `ASSUMPTION`: PyTorch CUDA aarch64 wheels + nanochat's SDPA fallback work here — verify at M0 |
| **athena** | RTX 4080 16 GB | SFT/GRPO of ≤1.7B, ComidaBench evals, export smoke tests (fp16; no bf16 issue on Ada) |
| **Mac Minis ×4** (M4, 24 GB) | MLX | Data building, teacher traces with Qwen3 via mlx-lm, bench baselines, **deployment target** (MLX export). Not for training beyond `runcpu.sh`-scale demos |
| **Galaxy S23** | Snapdragon 8 Gen 2 | Edge target for GGUF (llama.cpp / MLC) latency tests |

### 8.2 Phase 0 throughput table (G0 requirement) — fill `docs/compute.md`, never estimate
Run on each GPU: `base_train --depth=12 --num-iterations=60 --core-metric-every=-1 --sample-every=-1 --save-every=-1`
with the largest `--device-batch-size` that fits; record `tok/s`, `mfu`, peak VRAM, and the same for d16/d20 where they
fit. Plan runs from these numbers only.

Order-of-magnitude expectations to be *replaced* by measurements (L40S ≈ 180 TFLOPS dense bf16 at ~35–40 % MFU):
d12 (~1.6B tokens) ≈ 5–7 h; d16 ≈ 1 day; d20 (the old ~4e19 speedrun) ≈ 1 week single-GPU. Track Q 0.6B on 3B tokens ≈
2 days; 1.7B ≈ 5–6 days. These are feasible for a side project if runs are queued, resumable (`--resume-from-step`,
`save-every`) and tolerant of pre-emption by other users.

### 8.3 Orchestration
Reuse the FVE pattern: `config/cluster.yaml`, SSH + rsync, one process per machine, no Ray, no queue. Training runs
are launched by hand in `tmux`/`screen` with `nohup`; evals and data jobs may use the pull-dispatcher pattern if ever
needed (not before M3). Checkpoints sync back to the master Mini with `rsync` (no `--delete`).

---

## 9. ComidaBench baselines

Fixed set, run once per bench version, reported in README:
- **Open, local:** Qwen3-0.6B/1.7B/4B/8B (instruct), Gemma 3 1B/4B, gpt-oss-20b, Llama-3.2-1B/3B — zero-shot and
  with the tool loop (`scripts/comida_bench.py --runtime ollama|mlx`).
- **Frontier, API, eval only:** one OpenAI, one Anthropic, one Google model at their current default tier, zero-shot
  and tool-enabled (tools exposed as function calls). Record model id and date; costs logged.
- **Human reference:** a dietitian-computed subset (n ≥ 100) of `nutrition_calc` if SJRI colleagues can spare the time
  (`ASSUMPTION`).

Metrics follow the Nutrition5k / seefood convention: MAE, MAE %, MdAPE per nutrient; severe-failure rate (>50 %,
>100 %); interval coverage; everything stratified Indian vs non-Indian and by ingredient count; paired bootstrap
(2,000 resamples) on every model-vs-model delta. Identity tasks: accuracy with a hand-reviewed alias map, never fuzzy
auto-matching.

---

## 10. Edge deployment (`export/`)
- GGUF via llama.cpp (Q4_K_M, Q8_0) and MLX 4/8-bit; a 200-line reference runner per runtime implementing the tool
  registry; `comida/nutrition/` tables ship as a single SQLite file (< 30 MB).
- Gate metrics: ComidaBench accuracy of the quantised model within 1 pt of bf16; `nutrition_calc` end-to-end ≤ 2 s on
  the S23 and ≤ 0.5 s on a Mini for a 10-ingredient recipe; binary + tables ≤ 1.5 GB for 1.7B Q4.
- The Flutter integration (SODIET/seefood) is **out of scope** here; `export/` only proves the model and tool run
  offline. An adapter contract (`result_v1`-compatible JSON) is documented so seefood can consume it later.

---

## 11. Research questions this harness is built to answer
- **R1 (tokenizer):** how much does a food+Indic-aware BPE improve bits-per-byte on Indian food text at fixed vocab?
- **R2 (mix ratio):** at d12, what food-mix fraction maximises `food_mc` and food val-bpb without losing more than
  1 CORE point? Does the optimum move at d16/d20?
- **R3 (tool vs recall):** tool-augmented SFT vs direct-answer SFT on `nutrition_calc`, same data — size of the gap,
  and does it shrink with scale? (The core thesis.)
- **R4 (RL):** does GRPO with an engine-verified reward improve `nutrition_calc` beyond SFT, and does it transfer to
  held-out Indian recipes?
- **R5 (from-scratch vs continued):** matched post-training on `comidagpt-d16` vs `comidagpt-0.6b` — how far behind is the
  from-scratch model, and on which tasks?
- **R6 (edge):** accuracy and latency after quantisation; where does Q4 break first?

Each answer is one section of the write-up (§12 M6) and one `dev/LOG.md`-style entry.

---

## 12. Milestones and gates

| M | Deliverable | Gate (must pass before next M) |
|---|---|---|
| **M0** | Fork + scaffold (§3); `uv sync` on all machines; `runs/runcpu.sh` completes on a Mini; upstream tests green on each GPU box incl. SDPA fallback on spark; `docs/compute.md` throughput table; `docs/cluster.md` | **G0:** throughput measured on catalina, spark, athena; d12/60-iter smoke on each; one upstream PR opened (MPS/CPU sharp edge or test) |
| **M1** | `comida/nutrition/` v1 with tests; five tasks with data builders; `scripts/comida_bench.py`; baseline table for open models + 3 API models | **G1:** engine reproduces 50 hand-computed recipes within 1 %; bench runs end-to-end on a Mini and on athena; baseline table published. **Decision gate:** if a ≤2B open model already scores ≥ 90 % within ±10 % on `nutrition_calc` zero-shot, Track N narrows to R1/R2 only and Track Q becomes a thin SFT |
| **M2** | ComidaMix v1 with manifest + dedup report; comidaGPT tokenizer; d12 ablation `--food-mix ∈ {0, 0.25, 0.5}` on catalina | **G2:** R1 and R2 answered with numbers; food val-bpb improves at ≤ 1 CORE-point cost for at least one ratio; else record and keep food-mix as an option |
| **M3** | Tool registry in `engine.py` (bit-identical calc path); teacher-trace generation + engine filtering; SFT with tools on `comidagpt-d12`; `chat_rl` generalised to `nutrition_calc` | **G3:** R3 measured; tool-SFT beats direct-SFT on `nutrition_calc` MAPE (paired bootstrap p < 0.05); schema-valid rate ≥ 98 % |
| **M4** | Track Q: `comidagpt-0.6b` through CPT → SFT → GRPO in `ext/`; `comidagpt-d16` with the same post-training | **G4:** R4 and R5 measured; a model beats every ≤2B open baseline on ComidaBench-Indian `nutrition_calc` and `food_code` |
| **M5** | `export/` GGUF + MLX; runners; S23 and Mini latency; quantisation accuracy | **G5:** §10 gate metrics met for the best model |
| **M6** | Write-up: nanochat Discussion post + arXiv-style report; dataset + checkpoints released on HF with model cards; README leaderboard | Ashikh sign-off |

Working order is M0 → M1 before any training. M1 is the cheapest and highest-value step: it tells us whether the
rest is needed.

---

## 13. Tech stack
- **Core:** upstream nanochat pins (`uv sync --extra gpu` on CUDA boxes, `--extra cpu` on Minis), Python per
  `.python-version`. Do not add dependencies to the core; `comida/nutrition/` depends only on `pandas`, `pyarrow`,
  `rank_bm25`, `sqlite3`.
- **Track Q (`--extra q`):** `transformers`, `trl`, `peft`, `accelerate`, `datasets`.
- **Bench:** `ollama`/`mlx-lm` clients, `openai`/`anthropic`/`google-genai` SDKs for API baselines only.
- **Export:** `llama.cpp` (submodule or pinned release), `mlx-lm` convert.
- Pin exact versions in `uv.lock`; record in `PROGRESS.md` at M0.

---

## 14. Testing and definition of done
- `comida/nutrition/`: hand-computed fixtures (incl. kJ→kcal, yield factors, group mapping); property tests
  (totals = Σ items, no negatives); 100 % branch coverage on `units.py`.
- Tool loop: `tests/test_tools.py` proves the `calc` path is bit-identical to upstream `test_engine.py`; `lookup`
  and `search` round-trip; timeouts honoured.
- Tasks: slicing/mixture tests mirror upstream `test_tasks.py`; each task has a 20-example golden file; `evaluate()`
  tested on crafted completions (correct, off-by-tolerance, malformed).
- Bench: metric functions tested against tiny hand-computed tables; bootstrap seeded.
- Done = tests pass on a Mini (CPU) and on one CUDA box; `ruff` clean; no TODO without an owner; `PROGRESS.md`
  updated; provenance present on every new result.

---

## 15. Working agreement for the coding agent
- Keep `PROGRESS.md` with: milestone status; gate results with numbers; `Runs` ledger (run id, machine, git sha,
  config, data sha, start/end, result path); decisions (dated, one line each); assumptions and status; pinned
  versions; open questions.
- Work milestone by milestone. At each gate write the measured numbers and stop for review if it fails.
- Before proposing a long run, show the throughput-derived time estimate and which machine, and ask.
- Never launch anything on a shared box without checking `nvidia-smi` / `uptime` for other users' jobs first.
- When this file and upstream nanochat conventions conflict on *how* to write code, upstream wins; on *what* to build,
  this file wins. Note conflicts in `PROGRESS.md`.
- Prefer deleting code to adding switches. A flag exists only if the bench or a milestone needs it.

## 16. Open questions for Ashikh (track answers in PROGRESS.md)
1. IFCT 2017 redistribution: can the verbalised table (or the raw file) ship in an MIT repo / HF dataset?
2. ADS subset: is there an openly licensable slice of the 4,955 recipes, and who decides?
3. A lab-measured Indian recipe set for ComidaBench — exists? releasable?
4. Dietitian time for the human reference subset (§9)?
5. Which machine can be reserved for week-long runs (d20, 1.7B CPT) without disrupting colleagues?
6. Qwen3 vs Gemma 3 vs SmolLM3 as the Track Q base — decide at M4 start with the then-current licences.
7. Name: `comidaGPT` (confirmed 2026-10-19; `annam` rejected — already one of Ashikh's projects).
