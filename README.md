# NGSS Middle-School Science LLM Benchmark

A reproducible pipeline that (1) builds a labeled ground-truth dataset of questions a
middle-school student might ask while learning each Next Generation Science Standard, and
(2) benchmarks multiple LLMs on how well they answer those questions — to guide educators on
model choice per science domain.

It is organized so you can add/remove models by editing one config file, run any stage on its
own or the whole pipeline at once, resume interrupted runs, and produce paper-ready analyses.

---

## Workflow at a glance

Everything below can be done from the **UI** (recommended) or the CLI. Typical order:

1. **Install** deps and create `.env` (§1).
2. **Configure** → set **API keys**, add your **models** (fetch the model list from each
   endpoint so there are no typos), and pick generator / judges / benchmark models (§2 / UI
   *Configure* page).
3. **Review the standards** and edit `data/standards.yaml` if needed (§3 / UI *Standards*).
4. **Quick generation test** — generate ~20 questions and eyeball them before committing to the
   full run (UI *Ground truth → Build → Quick generation test*).
5. **Build the ground truth** — `generate → quality → validate` (UI *Ground truth → Build →
   **Build ground truth***). Then **review & fix** any doubtful reference answers on *Ground
   truth → Review & fix* (or drop them).
6. **Run the benchmark** — `benchmark → evaluate → calibrate → analyze` for all models (UI
   *Benchmark → **Run benchmark***). This does not regenerate the ground truth.
7. *(Optional but recommended for a paper)* fill the judge **calibration** sample's `human_score`.
8. Read the **Results** dashboard / report CSVs.
9. **Human review**: resolve flagged disagreements, then re-run `analyze` (UI *Benchmark →
   analyze*) to fold the human labels into the leaderboard. *(To skip this and have the pipeline
   auto-drop every doubted question instead, enable `validation.auto_drop_disagreements` — see the
   config table.)*

Ground-truth building and benchmarking are deliberately on **separate pages**, so running the
benchmark never re-does the (slow, paid) ground-truth generation. Long runs execute in the
background with a live progress dashboard.

---

## 1. Install

```bash
pip install -r requirements.txt          # add --break-system-packages on Debian/Ubuntu if needed
cp .env.example .env                      # then put your real API keys in .env
```

`.env` holds secrets only; everything else lives in `config.yaml`. The `.env` file is
gitignored — never commit it.

Validate your setup at any time:

```bash
python -m src.config --check
```

---

## 2. Configure (`config.yaml`)

Everything is driven from one file. Key sections:

| Section | What it controls |
| --- | --- |
| `models` | The registry. Each entry = `base_url` + `api_key_env` + `model` (+ `family`). **Add a model by adding an entry**; it can then be used as a generator, judge, or benchmark target. Everything goes through the OpenAI-compatible API, so OpenAI, Anthropic, OpenRouter, Together, Groq, vLLM, Ollama, etc. all work. |
| `generator_model` | Which model writes the ground-truth Q&A. |
| `gt_judges` | The 3-model panel that validates the ground truth. |
| `validation.auto_drop_disagreements` | `false` (default) → doubted questions go to human review. `true` → questions the judges disagreed on **or** unanimously marked wrong are **auto-dropped** from the benchmark set at validation time (recorded in `results/human_review/auto_dropped_groundtruth.csv`; `ground_truth.csv` untouched). |
| `benchmark_models` | The candidates under test. **Add/remove freely.** |
| `eval_mode` | `single` or `triple` — which grading framework scores benchmark responses. |
| `eval_judge` / `eval_judges` | The grader(s) used in single / triple mode. |
| `runs_per_model`, `questions_per_area`, `question_type_split` | Dataset/run sizes. |
| `benchmark_protocol` | The **standardized** decoding applied to every benchmarked model (the `middle_school` system prompt, zero-shot, and an optional shared temperature) so the comparison is fair. By default temperature and `max_tokens` are unset (`null`): temperature is omitted so each model uses its default, and there is **no output cap** so every model answers to completion. |
| `grading.rubric_for` | Question types that also get a 0/0.5/1 rubric score. |
| `concurrency.max_parallel_requests` | Parallelism. Raise to go faster, set `1` to debug. |
| `retry` | Automatic retry with backoff on transient API errors. |
| `checkpoint` | Resume settings. |
| `appropriateness.fk_grade_range` | Middle-school readability band for filtering questions. |
| `generation.max_rounds` / `generation.overgen` | The auto-retry loop: failed questions (readability/appropriateness/dedupe) are regenerated for that exact slot for up to `max_rounds` passes; `overgen` requests extra per slot so filtering still leaves enough. |
| `defaults.temperature_*` | Per-stage temperatures. **`null` = don't send temperature** (model uses its own default). We never force `0`, so models that require/forbid a specific temperature still work. |
| `defaults.max_tokens_*` | Per-stage output caps. **`null` = no cap** so models (incl. reasoning models) answer to completion. Set a number only if you want to limit length. |
| per-model `temperature` / `omit_temperature` | In a `models` entry: force a fixed temperature, or never send one (e.g. reasoning models). Overrides the stage defaults. |

> **Pin dated model snapshots** in `model:` (e.g. `gpt-4o-2024-11-20`) for reproducibility.
>
> **Temperature & length:** by default the pipeline does not force a temperature or an output
> limit anywhere, so every model runs on its own terms; tokens are recorded as the provider's
> **actual** counts (see §6), not estimates.

To use Claude as a generator/judge/candidate, point an entry at the Anthropic
OpenAI-compatible endpoint and set the model id (e.g. `claude-opus-4-8`).

---

## 3. Review the standards (do this first)

The 12 areas (NGSS disciplinary-core-idea codes) and their standards live in the **editable**
`data/standards.yaml`. Generate a readable view and confirm/correct it before generating data:

```bash
python -m src.standards --doc      # writes docs/areas_overview.md
```

Edit `data/standards.yaml` (the source of truth) if anything is wrong, then regenerate the doc.

---

## 4. Optional: the UI

A Streamlit control panel wraps everything (config, runs, results, human review):

```bash
python3 -m streamlit run app.py     # most robust; `streamlit run app.py` also works if it's on PATH
```

Then open the URL it prints (default **http://localhost:8501**) in your browser — it may not
auto-open. Anonymous Streamlit telemetry is already disabled via `.streamlit/config.toml`.

Pages:
- **Configure** — everything is editable here, no code required, across tabs:
  - *Status & settings* — run settings and role assignments (writes `config.yaml` back **with
    comments preserved**).
  - *Models* — add/edit/remove registry entries (base_url, model id, `api_key_env`, family,
    temperature/omit) in a table. Or use **"Add a model — fetch the model list from an
    endpoint"**: enter the base_url + pick a saved key, click *Fetch*, and choose the exact
    model id from a dropdown (no typos).
  - *API keys* — view/set the keys in `.env` (masked), with referenced/missing detection.
  - *Prompts* — edit the system prompts and every prompt template (saved to `data/prompts.yaml`),
    with placeholder validation so you can't break the `$variables`.
  - *Raw config* — full `config.yaml` editor for the long tail, validated on save.
- **Standards** — review/edit the 12 areas and regenerate the overview doc.
- **Ground truth** — two tabs:
  - *Build* — run **generate → quality → validate** (the **Build ground truth** button runs
    just these, never benchmarking), plus individual steps and the quick-generation / mock
    smoke tests.
  - *Review & fix* — surfaces items the judge panel doubted (disagreements *or* a unanimous
    "incorrect"), lets you correct an inaccurate question or reference answer, or drop an item —
    writing back to `data/ground_truth.csv` (the gold answers models are graded against) and
    keeping the validated file in sync (`fk_grade` recomputed for edited questions).
- **Benchmark** — runs **benchmark → evaluate → calibrate → analyze** for **all**
  `benchmark_models`. The **Run benchmark** button does *not* regenerate the ground truth.
- **Results** — leaderboard (with a chart) and every report CSV, downloadable.

  Runs launch as **detached background jobs** (on both the Ground truth and Benchmark pages),
  so they keep going even if you close the tab, log out, or lose the connection. A live
  dashboard (auto-refreshes every few seconds) shows **per-stage progress bars** for whichever
  stages the run includes (computed from the checkpoint files), elapsed time, a running **token
  meter** and **API-error count**, and a tail of the log — plus a **Stop** button. Reopen the
  app any time to see current status; a finished/crashed run is detected via a heartbeat.

  From a terminal you can check the same progress without the UI:
  ```bash
  python -m src.progress      # per-stage % complete + tokens/errors so far
  ```
- **Human review** — edit the disagreement-flag and calibration sheets in-place (`human_score`),
  save, and recompute calibration/analysis.

The UI is optional — every action it triggers is just a CLI command you can also run directly,
as below.

## 5. Run the pipeline (CLI)

**Quick test first** (offline, no keys, no cost — verifies wiring end-to-end):

```bash
python scripts/smoke_test.py --mock
```

Then a small **live** smoke test against your real endpoint:

```bash
python scripts/smoke_test.py
```

**Whole pipeline:**

```bash
python -m src.run_pipeline --stage all
```

**A group of stages** (the UI's split-page buttons use these):

```bash
python -m src.run_pipeline --stage groundtruth     # generate, quality, validate
python -m src.run_pipeline --stage benchmark_all    # benchmark, evaluate, calibrate, analyze
python -m src.run_pipeline --stage benchmark,evaluate,analyze   # or any comma-separated list
```

**One stage at a time** (each reads the previous stage's CSVs from disk):

```bash
python -m src.run_pipeline --stage generate     # or: python -m src.generate_dataset
python -m src.run_pipeline --stage quality
python -m src.run_pipeline --stage validate
python -m src.run_pipeline --stage benchmark
python -m src.run_pipeline --stage evaluate
python -m src.run_pipeline --stage calibrate
python -m src.run_pipeline --stage analyze
```

Stages and order: **generate → quality → validate → benchmark → evaluate → calibrate → analyze.**

### Resuming
Every stage checkpoints incrementally. If a run is interrupted (Ctrl-C, crash, network drop),
**re-run the same command** and it skips completed items and finishes the rest. Use `--fresh`
to ignore checkpoints and recompute.

### Adding models later (without redoing existing work)
Results are stored **one file per model** (`results/responses/<model>.csv`,
`results/eval/<model>_<mode>.csv`), and `analyze` builds the leaderboard by reading **every**
per-model file on disk — not the config list. So adding models is **purely additive**: new models
slot into your existing summary tables and nothing already collected is re-run, re-billed, or
overwritten. To do it:

1. Add the model to the `models:` registry (and its key to `.env`), then add it to
   `benchmark_models`.
2. Benchmark **only the new model(s)** — in the UI, use the per-model **▶ Run** buttons on the
   Benchmark page; on the CLI, `python -m src.benchmark --models "New Model"` then
   `python -m src.evaluate --models "New Model"`.
3. Re-run **`analyze`** once (`python -m src.run_pipeline --stage analyze`, or the *analyze*
   button) to rebuild the leaderboard/tables with old **and** new models.

Three things to avoid, or old results won't line up:

- **Don't pass `--fresh`** when adding models — it clears checkpoints and re-fetches. (It now
  moves the old file to a `_backups/` folder rather than deleting it, but you still don't want to
  re-run existing models.)
- **Keep `eval_mode` the same** across runs. `analyze` only reads eval files for the *current*
  mode (`*_single.csv` vs `*_triple.csv`); mixing modes hides whichever doesn't match.
- **Don't regenerate the ground truth** between runs — that changes `question_id`s, so old and new
  responses would no longer be on the same questions. Keep the same `ground_truth_validated.csv`.

---

## 6. The human-review workflow

Two CSVs ask for human input. Both have an empty `human_score` column (`1` = correct,
`0` = incorrect) and a `human_notes` column for you to fill in.

1. **Disagreement queues** — where the judge panel did not agree:
   - `results/human_review/flags_groundtruth.csv` (from `validate`)
   - `results/human_review/flags_eval_<model>.csv` (from `evaluate`, triple mode)

   Fill these to resolve the authoritative label for hard cases. When you re-run `analyze`,
   these labels are applied: a filled `human_score` in `flags_eval_<model>.csv` **replaces**
   that response's machine `binary_score`, and any question marked `human_score = 0` in
   `flags_groundtruth.csv` (the reference answer itself is wrong) is **dropped** from the
   benchmark so models aren't scored against a bad item.

   To **fix** an inaccurate reference answer instead of dropping it, use the UI's **Ground
   truth** page (or edit `data/ground_truth.csv` directly) — it writes the corrected gold answer
   back so the item stays in the benchmark.

2. **Judge calibration sheet** — a stratified, representative sample (including items the
   judges agreed on):
   - `results/calibration/to_label.csv` (from `calibrate`)

   Fill `human_score` for each row.

Then re-run calibration + analysis to fold the human labels in:

```bash
python -m src.calibrate_judges        # computes results/calibration/judge_vs_human.csv
python -m src.analyze
```

`judge_vs_human.csv` reports, **per judge**, the human↔LLM agreement (accuracy, Cohen's κ,
false-correct/false-incorrect rates) — this is how you validate the judges themselves.

---

## 7. Outputs

### Data
| File | Contents |
| --- | --- |
| `data/ground_truth.csv` | `question_id, area, standard, question_type, question, reference_answer, generator_model, fk_grade` |
| `data/ground_truth_validated.csv` | the above + one `judge_<model>` column per ground-truth judge, `majority_label`, `human_review_flag` (in auto-drop mode, doubted rows are excluded here) |
| `results/human_review/auto_dropped_groundtruth.csv` | *(auto-drop mode only)* the doubted questions removed from the validated set, with judge verdicts/rationales and a `drop_reason` (`judge_disagreement` / `unanimous_incorrect`) |

### Per model
| File | Contents |
| --- | --- |
| `results/responses/<model>.csv` | one row per (question, run): `response`, `model_id`, `finish_reason`, `is_error`, `response_fk_grade`, `response_reading_ease`, and **actual** token counts (`prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `total_tokens`) |
| `results/eval/<model>_<mode>.csv` | `binary_score` (0/1), `rubric_score` (0/0.5/1 for conceptual/applied), `is_refusal`, per-judge columns (triple), `human_review_flag` |

### Reports (`results/reports/`)
| File | Contents |
| --- | --- |
| `leaderboard.csv` | per model: **macro** (equal-weight per area, primary) and **micro** accuracy, rubric, refusal rate, mean response reading level, mean total/reasoning tokens, rank |
| `token_usage_by_model.csv` / `_by_area.csv` / `_by_type.csv` | **actual** token analytics per model (× area / type): mean + SD of input, output, reasoning, and total tokens, plus `total_sum` (cost) and `reasoning_available_frac` |
| `accuracy_by_area.csv` / `_by_type.csv` / `_by_area_type.csv` / `_by_standard.csv` | accuracy + **Wilson 95% CIs** + run-to-run std + rubric + refusal rate |
| `reading_level.csv` | response Flesch-Kincaid aggregated over runs, by area and question type, vs the target band |
| `item_stats.csv` | per-item fraction-correct, discrimination, and trivial/broken flags (does the set separate models?) |
| `significance.csv` | pairwise **McNemar** tests with **Holm-Bonferroni** correction |
| `judge_reliability.csv` | Fleiss' κ + raw agreement for each judge panel |
| `self_preference.csv` | (triple mode) each judge's score delta on same-family vs other-family responses |
| `dataset_quality.csv` | per-area counts, standard coverage, readability vs band |
| `generation_rejects.csv` | questions dropped during generation, with reasons |

### Docs
- `docs/areas_overview.md` — reviewable areas/standards table (generated).
- `docs/DECISIONS.md` — every methodological choice and its rationale.
- `docs/DATASHEET.md` — dataset card (`python scripts/make_datasheet.py`).

---

## 8. Tests

```bash
python -m pytest tests/ -q
```

Runs the offline mock pipeline and asserts output schemas and value domains, plus unit tests
for dedupe and the kappa helpers.

---

## 9. Limitations (read before drawing conclusions)

This benchmark measures **answer correctness only** — not pedagogical quality, age-appropriate
explanation, hallucination rate, or safety. Questions are LLM-generated and may favor LLMs;
consider augmenting with human/textbook-sourced items. See `docs/DECISIONS.md` and. 

## 10. License

The pipeline code in this repository is licensed under MIT — see `LICENSE`.
