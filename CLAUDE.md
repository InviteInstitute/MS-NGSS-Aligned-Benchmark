# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A reproducible pipeline that (1) builds a labeled ground-truth dataset of questions a middle-school
student might ask while learning each NGSS science standard, and (2) benchmarks multiple LLMs on how
well they answer those questions. The `README.md` is the authoritative user-facing guide; this file
is the orientation for changing the code.

## Commands

```bash
pip install -r requirements.txt          # add --break-system-packages on Debian/Ubuntu if needed
python -m src.config --check             # validate config.yaml + .env, print resolved roles

python scripts/smoke_test.py --mock      # OFFLINE end-to-end wiring check (no keys, no cost)
python scripts/smoke_test.py             # same, but LIVE against your endpoint
python -m pytest tests/ -q               # runs the mock pipeline + asserts output schemas
python -m pytest tests/test_pipeline.py::test_dedupe_drops_near_duplicates   # single test

python -m src.run_pipeline --stage all                 # whole pipeline
python -m src.run_pipeline --stage groundtruth         # generate,quality,validate
python -m src.run_pipeline --stage benchmark_all       # benchmark,evaluate,calibrate,analyze
python -m src.run_pipeline --stage benchmark,evaluate  # any comma-separated subset, or one stage
python -m src.run_pipeline --stage all --fresh         # ignore checkpoints, recompute

python -m src.progress                   # per-stage % + tokens/errors of the running job
python3 -m streamlit run app.py          # the optional UI control panel (localhost:8501)
```

There is no lint/format tooling configured. Every stage module is also runnable standalone
(`python -m src.benchmark --fresh --models "GPT 5.4"`, `python -m src.standards --doc`, etc.).

## Architecture

**Stage pipeline.** Seven stages run in a fixed order, each a module under `src/` with a top-level
function and a `main()`:

    generate → quality → validate → benchmark → evaluate → calibrate → analyze

`src/run_pipeline.py` is the orchestrator: `STAGE_ORDER`, the named `GROUPS` (`all`, `groundtruth`,
`benchmark_all`), and `_NEEDS_KEYS` (which stages require API keys) all live there. Stages do **not**
pass data in memory — each reads the prior stage's CSV from disk (paths from `config.yaml` `paths:`)
and writes its own. This is why any stage/subset can be run independently and why the ground-truth
build (slow, paid) is deliberately separate from benchmarking.

**Config is the single source of control.** `config.yaml` (with comments preserved on UI save) holds
everything; `.env` holds only secrets, referenced by name via each model's `api_key_env`.
`src/config.py` `Config.load()` parses both, resolves every model *key* referenced by a role
(`generator_model`, `gt_judges`, `benchmark_models`, `eval_judge`/`eval_judges`) into a
`ResolvedModel`, and **fails fast** if a referenced model is missing from the registry or its key is
absent. Adding a provider/model is purely a config edit — no code change.

**All model calls go through one client.** `src/llm_client.py` `LLMClient.complete()` wraps the
OpenAI-compatible Chat Completions API (so OpenAI, Anthropic, OpenRouter, vLLM, Ollama, etc. all
work via `base_url`+`api_key`). It **never raises** on API/model errors — it returns
`LLMResult(ok=False, ...)` so one bad call can't kill a stage. Token counts are the provider's
**actual** `usage` numbers, not estimates.

**Temperature / max_tokens semantics (important, load-bearing).** `None` means *do not send the
parameter*, so the model uses its own default and reasoning models answer to completion. Never force
`temperature=0` or a token cap by default. Precedence for temperature: per-model `omit_temperature`
→ per-model fixed `temperature` → the caller's requested value. Use `to_temp()`/`to_int()` from
`src/utils.py` to coerce config values (blank/`null` → `None`).

**Concurrency + checkpointing.** `src/concurrency.py` `parallel_map()` fans per-item work across a
bounded thread pool (I/O-bound network calls; results keep input order) with an `on_result` hook for
incremental checkpointing. `src/utils.py` `CheckpointWriter` is an append-only, resumable CSV writer
keyed by id column(s): it loads existing rows, exposes `pending()`/`done_keys` so re-running skips
finished work, and flushes every N rows. `--fresh` deletes the checkpoint first. **New stage work
should follow this pattern** (task list → `writer.pending()` → `parallel_map` with `writer.add`) so
runs stay resumable.

**Prompts are data.** `src/prompts.py` holds code defaults for system prompts, question-type
definitions, and templates, overridable by `data/prompts.yaml` (written by the UI). Templates use
`string.Template` `$name` placeholders (not `.format`) so literal JSON braces don't need escaping.
`benchmark_protocol.system_prompt` must be a key in `prompts.SYSTEM_PROMPTS` (validated at config load).

**Background jobs + UI.** `app.py` is a Streamlit control panel and is optional — every button just
runs a CLI command. Long runs launch as **detached** processes (`src/jobs.py` → `scripts/run_job.py`,
`start_new_session=True`) surviving the UI session; state is JSON in `results/_runs/` with a
heartbeat, and `src/progress.py` derives per-stage progress from checkpoint files. Nothing in `src/`
imports Streamlit.

**Data source.** The 12 NGSS areas + standards live in the editable `data/standards.yaml` (the source
of truth); `src/standards.py --doc` regenerates the human-reviewable `docs/areas_overview.md`.

**Additive report generators.** `src/teacher_report.py` and `src/academic_report.py` sit outside
`STAGE_ORDER` (not wired into `run_pipeline.py`) and are purely additive: each only writes its own
new `results/reports/*` file(s) and touches no pipeline CSV. They recompute from the eval/response
files by importing private (`_`-prefixed) helpers straight out of `src/analyze.py`
(`_load_eval`, `_apply_human_overrides`, `_leaderboard`, etc.) rather than duplicating that logic, and
`academic_report.py` further reuses `teacher_report.py`'s helpers. Both are covered by
`tests/test_pipeline.py` — so a signature change to one of those private helpers in `analyze.py` can
silently break both report generators; grep for the helper name before renaming it.

## Conventions when editing

- Keep stages disk-decoupled: read prior CSV via `read_csv`, write via `CheckpointWriter`, honor
  `--fresh`. Don't wire stages together in memory.
- Task functions passed to `parallel_map` should be total (return a row, not raise) — the LLM client
  already turns API failures into `ok=False` rows; keep errors *in the data* (e.g. `is_error`) rather
  than dropping rows, so refusals/errors remain scorable downstream.
- Add a model by adding a `models:` registry entry (`base_url`, `api_key_env`, `model`, `family`,
  optional `temperature`/`omit_temperature`); reference it by key in a role. Pin dated snapshots for
  reproducibility.
- When changing an output CSV's columns, update `tests/test_pipeline.py` (it asserts schemas and
  value domains against the mock run) and the Outputs tables in `README.md`.

## Docs to consult

- `README.md` — full workflow, config reference, output-file schemas, human-review process.
- `docs/DECISIONS.md` — every methodological choice and its rationale.
- `docs/DATASHEET.md` — dataset card (regenerate via `python scripts/make_datasheet.py`).
- This benchmark measures **answer correctness only**; questions are LLM-generated (may favor LLMs).
  See the Limitations section of the README and the licensing note before any public release.
