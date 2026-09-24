# Decisions & Rationale

A step-by-step record of the methodological choices behind this pipeline, so the method is
auditable and paper-ready. Each entry: **what** we did and **why**.

---

## Scope & framing

### 1. Areas = the 12 NGSS disciplinary-core-idea (DCI) codes
**What:** The top-level grouping is the 12 DCI codes — `MS-PS1, MS-PS2, MS-PS3, MS-PS4,
MS-LS1, MS-LS2, MS-LS3, MS-LS4, MS-ESS1, MS-ESS2, MS-ESS3, MS-ETS1`. Each area's 100 questions
are spread across that area's performance-expectation standards (e.g. `MS-PS3-1 … MS-PS3-5`).
**Why:** Matches the user's intended granularity and the `ms-ps3` example, and gives exactly 12
clean, non-overlapping areas. The source PDF is arranged into 16 *topics*, but topics share DCI
codes (e.g. "Structure and Properties of Matter" and "Chemical Reactions" are both MS-PS1);
rolling up to DCI codes avoids double-counting and ambiguous membership.

### 2. 100 questions/area, even 3-way type split
**What:** Per area, ~34 `recall`, ~33 `conceptual`, ~33 `applied`, distributed across the
area's standards. `question_type` is a labeled column.
**Why:** A balanced mix tests recall, understanding, and transfer rather than rote facts alone;
the even split lets us report performance **by question type** (a user requirement) without one
type dominating. Standards-level distribution seeks coverage of every performance expectation,
not just the easy ones (but a standard that can't be filled under the quality gates yields to its
siblings so the area still meets quota — see 9a).

### 3. Goal is educator guidance, not a single score
**What:** Outputs are broken down by area and question type, with reading-level and refusal
signals alongside accuracy.
**Why:** "Which model for my domain?" needs domain-level and behavior-level detail, not one
leaderboard number.

---

## Architecture

### 4. OpenAI-compatible-only client
**What:** Every model is queried through the OpenAI Chat Completions API via `base_url` +
`api_key` + `model`. Adding/removing a model is a `config.yaml` edit.
**Why:** One code path covers nearly every hosted and local provider (OpenAI, Anthropic's
compatible endpoint, OpenRouter, Together, Groq, vLLM, Ollama). Maximum flexibility, minimum
provider-specific code — directly serves "I'm not sure how many models I'll use."

### 5. One config file + `.env` for secrets
**What:** `config.yaml` holds the registry, endpoints, run settings, and all toggles; `.env`
(gitignored) holds keys, referenced by name.
**Why:** Single source of configuration as requested, while keeping secrets out of any file you
might share or commit.

### 6. Prompts isolated in `src/prompts.py`
**What:** All prompt text lives in one module.
**Why:** Prompts are the most-tuned and most-scrutinized artifact in an LLM study; centralizing
them keeps them reviewable and versionable.

---

## Dataset construction

### 7. Generation grounded in verbatim standard text
**What:** The generator is given each standard's exact performance-expectation text from
`data/standards.yaml`.
**Why:** Keeps questions standard-aligned and on-topic; the YAML is human-editable so content
can be corrected without touching code.

### 8. Embedding-free dedupe
**What:** Near-duplicate questions are detected with `rapidfuzz` token-set similarity on
normalized text (configurable threshold), not embeddings.
**Why:** The user asked not to use an embedding model. Lexical similarity is transparent,
dependency-light, deterministic, and adequate for catching reworded duplicates.

### 9. Level-appropriateness filter at generation time
**What:** Each candidate question must (a) fall inside a middle-school Flesch-Kincaid band and
(b) pass an LLM grade-appropriateness gate before acceptance; rejects are logged. Short cells
are backfilled over multiple rounds.
**Why:** Off-level questions (too advanced/trivial/off-topic) would invalidate a middle-school
benchmark. Filtering at generation (not just auditing after) keeps the released set clean.

### 9a. Quota is met by redistributing stuck slots, not by relaxing quality
**What:** Generation targets are planned per `(area, standard, question_type)` slot and filled
over up to `generation.max_rounds` passes. A slot that makes no progress for
`generation.stuck_patience` consecutive rounds — because the gates in #8/#9 keep rejecting its
candidates (e.g. only so many distinct in-band questions exist for one narrow standard) — is
declared **stuck**: it is capped at what it has, and its remaining need is **redistributed to
sibling standards in the same area and question type**. The area therefore still reaches its
per-type quota (34/33/33, see #2). The quality gates are never loosened to hit a number; the run
finishes short *only* if **every** standard for an area+type is exhausted, and it then logs the
exact shortfall. (This replaced an earlier heuristic that stopped the whole run at the first
zero-accept round, which is why early runs landed a little under 1,200.)
**Why:** The user's requirement is to reach the full quota (1,200 = 100 × 12) without diluting
question quality. Redistribution preserves both the area-level target and the readability/dedupe
standards; the only concession is that a genuinely hard standard contributes fewer questions and
its siblings more, so the *per-standard* balance within an area+type can be uneven. That is an
accepted, logged trade-off — preferable to either silently falling short or relaxing the
appropriateness/dedupe gates to force-fill a stubborn slot.

### 10. Short, factual reference answers
**What:** Reference answers are 1–3 sentences.
**Why:** Concise single-answer references make 0/1 grading reliable and reduce judge ambiguity.

---

## Ground-truth validation

### 11. Independent 3-judge panel + disagreement flagging
**What:** Three judges score each pair 0/1, blind to one another (judge temperature is
configurable and left unset by default — see 20a); the majority label is recorded and
non-unanimous rows are flagged for human review.
**Why:** A single judge is an unvalidated instrument. Independent diverse judges reduce
correlated error; disagreement is a cheap, high-precision signal for where humans should look.

### 11a. Optional auto-drop of doubted questions (`validation.auto_drop_disagreements`)
**What:** A config toggle (default **off**). When **off**, questions the panel *doubted* — judges
disagree, or all judges mark the reference answer wrong — are flagged for human review and remain
in the benchmark set until a human resolves them (the original behavior). When **on**, those
doubted questions are **automatically dropped** from the validated (benchmark) set at validation
time; the removed rows (with each judge's verdict, rationale, and a `drop_reason` of
`judge_disagreement` or `unanimous_incorrect`) are written to
`results/human_review/auto_dropped_groundtruth.csv`. `data/ground_truth.csv` is never modified,
and validation warns if auto-drop would empty an entire area.
**Why:** In practice, human review of disagreements almost always *ends in a drop* — the reviewer
confirms the item is ambiguous or the reference answer is wrong. For a large set that manual pass
is expensive busywork. Auto-drop makes the "drop it" outcome the default while (a) keeping the
human-review path available for anyone who wants it, (b) preserving a full, auditable record of
what was removed and why, and (c) dropping the items *before* benchmarking so no API spend is
wasted answering questions that will be excluded. It also closes a prior gap: unanimous-incorrect
items (all judges reject the gold answer) were never dropped before — only disagreements were
queued — so invalid gold answers could silently reach the benchmark. Note the trade-off: this is a
stricter, more conservative dataset (only unanimously-validated items survive), which can lower the
retained count; the recorded drop file makes that fully transparent for the datasheet.

**Judge-agreement stats are unaffected by auto-drop.** Pairwise judge agreement, per-judge rates,
and Fleiss' κ (#24) are computed over **every judged item** — the retained set *plus* the
auto-dropped rows (both carry the per-judge verdict columns), via `utils.load_judged`. If they
were computed from the retained `ground_truth_validated.csv` alone, auto-drop would have removed
exactly the disagreements and agreement would read a meaningless ~100%. The reliability metric
therefore still reflects how the panel actually behaved, independent of the drop policy.

### 12. Generator ≠ judge ≠ candidate (recommended separation)
**What:** Roles are configurable; the README recommends not letting a model grade its own
output, and `analyze.py` can report with same-family pairs excluded.
**Why:** Avoids self-preference bias contaminating both the ground truth and the benchmark.

---

## Benchmarking

### 13. Standardized, zero-shot protocol for all models
**What:** The shared `middle_school` system prompt and zero-shot for every benchmarked model —
no per-model tuning. Temperature is held constant where supported but may be omitted/overridden
per model (see 20a); output length is uncapped by default so each model answers to completion
(see 15a). Pinned dated model ids are recorded per row.
**Why:** A fair comparison requires holding the protocol constant; per-model prompt tuning
would confound model quality with prompt engineering. Temperature handling stays flexible so
models that can't accept a fixed temperature are still benchmarkable.

### 14. Middle-school system prompt
**What:** Every benchmarked model is told it is answering a middle-school student and must
write at a middle-school reading level.
**Why:** Matches the real use case (classroom audience) and makes the response reading-level
metric meaningful and comparable.

### 15. Three runs per model
**What:** Each model answers every question 3 times.
**Why:** Captures run-to-run variability so we can report stability/self-consistency, not just
a point estimate.

### 15a. No output-length cap; actual token counts recorded
**What:** We do not send a `max_tokens` limit by default, so every model produces its full
answer/reasoning (models reason for differing lengths). For each call we record the
**provider-reported** token usage — `prompt_tokens`, `completion_tokens`, `reasoning_tokens`
(for reasoning models), and `total_tokens` — taken straight from the API `usage` field.
**Why:** Capping output would truncate longer-reasoning models and bias the comparison (and
some reasoning models reject `max_tokens` outright). The API's `usage` is the actual count the
provider bills, not a local tokenizer estimate, so cost/length analysis is exact rather than
approximate.

### 15b. Token analytics (mean + SD) on actual counts
**What:** Stage 5 reports, per model (and per area / question type), the **mean and SD** of
input (prompt), output (completion), reasoning, and total tokens, plus `total_sum` (for cost)
and `reasoning_available_frac` (the share of responses for which the provider actually reported
reasoning tokens). Mean total/reasoning tokens also appear on the leaderboard.
**Why:** Mean ± SD captures both typical cost/length and its variability across questions and
runs — more informative than a single number. `total_sum` answers the cost question directly,
and `reasoning_available_frac` flags when reasoning means are zero only because the provider
didn't report them (vs the model genuinely not reasoning), so the metric isn't misread.

### 16. Reading level recorded per response
**What:** Flesch-Kincaid grade and reading-ease are computed for every response and aggregated
by model across runs, areas, and question types.
**Why:** For educators, *how* a model explains (at grade level) matters alongside whether it is
correct.

---

## Evaluation & grading

### 17. Single- and triple-judge frameworks, toggled in config
**What:** `eval_mode: single | triple`. Single uses one grader; triple uses a panel with
majority scoring and disagreement flags.
**Why:** Requested. Single is cheap for iteration; triple is the rigorous, reportable setting.

### 18. Binary headline + 3-level rubric for open-ended types
**What:** Binary 0/1 everywhere; an additional 0/0.5/1 rubric for conceptual and applied items.
**Why:** Binary keeps the primary metric simple and comparable, while the rubric captures
partial correctness where a one-bit verdict is too coarse.

### 19. Pre-registered refusal/non-answer policy
**What:** Empty, errored, or explicitly refusing responses score binary 0 / rubric 0 and are
tagged `is_refusal`; refusal rate is reported separately from wrong answers.
**Why:** Refusals are a distinct behavior from incorrect answers; conflating them would
mislead. Fixing the policy in advance avoids post-hoc bias.

### 20. Blind grading
**What:** Judges see the question, reference, and candidate response, but not which model
produced it.
**Why:** Reduces brand/position bias and makes grading more reproducible.

### 20a. Temperature is optional, never forced
**What:** Temperatures (generation, answering, judging, benchmark protocol) are configurable
and may be left unset (`null`), in which case we **do not send a temperature** and the model
uses its own default. Each model can also override with a fixed `temperature` or `omit_temperature`.
**Why:** Forcing `temperature=0` breaks models that require a specific temperature or reject the
parameter entirely (e.g. some reasoning models). We deliberately do not hard-code temperature 0
for judges or candidates; reproducibility is preserved via pinned dated model snapshots and
logged settings rather than by forcing a temperature an endpoint may not accept.

---

## Statistics & reporting

### 21. Macro and micro aggregation, both reported
**What:** Macro-average (equal weight per area) is primary; micro-average (per question) is also
reported.
**Why:** Areas have unequal standard counts; macro prevents large areas from dominating, while
micro reflects raw item-level performance.

### 22. Uncertainty and significance
**What:** Wilson 95% CIs on accuracies; pairwise McNemar tests with Holm-Bonferroni correction
and effect sizes for model comparisons.
**Why:** "Model A beats B" needs paired significance testing and multiple-comparison control,
not just a gap between point estimates.

### 23. Item difficulty / discrimination
**What:** Per-item fraction-correct across models, with trivial (≈all correct) and broken
(≈all wrong) flags and a discrimination measure.
**Why:** Items everyone passes or fails carry no signal; reporting this demonstrates the
benchmark actually separates models.

### 24. Inter-judge reliability + judge↔human calibration
**What:** Fleiss' κ for each judge panel, and a stratified human-labeled subset that yields
per-judge accuracy and Cohen's κ vs humans.
**Why:** Validates the measuring instrument itself — essential for any LLM-as-judge result. The
calibration sample is representative (includes agreed items), which is the correct basis for an
accuracy estimate; the disagreement queue is for label resolution, not accuracy estimation.

### 24a. Human labels are folded back into results
**What:** When `analyze` runs, it applies any filled human labels: a `human_score` in
`flags_eval_<model>.csv` overrides that response's `binary_score`, and a question marked
`human_score = 0` in `flags_groundtruth.csv` (reference answer wrong) is dropped from scoring.
**Why:** Human review only matters if it changes the numbers. This makes the disagreement
queues authoritative over the LLM judges and removes invalid items from the benchmark, while
remaining a no-op when nothing has been labeled yet.

### 25. Self-preference bias measured, not just avoided
**What:** In triple mode, report each judge's score delta on same-family vs other-family
responses.
**Why:** Quantifying the bias is more defensible than assuming it away.

### 25a. Teacher-facing leaderboards (first-run vs. ensemble), separate from the methodology leaderboard
**What:** An additive report (`src/teacher_report.py` → `results/reports/teacher_*.csv`) built for
classroom decision-making, distinct from the methodology `leaderboard.csv` (#21-22). For every
model it shows two plain accuracies — **first-run** (run #1 only, single-shot behaviour) and
**ensemble** (majority vote across the model's `runs_per_model` runs, i.e. self-consistency) — at
**overall / per-area / per-standard** granularity, plus compact **best-pick** tables mapping each
area/standard to its top model(s). Overall uses micro accuracy (plain % over all questions) for
intuitiveness rather than the macro average #21 uses. It reuses `analyze._load_eval` +
`_apply_human_overrides`, so it honours human label corrections and dropped items, and it writes
only new files (no existing report or data file is touched).
**Why:** A teacher asking "which model should I use to support standard MS-PS1-3?" needs a direct,
readable answer, not CIs and McNemar tests. First-run vs. ensemble makes the value of repeated
sampling explicit. Crucially, **when several models tie at the top the best-pick tables list *all*
of them** (with `n_tied` and the `gap` to the next scoring tier) rather than arbitrarily naming one
— because on this LLM-generated middle-school set many standards are aced by multiple models
(typically 3-8), so the honest guidance is "any of these," leaving the final choice to secondary
factors (reading level #16, token cost #15b, availability).

### 25b. One curated academic report, synthesized from the existing analyses
**What:** An additive module (`src/academic_report.py` → `results/reports/academic_report.md`,
one-click from the Results page) that assembles a single, shareable, academic Markdown write-up of
the most decision-relevant findings — abstract, methods, overall leaderboard, pairwise-significance
matrix (#22), performance by question type and by area, **ensemble-recomputed** item difficulty /
discrimination by area × question type (#23; recomputed because `analyze._item_stats` is a per-run
mean, not a majority-vote ensemble), best model per standard (#25a), run-to-run stability, reading
level (#16), token cost (#15b), and judge reliability (#24) — with a linked table of contents, a
numbered list of tables, and appendices (full 59-standard detail; question-type × area detail; raw
CSV pointers). It recomputes everything fresh from the eval/response files by reusing the analyze /
teacher_report functions (so it honours human overrides #24a), and writes only the one Markdown file
— no existing report, config, or data file changes. Sections gracefully degrade when data is absent
(e.g. single-mode runs omit self-preference #25 and eval-panel reliability; judge↔human calibration
#24 is flagged "pending" until the sample is labeled).
**Why:** The pipeline emits ~20 scattered report CSVs; colleagues need one coherent narrative, not a
data dump. Curating "the important findings" into a linked document — while leaving every raw CSV
available — makes the results reviewable and paper-ready. The report keeps the construct-validity
caveat (#32: LLM-generated questions) in the abstract and Limitations so high headline accuracies
are not over-read.

---

## Engineering for trustworthy long runs

### 26. Adjustable parallelism
**What:** A bounded thread pool sized by `concurrency.max_parallel_requests`.
**Why:** 1,200 questions × multiple models × runs × judges is large; parallelism keeps it
tractable, while a single knob lets users trade speed for rate-limit headroom.

### 27. Automatic retry with backoff
**What:** Transient API errors (timeouts, 429, 5xx) retry with exponential backoff + jitter;
exhausted calls become `ok=False` results rather than crashing the run.
**Why:** Long multi-model runs hit transient failures; they must self-heal and never abort the
whole job over one bad call.

### 28. Checkpoint / resume
**What:** Each stage appends results keyed by a stable id and skips completed work on restart.
**Why:** Interruptions are inevitable in long, paid runs; resumability avoids wasted cost and
duplicated calls.

### 28a. Long runs execute as detached, heartbeated background jobs
**What:** The optional UI launches each run as a detached process (`scripts/run_job.py`, its own
session) that records status JSON and stamps a heartbeat every few seconds. Progress is computed
from the checkpoint files (rows done / expected), so it is accurate even for a job started in a
prior session. Liveness is judged by heartbeat freshness, so hard kills/crashes are detected
rather than showing "running" forever.
**Why:** Full runs are long (many thousands of API calls). The UX must not require a tab to stay
open or block on a single browser session, must survive disconnects, and must show real progress
and cost — all of which fall out of the existing checkpoint/resume design plus a heartbeat.

### 29. Offline mock smoke test + schema tests (sandboxed from real data)
**What:** `scripts/smoke_test.py --mock` runs the whole pipeline with canned responses; pytest
asserts output schemas/value domains. Both write **only** into a disposable `_smoke_scratch/`
tree (redirected in `_tiny_cfg`, enforced by an abort-on-misconfig guard in `run()`), never the
real `data/`  or `results/`. Separately, `CheckpointWriter(fresh=True)` now *backs up* the file
it would clear into a sibling `_backups/` dir instead of deleting it.
**Why:** Lets us verify wiring with zero cost/keys and guards against regressions — without the
smoke test or a `--fresh` run being able to destroy an expensive real dataset (a hazard that
previously cost a full ground-truth build).

---

## Open science & ethics

### 30. Open artifacts
**What:** Dataset, prompts, code, raw outputs, judge rationales, and a datasheet are all
produced; tokens/cost are logged.
**Why:** Reproducibility and review.

### 31. Licensing/IP of NGSS text
**What:** The source PDF is © 2013 Achieve, Inc. ("all rights reserved"). Standard text is confined to `data/standards.yaml`
with attribution; generated Q&A are derivative works.

### 32. Construct-validity limitation stated up front
**What:** The benchmark measures answer correctness only — not pedagogy, age-appropriateness of
explanations, hallucination rate, or safety.
**Why:** Honest scoping prevents over-claiming "best classroom model" from a correctness
benchmark.
