"""Streamlit control panel for the NGSS MS-Science LLM benchmark pipeline.

Run with:  streamlit run app.py

Pages:
  Configure     – tabs for settings, model registry, API keys (.env), prompts, and raw config —
                  everything editable from the UI, no code changes needed
  Standards     – review/edit the 12 areas; regenerate the overview doc
  Run pipeline  – launch any stage (or all / quick-gen / smoke) as a detached background job
                  with a live progress dashboard (survives tab close)
  Results       – leaderboard + every report, with a couple of charts
  Human review  – edit the disagreement-flag and calibration sheets, then recompute

Long runs execute as detached jobs (scripts/run_job.py); the pipeline's checkpoint/resume plus
a heartbeat mean a closed tab or crash never loses progress and is detected correctly.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import Config, ConfigError  # noqa: E402
from src import jobs, progress  # noqa: E402

st.set_page_config(page_title="NGSS LLM Benchmark", layout="wide")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _safe_config_summary() -> tuple[dict, str]:
    """Load config (no key requirement) and return a summary + any error string."""
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env", override=True)  # reflect .env edits without a restart
    try:
        cfg = Config.load(require_keys=False)
    except Exception as exc:  # noqa: BLE001 - ConfigError or malformed YAML; show a banner, don't crash
        return {}, str(exc)
    referenced = sorted(cfg._referenced_model_keys())
    missing = [k for k in referenced if cfg.model(k).api_key == "missing"]
    return {
        "generator_model": cfg.get("generator_model"),
        "gt_judges": cfg.get("gt_judges"),
        "benchmark_models": cfg.get("benchmark_models"),
        "eval_mode": cfg.get("eval_mode"),
        "auto_drop_disagreements": bool(cfg.get("validation", "auto_drop_disagreements", default=False)),
        "runs_per_model": cfg.get("runs_per_model"),
        "questions_per_area": cfg.get("questions_per_area"),
        "referenced": referenced,
        "missing_keys": missing,
        "registry": list((cfg.get("models", default={}) or {}).keys()),
    }, ""


def _cfg() -> Config:
    return Config.load(require_keys=False)


def _path(name: str) -> Path:
    return _cfg().path(name)


def stream_command(cmd: list[str]) -> int:
    """Run a subprocess from the repo root, streaming combined output into the page."""
    st.caption("`" + " ".join(cmd) + "`")
    box = st.empty()
    lines: list[str] = []
    proc = subprocess.Popen(
        cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env={**os.environ},
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line.rstrip("\n"))
        box.code("\n".join(lines[-500:]) or " ", language="text")
    proc.wait()
    if proc.returncode == 0:
        st.success("Done (exit 0).")
    else:
        st.error(f"Exited with code {proc.returncode}.")
    return proc.returncode


def _py(*args: str) -> list[str]:
    return [sys.executable, *args]


def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _list_models(base_url: str, api_key: str) -> tuple[list[str], str]:
    """Fetch ALL available model ids from an OpenAI-compatible endpoint (follows pagination)."""
    if not base_url.strip():
        return [], "Enter a base_url first."
    if not api_key:
        return [], "No value for that key — set it in the API keys tab first."
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url.strip(), api_key=api_key, timeout=20, max_retries=1)
        page = client.models.list()
        ids = {m.id for m in page.data}
        try:  # some endpoints paginate /models — walk every page, not just the first
            while page.has_next_page():
                page = page.get_next_page()
                ids.update(m.id for m in page.data)
        except Exception:  # noqa: BLE001 - older clients without pagination helpers
            pass
        if not ids:
            return [], "Endpoint returned no models."
        return sorted(ids), ""
    except Exception as exc:  # noqa: BLE001 - surface any network/auth error to the user
        return [], f"Could not fetch models: {type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def _idx(options: list[str], value) -> int:
    return options.index(value) if value in options else 0


def _load_cfg_rt():
    """Round-trip load config.yaml (preserves comments) -> (YAML, data)."""
    from ruamel.yaml import YAML
    y = YAML()
    return y, y.load((REPO_ROOT / "config.yaml").read_text())


def _dump_cfg_rt(y, data) -> None:
    with open(REPO_ROOT / "config.yaml", "w") as f:
        y.dump(data, f)
    st.cache_data.clear()


# ---- Configuration page (tabbed) ------------------------------------------ #
def page_config() -> None:
    st.header("Configure")
    summary, err = _safe_config_summary()
    if err:
        st.error(f"config.yaml problem: {err}")
    tabs = st.tabs(["Status & settings", "Models", "API keys", "Prompts", "Raw config"])
    with tabs[0]:
        _tab_settings(summary)
    with tabs[1]:
        _tab_models()
    with tabs[2]:
        _tab_keys()
    with tabs[3]:
        _tab_prompts()
    with tabs[4]:
        _tab_raw()


def _tab_settings(summary: dict) -> None:
    if summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Areas (questions/area)", f"12 × {summary['questions_per_area']}")
        c2.metric("Benchmark models", len(summary["benchmark_models"]))
        c3.metric("Eval mode", summary["eval_mode"])
        c4.metric("GT review", "auto-drop" if summary.get("auto_drop_disagreements") else "human")
        if summary["missing_keys"]:
            st.warning("Missing API keys for: " + ", ".join(summary["missing_keys"])
                       + " — set them in the **API keys** tab.")
        else:
            st.success("All referenced models have API keys.")

    st.caption("Edits write back to `config.yaml` with comments preserved.")
    y, data = _load_cfg_rt()
    registry = list((data.get("models") or {}).keys())
    split = data.get("question_type_split") or {}
    with st.form("settings"):
        # ---- Ground truth: how the dataset is generated and validated ----------
        st.subheader("📚 Ground truth")
        st.caption("Builds and validates the question set (`generate → quality → validate`).")
        generator = st.selectbox("generator_model", registry, index=_idx(registry, data.get("generator_model")),
                                 help="The model that writes the ground-truth Q&A.")
        qpa = st.number_input("questions_per_area", 1, 1000, int(data.get("questions_per_area", 100)),
                              help="Target questions per NGSS area (× 12 areas = dataset size).")
        st.caption("question_type_split — must sum to questions_per_area.")
        s1, s2, s3 = st.columns(3)
        n_recall = s1.number_input("recall", 0, 1000, int(split.get("recall", 34)))
        n_concept = s2.number_input("conceptual", 0, 1000, int(split.get("conceptual", 33)))
        n_applied = s3.number_input("applied", 0, 1000, int(split.get("applied", 33)))
        gtj = st.multiselect("gt_judges", registry,
                             default=[m for m in (data.get("gt_judges") or []) if m in registry],
                             help="Independent panel that validates each generated Q&A.")
        auto_drop = st.checkbox(
            "Auto-drop judge disagreements (skip human review)",
            value=bool((data.get("validation") or {}).get("auto_drop_disagreements", False)),
            help="When ground-truth judges disagree — or all mark the reference answer wrong — drop "
                 "that question from the benchmark set automatically instead of queueing it for "
                 "human review. Dropped items are recorded in "
                 "results/human_review/auto_dropped_groundtruth.csv; ground_truth.csv is untouched.")

        st.divider()

        # ---- Benchmark: which models are tested and how they're graded ---------
        st.subheader("🏁 Benchmark")
        st.caption("Runs the candidate models on the dataset and grades them "
                   "(`benchmark → evaluate → calibrate → analyze`).")
        bench = st.multiselect("benchmark_models", registry,
                               default=[m for m in (data.get("benchmark_models") or []) if m in registry],
                               help="The candidate models under test.")
        b1, b2 = st.columns(2)
        runs = b1.number_input("runs_per_model", 1, 20, int(data.get("runs_per_model", 3)),
                               help="How many times each model answers every question.")
        eval_mode = b2.selectbox("eval_mode", ["single", "triple"],
                                 index=_idx(["single", "triple"], data.get("eval_mode", "single")),
                                 help="single = one grader; triple = 3-judge panel with majority vote.")
        ej = st.selectbox("eval_judge (single mode)", registry, index=_idx(registry, data.get("eval_judge")))
        ejs = st.multiselect("eval_judges (triple mode)", registry,
                             default=[m for m in (data.get("eval_judges") or []) if m in registry])

        st.divider()

        # ---- General: applies to every stage -----------------------------------
        st.subheader("⚙️ General")
        conc = st.number_input("concurrency.max_parallel_requests", 1, 256,
                               int((data.get("concurrency") or {}).get("max_parallel_requests", 8)),
                               help="Parallel API requests across all stages. Lower it if you hit rate limits.")

        if st.form_submit_button("Save settings", type="primary"):
            if n_recall + n_concept + n_applied != int(qpa):
                st.error(f"question_type_split ({n_recall}+{n_concept}+{n_applied}="
                         f"{n_recall + n_concept + n_applied}) must sum to questions_per_area "
                         f"({int(qpa)}). Not saved.")
                return
            data["eval_mode"] = eval_mode
            data["runs_per_model"] = int(runs)
            data["questions_per_area"] = int(qpa)
            data["question_type_split"] = {"recall": int(n_recall), "conceptual": int(n_concept),
                                           "applied": int(n_applied)}
            data.setdefault("concurrency", {})["max_parallel_requests"] = int(conc)
            data["generator_model"] = generator
            data["benchmark_models"] = bench
            data["gt_judges"] = gtj
            data["eval_judge"] = ej
            data["eval_judges"] = ejs
            data.setdefault("validation", {})["auto_drop_disagreements"] = bool(auto_drop)
            _dump_cfg_rt(y, data)
            st.success("Saved config.yaml.")
            st.rerun()


def _tab_models() -> None:
    import hashlib
    st.caption("Edit existing models' fields below and click **Save model edits**. To delete, "
               "use **Remove models**; to add, use **Add a model** — both apply immediately. "
               "`api_key_env` is a key name from the **API keys** tab; leave `temperature` blank "
               "to omit it. (Rename a model by removing and re-adding it.)")
    y, data = _load_cfg_rt()
    models = data.get("models") or {}

    rows = [{"remove": False, "key": k, "base_url": v.get("base_url", ""),
             "model": v.get("model", ""), "api_key_env": v.get("api_key_env", ""),
             "family": v.get("family", ""),
             "temperature": ("" if v.get("temperature") is None else v.get("temperature")),
             "omit_temperature": bool(v.get("omit_temperature", False))}
            for k, v in models.items()]
    # Tie the editor's identity to the current model set so it resets (no stale edit deltas)
    # whenever a model is added or removed — this is what prevents "deleted models coming back".
    sig = hashlib.md5("|".join(sorted(models.keys())).encode()).hexdigest()[:8]
    edited = st.data_editor(
        pd.DataFrame(rows, columns=["remove", "key", "base_url", "model", "api_key_env",
                                    "family", "temperature", "omit_temperature"]),
        num_rows="fixed", width="stretch", height=320, key=f"models_editor_{sig}",
        column_config={
            "remove": st.column_config.CheckboxColumn("remove?", help="Tick to delete this model"),
            "key": st.column_config.TextColumn("key", disabled=True, help="Identity — rename via remove + re-add"),
            "omit_temperature": st.column_config.CheckboxColumn("omit_temp"),
        },
    )

    checked = [str(r["key"]) for _, r in edited.iterrows() if bool(r.get("remove"))]
    used = _models_in_use(data, checked)
    if used:
        st.warning("Ticked models in use by a role (generator/judge/benchmark): "
                   + ", ".join(sorted(used)) + " — they'll also be removed from those roles.")

    c_save, c_remove = st.columns(2)
    if c_save.button("💾 Save edits"):
        new: dict = {}
        for _, r in edited.iterrows():
            key = str(r["key"]).strip()
            if not key:
                continue
            entry = {"base_url": str(r["base_url"]).strip(),
                     "api_key_env": str(r["api_key_env"]).strip(),
                     "model": str(r["model"]).strip(),
                     "family": str(r["family"]).strip() or "unknown"}
            temp = str(r.get("temperature", "")).strip()
            if temp:
                try:
                    entry["temperature"] = float(temp)
                except ValueError:
                    st.error(f"Row '{key}': temperature must be a number or blank.")
                    return
            if bool(r.get("omit_temperature")):
                entry["omit_temperature"] = True
            new[key] = entry
        data["models"] = new
        _dump_cfg_rt(y, data)
        st.success(f"Saved edits to {len(new)} models.")
        st.rerun()

    if c_remove.button(f"🗑 Remove checked ({len(checked)})", disabled=not checked):
        removed = set(checked)
        for k in removed:
            models.pop(k, None)
        data["models"] = models
        # Strip the removed models from role references so the config stays valid.
        for many in ("gt_judges", "benchmark_models", "eval_judges"):
            if data.get(many):
                data[many] = [m for m in data[many] if m not in removed]
        remaining = list(models.keys())
        for single in ("generator_model", "eval_judge"):
            if data.get(single) in removed:
                data[single] = remaining[0] if remaining else ""
        _dump_cfg_rt(y, data)
        st.success(f"Removed {len(removed)} model(s).")
        st.rerun()

    _add_model_from_endpoint()


def _models_in_use(data: dict, keys: list[str]) -> set:
    """Of `keys`, which are referenced by any role assignment in the config?"""
    refs = set()
    for single in ("generator_model", "eval_judge"):
        if data.get(single):
            refs.add(data[single])
    for many in ("gt_judges", "benchmark_models", "eval_judges"):
        refs.update(data.get(many) or [])
    return {k for k in keys if k in refs}


def _add_model_from_endpoint() -> None:
    """Fetch the model list from an endpoint and add a chosen model — no typing model names."""
    from dotenv import dotenv_values
    env = dict(dotenv_values(REPO_ROOT / ".env")) if (REPO_ROOT / ".env").exists() else {}
    with st.expander("➕ Add a model — fetch the model list from an endpoint"):
        c1, c2 = st.columns(2)
        base_url = c1.text_input("base_url", "https://api.openai.com/v1", key="add_base")
        key_names = sorted(env.keys())
        key_name = c2.selectbox("api_key_env (from your saved keys)",
                                key_names or ["— set a key first —"], key="add_keyenv")

        if st.button("🔍 Fetch available models"):
            ids, err = _list_models(base_url, env.get(key_name, ""))
            if err:
                st.error(err)
                st.session_state.pop("fetched_models", None)
            else:
                st.session_state["fetched_models"] = ids
                st.success(f"Found {len(ids)} models at this endpoint.")

        fetched = st.session_state.get("fetched_models", [])
        if fetched:
            st.caption(f"{len(fetched)} models available — type in the box to search/filter.")
            with st.expander(f"See all {len(fetched)} model ids"):
                st.code("\n".join(fetched), language="text")
            picked = st.selectbox("model (pick from the endpoint)", fetched, key="add_modelid")
            manual = st.text_input("…or type a model id manually (overrides the pick if non-empty)",
                                   key="add_manual")
            model_id = manual.strip() or picked
            d1, d2 = st.columns(2)
            reg_key = d1.text_input("registry key (short name you'll reference)",
                                    value=model_id, key="add_regkey")
            family = d2.text_input("family (e.g. openai, anthropic)", key="add_family")
            if st.button("Add to registry", type="primary"):
                if not reg_key.strip():
                    st.error("Give the model a short registry key.")
                    return
                y, data = _load_cfg_rt()
                models = data.get("models") or {}
                models[reg_key.strip()] = {
                    "base_url": base_url.strip(), "api_key_env": key_name,
                    "model": model_id, "family": family.strip() or "unknown",
                }
                data["models"] = models
                _dump_cfg_rt(y, data)
                st.session_state.pop("fetched_models", None)
                st.success(f"Added '{reg_key.strip()}' → {model_id}.")
                st.rerun()


def _tab_keys() -> None:
    from dotenv import dotenv_values, set_key, unset_key
    env_path = REPO_ROOT / ".env"
    st.caption("API keys are stored in `.env` (gitignored). They are loaded fresh by each run, "
               "so changes take effect on the next launch.")
    current = dict(dotenv_values(env_path)) if env_path.exists() else {}

    # Which key names are referenced by the model registry?
    _, data = _load_cfg_rt()
    referenced = sorted({v.get("api_key_env") for v in (data.get("models") or {}).values()
                         if v.get("api_key_env")})
    names = sorted(set(referenced) | set(current))

    st.markdown("**Set / update keys**")
    with st.form("keys"):
        new_values: dict[str, str] = {}
        for name in names:
            tag = " (referenced)" if name in referenced else ""
            new_values[name] = st.text_input(f"{name}{tag}", value=current.get(name, ""),
                                             type="password", key=f"env_{name}")
        st.markdown("**Add a new key**")
        nk1, nk2 = st.columns(2)
        new_name = nk1.text_input("New key name", key="env_new_name")
        new_val = nk2.text_input("New key value", type="password", key="env_new_val")
        if st.form_submit_button("Save keys"):
            env_path.touch(exist_ok=True)
            for name, val in new_values.items():
                if val:
                    set_key(str(env_path), name, val)
            if new_name.strip() and new_val:
                set_key(str(env_path), new_name.strip(), new_val)
            st.cache_data.clear()
            st.success("Saved .env.")
            st.rerun()

    # Explicit deletion (outside the form so it can have its own button).
    st.markdown("**Delete keys**")
    if current:
        to_delete = st.multiselect("Select keys to remove from `.env`", sorted(current.keys()),
                                   key="env_delete")
        warn = [k for k in to_delete if k in referenced]
        if warn:
            st.warning("These are referenced by a model in the registry: " + ", ".join(warn)
                       + ". Removing them will make those models fail until re-added.")
        if st.button("🗑 Delete selected keys", disabled=not to_delete):
            for k in to_delete:
                unset_key(str(env_path), k)
            st.cache_data.clear()
            st.success(f"Deleted {len(to_delete)} key(s) from .env.")
            st.rerun()
    else:
        st.caption("No keys stored yet.")

    missing = [k for k in referenced if not current.get(k)]
    if missing:
        st.warning("Referenced keys still empty: " + ", ".join(missing))
    elif referenced:
        st.success("All referenced model keys are set.")


def _tab_prompts() -> None:
    from src import prompts
    import re
    st.caption("Edit the prompt text used across the pipeline. Placeholders look like "
               "`$name` and **must be kept** — they are filled in at runtime. Saved to "
               "`data/prompts.yaml`; changes take effect on the next run.")
    tpl = prompts.effective_templates()

    st.subheader("System prompts")
    sys_edit = {name: st.text_area(name, val, height=90, key=f"sys_{name}")
                for name, val in tpl["system_prompts"].items()}

    st.subheader("Question-type definitions")
    qt_edit = {name: st.text_area(name, val, height=80, key=f"qt_{name}")
               for name, val in tpl["question_types"].items()}

    st.subheader("Prompt templates")
    tmpl_edit = {}
    for name, val in tpl["templates"].items():
        tmpl_edit[name] = st.text_area(name, val, height=170, key=f"tpl_{name}")
        req = prompts.REQUIRED_PLACEHOLDERS.get(name)
        if req:
            st.caption("Required placeholders: " + ", ".join(f"`${p}`" for p in req))

    if st.button("Save prompts"):
        # validate required placeholders survive
        problems = []
        for name, req in prompts.REQUIRED_PLACEHOLDERS.items():
            text = tmpl_edit.get(name, "")
            for ph in req:
                if not re.search(r"\$\{?" + re.escape(ph) + r"\b", text):
                    problems.append(f"`{name}` is missing `${ph}`")
        if problems:
            st.error("Not saved — fix these placeholders:\n\n- " + "\n- ".join(problems))
            return
        prompts.save_templates({"system_prompts": sys_edit, "question_types": qt_edit,
                                "templates": tmpl_edit})
        st.success("Saved data/prompts.yaml.")


def _tab_raw() -> None:
    st.caption("Full `config.yaml` for anything not covered by the other tabs. Validated on save.")
    raw = (REPO_ROOT / "config.yaml").read_text()
    edited = st.text_area("config.yaml", raw, height=480, key="raw_cfg")
    if st.button("Save raw config"):
        import yaml
        try:
            yaml.safe_load(edited)
        except yaml.YAMLError as exc:
            st.error(f"Invalid YAML, not saved: {exc}")
            return
        (REPO_ROOT / "config.yaml").write_text(edited)
        st.cache_data.clear()
        # surface deeper validation (key presence, splits, etc.)
        try:
            Config.load(require_keys=False)
            st.success("Saved and validated config.yaml.")
        except ConfigError as exc:
            st.warning(f"Saved, but validation warns: {exc}")


def page_standards() -> None:
    st.header("Standards (12 areas)")
    from src.standards import load_areas
    try:
        areas = load_areas(_cfg())
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load standards.yaml: {exc}")
        return
    st.caption(f"{len(areas)} areas · {sum(len(a.standards) for a in areas)} performance expectations")
    for a in areas:
        with st.expander(f"{a.code} — {a.title}  ({len(a.standards)} standards)"):
            st.write(a.description)
            st.table(pd.DataFrame([{"standard": s.code, "text": s.text} for s in a.standards]))

    st.subheader("Edit standards.yaml")
    raw = (REPO_ROOT / "data" / "standards.yaml").read_text()
    edited = st.text_area("standards.yaml", raw, height=300)
    c1, c2 = st.columns(2)
    if c1.button("Save standards.yaml"):
        import yaml
        try:
            yaml.safe_load(edited)
        except yaml.YAMLError as exc:
            st.error(f"Invalid YAML, not saved: {exc}")
        else:
            (REPO_ROOT / "data" / "standards.yaml").write_text(edited)
            st.success("Saved. Click 'Regenerate overview doc' to refresh docs/areas_overview.md.")
    if c2.button("Regenerate overview doc"):
        stream_command(_py("-m", "src.standards", "--doc"))


def _run_section(controls_fn) -> None:
    """Shared run UI: if a job is active show the live panel, else last-run + the given controls."""
    rec = jobs.current()
    if jobs.is_running(rec):
        st.info("A run is in progress. It continues in the background even if you close this "
                "tab — come back any time to check on it.")
        _live_panel()
        return
    if rec:
        _show_last_run(rec)
    controls_fn()


def _show_last_run(rec: dict) -> None:
    status = jobs.display_status(rec)
    label = f"Last run — {rec.get('kind')}/{rec.get('stage')}: **{status}**"
    {"done": st.success, "failed": st.error, "stopped": st.warning,
     "interrupted": st.warning}.get(status, st.info)(label)
    with st.expander("Last run log"):
        st.code(jobs.tail_log(rec, 300) or "(empty)", language="text")


def _stage_buttons(stages: list[str], fresh: bool) -> None:
    cols = st.columns(len(stages))
    for col, stage in zip(cols, stages):
        if col.button(stage, key=f"stage_{stage}"):
            jobs.launch("pipeline", stage=stage, fresh=fresh)
            st.rerun()


def _gt_build_controls() -> None:
    fresh = st.checkbox("--fresh (ignore checkpoints and regenerate from scratch)", value=False,
                        key="gt_fresh")
    st.subheader("Build the dataset")
    st.caption("Runs **generate → quality → validate** only. Does not touch benchmarking.")
    if st.button("▶ Build ground truth", type="primary"):
        jobs.launch("pipeline", stage="groundtruth", fresh=fresh)
        st.rerun()

    st.subheader("Individual steps")
    _stage_buttons(["generate", "quality", "validate"], fresh)

    st.subheader("Quick tests")
    c1, c2 = st.columns(2)
    with c1:
        total = st.number_input("Quick generation: total questions", 4, 200, 20, step=4)
        areas = st.text_input("Areas (comma-separated)", "MS-PS3,MS-LS1,MS-ESS2,MS-ETS1")
        if st.button("▶ Quick generation test"):
            jobs.launch("quickgen", total=int(total), areas=areas)
            st.rerun()
    with c2:
        st.write("Offline wiring check (no keys/cost):")
        if st.button("▶ Smoke test (mock)"):
            jobs.launch("smoke")
            st.rerun()


def _bench_controls() -> None:
    cfg = _cfg()
    if not cfg.path("ground_truth").exists():
        st.warning("No ground truth yet — build it on the **Ground truth** page first.")
    if not cfg.get("benchmark_models"):
        st.error("No **benchmark_models** selected — pick them in *Configure → Status & settings*, "
                 "or there's nothing to benchmark.")
    fresh = st.checkbox("--fresh (ignore checkpoints and recompute from scratch)", value=False,
                        key="bench_fresh")
    st.subheader("Run the benchmark")
    st.caption("Runs **benchmark → evaluate → calibrate → analyze** for **all** `benchmark_models`. "
               "Does **not** regenerate the ground truth.")
    if st.button("▶ Run benchmark (all models)", type="primary"):
        jobs.launch("pipeline", stage="benchmark_all", fresh=fresh)
        st.rerun()

    _per_model_controls(cfg, fresh)

    st.subheader("Individual steps")
    st.caption("`calibrate` and `analyze` aggregate across **all** models — run them once, "
               "after every model above has been benchmarked and evaluated.")
    _stage_buttons(["benchmark", "evaluate", "calibrate", "analyze"], fresh)


def _per_model_controls(cfg: Config, fresh: bool) -> None:
    """One button per `benchmark_models` entry — run **benchmark,evaluate** for that model alone.

    Lets you work through a large model set one at a time (no need to stop the all-models run
    partway). The cross-model `calibrate`/`analyze` steps are run separately once all are done.
    """
    models = list(cfg.get("benchmark_models") or [])
    if not models:
        return
    st.subheader("Run one model at a time")
    st.caption("Each button runs **benchmark → evaluate** for that model only. ✓ = responses "
               "already collected (re-running resumes; tick **--fresh** above to recompute).")
    resp_dir = cfg.path("responses_dir")
    for key in models:
        done = (resp_dir / f"{key}.csv").exists()
        c1, c2 = st.columns([3, 1])
        c1.markdown(f"{'✓' if done else '•'} **{key}**")
        if c2.button("▶ Run", key=f"bench_model_{key}"):
            jobs.launch("pipeline", stage="benchmark,evaluate", models=key, fresh=fresh)
            st.rerun()


def _elapsed(rec: dict) -> str:
    from datetime import datetime, timezone
    try:
        start = datetime.fromisoformat(rec["started_at"])
    except (KeyError, ValueError):
        return ""
    end = (datetime.fromisoformat(rec["finished_at"])
           if rec.get("finished_at") else datetime.now(timezone.utc))
    secs = int((end - start).total_seconds())
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"


@st.fragment(run_every=3)
def _live_panel() -> None:
    """Auto-refreshing (every 3s) status panel — re-reads checkpoint files, not the run process."""
    rec = jobs.current()
    if not rec:
        st.info("No active run.")
        return
    running = jobs.is_running(rec)

    c1, c2, c3 = st.columns([3, 1, 1])
    scope = f" · {rec['models']}" if rec.get("models") else ""
    c1.markdown(f"**Run `{rec['run_id']}`** — {rec.get('kind')}/{rec.get('stage')}{scope}")
    c2.markdown(f"Status: **{jobs.display_status(rec)}**")
    c3.markdown(f"Elapsed: {_elapsed(rec)}")
    if running and st.button("⏹ Stop run", type="secondary"):
        jobs.stop(rec)
        st.rerun()

    cfg = _cfg()
    if rec.get("kind") == "pipeline":
        from src.run_pipeline import resolve_stages
        try:
            active = set(resolve_stages(rec.get("stage", "all")))
        except ValueError:
            active = set()
        bars = [s for s in progress.stage_progress(cfg) if s.stage in active]
        if bars:
            for s in bars:
                st.progress(s.pct, text=f"{s.stage}: {s.done}/{s.total} ({s.pct*100:.0f}%)")
        else:
            st.caption("This stage finishes quickly — see the log below.")
        u = progress.usage_stats(cfg)
        m1, m2, m3 = st.columns(3)
        m1.metric("Responses collected", f"{u['responses']:,}")
        m2.metric("Total tokens used", f"{u['total_tokens']:,}")
        m3.metric("API errors", u["errors"])

    with st.expander("Live log", expanded=True):
        st.code(jobs.tail_log(rec, 300) or "(starting…)", language="text")

    if not running:
        st.rerun()   # job finished — flip the page back to the launch controls


def _gt_status() -> None:
    cfg = _cfg()
    gt = _read_csv(cfg.path("ground_truth"))
    val = _read_csv(cfg.path("ground_truth_validated"))
    flagged = 0
    if val is not None and "human_review_flag" in val.columns:
        flagged = int(pd.to_numeric(val["human_review_flag"], errors="coerce").fillna(0).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Ground-truth Qs", 0 if gt is None else len(gt))
    c2.metric("Validated", "yes" if val is not None else "no")
    c3.metric("Flagged for review", flagged)


def _bench_status() -> None:
    cfg = _cfg()
    resp = list(Path(cfg.path("responses_dir")).glob("*.csv")) if cfg.path("responses_dir").exists() else []
    evals = list(Path(cfg.path("eval_dir")).glob("*_single.csv")) + \
        list(Path(cfg.path("eval_dir")).glob("*_triple.csv")) if cfg.path("eval_dir").exists() else []
    reports = list(Path(cfg.path("reports_dir")).glob("*.csv")) if cfg.path("reports_dir").exists() else []
    c1, c2, c3 = st.columns(3)
    c1.metric("Response files", len(resp))
    c2.metric("Eval files", len(evals))
    c3.metric("Reports", len(reports))


def page_benchmark() -> None:
    st.header("Benchmark")
    _bench_status()
    _run_section(_bench_controls)


def _teacher_reports_button() -> None:
    """Build the teacher-facing leaderboards (first-run vs. ensemble) from the eval files.

    Pure local computation (no API keys); writes new teacher_*.csv that then appear in the
    report dropdown below. Touches no existing files.
    """
    st.caption("**Teacher guide** — simple first-run vs. ensemble (majority-vote) leaderboards by "
               "area and standard, for picking a model to support a given standard.")
    if st.button("🧑‍🏫 Build teacher reports"):
        from src.teacher_report import teacher_report
        try:
            written = teacher_report(_cfg())
        except FileNotFoundError as exc:
            st.warning(f"{exc} Run **benchmark → evaluate** first.")
            return
        except Exception as exc:  # noqa: BLE001 - never crash the page over a report build
            st.error(f"Teacher report build failed: {exc}")
            return
        st.success(f"Wrote {len(written)} teacher report(s) — open the `teacher_*` files below.")
        st.rerun()


def _academic_report_button() -> None:
    """Build the single curated academic Markdown report, then preview + offer it for download.

    Output is Markdown (not CSV), so it doesn't appear in the report dropdown; this button surfaces
    it directly. Writes only results/reports/academic_report.md — touches no existing file.
    """
    st.caption("**Academic report** — one curated Markdown write-up of the key findings "
               "(methods, leaderboards, significance, by area / question type / standard, "
               "psychometrics, reliability) with a linked table of contents and appendices.")
    if st.button("📄 Build academic report"):
        from src.academic_report import academic_report
        try:
            out = academic_report(_cfg())
        except FileNotFoundError as exc:
            st.warning(f"{exc} Run **benchmark → evaluate** first.")
            return
        except Exception as exc:  # noqa: BLE001 - never crash the page over a report build
            st.error(f"Academic report build failed: {exc}")
            return
        md = Path(out).read_text()
        st.success(f"Wrote `{out}`.")
        st.download_button("⬇ Download academic_report.md", md,
                           file_name="academic_report.md", mime="text/markdown")
        with st.expander("Preview the report", expanded=True):
            st.markdown(md, unsafe_allow_html=True)


def page_results() -> None:
    st.header("Results")
    reports_dir = _path("reports_dir")
    _teacher_reports_button()
    _academic_report_button()
    files = sorted(Path(reports_dir).glob("*.csv")) if reports_dir.exists() else []
    if not files:
        st.info("No reports yet. Run the **analyze** stage first.")
        return

    lb = _read_csv(reports_dir / "leaderboard.csv")
    if lb is not None:
        st.subheader("Leaderboard")
        st.dataframe(lb, width="stretch")
        if "macro_accuracy" in lb.columns:
            st.bar_chart(lb.set_index("model")["macro_accuracy"])

    names = [f.name for f in files]
    choice = st.selectbox("Open a report", names,
                          index=names.index("leaderboard.csv") if "leaderboard.csv" in names else 0)
    df = _read_csv(reports_dir / choice)
    if df is not None:
        st.dataframe(df, width="stretch")
        # handy charts for a couple of well-known reports
        if choice == "accuracy_by_area.csv" and {"model", "area", "binary_accuracy"} <= set(df.columns):
            st.bar_chart(df.pivot_table(index="area", columns="model", values="binary_accuracy"))
        if choice == "token_usage_by_model.csv" and {"model", "total_mean"} <= set(df.columns):
            st.bar_chart(df.set_index("model")["total_mean"])
        st.download_button("Download CSV", df.to_csv(index=False), file_name=choice)


def page_ground_truth() -> None:
    st.header("Ground truth")
    build, review, summary = st.tabs(["Build", "Review & fix", "Summary"])
    with build:
        _gt_status()
        _run_section(_gt_build_controls)
    with review:
        _gt_review()
    with summary:
        _gt_summary()


def _as_bool(v) -> bool:
    """Coerce a CSV/editor value to bool (handles True/False, 1/0, and their string forms)."""
    if pd.isna(v):
        return False
    return str(v).strip().lower() in ("true", "1", "1.0", "yes")


def _judge_agreement_data(
    val: pd.DataFrame, judge_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Compute (matrix, pairs, n_items_all) for pairwise judge agreement — no rendering."""
    labels = [c[len("judge_"):] for c in judge_cols]
    scores = {c: pd.to_numeric(val[c], errors="coerce") for c in judge_cols}

    matrix = pd.DataFrame(index=labels, columns=labels, dtype=object)
    pairs = []
    for i, ci in enumerate(judge_cols):
        matrix.iloc[i, i] = "—"
        for j in range(i + 1, len(judge_cols)):
            cj = judge_cols[j]
            both = scores[ci].notna() & scores[cj].notna()
            n = int(both.sum())
            pct = float((scores[ci][both] == scores[cj][both]).mean() * 100) if n else float("nan")
            cell = f"{pct:.0f}%" if n else "n/a"
            matrix.iloc[i, j] = cell
            matrix.iloc[j, i] = cell
            pairs.append({"judge A": labels[i], "judge B": labels[j],
                          "agree %": round(pct, 1) if n else None, "items": n})

    n_items = int(pd.concat([scores[c] for c in judge_cols], axis=1).notna().all(axis=1).sum())
    pairs_df = pd.DataFrame(pairs).sort_values("agree %", na_position="last")
    return matrix, pairs_df, n_items


def _judge_pass_rates(val: pd.DataFrame, judge_cols: list[str]) -> pd.DataFrame:
    """Per-judge verdict metrics: how many items each judge scored, its 'correct' rate,
    and how often it agreed with the panel majority."""
    maj = pd.to_numeric(val["majority_label"], errors="coerce") if "majority_label" in val else None
    rows = []
    for c in judge_cols:
        s = pd.to_numeric(val[c], errors="coerce")
        n = int(s.notna().sum())
        pass_rate = round(float((s == 1).sum()) / n * 100, 1) if n else None
        agree_maj = None
        if maj is not None:
            both = s.notna() & maj.notna()
            if bool(both.any()):
                agree_maj = round(float((s[both] == maj[both]).mean() * 100), 1)
        rows.append({"judge": c[len("judge_"):], "items scored": n,
                     "said correct %": pass_rate, "agreed w/ majority %": agree_maj})
    return pd.DataFrame(rows)


def _judge_agreement(val: pd.DataFrame, judge_cols: list[str]) -> None:
    """Pairwise percent agreement between every pair of ground-truth judges."""
    matrix, pairs_df, n_items = _judge_agreement_data(val, judge_cols)
    matrix_out = matrix.rename_axis("judge").reset_index()

    st.subheader("Judge agreement")
    st.caption(f"Percent of items where the two judges gave the same 0/1 verdict "
               f"(over items both scored). {n_items} item(s) scored by all judges.")

    # Persist to results/reports/ so it's kept for future reference.
    reports = _cfg().path("reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    pairs_path = reports / "gt_judge_agreement_pairs.csv"
    matrix_path = reports / "gt_judge_agreement_matrix.csv"
    pairs_df.to_csv(pairs_path, index=False)
    matrix_out.to_csv(matrix_path, index=False)

    st.dataframe(matrix, width="stretch")
    st.dataframe(pairs_df, width="stretch", hide_index=True)
    st.caption(f"Saved to `{pairs_path}` and `{matrix_path}`.")
    c1, c2 = st.columns(2)
    c1.download_button("⬇ Download pairs CSV", pairs_df.to_csv(index=False),
                       file_name="gt_judge_agreement_pairs.csv", mime="text/csv")
    c2.download_button("⬇ Download matrix CSV", matrix_out.to_csv(index=False),
                       file_name="gt_judge_agreement_matrix.csv", mime="text/csv")


def _gt_review() -> None:
    cfg = _cfg()
    gt_path = cfg.path("ground_truth")
    if not gt_path.exists():
        st.info("No `data/ground_truth.csv` yet. Build it in the **Build** tab first.")
        return
    gt = pd.read_csv(gt_path)

    # If auto-drop is on, doubted questions were removed at validation time — there's no queue to
    # review. Tell the reviewer what happened and where the record is, rather than showing an empty
    # list with no explanation.
    if bool(cfg.get("validation", "auto_drop_disagreements", default=False)):
        auto_rec = _read_csv(cfg.path("human_review_dir") / "auto_dropped_groundtruth.csv")
        n = len(auto_rec) if auto_rec is not None else 0
        st.warning(
            f"**Auto-drop is ON** (Configure → settings). Questions the judges doubted were dropped "
            f"automatically at validation — **{n}** so far — so the review queue below is empty by "
            "design. Dropped items are recorded in "
            "`results/human_review/auto_dropped_groundtruth.csv` (nothing is deleted from "
            "`ground_truth.csv`). Turn the checkbox off and re-run **validate** to review them by hand.")

    # Which items did the judge panel doubt? Disagreements OR a unanimous "incorrect".
    needs_review: set = set()
    val = None
    judge_cols: list[str] = []
    rationale_cols: list[str] = []
    val_path = cfg.path("ground_truth_validated")
    if val_path.exists():
        val = pd.read_csv(val_path)
        flag = val.get("human_review_flag")
        maj = val.get("majority_label")
        mask = pd.Series(False, index=val.index)
        if flag is not None:
            mask = mask | (pd.to_numeric(flag, errors="coerce") == 1)
        if maj is not None:
            mask = mask | (pd.to_numeric(maj, errors="coerce") == 0)
        needs_review = set(val.loc[mask, "question_id"])
        judge_cols = [c for c in val.columns if c.startswith("judge_")]
        rationale_cols = [c for c in val.columns if c.startswith("rationale_")]

    if val is not None and len(judge_cols) >= 2:
        # Agreement over the full judged set (incl. auto-dropped), not just the retained rows.
        from src.utils import load_judged
        _judge_agreement(load_judged(cfg), judge_cols)

    st.caption("Edit a **question** or its **reference answer** if it's inaccurate, then Save — "
               "this updates the gold answers models are graded against. Tick **keep** to mark an "
               "item you've reviewed and are happy with. Tick **drop** to exclude an item from the "
               "**validated** dataset (`ground_truth_validated.csv`, what the benchmark uses); the "
               "item stays in the originally-generated `ground_truth.csv`, flagged `dropped`. "
               "`fk_grade` is recomputed for edited questions. Each judge's verdict "
               "(1=correct, 0=incorrect) and rationale are shown read-only for context.")
    only_review = st.checkbox(
        f"Show only items the judges doubted ({len(needs_review)})",
        value=bool(needs_review), disabled=not needs_review)
    view = gt[gt["question_id"].isin(needs_review)] if (only_review and needs_review) else gt
    if view.empty:
        st.success("Nothing flagged for review. You can still untick the box to edit any item.")
        return

    view = view.copy()
    # `keep` persists across sessions (stored in the CSV); `drop` is a per-session action.
    if "keep" in view.columns:
        view["keep"] = view["keep"].map(_as_bool)
    else:
        view["keep"] = False
    # `dropped` is a read-only record (in ground_truth.csv) of items excluded from validated.
    if "dropped" in view.columns:
        view["dropped"] = view["dropped"].map(_as_bool)
    view["drop"] = False

    # Bring in each judge's verdict + rationale (read-only) so the reviewer sees why an item
    # was doubted. Interleave score then rationale per judge for readability.
    ordered_judge_cols: list[str] = []
    if val is not None and (judge_cols or rationale_cols):
        merge_cols = ["question_id", *judge_cols, *rationale_cols]
        view = view.merge(val[merge_cols], on="question_id", how="left")
        for jc in judge_cols:
            ordered_judge_cols.append(jc)
            rc = "rationale_" + jc[len("judge_"):]
            if rc in rationale_cols:
                ordered_judge_cols.append(rc)
        # any rationale without a matching score column
        ordered_judge_cols += [c for c in rationale_cols if c not in ordered_judge_cols]

    locked = {c: st.column_config.TextColumn(disabled=True)
              for c in ["question_id", "area", "standard", "question_type", "generator_model"]}
    locked["fk_grade"] = st.column_config.NumberColumn(disabled=True)
    locked["keep"] = st.column_config.CheckboxColumn(
        "keep", help="Mark items you've reviewed and want to keep (saved to the CSV)")
    locked["dropped"] = st.column_config.CheckboxColumn(
        "dropped", disabled=True, help="Already excluded from the validated dataset")
    locked["drop"] = st.column_config.CheckboxColumn(
        help="Exclude this item from the validated dataset (kept in the generated file)")
    for jc in judge_cols:
        label = jc[len("judge_"):]
        locked[jc] = st.column_config.NumberColumn(
            label, disabled=True, help=f"{label} verdict: 1=answer correct, 0=incorrect")
    for rc in rationale_cols:
        label = rc[len("rationale_"):]
        locked[rc] = st.column_config.TextColumn(
            f"{label} — why", disabled=True, width="large",
            help=f"{label}'s rationale for its verdict")

    # Editable columns first, then the read-only judge context columns at the end.
    status_cols = [c for c in ("keep", "dropped", "drop") if c in view.columns]
    base_cols = [c for c in view.columns
                 if c not in ordered_judge_cols and c not in status_cols]
    col_order = [*base_cols, *status_cols, *ordered_judge_cols]
    edited = st.data_editor(view, width="stretch", num_rows="fixed",
                            column_config=locked, column_order=col_order, key="gt_editor")

    if st.button("💾 Save ground-truth edits", type="primary"):
        n_changed, n_dropped, n_kept = _save_gt_edits(cfg, gt, edited)
        st.success(f"Saved: {n_changed} item(s) edited, {n_dropped} dropped from the validated "
                   f"dataset (kept in the generated file), {n_kept} marked keep (in this view). "
                   "Re-run **benchmark/evaluate** (or just **analyze** if only answers changed "
                   "and responses already exist) to use the corrected gold answers.")
        st.rerun()


def _save_gt_edits(cfg: Config, full_gt: pd.DataFrame, edited: pd.DataFrame) -> tuple[int, int, int]:
    from src.quality_checks import fk_grade
    ed = edited.set_index("question_id")
    drop_ids = set(ed.index[ed["drop"].fillna(False).astype(bool)])
    has_keep = "keep" in ed.columns
    n_kept = int(ed["keep"].map(_as_bool).sum()) if has_keep else 0

    def _apply(df: pd.DataFrame, *, remove_dropped: bool) -> tuple[pd.DataFrame, int]:
        df = df.set_index("question_id")
        if has_keep and "keep" not in df.columns:
            df["keep"] = False
        changed = 0
        for qid, r in ed.iterrows():
            if qid not in df.index:
                continue
            if has_keep:  # persist the review "keep" flag regardless of text edits
                df.loc[qid, "keep"] = _as_bool(r["keep"])
            new_q, new_a = str(r["question"]), str(r["reference_answer"])
            if df.loc[qid, "question"] != new_q or df.loc[qid, "reference_answer"] != new_a:
                if df.loc[qid, "question"] != new_q and "fk_grade" in df.columns:
                    df.loc[qid, "fk_grade"] = round(fk_grade(new_q), 2)
                df.loc[qid, "question"] = new_q
                df.loc[qid, "reference_answer"] = new_a
                changed += 1
        if remove_dropped:
            # Excluded from the validated (benchmark) dataset entirely.
            df = df.drop(index=[i for i in drop_ids if i in df.index])
        elif drop_ids:
            # Kept in the originally-generated file, but marked so there's a record of the drop.
            if "dropped" not in df.columns:
                df["dropped"] = False
            for i in drop_ids:
                if i in df.index:
                    df.loc[i, "dropped"] = True
        return df.reset_index(), changed

    # Originally-generated file: apply text/keep edits, KEEP dropped rows (flagged `dropped`).
    out, n_changed = _apply(full_gt, remove_dropped=False)
    out.to_csv(cfg.path("ground_truth"), index=False)
    # Validated (benchmark) dataset: apply the same edits and REMOVE dropped rows entirely.
    vp = cfg.path("ground_truth_validated")
    if vp.exists():
        vout, _ = _apply(pd.read_csv(vp), remove_dropped=True)
        vout.to_csv(vp, index=False)
    return n_changed, len(drop_ids), n_kept


# --- Summary / provenance report ---------------------------------------------

def _area_titles(cfg: Config) -> dict[str, str]:
    """Map area code -> human title from standards.yaml (empty on any problem)."""
    try:
        import yaml
        d = yaml.safe_load(cfg.path("standards_yaml").read_text())
        return {k: (v or {}).get("title", "") for k, v in (d or {}).items()}
    except Exception:  # noqa: BLE001 - the summary is best-effort; never crash the page
        return {}


def _pipeline_mermaid(*, generator, judges, n_areas, generated, retained,
                      n_dropped, n_flag, fk_band, dedupe_thr,
                      auto_drop=False, n_auto_dropped=0) -> str:
    """A Mermaid flowchart of the build pipeline — renders as a diagram in GitHub/Obsidian/
    VS Code markdown, and stays readable as plain text where Mermaid isn't supported."""
    judge_lines = "".join(f"<br/>&bull; {j}" for j in judges)
    review = (f'Auto-drop doubted<br/>disagree / unanimous-wrong ({n_auto_dropped})'
              if auto_drop else f'Human review &amp; fix<br/>edit &middot; keep &middot; drop ({n_dropped})')
    return "\n".join([
        "```mermaid",
        "flowchart TD",
        f'    A["{n_areas} NGSS DCI areas<br/>(standards.yaml)"]'
        f' --> B["Generate Q&amp;A<br/>{generator}<br/>per standard &times; type"]',
        f'    B --> C["Auto-filter every candidate<br/>&bull; near-duplicate &ge; {dedupe_thr}'
        f'<br/>&bull; readability FK {fk_band[0]}&ndash;{fk_band[1]}<br/>&bull; LLM appropriateness"]',
        f'    C --> D["ground_truth.csv<br/>{generated} questions"]',
        f'    D --> E["{len(judges)}-judge validation panel'
        f'<br/>independent 0/1 verdict + rationale{judge_lines}"]',
        f'    E --> F["Majority label<br/>+ flag disagreements ({n_flag})"]',
        f'    F --> G["{review}"]',
        f'    G --> H["ground_truth_validated.csv<br/>{retained} questions &rarr; benchmark"]',
        "    style H fill:#dcfce7,stroke:#16a34a,stroke-width:2px",
        "    style D fill:#eef2ff,stroke:#6366f1",
        "```",
    ])


def _pipeline_boxes(*, generator, judges, n_areas, generated, retained,
                    n_dropped, n_flag, fk_band, dedupe_thr,
                    auto_drop=False, n_auto_dropped=0) -> list[dict]:
    """The pipeline steps as (title, detail lines, colour) — shared by the PNG renderer."""
    BLUE, GREEN, GREY = ("#eef2ff", "#6366f1"), ("#dcfce7", "#16a34a"), ("#f1f5f9", "#94a3b8")
    return [
        {"title": f"{n_areas} NGSS DCI areas", "lines": ["standards.yaml"], "c": GREY},
        {"title": "Generate Q&A", "lines": [generator, "per standard × type"], "c": GREY},
        {"title": "Auto-filter every candidate",
         "lines": [f"• near-duplicate  (rapidfuzz ≥ {dedupe_thr})",
                   f"• readability  FK {fk_band[0]}–{fk_band[1]}",
                   "• LLM appropriateness gate"], "c": GREY},
        {"title": "ground_truth.csv", "lines": [f"{generated} questions"], "c": BLUE},
        {"title": f"{len(judges)}-judge validation panel",
         "lines": ["independent 0/1 verdict + rationale",
                   *[f"• {j}" for j in judges]], "c": GREY},
        {"title": "Majority label", "lines": [f"+ flag disagreements ({n_flag})"], "c": GREY},
        ({"title": "Auto-drop doubted",
          "lines": [f"disagree / unanimous-wrong ({n_auto_dropped})"], "c": GREY} if auto_drop else
         {"title": "Human review & fix",
          "lines": [f"edit · keep · drop ({n_dropped})"], "c": GREY}),
        {"title": "ground_truth_validated.csv",
         "lines": [f"{retained} questions → benchmark"], "c": GREEN},
    ]


def _render_pipeline_png(boxes: list[dict]) -> bytes:
    """Draw the pipeline as a clean vertical flowchart PNG (Pillow only, 2× supersampled)."""
    from PIL import Image, ImageDraw, ImageFont
    F = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    S = 2  # supersample factor for crisp text
    title_f = ImageFont.truetype(FB, 21 * S)
    body_f = ImageFont.truetype(F, 16 * S)
    box_w, pad, line_gap, gap = 560 * S, 16 * S, 6 * S, 34 * S
    canvas_w = box_w + 80 * S

    def _th(font, text):
        b = font.getbbox(text)
        return b[3] - b[1]

    # measure box heights
    heights = []
    for bx in boxes:
        h = pad + _th(title_f, bx["title"]) + line_gap
        for ln in bx["lines"]:
            h += _th(body_f, ln) + line_gap
        heights.append(h + pad)
    total_h = sum(heights) + gap * (len(boxes) - 1) + 20 * S

    img = Image.new("RGB", (canvas_w, total_h), "white")
    d = ImageDraw.Draw(img)
    x0 = (canvas_w - box_w) // 2
    y = 10 * S
    centers = []
    for bx, h in zip(boxes, heights):
        fill, border = bx["c"]
        d.rounded_rectangle([x0, y, x0 + box_w, y + h], radius=12 * S,
                            fill=fill, outline=border, width=2 * S)
        ty = y + pad
        tw = title_f.getbbox(bx["title"])[2]
        d.text(((canvas_w - tw) // 2, ty), bx["title"], font=title_f, fill="#0f172a")
        ty += _th(title_f, bx["title"]) + line_gap + 2 * S
        for ln in bx["lines"]:
            lw = body_f.getbbox(ln)[2]
            d.text(((canvas_w - lw) // 2, ty), ln, font=body_f, fill="#334155")
            ty += _th(body_f, ln) + line_gap
        centers.append((x0 + box_w // 2, y, y + h))
        y += h + gap

    # arrows between boxes
    cx = canvas_w // 2
    for i in range(len(boxes) - 1):
        y1 = centers[i][2]
        y2 = centers[i + 1][1]
        d.line([cx, y1, cx, y2 - 6 * S], fill="#94a3b8", width=2 * S)
        d.polygon([(cx - 6 * S, y2 - 8 * S), (cx + 6 * S, y2 - 8 * S), (cx, y2)],
                  fill="#94a3b8")

    img = img.resize((canvas_w // S, total_h // S), Image.LANCZOS)
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _df_to_html(df: pd.DataFrame) -> str:
    """A DataFrame as a styled HTML table for the ODT/HTML report."""
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = []
    for row in df.itertuples(index=False, name=None):
        cells = "".join(f"<td>{'' if pd.isna(v) else v}</td>" for v in row)
        rows.append(f"<tr>{cells}</tr>")
    return (f"<table border='1' cellspacing='0' cellpadding='5'>"
            f"<tr>{head}</tr>{''.join(rows)}</table>")


def _html_to_odt(html: str, out_path: Path) -> bytes | None:
    """Convert an HTML string to ODT via the local LibreOffice (headless). None on failure.

    The working dir is kept next to `out_path` (under the repo, i.e. $HOME) rather than /tmp,
    because a snap-packaged LibreOffice runs in a private mount namespace and cannot read the
    host's /tmp. Args are passed as a list so spaces in the path are handled correctly.
    """
    import shutil
    soffice = shutil.which("libreoffice") or shutil.which("soffice")
    if not soffice:
        return None
    work = out_path.parent / ".lo_convert"
    try:
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)
        src = work / "report.html"
        src.write_text(html, encoding="utf-8")
        profile = (work / "profile").as_uri()
        try:
            subprocess.run(
                [soffice, "--headless", f"-env:UserInstallation={profile}",
                 "--convert-to", "odt:writer8", "--outdir", str(work), str(src)],
                check=True, capture_output=True, timeout=120)
        except (subprocess.SubprocessError, OSError):
            return None
        produced = work / "report.odt"
        if not produced.exists():
            return None
        data = produced.read_bytes()
        out_path.write_bytes(data)
        return data
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _df_to_md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table (no `tabulate` dependency)."""
    cols = [str(c) for c in df.columns]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *body])


def _pretty_reject_reason(reason: str) -> str:
    """Collapse the many `fk_grade=…_out_of_band` reasons into readability buckets."""
    r = str(reason)
    if r.startswith("fk_grade"):
        return "readability out of band"
    return {"duplicate": "near-duplicate",
            "llm_inappropriate": "LLM appropriateness gate"}.get(r, r)


def _gt_summary() -> None:
    cfg = _cfg()
    gt = _read_csv(cfg.path("ground_truth"))
    val = _read_csv(cfg.path("ground_truth_validated"))
    if gt is None:
        st.info("No ground truth yet — build it in the **Build** tab first.")
        return

    titles = _area_titles(cfg)
    type_order = ["recall", "conceptual", "applied"]
    area_order = list(titles) or sorted(gt["area"].dropna().unique())

    # --- provenance (from config) -------------------------------------------
    generator = cfg.get("generator_model", default="?")
    gt_judges = cfg.get("gt_judges", default=[]) or []
    qpa = int(cfg.get("questions_per_area", default=100))
    n_areas = gt["area"].nunique()
    target = qpa * n_areas
    fk_band = (cfg.get("appropriateness", default={}) or {}).get("fk_grade_range", [5, 9])
    dedupe_thr = (cfg.get("dedupe", default={}) or {}).get("similarity_threshold", 88)
    split = cfg.get("question_type_split", default={}) or {}
    # Actual judges are whoever scored in the validated file; fall back to the configured panel.
    judge_cols = [c for c in (val.columns if val is not None else []) if c.startswith("judge_")]
    judge_names = [c[len("judge_"):] for c in judge_cols] or list(gt_judges)

    # --- counts --------------------------------------------------------------
    generated = len(gt)
    if "dropped" in gt.columns:
        n_dropped = int(gt["dropped"].map(_as_bool).sum())
    elif val is not None:
        n_dropped = max(0, generated - len(val))
    else:
        n_dropped = 0
    retained = len(val) if val is not None else generated - n_dropped
    n_flag = 0
    n_maj0 = 0
    if val is not None:
        if "human_review_flag" in val.columns:
            n_flag = int(pd.to_numeric(val["human_review_flag"], errors="coerce").fillna(0).sum())
        if "majority_label" in val.columns:
            n_maj0 = int((pd.to_numeric(val["majority_label"], errors="coerce") == 0).sum())
    n_kept = int(gt["keep"].map(_as_bool).sum()) if "keep" in gt.columns else 0

    # Ground-truth review mode: human-review queue vs. automatic drop of doubted questions.
    auto_drop = bool(cfg.get("validation", "auto_drop_disagreements", default=False))
    auto_rec = _read_csv(cfg.path("human_review_dir") / "auto_dropped_groundtruth.csv")
    n_auto_dropped = len(auto_rec) if auto_rec is not None else (n_dropped if auto_drop else 0)

    st.subheader("At a glance")
    m = st.columns(5)
    m[0].metric("Areas", n_areas)
    m[1].metric("Generated", generated)
    m[2].metric("Validated (final)", retained)
    m[3].metric("Auto-dropped" if auto_drop else "Dropped", n_auto_dropped if auto_drop else n_dropped)
    m[4].metric("Flagged", n_flag)
    if auto_drop:
        st.info(f"**Auto-drop is ON.** Questions the judges doubted (disagreement or unanimous-wrong) "
                f"are removed from the benchmark set automatically — **{n_auto_dropped}** dropped, "
                f"recorded in `results/human_review/auto_dropped_groundtruth.csv`. "
                f"`ground_truth.csv` is unchanged.")
    st.caption(
        f"Generator: **{generator}** · Judges: **{', '.join(gt_judges) or '—'}** · "
        f"GT review: **{'auto-drop doubted' if auto_drop else 'human review'}** · "
        f"Target {qpa}/area ({', '.join(f'{k} {v}' for k, v in split.items()) or 'even split'}) · "
        f"Readability band FK {fk_band[0]}–{fk_band[1]} · Dedupe ≥ {dedupe_thr}")

    # --- pipeline diagram ----------------------------------------------------
    st.subheader("How the ground truth was built")
    judge_dot = "".join(f"\\n• {j}" for j in judge_names)  # DOT uses \n for line breaks
    dot = f"""
    digraph gt {{
      rankdir=TB; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fontname="Helvetica",
            fillcolor="#eef2ff", color="#6366f1", fontsize=11];
      edge [color="#94a3b8"];
      a [label="{n_areas} NGSS DCI areas\\n(standards.yaml)"];
      b [label="Generate Q&A\\n{generator}\\nper standard × type"];
      c [label="Auto-filter every candidate\\n• near-duplicate (rapidfuzz ≥ {dedupe_thr})\\n• readability FK {fk_band[0]}–{fk_band[1]}\\n• LLM appropriateness gate"];
      d [label="ground_truth.csv\\n{generated} questions"];
      e [label="{len(judge_names)}-judge validation panel\\nindependent 0/1 verdict + rationale{judge_dot}"];
      f [label="Majority label\\n+ flag disagreements ({n_flag})"];
      g [label="{'Auto-drop doubted' if auto_drop else 'Human review & fix'}\\n{('disagree / unanimous-wrong (' + str(n_auto_dropped) + ')') if auto_drop else ('edit · keep · drop (' + str(n_dropped) + ')')}"];
      h [label="ground_truth_validated.csv\\n{retained} questions → benchmark",
         fillcolor="#dcfce7", color="#16a34a"];
      a -> b -> c -> d -> e -> f -> g -> h;
    }}
    """
    st.graphviz_chart(dot, width="stretch")

    # --- generation funnel ---------------------------------------------------
    st.subheader("Generation & filtering")
    rej = _read_csv(cfg.path("reports_dir") / "generation_rejects.csv")
    if rej is not None and "reason" in rej.columns and len(rej):
        rej = rej.copy()
        rej["bucket"] = rej["reason"].map(_pretty_reject_reason)
        by_reason = (rej.groupby("bucket").size().sort_values(ascending=False)
                     .rename("candidates rejected").reset_index())
        st.caption(f"{len(rej):,} candidate questions were generated and rejected before "
                   f"the {generated} that were kept — each regenerated until the slot filled.")
        st.dataframe(by_reason, width="stretch", hide_index=True)
    else:
        st.caption("No `generation_rejects.csv` found (rejections weren't logged).")

    # --- final composition by area ------------------------------------------
    st.subheader("Final dataset composition")
    final = val if val is not None else gt
    rows = []
    for area in area_order:
        sub = final[final["area"] == area]
        if sub.empty:
            continue
        row = {"area": area, "title": titles.get(area, ""), "questions": len(sub),
               "standards": sub["standard"].nunique()}
        for t in type_order:
            row[t] = int((sub["question_type"] == t).sum())
        if "fk_grade" in sub.columns:
            row["mean FK"] = round(pd.to_numeric(sub["fk_grade"], errors="coerce").mean(), 2)
        rows.append(row)
    comp = pd.DataFrame(rows)
    if not comp.empty:
        total_row = {"area": "TOTAL", "title": "", "questions": int(comp["questions"].sum()),
                     "standards": int(comp["standards"].sum())}
        for t in type_order:
            total_row[t] = int(comp[t].sum())
        comp = pd.concat([comp, pd.DataFrame([total_row])], ignore_index=True)
    st.caption("Counts reflect the **validated** (final) dataset — what the benchmark uses.")
    st.dataframe(comp, width="stretch", hide_index=True)
    if not comp.empty:
        chart = comp[comp["area"] != "TOTAL"].set_index("area")[type_order]
        st.bar_chart(chart)

    # --- per-standard counts -------------------------------------------------
    with st.expander("Per-standard counts"):
        per_std = (final.groupby(["area", "standard", "question_type"]).size()
                   .unstack("question_type", fill_value=0))
        for t in type_order:
            if t not in per_std.columns:
                per_std[t] = 0
        per_std = per_std[type_order]
        per_std["total"] = per_std.sum(axis=1)
        st.dataframe(per_std.reset_index(), width="stretch", hide_index=True)

    # --- where drops came from ----------------------------------------------
    if n_dropped and "dropped" in gt.columns:
        st.subheader("Dropped items")
        dropped = gt[gt["dropped"].map(_as_bool)]
        by_cat = (dropped.groupby(["area", "question_type"]).size()
                  .rename("dropped").reset_index())
        st.caption(f"{n_dropped} item(s) excluded from the validated dataset (retained, "
                   "flagged, in `ground_truth.csv`).")
        st.dataframe(by_cat, width="stretch", hide_index=True)

    # --- judge agreement (original) -----------------------------------------
    # Compute over EVERY judged item (validated survivors + auto-dropped), so auto-drop — which
    # removes exactly the disagreements — doesn't inflate agreement to a meaningless ~100%.
    from src.utils import load_judged
    judged = load_judged(cfg)
    if judged is not None and len(judge_cols) >= 2:
        st.subheader("Original judge agreement")
        matrix, pairs_df, n_all = _judge_agreement_data(judged, judge_cols)
        st.caption(f"Pairwise: percent of items where two judges gave the same 0/1 verdict "
                   f"(over items both scored). {n_all} item(s) scored by all judges"
                   + (" (includes auto-dropped questions)." if auto_drop else "."))
        st.dataframe(matrix, width="stretch")
        st.dataframe(pairs_df, width="stretch", hide_index=True)

        st.markdown("**By judge model**")
        st.caption("Each judge's verdict rate and how often it matched the panel majority.")
        st.dataframe(_judge_pass_rates(judged, judge_cols), width="stretch", hide_index=True)
        if n_maj0:
            st.caption(f"Panel majority said **incorrect** on {n_maj0} item(s).")

    # --- exportable written report ------------------------------------------
    st.subheader("Export a shareable report")
    reports = cfg.path("reports_dir")
    reports.mkdir(parents=True, exist_ok=True)

    boxes = _pipeline_boxes(
        generator=generator, judges=judge_names, n_areas=n_areas, generated=generated,
        retained=retained, n_dropped=n_dropped, n_flag=n_flag, fk_band=fk_band,
        dedupe_thr=dedupe_thr, auto_drop=auto_drop, n_auto_dropped=n_auto_dropped)

    report = _gt_summary_markdown(
        cfg, generator=generator, gt_judges=judge_names, qpa=qpa, n_areas=n_areas,
        target=target, fk_band=fk_band, dedupe_thr=dedupe_thr, split=split,
        generated=generated, retained=retained, n_dropped=n_dropped, n_flag=n_flag,
        n_maj0=n_maj0, comp=comp, val=val, judge_cols=judge_cols, rej=rej,
        auto_drop=auto_drop, n_auto_dropped=n_auto_dropped)
    md_path = reports / "gt_provenance_summary.md"
    md_path.write_text(report)
    # The PNG diagram is best-effort: Pillow or the bundled font may be missing on some
    # machines. If it fails, degrade to the in-app Graphviz diagram (already shown above)
    # rather than crashing the whole tab.
    diagram_png = None
    try:
        diagram_png = _render_pipeline_png(boxes)
    except Exception:  # noqa: BLE001 - diagram is optional; the tab must still render
        pass

    # Signature of the current dataset state, so a cached ODT built for an earlier state is
    # never served after the data changes (the tables/diagram recompute every rerun; the ODT
    # is only rebuilt on demand). Hash the report text + diagram bytes — both derive from all
    # the inputs the document contains.
    import hashlib
    sig = hashlib.md5(report.encode("utf-8") + (diagram_png or b"")).hexdigest()

    st.markdown("**ODT** — a formatted word-processor document (opens in Word, LibreOffice, "
                "Google Docs) with the embedded diagram and real tables. **Markdown** — plain "
                "text with a Mermaid diagram, best for GitHub/wikis.")
    c1, c2 = st.columns(2)
    # ODT conversion spawns LibreOffice (~seconds), so build it only on demand and cache it.
    import shutil as _sh
    have_lo = bool(_sh.which("libreoffice") or _sh.which("soffice"))
    if c1.button("🖹 Build ODT document", disabled=not (have_lo and diagram_png is not None)):
        with st.spinner("Rendering ODT via LibreOffice…"):
            try:
                html = _gt_summary_html(
                    diagram_png=diagram_png, generator=generator, judges=judge_names, qpa=qpa,
                    n_areas=n_areas, fk_band=fk_band, dedupe_thr=dedupe_thr, split=split,
                    generated=generated, retained=retained, n_dropped=n_dropped, n_flag=n_flag,
                    n_maj0=n_maj0, comp=comp, val=val, judge_cols=judge_cols, rej=rej,
                    auto_drop=auto_drop, n_auto_dropped=n_auto_dropped)
                st.session_state["gt_odt"] = {
                    "sig": sig, "bytes": _html_to_odt(html, reports / "gt_provenance_summary.odt")}
            except Exception as exc:  # noqa: BLE001 - never break the page over a convert issue
                st.session_state["gt_odt"] = None
                st.error(f"ODT build failed: {exc}")
    cached = st.session_state.get("gt_odt")
    # Only serve the cached bytes if they were built for the current dataset state.
    odt_bytes = cached["bytes"] if cached and cached.get("sig") == sig else None
    if odt_bytes:
        c1.download_button("⬇ Download ODT", odt_bytes,
                           file_name="gt_provenance_summary.odt",
                           mime="application/vnd.oasis.opendocument.text")
        c1.caption(f"Saved to `{reports / 'gt_provenance_summary.odt'}`.")
    elif cached and cached.get("bytes"):
        c1.caption("Dataset changed since the last build — click **Build ODT** to refresh it.")
    elif diagram_png is None:
        c1.caption("ODT needs the pipeline diagram, which couldn't be rendered on this machine.")
    elif not have_lo:
        c1.caption("ODT needs LibreOffice installed (not found on this machine).")
    c2.download_button("⬇ Download Markdown", report,
                       file_name="gt_provenance_summary.md", mime="text/markdown")
    c2.caption(f"Saved to `{md_path}`.")

    if diagram_png is not None:
        st.image(diagram_png, caption="Pipeline diagram (as embedded in the ODT)")
    with st.expander("Preview the written summary"):
        st.markdown(report)


def _gt_summary_markdown(cfg, *, generator, gt_judges, qpa, n_areas, target, fk_band,
                         dedupe_thr, split, generated, retained, n_dropped, n_flag,
                         n_maj0, comp, val, judge_cols, rej,
                         auto_drop=False, n_auto_dropped=0) -> str:
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    split_txt = ", ".join(f"{k} {v}" for k, v in split.items()) or "even split"
    lines = [
        "# Ground-truth dataset: how it was built",
        f"_Generated {stamp}._",
        "",
        "## Overview",
        f"A ground-truth question–answer set for **{n_areas} NGSS middle-school science "
        f"areas** was generated by **{generator}** (target {qpa} questions/area, "
        f"types: {split_txt}), then independently validated by a "
        f"**{len(gt_judges)}-judge LLM panel**: {', '.join(gt_judges) or '—'}.",
        "",
        "Every candidate question passed three automatic gates before entering the set: "
        f"near-duplicate removal (rapidfuzz token-set ratio ≥ {dedupe_thr}), a "
        f"middle-school readability band (Flesch–Kincaid {fk_band[0]}–{fk_band[1]}), and an "
        "LLM appropriateness check; failures were dropped and regenerated for the same "
        "(area, standard, type) slot until the target was met.",
        "",
        "## Pipeline",
        _pipeline_mermaid(generator=generator, judges=gt_judges, n_areas=n_areas,
                          generated=generated, retained=retained, n_dropped=n_dropped,
                          n_flag=n_flag, fk_band=fk_band, dedupe_thr=dedupe_thr,
                          auto_drop=auto_drop, n_auto_dropped=n_auto_dropped),
        "",
        "<details><summary>Steps (text)</summary>",
        "",
        "1. **Standards** — 12 NGSS DCI areas (`data/standards.yaml`).",
        f"2. **Generate** — {generator} writes Q&A per standard × question type.",
        "3. **Auto-filter** — dedupe + readability band + LLM appropriateness gate.",
        f"4. **ground_truth.csv** — {generated} questions kept.",
        f"5. **Validate** — {len(gt_judges)} judges give independent 0/1 verdicts + rationale.",
        f"6. **Majority + flag** — disagreements flagged ({n_flag}).",
        (f"7. **Auto-drop** — doubted questions (disagreement or unanimous-wrong) removed "
         f"automatically ({n_auto_dropped} dropped); recorded in "
         "`results/human_review/auto_dropped_groundtruth.csv`."
         if auto_drop else
         f"7. **Human review** — edit / keep / drop ({n_dropped} dropped)."),
        f"8. **ground_truth_validated.csv** — {retained} questions used by the benchmark.",
        "",
        "</details>",
        "",
        "## Counts",
        f"- Generated: **{generated}**",
        f"- Validated (final, used by benchmark): **{retained}**",
        (f"- Auto-dropped as doubted (disagreement or unanimous-wrong): **{n_auto_dropped}** "
         "(recorded in `results/human_review/auto_dropped_groundtruth.csv`; `ground_truth.csv` intact)"
         if auto_drop else
         f"- Dropped from validated set: **{n_dropped}** (retained & flagged in `ground_truth.csv`)"),
        f"- Flagged for human review (judge disagreement): **{n_flag}**",
        f"- Panel majority said *incorrect*: **{n_maj0}**",
        "",
        "## Final composition by area",
    ]
    if comp is not None and not comp.empty:
        lines.append(_df_to_md(comp))
    if rej is not None and "reason" in rej.columns and len(rej):
        buckets = (rej["reason"].map(_pretty_reject_reason).value_counts()
                   .rename_axis("reason").rename("rejected").reset_index())
        lines += ["", "## Candidates rejected during generation",
                  f"{len(rej):,} total.", "", _df_to_md(buckets)]
    if val is not None and len(judge_cols) >= 2:
        matrix, pairs_df, n_all = _judge_agreement_data(val, judge_cols)
        lines += ["", "## Original judge agreement",
                  f"Pairwise percent agreement (over items both judges scored; "
                  f"{n_all} scored by all).", "",
                  _df_to_md(matrix.rename_axis("judge").reset_index()),
                  "", "### By judge model",
                  _df_to_md(_judge_pass_rates(val, judge_cols))]
    return "\n".join(lines) + "\n"


def _gt_summary_html(*, diagram_png, generator, judges, qpa, n_areas, fk_band, dedupe_thr,
                     split, generated, retained, n_dropped, n_flag, n_maj0, comp, val,
                     judge_cols, rej, auto_drop=False, n_auto_dropped=0) -> str:
    """A styled, self-contained HTML report (embedded PNG diagram) for conversion to ODT."""
    import base64
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    split_txt = ", ".join(f"{k} {v}" for k, v in split.items()) or "even split"
    img_b64 = base64.b64encode(diagram_png).decode()
    parts = [
        "<html><head><meta charset='utf-8'><style>",
        "body{font-family:'Liberation Sans',Arial,sans-serif;color:#0f172a;}",
        "h1{font-size:22pt;} h2{font-size:15pt;color:#3730a3;margin-top:18pt;}",
        "table{border-collapse:collapse;} th{background:#eef2ff;text-align:left;}",
        "th,td{border:1px solid #cbd5e1;padding:4px 8px;font-size:10pt;}",
        "p,li{font-size:11pt;line-height:1.4;}",
        "</style></head><body>",
        "<h1>Ground-truth dataset: how it was built</h1>",
        f"<p><i>Generated {stamp}.</i></p>",
        "<h2>Overview</h2>",
        f"<p>A ground-truth question&ndash;answer set for <b>{n_areas} NGSS middle-school "
        f"science areas</b> was generated by <b>{generator}</b> (target {qpa} questions/area, "
        f"types: {split_txt}), then independently validated by a "
        f"<b>{len(judges)}-judge LLM panel</b>: {', '.join(judges) or '&mdash;'}.</p>",
        f"<p>Every candidate question passed three automatic gates before entering the set: "
        f"near-duplicate removal (rapidfuzz token-set ratio &ge; {dedupe_thr}), a "
        f"middle-school readability band (Flesch&ndash;Kincaid {fk_band[0]}&ndash;{fk_band[1]}), "
        f"and an LLM appropriateness check; failures were dropped and regenerated for the same "
        f"(area, standard, type) slot until the target was met.</p>",
        "<h2>Pipeline</h2>",
        f"<p><img src='data:image/png;base64,{img_b64}' width='460'/></p>",
        "<h2>Counts</h2><ul>",
        f"<li>Generated: <b>{generated}</b></li>",
        f"<li>Validated (final, used by benchmark): <b>{retained}</b></li>",
        (f"<li>Auto-dropped as doubted (disagreement or unanimous-wrong): <b>{n_auto_dropped}</b> "
         "(recorded in auto_dropped_groundtruth.csv; ground_truth.csv intact)</li>"
         if auto_drop else
         f"<li>Dropped from validated set: <b>{n_dropped}</b> (retained &amp; flagged in "
         "ground_truth.csv)</li>"),
        f"<li>Flagged for human review (judge disagreement): <b>{n_flag}</b></li>",
        f"<li>Panel majority said <i>incorrect</i>: <b>{n_maj0}</b></li></ul>",
    ]
    if comp is not None and not comp.empty:
        parts += ["<h2>Final composition by area</h2>", _df_to_html(comp)]
    if rej is not None and "reason" in rej.columns and len(rej):
        buckets = (rej["reason"].map(_pretty_reject_reason).value_counts()
                   .rename_axis("reason").rename("rejected").reset_index())
        parts += ["<h2>Candidates rejected during generation</h2>",
                  f"<p>{len(rej):,} total.</p>", _df_to_html(buckets)]
    if val is not None and len(judge_cols) >= 2:
        matrix, _pairs, n_all = _judge_agreement_data(val, judge_cols)
        parts += ["<h2>Original judge agreement</h2>",
                  f"<p>Pairwise percent agreement (over items both judges scored; "
                  f"{n_all} scored by all).</p>",
                  _df_to_html(matrix.rename_axis("judge").reset_index()),
                  "<h3>By judge model</h3>",
                  _df_to_html(_judge_pass_rates(val, judge_cols))]
    parts.append("</body></html>")
    return "".join(parts)


def page_human_review() -> None:
    st.header("Human review")
    st.caption("Fill `human_score` (1 = correct, 0 = incorrect), Save, then recompute. "
               "These labels override the LLM judges and drop invalid items. "
               "To **fix** an inaccurate reference answer (instead of dropping it), use the "
               "**Ground truth** page.")
    cfg = _cfg()
    candidates = []
    hr = cfg.path("human_review_dir")
    if hr.exists():
        candidates += sorted(Path(hr).glob("flags_*.csv"))
    cal = cfg.path("calibration_dir") / "to_label.csv"
    if cal.exists():
        candidates.append(cal)
    if not candidates:
        st.info("No review files yet. Run **validate** / **evaluate** (triple) / **calibrate** first.")
        return

    labels = [str(p.relative_to(REPO_ROOT)) for p in candidates]
    pick = st.selectbox("File to edit", labels)
    path = REPO_ROOT / pick
    df = pd.read_csv(path)
    st.write(f"{len(df)} rows")
    edited = st.data_editor(df, width="stretch", num_rows="fixed", key=pick)
    if st.button("Save labels"):
        edited.to_csv(path, index=False)
        st.success(f"Saved {pick}.")

    st.subheader("Recompute with the new labels")
    c1, c2 = st.columns(2)
    if c1.button("Recompute judge calibration"):
        stream_command(_py("-m", "src.calibrate_judges"))
    if c2.button("Re-run analysis"):
        stream_command(_py("-m", "src.analyze"))

    jvh = cfg.path("calibration_dir") / "judge_vs_human.csv"
    if jvh.exists():
        st.subheader("Judge ↔ human agreement")
        st.dataframe(pd.read_csv(jvh), width="stretch")


# --------------------------------------------------------------------------- #
# Nav
# --------------------------------------------------------------------------- #
PAGES = {
    "Configure": page_config,
    "Standards": page_standards,
    "Ground truth": page_ground_truth,
    "Benchmark": page_benchmark,
    "Results": page_results,
    "Human review": page_human_review,
}

st.sidebar.title("NGSS LLM Benchmark")
st.sidebar.caption("Middle-school science Q&A benchmark control panel")
choice = st.sidebar.radio("Go to", list(PAGES))
st.sidebar.divider()
st.sidebar.caption("Stages run as subprocesses with checkpoint/resume — safe to re-run.")
PAGES[choice]()
