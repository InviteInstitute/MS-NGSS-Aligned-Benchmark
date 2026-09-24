"""Load and validate the single config.yaml + .env, and resolve model entries.

A model "key" (e.g. "gpt-4o") referenced anywhere in the config is resolved to a
ResolvedModel with its base_url, api_key (read from the named env var), model id, and
family. `Config.load()` fails fast with a clear message if anything is missing.

CLI:  python -m src.config --check
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"


class ConfigError(Exception):
    """Raised when configuration is invalid or incomplete."""


@dataclass(frozen=True)
class ResolvedModel:
    """A registry entry with its secret resolved from the environment."""

    key: str
    base_url: str
    api_key: str
    model: str
    family: str
    # Optional per-model temperature handling:
    #   temperature      -> always send this exact value (e.g. a model that requires temp=1)
    #   omit_temperature -> never send a temperature for this model (use the model's own default)
    temperature: float | None = None
    omit_temperature: bool = False


@dataclass
class Config:
    raw: dict[str, Any]
    models: dict[str, ResolvedModel] = field(default_factory=dict)

    # ---- convenience accessors -------------------------------------------------
    def get(self, *path: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def path(self, name: str) -> Path:
        rel = self.get("paths", name)
        if rel is None:
            raise ConfigError(f"paths.{name} is not defined in config.yaml")
        return REPO_ROOT / rel

    def model(self, key: str) -> ResolvedModel:
        if key not in self.models:
            raise ConfigError(f"Model '{key}' is referenced but not resolved.")
        return self.models[key]

    # ---- loading ---------------------------------------------------------------
    @classmethod
    def load(cls, config_path: Path | None = None, *, require_keys: bool = True) -> "Config":
        path = config_path or CONFIG_PATH
        if not path.exists():
            raise ConfigError(f"config.yaml not found at {path}")
        load_dotenv(REPO_ROOT / ".env")
        raw = yaml.safe_load(path.read_text()) or {}
        cfg = cls(raw=raw)
        cfg._resolve_models(require_keys=require_keys)
        cfg._validate()
        return cfg

    def _referenced_model_keys(self) -> set[str]:
        keys: set[str] = set()
        for single in ("generator_model", "eval_judge"):
            if self.get(single):
                keys.add(self.get(single))
        for many in ("gt_judges", "benchmark_models", "eval_judges"):
            keys.update(self.get(many, default=[]) or [])
        return keys

    def _resolve_models(self, *, require_keys: bool) -> None:
        registry = self.get("models", default={}) or {}
        referenced = self._referenced_model_keys()
        missing_registry = sorted(k for k in referenced if k not in registry)
        if missing_registry:
            raise ConfigError(
                "These models are referenced by a role but absent from the `models` "
                f"registry: {missing_registry}"
            )
        missing_env: list[str] = []
        for key, entry in registry.items():
            env_name = entry.get("api_key_env")
            api_key = os.environ.get(env_name, "") if env_name else ""
            if require_keys and key in referenced and not api_key:
                missing_env.append(f"{key} (needs {env_name} in .env)")
            temp = entry.get("temperature")
            self.models[key] = ResolvedModel(
                key=key,
                base_url=entry.get("base_url", ""),
                api_key=api_key or "missing",
                model=entry.get("model", ""),
                family=entry.get("family", "unknown"),
                temperature=None if temp is None else float(temp),
                omit_temperature=bool(entry.get("omit_temperature", False)),
            )
        if missing_env:
            raise ConfigError(
                "Missing API keys for referenced models:\n  - "
                + "\n  - ".join(missing_env)
                + "\nAdd them to .env (see .env.example)."
            )

    def _validate(self) -> None:
        split = self.get("question_type_split", default={}) or {}
        total = sum(split.values())
        qpa = self.get("questions_per_area")
        if qpa is not None and total != qpa:
            raise ConfigError(
                f"question_type_split sums to {total} but questions_per_area is {qpa}."
            )
        mode = self.get("eval_mode")
        if mode not in ("single", "triple"):
            raise ConfigError(f"eval_mode must be 'single' or 'triple', got {mode!r}.")
        sysp = self.get("benchmark_protocol", "system_prompt")
        # Lazy import to avoid a cycle at module import time.
        from src import prompts

        if sysp not in prompts.SYSTEM_PROMPTS:
            raise ConfigError(
                f"benchmark_protocol.system_prompt '{sysp}' is not defined in prompts.SYSTEM_PROMPTS "
                f"(available: {sorted(prompts.SYSTEM_PROMPTS)})."
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate config.yaml + .env.")
    parser.add_argument("--check", action="store_true", help="Validate and print a summary.")
    parser.parse_args(argv)
    try:
        cfg = Config.load()
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1
    referenced = sorted(cfg._referenced_model_keys())
    print("config.yaml + .env OK")
    print(f"  areas source : {cfg.path('standards_yaml').relative_to(REPO_ROOT)}")
    print(f"  generator    : {cfg.get('generator_model')}")
    print(f"  gt judges    : {cfg.get('gt_judges')}")
    print(f"  benchmark    : {cfg.get('benchmark_models')}")
    print(f"  eval mode    : {cfg.get('eval_mode')}")
    print(f"  referenced   : {referenced}")
    print(f"  per area     : {cfg.get('questions_per_area')} "
          f"(split {cfg.get('question_type_split')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
