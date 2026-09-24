"""Load data/standards.yaml (the 12 DCI areas) and render docs/areas_overview.md.

The YAML is the single editable source of truth. This module validates it, exposes typed
access for the generation stage, and can regenerate the human-readable review doc.

CLI:  python -m src.standards --doc      # (re)write docs/areas_overview.md
      python -m src.standards --list     # print a compact summary
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.config import REPO_ROOT, Config


@dataclass(frozen=True)
class Standard:
    code: str
    text: str


@dataclass(frozen=True)
class Area:
    code: str          # DCI code, e.g. "MS-PS3"
    title: str
    description: str
    standards: tuple[Standard, ...]


def load_areas(cfg: Config | None = None) -> list[Area]:
    cfg = cfg or Config.load(require_keys=False)
    path = cfg.path("standards_yaml")
    raw = yaml.safe_load(Path(path).read_text()) or {}
    areas: list[Area] = []
    for code, entry in raw.items():
        standards = tuple(
            Standard(code=s["code"], text=s["text"].strip())
            for s in entry.get("standards", [])
        )
        if not standards:
            raise ValueError(f"Area {code} has no standards in {path}.")
        areas.append(
            Area(
                code=code,
                title=entry.get("title", code).strip(),
                description=" ".join(entry.get("description", "").split()),
                standards=standards,
            )
        )
    if len(areas) != 12:
        raise ValueError(f"Expected 12 areas in {path}, found {len(areas)}: {[a.code for a in areas]}")
    return areas


def render_doc(areas: list[Area]) -> str:
    lines = [
        "# NGSS Middle-School Science — Areas Overview",
        "",
        "Generated from `data/standards.yaml` by `python -m src.standards --doc`.",
        "**Review this file**, then edit `data/standards.yaml` (the source of truth) and",
        "regenerate. The 12 areas are the NGSS disciplinary-core-idea (DCI) codes.",
        "",
        f"**Total:** {len(areas)} areas, "
        f"{sum(len(a.standards) for a in areas)} performance expectations.",
        "",
    ]
    for a in areas:
        lines.append(f"## {a.code} — {a.title}")
        lines.append("")
        lines.append(a.description)
        lines.append("")
        lines.append("| Standard | Performance expectation |")
        lines.append("| --- | --- |")
        for s in a.standards:
            text = s.text.replace("|", "\\|")
            lines.append(f"| **{s.code}** | {text} |")
        lines.append("")
    return "\n".join(lines)


def write_doc(cfg: Config | None = None) -> Path:
    cfg = cfg or Config.load(require_keys=False)
    areas = load_areas(cfg)
    out = cfg.path("areas_doc")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_doc(areas))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NGSS standards utilities.")
    parser.add_argument("--doc", action="store_true", help="(Re)write docs/areas_overview.md.")
    parser.add_argument("--list", action="store_true", help="Print a compact summary.")
    args = parser.parse_args(argv)
    areas = load_areas()
    if args.doc:
        out = write_doc()
        print(f"Wrote {out.relative_to(REPO_ROOT)} ({len(areas)} areas).")
    if args.list or not args.doc:
        for a in areas:
            print(f"{a.code:9s} {a.title}  ({len(a.standards)} standards)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
