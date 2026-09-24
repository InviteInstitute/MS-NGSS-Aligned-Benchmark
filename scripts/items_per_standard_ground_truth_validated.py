"""Item-count coverage table for the validated ground-truth dataset.

For each of the 59 NGSS standards, counts how many recall/conceptual/applied questions are
in data/ground_truth_validated.csv (i.e., survived generation + judge validation). One row
per standard, one column per question type plus a total -- sized for direct use in an
academic paper, not the long (area, type, standard) format the pipeline's other reports use.
Each area's standard rows are followed by a "Subtotal" row for that area, and the table ends
with a grand "Total" row.

Standards are enumerated from data/standards.yaml (via src.standards.load_areas), so a
standard with zero validated questions still gets a row (visible as 0s) instead of silently
disappearing.

CLI:  python scripts/items_per_standard_ground_truth_validated.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.standards import load_areas

INPUT_FILE = ROOT / "data" / "ground_truth_validated.csv"
OUTPUT_FILE = ROOT / "data" / "items_per_standard_ground_truth_validated.csv"

QTYPE_ORDER = ["recall", "conceptual", "applied"]


def main() -> None:
    df = pd.read_csv(INPUT_FILE)

    required_cols = {"area", "standard", "question_type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    areas = load_areas(Config.load(require_keys=False))
    counts = df.groupby(["area", "standard", "question_type"]).size()

    rows = []
    grand = {qt: 0 for qt in QTYPE_ORDER}
    for area in areas:
        area_sum = {qt: 0 for qt in QTYPE_ORDER}
        for std in area.standards:
            rec = {"area": area.code, "standard": std.code}
            for qt in QTYPE_ORDER:
                n = int(counts.get((area.code, std.code, qt), 0))
                rec[qt] = n
                area_sum[qt] += n
            rec["total"] = sum(rec[qt] for qt in QTYPE_ORDER)
            rows.append(rec)

        subtotal = {"area": area.code, "standard": "Subtotal", **area_sum}
        subtotal["total"] = sum(area_sum.values())
        rows.append(subtotal)
        for qt in QTYPE_ORDER:
            grand[qt] += area_sum[qt]

    rows.append({"area": "All areas", "standard": "Total", **grand,
                 "total": sum(grand.values())})

    table_df = pd.DataFrame(rows)
    table_df.to_csv(OUTPUT_FILE, index=False)

    std_rows = table_df[~table_df["standard"].isin(["Subtotal", "Total"])]
    zero_cov = std_rows[std_rows["total"] == 0]
    if len(zero_cov):
        print(f"WARNING: {len(zero_cov)} standard(s) have zero validated questions: "
              f"{', '.join(zero_cov['standard'])}")

    print(f"Done. {len(std_rows)} standards, {grand} by type, "
          f"{sum(grand.values())} questions total.")
    print(f"Saved to {OUTPUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
