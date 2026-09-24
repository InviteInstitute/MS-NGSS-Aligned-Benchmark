import pandas as pd

# ---- CONFIG ----
INPUT_FILE = "data/ground_truth_validated.csv"
OUTPUT_FILE = "data/human_review_selection.csv"
SUMMARY_FILE = "data/human_review_standard_coverage.csv"
SUMMARY_TABLE_FILE = "data/human_review_standard_coverage_table.csv"

QTYPE_ORDER = ["recall", "conceptual", "applied"]

TARGETS = {"recall": 3, "conceptual": 3, "applied": 4}

# ---- LOAD DATA ----
df = pd.read_csv(INPUT_FILE)

# Expect columns: question_id, area, standard, question_type
required_cols = {"question_id", "area", "standard", "question_type"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

def round_robin_select(group_df, n):
    """Cycle through unique standards, picking one question at a time,
    until n questions are selected or the group is exhausted."""
    standards = list(dict.fromkeys(group_df["standard"]))  # unique, in order of appearance
    buckets = {s: group_df[group_df["standard"] == s].to_dict("records") for s in standards}
    selected = []
    i = 0
    while len(selected) < n and any(buckets.values()):
        s = standards[i % len(standards)]
        if buckets[s]:
            selected.append(buckets[s].pop(0))
        i += 1
        if i > 100000:  # safety valve
            break
    return selected[:n]

# ---- BUILD REVIEW SET ----
results = []
for area, area_df in df.groupby("area"):
    for qtype, n in TARGETS.items():
        sub = area_df[area_df["question_type"] == qtype]
        picked = round_robin_select(sub, n)
        if len(picked) < n:
            print(f"WARNING: {area} / {qtype} only had {len(picked)} of {n} needed.")
        results.extend(picked)

review_df = pd.DataFrame(results)

# ---- COVERAGE SUMMARY ----
summary_df = (
    review_df.groupby(["area", "question_type", "standard"])
    .size()
    .reset_index(name="count")
)

# ---- PAPER-READY SUMMARY TABLE (one row per standard, one column per type) ----
table_df = (
    summary_df.pivot_table(index=["area", "standard"], columns="question_type",
                            values="count", fill_value=0)
    .reindex(columns=QTYPE_ORDER, fill_value=0)
    .astype(int)
    .reset_index()
    .sort_values(["area", "standard"])
)
table_df["total"] = table_df[QTYPE_ORDER].sum(axis=1)

# ---- SAVE OUTPUT ----
review_df.to_csv(OUTPUT_FILE, index=False)
summary_df.to_csv(SUMMARY_FILE, index=False)
table_df.to_csv(SUMMARY_TABLE_FILE, index=False)

print(f"Done. Selected {len(review_df)} questions across {df['area'].nunique()} areas.")
print(f"Saved to {OUTPUT_FILE}, {SUMMARY_FILE}, and {SUMMARY_TABLE_FILE}")