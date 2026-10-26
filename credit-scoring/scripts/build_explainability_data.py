from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "home-credit-default-risk" / "application_train.csv"
OUTPUT_PATH = ROOT / "frontend" / "explainability-data.json"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Training dataset not found: {DATA_PATH}")

df = pd.read_csv(
    DATA_PATH,
    usecols=["TARGET", "DAYS_BIRTH", "AMT_INCOME_TOTAL",
             "AMT_CREDIT"],
)

df = df[df["TARGET"].isin([0, 1])].copy()
df["age"] = (-df["DAYS_BIRTH"] / 365.25).where(df["DAYS_BIRTH"].notna())
df["age_group"] = (np.floor(df["age"] / 5) * 5).astype("Int64")
df["credit_income_ratio"] = (
    df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].where(df["AMT_INCOME_TOTAL"] > 0)
)

def histogram_by_target(series, target, bins):
    result = []
    for label, mask in [
        ("No recorded difficulties", target == 0),
        ("Recorded difficulties", target == 1),
    ]:
        values = series[mask].dropna()
        counts, edges = np.histogram(values, bins=bins)
        total = int(counts.sum())
        result.append({
            "label": label,
            "counts": counts.astype(int).tolist(),
            "percentages": (
                (counts / total * 100).round(3).tolist()
                if total else [0] * len(counts)
            ),
        })
    return result

# Use common bins for both outcomes so their profiles can be compared.
valid_age = df["age"].dropna()
age_edges = np.arange(
    np.floor(valid_age.min() / 5) * 5,
    np.ceil(valid_age.max() / 5) * 5 + 5,
    5,
)
age_labels = [
    f"{int(age_edges[i])}-{int(age_edges[i + 1])}"
    for i in range(len(age_edges) - 1)
]
age_profiles = histogram_by_target(df["age"], df["TARGET"], age_edges)

age_rates = (
    df.dropna(subset=["age_group"])
      .groupby("age_group", observed=True)["TARGET"]
      .agg(["count", "sum"])
      .reset_index()
)
age_rates["rate_percent"] = (
    age_rates["sum"] / age_rates["count"] * 100
).round(3)

# Exclude the pooled top 1% of valid positive incomes for readability.
income = df["AMT_INCOME_TOTAL"]
income_cutoff = income[income > 0].quantile(0.99)
income_valid = income.where((income > 0) & (income <= income_cutoff))
income_edges = np.linspace(0, float(income_cutoff), 21)
income_profiles = histogram_by_target(income_valid, df["TARGET"], income_edges)
income_labels = [
    f"{income_edges[i] / 1000:.0f}-{income_edges[i + 1] / 1000:.0f}k"
    for i in range(len(income_edges) - 1)
]

# Omit invalid ratios and ratios above 10, as requested.
ratio = df["credit_income_ratio"]
ratio_valid = ratio.where(np.isfinite(ratio) & (ratio >= 0) & (ratio <= 10))

def box_summary(values):
    values = values.dropna().to_numpy()
    if not len(values):
        return {"min": 0, "q1": 0, "median": 0, "q3": 0, "max": 0, "n": 0}
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    return {
        "min": round(float(values.min()), 4),
        "q1": round(float(q1), 4),
        "median": round(float(median), 4),
        "q3": round(float(q3), 4),
        "max": round(float(values.max()), 4),
        "n": int(len(values)),
    }

ratio_summary = {
    "no_difficulties": box_summary(ratio_valid[df["TARGET"] == 0]),
    "difficulties": box_summary(ratio_valid[df["TARGET"] == 1]),
}

payload = {
    "source": "Home Credit application_train.csv",
    "total_applications": int(len(df)),
    "target_counts": {
        "no_difficulties": int((df["TARGET"] == 0).sum()),
        "difficulties": int((df["TARGET"] == 1).sum()),
    },
    "age": {
        "labels": age_labels,
        "profiles": age_profiles,
        "rates": {
            "labels": age_rates["age_group"].astype(int).astype(str).tolist(),
            "counts": age_rates["count"].astype(int).tolist(),
            "percentages": age_rates["rate_percent"].tolist(),
        },
    },
    "income": {
        "labels": income_labels,
        "cutoff": round(float(income_cutoff), 2),
        "profiles": income_profiles,
    },
    "credit_income_ratio": {
        "summary": ratio_summary,
        "omitted_above": 10,
    },
}

OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
print(f"Saved aggregated chart data to: {OUTPUT_PATH}")
print(f"Applications included: {len(df):,}")
print(f"Income cutoff (99th percentile): {income_cutoff:,.2f}")
print("No individual applicant records were written to the output.")
