"""
woe.py
Weight-of-Evidence encoding and Information Value — the standard
feature-engineering step for a credit scorecard, done separately for
categorical attributes and binned numeric attributes.
"""

import numpy as np
import pandas as pd


def _woe_iv_table(df, feature, target):
    grp = df.groupby(feature, observed=True)[target].agg(["count", "sum"])
    grp.columns = ["total", "bad"]
    grp["good"] = grp["total"] - grp["bad"]

    total_bad = grp["bad"].sum()
    total_good = grp["good"].sum()

    # Laplace-smooth to avoid div-by-zero / log(0) on sparse bins.
    grp["dist_bad"] = (grp["bad"] + 0.5) / (total_bad + 0.5 * len(grp))
    grp["dist_good"] = (grp["good"] + 0.5) / (total_good + 0.5 * len(grp))
    grp["woe"] = np.log(grp["dist_good"] / grp["dist_bad"])
    grp["iv"] = (grp["dist_good"] - grp["dist_bad"]) * grp["woe"]
    return grp


def fit_woe(df: pd.DataFrame, features: list, target: str, n_bins=5):
    """
    Returns:
      woe_maps: dict[feature] -> {bin_label: woe_value}
      iv_summary: DataFrame of total IV per feature (for feature selection)
      bin_edges: dict[feature] -> np.array of bin edges (numeric features only)
    """
    woe_maps, bin_edges, iv_rows = {}, {}, []
    work = df[features + [target]].copy()

    for f in features:
        if pd.api.types.is_numeric_dtype(work[f]) and work[f].nunique() > n_bins:
            edges = np.unique(np.quantile(work[f], np.linspace(0, 1, n_bins + 1)))
            work[f + "_bin"] = pd.cut(work[f], bins=edges, include_lowest=True)
            bin_edges[f] = edges
            key = f + "_bin"
        else:
            key = f

        table = _woe_iv_table(work, key, target)
        woe_maps[f] = table["woe"].to_dict()
        iv_rows.append({"feature": f, "IV": table["iv"].sum(), "n_bins": len(table)})

    iv_summary = pd.DataFrame(iv_rows).sort_values("IV", ascending=False).reset_index(drop=True)
    return woe_maps, iv_summary, bin_edges


def transform_woe(df: pd.DataFrame, features: list, woe_maps: dict, bin_edges: dict):
    out = pd.DataFrame(index=df.index)
    for f in features:
        if f in bin_edges:
            binned = pd.cut(df[f], bins=bin_edges[f], include_lowest=True)
            out[f + "_woe"] = binned.map(woe_maps[f]).astype(float)
            out[f + "_woe"] = out[f + "_woe"].fillna(0.0)
        else:
            out[f + "_woe"] = df[f].map(woe_maps[f]).fillna(0.0)
    return out


IV_GUIDE = [
    (0.02, "Not useful for prediction"),
    (0.10, "Weak predictor"),
    (0.30, "Medium predictor"),
    (0.50, "Strong predictor"),
    (np.inf, "Suspiciously strong — check for leakage"),
]


def iv_strength(iv: float) -> str:
    for cutoff, label in IV_GUIDE:
        if iv < cutoff:
            return label
    return "Suspiciously strong — check for leakage"


if __name__ == "__main__":
    from data_loader import load_raw

    df = load_raw()
    features = [c for c in df.columns if c != "default"]
    woe_maps, iv_summary, bin_edges = fit_woe(df, features, "default")
    iv_summary["strength"] = iv_summary["IV"].apply(iv_strength)
    print(iv_summary.to_string(index=False))
