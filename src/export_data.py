"""
export_data.py
Runs the same pipeline as run_report.py but dumps the underlying numbers
needed for an interactive (Plotly) front end as JSON, instead of static PNGs.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from data_loader import load_raw
from woe import fit_woe, transform_woe, iv_strength
from pd_model import train_pd_model, evaluate
from ifrs9_staging import assign_stage, expected_credit_loss

OUT = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUT, exist_ok=True)


def main():
    df = load_raw()
    features = [c for c in df.columns if c != "default"]
    woe_maps, iv_summary, bin_edges = fit_woe(df, features, "default")
    iv_summary["strength"] = iv_summary["IV"].apply(iv_strength)
    kept = iv_summary[iv_summary["IV"] >= 0.10]["feature"].tolist()

    X = transform_woe(df, kept, woe_maps, bin_edges)
    y = df["default"]
    model, (X_train, X_test, y_train, y_test) = train_pd_model(X, y)
    metrics = evaluate(model, X_test, y_test)

    coefs = pd.Series(model.coef_[0], index=X.columns).sort_values()

    pd_hat_all = model.predict_proba(X)[:, 1]
    rng = np.random.default_rng(0)
    pd_origination = np.clip(pd_hat_all * rng.uniform(0.4, 0.9, len(pd_hat_all)), 0.001, 0.99)
    pd_lifetime = np.clip(pd_hat_all * 2.2, 0.001, 0.99)
    lgd = np.full(len(df), 0.45)
    ead = df["credit_amount"].values

    stage = assign_stage(pd.Series(pd_origination), pd.Series(pd_hat_all), df["default"])
    ecl = expected_credit_loss(stage, pd.Series(pd_hat_all), pd.Series(pd_lifetime),
                                pd.Series(lgd), pd.Series(ead))
    staging_report = pd.DataFrame({"stage": stage, "ecl": ecl, "ead": ead}).groupby("stage").agg(
        accounts=("ecl", "size"), total_EAD=("ead", "sum"), total_ECL=("ecl", "sum"))
    staging_report["coverage_ratio"] = (staging_report["total_ECL"] / staging_report["total_EAD"])

    fpr, tpr, _ = roc_curve(y_test, metrics["predicted_pd"])

    data = {
        "dataset": {
            "n_accounts": int(len(df)),
            "default_rate": float(df["default"].mean()),
        },
        "iv_ranking": [
            {"feature": r.feature, "IV": float(r.IV), "n_bins": int(r.n_bins), "strength": r.strength}
            for r in iv_summary.itertuples()
        ],
        "features_kept": kept,
        "model": {
            "AUC": float(metrics["AUC"]),
            "KS": float(metrics["KS"]),
        },
        "roc_curve": {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
        },
        "coefficients": [{"feature": k, "coef": float(v)} for k, v in coefs.items()],
        "ifrs9_staging": [
            {
                "stage": int(idx),
                "accounts": int(row.accounts),
                "total_EAD": float(row.total_EAD),
                "total_ECL": float(row.total_ECL),
                "coverage_ratio": float(row.coverage_ratio),
            }
            for idx, row in staging_report.iterrows()
        ],
        "pd_distribution": [float(x) for x in pd_hat_all],
    }

    with open(os.path.join(OUT, "model_data.json"), "w") as f:
        json.dump(data, f)
    print(f"Wrote {os.path.join(OUT, 'model_data.json')}")


if __name__ == "__main__":
    main()
