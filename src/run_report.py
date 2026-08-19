"""
run_report.py
End-to-end run: WOE/IV feature selection -> logistic regression PD model
-> evaluation (AUC/KS) -> scorecard points -> IFRS 9 staging & ECL.
Writes a summary report and plots to output/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from data_loader import load_raw
from woe import fit_woe, transform_woe, iv_strength
from pd_model import train_pd_model, evaluate, scorecard_points
from ifrs9_staging import assign_stage, expected_credit_loss

OUT = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUT, exist_ok=True)


def main():
    lines = ["CREDIT RISK SCORECARD & PD MODEL — SUMMARY REPORT", "=" * 52]

    df = load_raw()
    lines.append(f"\nDataset: UCI Statlog German Credit, {len(df)} accounts, "
                 f"default rate {df['default'].mean():.1%}")

    # ---- WOE / IV ----
    features = [c for c in df.columns if c != "default"]
    woe_maps, iv_summary, bin_edges = fit_woe(df, features, "default")
    iv_summary["strength"] = iv_summary["IV"].apply(iv_strength)
    kept = iv_summary[iv_summary["IV"] >= 0.10]["feature"].tolist()

    lines.append("\n-- Information Value ranking (top 8) --")
    lines.append(iv_summary.head(8).to_string(index=False))
    lines.append(f"\nFeatures retained for the model (IV >= 0.10): {kept}")

    # ---- Model ----
    X = transform_woe(df, kept, woe_maps, bin_edges)
    y = df["default"]
    model, (X_train, X_test, y_train, y_test) = train_pd_model(X, y)
    metrics = evaluate(model, X_test, y_test)

    lines.append("\n-- Model performance (held-out test set) --")
    lines.append(f"  AUC: {metrics['AUC']:.3f}   KS statistic: {metrics['KS']:.3f}")

    coefs = pd.Series(model.coef_[0], index=X.columns).sort_values()
    lines.append("\n-- Logistic regression coefficients (on WOE scale) --")
    lines.append(coefs.round(3).to_string())

    # ---- IFRS 9 staging ----
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
    staging_report["coverage_ratio"] = (staging_report["total_ECL"] / staging_report["total_EAD"]).round(3)

    lines.append("\n-- IFRS 9 staging & Expected Credit Loss --")
    lines.append(staging_report.round(2).to_string())

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(OUT, "summary_report.txt"), "w") as f:
        f.write(report)

    # ---- Plots ----
    fpr, tpr, _ = roc_curve(y_test, metrics["predicted_pd"])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color="#1F3864", lw=2, label=f"AUC = {metrics['AUC']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("PD Model ROC Curve (held-out test set)")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "roc_curve.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    iv_summary.head(10).plot.barh(x="feature", y="IV", ax=ax, color="#1F3864", legend=False)
    ax.invert_yaxis()
    ax.set_xlabel("Information Value"); ax.set_title("Top 10 Features by Information Value")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "information_value.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    staging_report["coverage_ratio"].plot.bar(ax=ax, color=["#2E7D32", "#F9A825", "#C62828"])
    ax.set_ylabel("ECL coverage ratio"); ax.set_xlabel("IFRS 9 Stage")
    ax.set_title("ECL Coverage Ratio by IFRS 9 Stage")
    for i, v in enumerate(staging_report["coverage_ratio"]):
        ax.text(i, v + 0.005, f"{v:.1%}", ha="center")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "ifrs9_coverage.png"), dpi=130)

    print(f"\nSaved report + 3 plots to {os.path.abspath(OUT)}/")


if __name__ == "__main__":
    main()
