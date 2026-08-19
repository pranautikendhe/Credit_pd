"""
ifrs9_staging.py
IFRS 9 staging overlay on top of the PD model: classifies each account
into Stage 1 (12-month ECL), Stage 2 (lifetime ECL, significant increase
in credit risk), or Stage 3 (credit-impaired / default), and computes a
simple Expected Credit Loss for each stage.

This is a simplified, illustrative implementation of the staging logic
(relative PD deterioration + a Stage-3 default trigger), not a production
IFRS 9 engine — real implementations also fold in DPD triggers, watchlist
overrides, and macro overlays.
"""

import numpy as np
import pandas as pd


def assign_stage(pd_origination: pd.Series, pd_current: pd.Series, defaulted: pd.Series,
                  sicr_multiple=2.0, sicr_abs_threshold=0.20):
    """
    Stage 3: currently in default.
    Stage 2: PD has deteriorated significantly since origination
             (relative jump >= sicr_multiple, OR current PD is high in absolute terms) — a
             simplified stand-in for IFRS 9's "significant increase in credit risk" test.
    Stage 1: everything else (performing, no SICR).
    """
    relative_jump = pd_current / pd_origination.clip(lower=1e-6)
    sicr = (relative_jump >= sicr_multiple) | (pd_current >= sicr_abs_threshold)

    stage = np.where(defaulted == 1, 3, np.where(sicr, 2, 1))
    return pd.Series(stage, index=pd_origination.index, name="ifrs9_stage")


def expected_credit_loss(stage: pd.Series, pd_12m: pd.Series, pd_lifetime: pd.Series,
                          lgd: pd.Series, ead: pd.Series):
    """
    Stage 1 -> 12-month ECL = PD_12m * LGD * EAD
    Stage 2 -> lifetime ECL = PD_lifetime * LGD * EAD
    Stage 3 -> lifetime ECL using PD=1 (already in default)
    """
    ecl = np.where(
        stage == 1, pd_12m * lgd * ead,
        np.where(stage == 2, pd_lifetime * lgd * ead, lgd * ead),
    )
    return pd.Series(ecl, index=stage.index, name="ecl")


if __name__ == "__main__":
    from data_loader import load_raw
    from woe import fit_woe, transform_woe
    from pd_model import train_pd_model

    df = load_raw()
    features = [c for c in df.columns if c != "default"]
    woe_maps, iv_summary, bin_edges = fit_woe(df, features, "default")
    kept = iv_summary[iv_summary["IV"] >= 0.10]["feature"].tolist()
    X = transform_woe(df, kept, woe_maps, bin_edges)
    y = df["default"]

    model, (X_train, X_test, y_train, y_test) = train_pd_model(X, y)
    pd_hat = model.predict_proba(X)[:, 1]

    rng = np.random.default_rng(0)
    # Illustrative origination PD: a noised-down version of current PD (accounts season over time).
    pd_origination = np.clip(pd_hat * rng.uniform(0.4, 0.9, len(pd_hat)), 0.001, 0.99)
    pd_lifetime = np.clip(pd_hat * 2.2, 0.001, 0.99)  # simple lifetime multiplier, illustrative
    lgd = np.full(len(df), 0.45)
    ead = df["credit_amount"].values

    stage = assign_stage(pd.Series(pd_origination), pd.Series(pd_hat), df["default"])
    ecl = expected_credit_loss(stage, pd.Series(pd_hat), pd.Series(pd_lifetime), pd.Series(lgd), pd.Series(ead))

    summary = pd.DataFrame({"stage": stage, "ecl": ecl, "ead": ead})
    report = summary.groupby("stage").agg(accounts=("ecl", "size"), total_EAD=("ead", "sum"),
                                           total_ECL=("ecl", "sum"))
    report["coverage_ratio"] = report["total_ECL"] / report["total_EAD"]
    print(report.round(2))
