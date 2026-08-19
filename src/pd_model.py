"""
pd_model.py
Logistic-regression PD model trained on WOE-transformed features,
plus the standard scorecard evaluation metrics: AUC and KS statistic.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve


def train_pd_model(X: pd.DataFrame, y: pd.Series, test_size=0.25, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, (X_train, X_test, y_train, y_test)


def ks_statistic(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return np.max(tpr - fpr)


def evaluate(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    ks = ks_statistic(y_test, proba)
    return {"AUC": auc, "KS": ks, "predicted_pd": proba}


def scorecard_points(model, X: pd.DataFrame, base_points=600, pdo=20, base_odds=50):
    """
    Converts model log-odds to a conventional scorecard point scale
    (higher points = lower risk), using the standard points-to-double-odds
    (PDO) transform used across the industry.
    """
    factor = pdo / np.log(2)
    offset = base_points - factor * np.log(base_odds)
    log_odds = X.values @ model.coef_[0] + model.intercept_[0]
    points = offset + factor * log_odds
    return points


if __name__ == "__main__":
    from data_loader import load_raw
    from woe import fit_woe, transform_woe, iv_strength

    df = load_raw()
    features = [c for c in df.columns if c != "default"]
    woe_maps, iv_summary, bin_edges = fit_woe(df, features, "default")

    # keep features with real predictive power (IV >= 0.10), a standard scorecard cutoff
    kept = iv_summary[iv_summary["IV"] >= 0.10]["feature"].tolist()
    X = transform_woe(df, kept, woe_maps, bin_edges)
    y = df["default"]

    model, (X_train, X_test, y_train, y_test) = train_pd_model(X, y)
    metrics = evaluate(model, X_test, y_test)

    print(f"Features kept (IV >= 0.10): {kept}")
    print(f"Test AUC: {metrics['AUC']:.3f}   Test KS: {metrics['KS']:.3f}")
