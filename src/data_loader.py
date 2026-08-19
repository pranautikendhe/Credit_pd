"""
data_loader.py
Loads the UCI Statlog (German Credit) dataset: 1,000 real loan
applications, 20 attributes, binary good/bad credit label.
Source: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
"""

import os
import pandas as pd

COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment_since", "installment_rate_pct", "personal_status_sex",
    "other_debtors", "residence_since", "property", "age", "other_installment_plans",
    "housing", "existing_credits", "job", "num_dependents", "own_telephone",
    "foreign_worker", "target",
]

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "german.data")


def load_raw():
    df = pd.read_csv(DATA_PATH, sep=" ", header=None, names=COLUMNS)
    # UCI encodes target 1 = good, 2 = bad. Convert to default flag: 1 = default (bad).
    df["default"] = (df["target"] == 2).astype(int)
    df = df.drop(columns=["target"])
    return df


if __name__ == "__main__":
    df = load_raw()
    print(df.shape)
    print(df["default"].value_counts(normalize=True))
    print(df.head())
