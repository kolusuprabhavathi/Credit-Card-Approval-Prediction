"""
preprocessing.py
-----------------
Feature engineering pipeline for the Credit Card Approval Prediction project.

Responsibilities:
  1. Load application_record.csv and credit_record.csv
  2. Convert the multi-class STATUS codes in credit_record.csv into a single
     binary TARGET per applicant (0 = good/approved, 1 = bad/rejected),
     exactly as described in the project brief: "converts multi-class
     payment status codes into binary labels".
  3. Merge that target onto the application table.
  4. Clean and engineer features (ages, employment length, income ratios).
  5. Encode categoricals and scale numerics.
  6. Return train-ready X, y plus the fitted encoders/scaler so the same
     transformation can be replayed at inference time in the Flask app.

This module is imported both by the training notebook/script and by the
Flask application (app.py), so the SAME logic is guaranteed to run in both
places.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Columns that go into the model, in this exact order.
FEATURE_COLUMNS = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN",
    "AMT_INCOME_TOTAL", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "AGE_YEARS",
    "YEARS_EMPLOYED", "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL",
    "OCCUPATION_TYPE", "CNT_FAM_MEMBERS", "INCOME_PER_FAMILY_MEMBER",
    "CNT_EMI_PAID_OFF", "CNT_EMI_PASTDUE", "NO_OF_LOANS",
]

CATEGORICAL_COLUMNS = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
]

NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]

# STATUS codes 2,3,4,5 (60+ days overdue) are treated as "bad" (high risk).
# 0,1 = mildly late, C = paid off that month, X = no loan that month -> "good".
BAD_STATUS_CODES = {"2", "3", "4", "5"}


def build_target_from_credit_record(credit_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the monthly STATUS history per ID into one binary TARGET.

    TARGET = 1 (reject / high risk)  if the applicant was EVER 60+ days
             past due (STATUS in {2,3,4,5}) in their history.
    TARGET = 0 (approve / low risk)  otherwise.
    """
    credit_df = credit_df.copy()
    credit_df["IS_BAD_MONTH"] = credit_df["STATUS"].isin(BAD_STATUS_CODES).astype(int)

    target = (
        credit_df.groupby("ID")["IS_BAD_MONTH"]
        .max()
        .reset_index()
        .rename(columns={"IS_BAD_MONTH": "TARGET"})
    )
    return target


def build_credit_history_features(credit_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the monthly STATUS history per applicant ID into three
    account-level summary features used by the form / model:

      CNT_EMI_PAID_OFF -> number of months on record where that month's
                           installment was paid off in full (STATUS == 'C')
      CNT_EMI_PASTDUE  -> number of months on record where the applicant
                           was past due by any amount (STATUS in 0..5)
      NO_OF_LOANS      -> total number of active loan/credit months on
                           record (i.e. every month that isn't 'X' = no
                           loan that month) -- a proxy for how many EMI
                           cycles the applicant's account history covers
    """
    credit_df = credit_df.copy()
    credit_df["STATUS"] = credit_df["STATUS"].astype(str)

    paid_off = (
        credit_df[credit_df["STATUS"] == "C"]
        .groupby("ID").size().rename("CNT_EMI_PAID_OFF")
    )
    pastdue = (
        credit_df[credit_df["STATUS"].isin(["0", "1", "2", "3", "4", "5"])]
        .groupby("ID").size().rename("CNT_EMI_PASTDUE")
    )
    active_loans = (
        credit_df[credit_df["STATUS"] != "X"]
        .groupby("ID").size().rename("NO_OF_LOANS")
    )

    feats = pd.concat([paid_off, pastdue, active_loans], axis=1).reset_index()
    feats[["CNT_EMI_PAID_OFF", "CNT_EMI_PASTDUE", "NO_OF_LOANS"]] = (
        feats[["CNT_EMI_PAID_OFF", "CNT_EMI_PASTDUE", "NO_OF_LOANS"]].fillna(0).astype(int)
    )
    return feats


def engineer_features(app_df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw application fields and derive model-ready features."""
    df = app_df.copy()

    # Age in years (DAYS_BIRTH is negative days-from-today in the raw schema)
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365).round(1)

    # 365243 is the sentinel value for pensioners / not currently employed
    df["YEARS_EMPLOYED"] = np.where(
        df["DAYS_EMPLOYED"] > 0, 0, (-df["DAYS_EMPLOYED"] / 365).round(1)
    )

    df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna("Unknown")
    df["CNT_FAM_MEMBERS"] = df["CNT_FAM_MEMBERS"].fillna(df["CNT_FAM_MEMBERS"].median())

    df["INCOME_PER_FAMILY_MEMBER"] = (
        df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].replace(0, 1)
    ).round(1)

    return df


def merge_and_label(app_df: pd.DataFrame, credit_df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: engineer features, build target, merge, drop unlabeled rows."""
    app_feat = engineer_features(app_df)
    target = build_target_from_credit_record(credit_df)
    credit_feats = build_credit_history_features(credit_df)

    merged = app_feat.merge(target, on="ID", how="inner")
    merged = merged.merge(credit_feats, on="ID", how="left")
    merged[["CNT_EMI_PAID_OFF", "CNT_EMI_PASTDUE", "NO_OF_LOANS"]] = (
        merged[["CNT_EMI_PAID_OFF", "CNT_EMI_PASTDUE", "NO_OF_LOANS"]].fillna(0).astype(int)
    )
    return merged


class FeaturePipeline:
    """Fits LabelEncoders + a StandardScaler on training data, and replays the
    exact same transformation later (e.g. inside the Flask app) given raw
    user-entered values."""

    def __init__(self):
        self.encoders = {col: LabelEncoder() for col in CATEGORICAL_COLUMNS}
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame):
        df = df.copy()
        for col in CATEGORICAL_COLUMNS:
            df[col] = self.encoders[col].fit_transform(df[col].astype(str))

        X_num = self.scaler.fit_transform(df[NUMERIC_COLUMNS])
        X_cat = df[CATEGORICAL_COLUMNS].values
        X = np.hstack([X_cat, X_num])
        self.is_fitted = True
        return X

    def transform(self, df: pd.DataFrame):
        df = df.copy()
        for col in CATEGORICAL_COLUMNS:
            known = set(self.encoders[col].classes_)
            df[col] = df[col].astype(str).apply(lambda v: v if v in known else self.encoders[col].classes_[0])
            df[col] = self.encoders[col].transform(df[col])

        X_num = self.scaler.transform(df[NUMERIC_COLUMNS])
        X_cat = df[CATEGORICAL_COLUMNS].values
        X = np.hstack([X_cat, X_num])
        return X

    def get_feature_names(self):
        return CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def load_raw_data(app_path="application_record.csv", credit_path="credit_record.csv"):
    app_df = pd.read_csv(app_path)
    credit_df = pd.read_csv(credit_path)
    return app_df, credit_df


if __name__ == "__main__":
    app_df, credit_df = load_raw_data()
    merged = merge_and_label(app_df, credit_df)
    print("Merged shape:", merged.shape)
    print("Target distribution:\n", merged["TARGET"].value_counts(normalize=True))

    pipeline = FeaturePipeline()
    X = pipeline.fit_transform(merged[FEATURE_COLUMNS])
    print("Feature matrix shape:", X.shape)
