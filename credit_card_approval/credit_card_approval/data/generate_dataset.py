"""
generate_dataset.py
--------------------
Generates two CSV files that replicate the schema of the real-world
"Credit Card Approval Prediction" dataset (the same one used on Kaggle:
https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction):

    1. application_record.csv  -> applicant demographic & financial info
    2. credit_record.csv       -> month-by-month repayment status history

If you already have the real dataset downloaded from Kaggle, simply
place those two CSV files in this `data/` folder and SKIP this script.
This generator exists so you can run the entire pipeline immediately
without needing a Kaggle account.

Usage:
    python generate_dataset.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_APPLICANTS = 8000


def generate_application_record(n=N_APPLICANTS):
    ids = np.arange(5008800, 5008800 + n)

    gender = np.random.choice(["M", "F"], size=n, p=[0.45, 0.55])
    own_car = np.random.choice(["Y", "N"], size=n, p=[0.4, 0.6])
    own_realty = np.random.choice(["Y", "N"], size=n, p=[0.55, 0.45])
    cnt_children = np.random.choice([0, 1, 2, 3, 4], size=n, p=[0.55, 0.25, 0.13, 0.05, 0.02])

    annual_income = np.round(np.random.lognormal(mean=11.5, sigma=0.5, size=n), 1)
    annual_income = np.clip(annual_income, 27000, 1600000)

    income_type = np.random.choice(
        ["Working", "Commercial associate", "Pensioner", "State servant", "Student"],
        size=n, p=[0.55, 0.22, 0.13, 0.09, 0.01]
    )

    education_type = np.random.choice(
        ["Secondary / secondary special", "Higher education",
         "Incomplete higher", "Lower secondary", "Academic degree"],
        size=n, p=[0.65, 0.25, 0.06, 0.03, 0.01]
    )

    family_status = np.random.choice(
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        size=n, p=[0.65, 0.15, 0.08, 0.07, 0.05]
    )

    housing_type = np.random.choice(
        ["House / apartment", "With parents", "Municipal apartment",
         "Rented apartment", "Office apartment", "Co-op apartment"],
        size=n, p=[0.78, 0.08, 0.06, 0.05, 0.02, 0.01]
    )

    days_birth = -np.random.randint(20 * 365, 69 * 365, size=n)

    days_employed = np.empty(n, dtype=int)
    pensioner_mask = income_type == "Pensioner"
    days_employed[pensioner_mask] = 365243  # special code used in the real dataset for unemployed/retired
    days_employed[~pensioner_mask] = -np.random.randint(30, 40 * 365, size=(~pensioner_mask).sum())

    flag_mobil = np.ones(n, dtype=int)
    flag_work_phone = np.random.choice([0, 1], size=n, p=[0.75, 0.25])
    flag_phone = np.random.choice([0, 1], size=n, p=[0.65, 0.35])
    flag_email = np.random.choice([0, 1], size=n, p=[0.85, 0.15])

    occupation_pool = [
        "Laborers", "Core staff", "Sales staff", "Managers", "Drivers",
        "High skill tech staff", "Accountants", "Medicine staff",
        "Cooking staff", "Security staff", "Cleaning staff",
        "Private service staff", "Low-skill Laborers", "Secretaries",
        "Waiters/barmen staff", "Realty agents", "HR staff", "IT staff"
    ]
    occupation_type = np.random.choice(occupation_pool + [np.nan], size=n,
                                        p=[0.85 / len(occupation_pool)] * len(occupation_pool) + [0.15])

    cnt_fam_members = cnt_children + np.random.choice([1, 2], size=n, p=[0.3, 0.7])

    df = pd.DataFrame({
        "ID": ids,
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": own_car,
        "FLAG_OWN_REALTY": own_realty,
        "CNT_CHILDREN": cnt_children,
        "AMT_INCOME_TOTAL": annual_income,
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education_type,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing_type,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "FLAG_MOBIL": flag_mobil,
        "FLAG_WORK_PHONE": flag_work_phone,
        "FLAG_PHONE": flag_phone,
        "FLAG_EMAIL": flag_email,
        "OCCUPATION_TYPE": occupation_type,
        "CNT_FAM_MEMBERS": cnt_fam_members.astype(float),
    })
    return df


def generate_credit_record(app_df):
    """
    For every applicant, generate a variable-length history of monthly
    STATUS codes exactly as in the real dataset:
      0      = 1-29 days past due
      1      = 30-59 days past due
      2      = 60-89 days past due
      3      = 90-119 days past due
      4      = 120-149 days past due
      5      = overdue / write-off (>150 days)
      C      = paid off that month
      X      = no loan for that month
    We bias the simulation using income, employment stability and
    family status so the resulting labels are realistic and learnable.
    """
    rows = []
    income = app_df["AMT_INCOME_TOTAL"].values
    days_employed = app_df["DAYS_EMPLOYED"].values
    children = app_df["CNT_CHILDREN"].values
    ids = app_df["ID"].values

    income_norm = pd.Series(income).rank(pct=True).values  # percentile rank, robust to long tail
    employed_years = np.where(days_employed < 0, -days_employed / 365.0, 0)
    employed_norm = pd.Series(np.clip(employed_years / 20.0, 0, 1)).rank(pct=True).values

    # risk score: lower = safer borrower
    risk_score = (
        0.55 * (1 - income_norm)
        + 0.35 * (1 - employed_norm)
        + 0.10 * np.clip(children / 4.0, 0, 1)
    )
    risk_score += np.random.normal(0, 0.03, size=len(risk_score))
    risk_score = np.clip(risk_score, 0, 1)

    for idx, cust_id in enumerate(ids):
        n_months = np.random.randint(6, 25)
        r = risk_score[idx]

        # Per-applicant chance of EVER going seriously delinquent (60+ days),
        # scaled by risk score, tuned so the overall bad rate sits near 10-15%
        # (similar to the real-world dataset's approval/rejection split).
        will_default = np.random.rand() < (0.01 + 0.35 * (r ** 1.3))

        statuses = []
        for m in range(n_months):
            roll = np.random.rand()
            if roll < 0.55:
                statuses.append("C")
            elif roll < 0.85:
                statuses.append("X")
            else:
                # mild lateness only, still counts as "good" credit
                statuses.append(np.random.choice(["0", "1"]))

        if will_default:
            # Guarantee the delinquency actually appears in the history so
            # the simulated TARGET cleanly reflects the underlying risk score
            # (mirrors how, in the real dataset, a defaulted account reliably
            # shows up as 60+ days overdue at least once in its statement history).
            n_bad_months = np.random.randint(1, 3)
            bad_indices = np.random.choice(len(statuses), size=min(n_bad_months, len(statuses)), replace=False)
            severities = ["2", "3", "4", "5"]
            for bi in bad_indices:
                statuses[bi] = np.random.choice(severities, p=[0.4, 0.3, 0.2, 0.1])

        for month_offset, status in enumerate(statuses):
            rows.append((cust_id, -month_offset, status))

    credit_df = pd.DataFrame(rows, columns=["ID", "MONTHS_BALANCE", "STATUS"])
    return credit_df


if __name__ == "__main__":
    app_df = generate_application_record()
    credit_df = generate_credit_record(app_df)

    app_df.to_csv("application_record.csv", index=False)
    credit_df.to_csv("credit_record.csv", index=False)

    print(f"application_record.csv -> {app_df.shape[0]} rows, {app_df.shape[1]} columns")
    print(f"credit_record.csv      -> {credit_df.shape[0]} rows, {credit_df.shape[1]} columns")
    print("\nSample application_record.csv:")
    print(app_df.head())
    print("\nSample credit_record.csv:")
    print(credit_df.head())
