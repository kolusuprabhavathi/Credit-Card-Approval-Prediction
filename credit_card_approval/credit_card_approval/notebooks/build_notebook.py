"""
build_notebook.py
-------------------
Generates notebooks/Credit_Card_Approval_Prediction.ipynb by hand-assembling
the Jupyter notebook JSON schema (nbformat 4). This avoids depending on the
`nbformat` package, which may not be installed in every environment.

Run once:
    python build_notebook.py
"""

import json

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }

cells = []

cells.append(md("""# Credit Card Approval Prediction

This notebook walks through the full machine learning pipeline for the
**Credit Card Approval Prediction** project:

1. Load the raw applicant and credit history data
2. Explore and visualize the data
3. Engineer features and build the binary approval/rejection target
4. Train four classification models: Logistic Regression, Decision Tree,
   Random Forest, and XGBoost
5. Compare model performance and select the best one
6. Save the final model for use in the Flask web application

> **Note:** If you don't have the real Kaggle dataset
> (`application_record.csv` + `credit_record.csv`), run
> `python data/generate_dataset.py` first to generate a realistic synthetic
> version with the same schema, so every cell below runs without changes.
"""))

cells.append(md("## 1. Imports"))

cells.append(code("""import sys
sys.path.append('../data')  # so we can import preprocessing.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("xgboost not installed -- run `pip install xgboost` to include it")

from preprocessing import (
    FEATURE_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS,
    FeaturePipeline, load_raw_data, merge_and_label,
    build_target_from_credit_record, engineer_features
)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
"""))

cells.append(md("## 2. Load raw data"))

cells.append(code("""app_df, credit_df = load_raw_data(
    app_path="../data/application_record.csv",
    credit_path="../data/credit_record.csv",
)

print("application_record.csv:", app_df.shape)
print("credit_record.csv:", credit_df.shape)
app_df.head()
"""))

cells.append(code("""credit_df.head()
"""))

cells.append(md("""## 3. Exploratory Data Analysis (EDA)

Before building the target label, let's understand the raw data:
distribution of income, age, employment length, and the credit STATUS
codes themselves."""))

cells.append(code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(app_df["AMT_INCOME_TOTAL"], bins=50, color="#1f6f50")
axes[0].set_title("Annual Income Distribution")
axes[0].set_xlabel("Annual Income")

age_years = -app_df["DAYS_BIRTH"] / 365
axes[1].hist(age_years, bins=40, color="#0e1b2b")
axes[1].set_title("Applicant Age Distribution")
axes[1].set_xlabel("Age (years)")

axes[2].hist(app_df["CNT_CHILDREN"], bins=range(0, 6), color="#b3852a")
axes[2].set_title("Number of Children")
axes[2].set_xlabel("Children")

plt.tight_layout()
plt.show()
"""))

cells.append(code("""status_counts = credit_df["STATUS"].value_counts().sort_index()
print(status_counts)

plt.figure(figsize=(8, 5))
status_counts.plot(kind="bar", color="#1f3146")
plt.title("Distribution of Monthly Repayment STATUS Codes")
plt.xlabel("STATUS code (0-5 = days past due tiers, C = paid off, X = no loan)")
plt.ylabel("Count of monthly records")
plt.show()
"""))

cells.append(md("""## 4. Feature engineering & target construction

The brief calls for converting the multi-class `STATUS` payment codes into
a binary label: an applicant is marked **high risk (TARGET=1)** if they were
EVER 60+ days past due (`STATUS` in `{2,3,4,5}`) at any point in their credit
history, and **low risk (TARGET=0)** otherwise."""))

cells.append(code("""merged = merge_and_label(app_df, credit_df)
print("Merged shape:", merged.shape)

target_counts = merged["TARGET"].value_counts(normalize=True) * 100
print("\\nTarget distribution:")
print(f"  Approved (0): {target_counts[0]:.1f}%")
print(f"  Rejected (1): {target_counts[1]:.1f}%")

plt.figure(figsize=(5, 5))
merged["TARGET"].value_counts().plot(
    kind="pie", labels=["Approved", "Rejected"],
    autopct="%1.1f%%", colors=["#e3efe6", "#f3e4df"],
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
plt.title("Approval vs Rejection Split")
plt.ylabel("")
plt.show()
"""))

cells.append(code("""merged[["AGE_YEARS", "YEARS_EMPLOYED", "INCOME_PER_FAMILY_MEMBER"]].describe()
"""))

cells.append(md("""## 5. Correlation check

A quick look at how engineered numeric features relate to the target."""))

cells.append(code("""numeric_for_corr = merged[["AMT_INCOME_TOTAL", "AGE_YEARS", "YEARS_EMPLOYED",
                            "CNT_CHILDREN", "CNT_FAM_MEMBERS",
                            "INCOME_PER_FAMILY_MEMBER", "TARGET"]]

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_for_corr.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Feature Correlation Heatmap")
plt.show()
"""))

cells.append(md("""## 6. Encode categorical features & scale numeric features

We use the shared `FeaturePipeline` class from `preprocessing.py` so that
the *exact same* transformation logic used here is reused later inside the
Flask app at inference time -- this avoids train/serve skew."""))

cells.append(code("""pipeline = FeaturePipeline()
X = pipeline.fit_transform(merged[FEATURE_COLUMNS])
y = merged["TARGET"].values

print("Feature matrix shape:", X.shape)
print("Feature order:", pipeline.get_feature_names())
"""))

cells.append(md("## 7. Train / test split"))

cells.append(code("""X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape, " Test shape:", X_test.shape)
print("Train target distribution:", np.bincount(y_train))
print("Test target distribution:", np.bincount(y_test))
"""))

cells.append(md("""## 8. Train the four classification models

- **Logistic Regression** -- a simple, interpretable linear baseline
- **Decision Tree** -- captures non-linear splits, easy to visualize
- **Random Forest** -- an ensemble of trees, usually more robust
- **XGBoost** -- gradient boosting, typically the strongest tabular model

All models use `class_weight="balanced"` (or the XGBoost equivalent,
`scale_pos_weight`) since rejected applicants are the minority class."""))

cells.append(code("""models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1
    ),
}

if XGBOOST_AVAILABLE:
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    models["XGBoost"] = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=42, use_label_encoder=False,
    )

fitted_models = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    fitted_models[name] = model
print("\\nAll models trained.")
"""))

cells.append(md("## 9. Evaluate each model"))

cells.append(code("""def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    print(f"--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["Approved (0)", "Rejected (1)"]))
    return metrics, y_pred

all_metrics = []
predictions = {}
for name, model in fitted_models.items():
    metrics, y_pred = evaluate(model, X_test, y_test, name)
    all_metrics.append(metrics)
    predictions[name] = y_pred

results_df = pd.DataFrame(all_metrics).set_index("model")
results_df
"""))

cells.append(md("## 10. Visualize model comparison"))

cells.append(code("""ax = results_df[["accuracy", "precision", "recall", "f1_score", "roc_auc"]].plot(
    kind="bar", figsize=(11, 6), rot=20
)
ax.set_title("Model Comparison -- Credit Card Approval Prediction")
ax.set_ylabel("Score")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()
"""))

cells.append(code("""fig, axes = plt.subplots(1, len(fitted_models), figsize=(5 * len(fitted_models), 4))
if len(fitted_models) == 1:
    axes = [axes]

for ax, (name, y_pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Approved", "Rejected"],
                yticklabels=["Approved", "Rejected"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()
"""))

cells.append(md("""## 11. Select & save the best model

We select the best model by **F1-score**, since it balances precision and
recall -- the right tradeoff given how few applicants are actually
rejected in the historical data (a model that just predicts "approve"
for everyone would have high accuracy but be useless)."""))

cells.append(code("""best_row = results_df["f1_score"].idxmax()
best_model = fitted_models[best_row]
print(f"Best model: {best_row}  (F1-score = {results_df.loc[best_row, 'f1_score']})")
"""))

cells.append(code("""import joblib
import os

os.makedirs("../model", exist_ok=True)

joblib.dump(best_model, "../model/best_model.pkl")
joblib.dump(pipeline, "../model/feature_pipeline.pkl")
joblib.dump(FEATURE_COLUMNS, "../model/feature_columns.pkl")

with open("../model/best_model_name.txt", "w") as f:
    f.write(best_row)

import json
with open("../model/metrics_report.json", "w") as f:
    json.dump({"all_models": all_metrics, "best_model": best_row}, f, indent=2)

print("Saved best_model.pkl, feature_pipeline.pkl, feature_columns.pkl, "
      "best_model_name.txt, and metrics_report.json to ../model/")
"""))

cells.append(md("""## 12. Quick sanity check: score a new applicant

Let's simulate Scenario 1 from the project brief: a credit analyst enters
a new applicant's profile and gets an instant prediction."""))

cells.append(code("""new_applicant = pd.DataFrame([{
    "CODE_GENDER": "F",
    "FLAG_OWN_CAR": "N",
    "FLAG_OWN_REALTY": "Y",
    "CNT_CHILDREN": 1,
    "AMT_INCOME_TOTAL": 210000,
    "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Higher education",
    "NAME_FAMILY_STATUS": "Married",
    "NAME_HOUSING_TYPE": "House / apartment",
    "AGE_YEARS": 29,
    "YEARS_EMPLOYED": 4,
    "FLAG_WORK_PHONE": 0,
    "FLAG_PHONE": 1,
    "FLAG_EMAIL": 1,
    "OCCUPATION_TYPE": "Accountants",
    "CNT_FAM_MEMBERS": 3,
    "INCOME_PER_FAMILY_MEMBER": 70000,
}])[FEATURE_COLUMNS]

X_new = pipeline.transform(new_applicant)
pred = best_model.predict(X_new)[0]
proba = best_model.predict_proba(X_new)[0]

print("Prediction:", "REJECTED" if pred == 1 else "APPROVED")
print(f"Approval probability: {proba[0]*100:.1f}%")
print(f"Rejection probability: {proba[1]*100:.1f}%")
"""))

cells.append(md("""## Next steps

- Run the Flask web app (`python app.py` from the project root) to use this
  saved model through a browser-based form.
- Optionally deploy this model to IBM Watson Machine Learning using
  `deployment/deploy_to_watson.py` for cloud-hosted, scalable predictions.
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("Credit_Card_Approval_Prediction.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("Notebook written to Credit_Card_Approval_Prediction.ipynb")
print(f"Total cells: {len(cells)}")
