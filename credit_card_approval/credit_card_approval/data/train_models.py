"""
train_models.py
-----------------
Trains and compares four classification algorithms for the Credit Card
Approval Prediction project:

    1. Logistic Regression
    2. Random Forest
    3. XGBoost (Gradient Boosting)
    4. Decision Tree

The best-performing model (by F1-score on the test split, which is the
right metric here given the class imbalance between approvals and
rejections) is saved to model/best_model.pkl, along with the fitted
FeaturePipeline (encoders + scaler) so the Flask app can reproduce the
exact same preprocessing at inference time.

Run from the project root:
    python data/train_models.py
"""

import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")  # headless backend, safe for servers / containers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.dirname(__file__))
from preprocessing import FEATURE_COLUMNS, FeaturePipeline, load_raw_data, merge_and_label

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN] xgboost is not installed in this environment. "
          "Run: pip install xgboost  -- the script will still run the "
          "other 3 models and skip XGBoost.")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_and_prepare():
    app_df, credit_df = load_raw_data(
        os.path.join(THIS_DIR, "application_record.csv"),
        os.path.join(THIS_DIR, "credit_record.csv"),
    )
    merged = merge_and_label(app_df, credit_df)
    pipeline = FeaturePipeline()
    X = pipeline.fit_transform(merged[FEATURE_COLUMNS])
    y = merged["TARGET"].values
    return X, y, pipeline, merged


def evaluate(model, X_test, y_test, name):
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
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:10s}: {v}")
    print(classification_report(y_test, y_pred, target_names=["Approved (0)", "Rejected (1)"]))
    return metrics, y_pred


def plot_confusion_matrices(results, save_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, y_test, y_pred) in zip(axes, results):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Approved", "Rejected"],
                    yticklabels=["Approved", "Rejected"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot -> {save_path}")


def plot_metric_comparison(metrics_list, save_path):
    df = pd.DataFrame(metrics_list).set_index("model")
    ax = df[["accuracy", "precision", "recall", "f1_score", "roc_auc"]].plot(
        kind="bar", figsize=(10, 6), rot=20
    )
    ax.set_title("Model Comparison - Credit Card Approval Prediction")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved model comparison plot -> {save_path}")


def main():
    print("Loading and preparing data...")
    X, y, pipeline, merged = load_and_prepare()
    print(f"Feature matrix: {X.shape}, target distribution: {np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }

    if XGBOOST_AVAILABLE:
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=42, use_label_encoder=False,
        )

    all_metrics = []
    cm_results = []
    fitted_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        metrics, y_pred = evaluate(model, X_test, y_test, name)
        all_metrics.append(metrics)
        cm_results.append((name, y_test, y_pred))
        fitted_models[name] = model

    # Pick best model by F1-score (most meaningful metric under class imbalance)
    best_metrics = max(all_metrics, key=lambda m: m["f1_score"])
    best_name = best_metrics["model"]
    best_model = fitted_models[best_name]
    print(f"\n>>> Best model: {best_name} (F1-score = {best_metrics['f1_score']}) <<<")

    # Save plots
    plot_confusion_matrices(cm_results, os.path.join(MODEL_DIR, "confusion_matrices.png"))
    plot_metric_comparison(all_metrics, os.path.join(MODEL_DIR, "model_comparison.png"))

    # Save metrics report
    with open(os.path.join(MODEL_DIR, "metrics_report.json"), "w") as f:
        json.dump({"all_models": all_metrics, "best_model": best_name}, f, indent=2)
    print(f"Saved metrics report -> {os.path.join(MODEL_DIR, 'metrics_report.json')}")

    # Save the best model + the fitted preprocessing pipeline together
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "feature_pipeline.pkl"))
    joblib.dump(FEATURE_COLUMNS, os.path.join(MODEL_DIR, "feature_columns.pkl"))
    with open(os.path.join(MODEL_DIR, "best_model_name.txt"), "w") as f:
        f.write(best_name)

    print(f"\nSaved best_model.pkl, feature_pipeline.pkl, feature_columns.pkl -> {MODEL_DIR}")
    print("Training complete.")


if __name__ == "__main__":
    main()
