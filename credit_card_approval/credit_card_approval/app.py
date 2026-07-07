"""
app.py
-------
Flask web application for the Credit Card Approval Prediction project.

Routes:
  GET  /                -> renders the applicant-entry form (index.html)
  POST /predict          -> accepts form data, runs the saved model, renders result.html
  POST /api/predict       -> JSON API version of the same prediction (for Scenario 2 batch
                              screening / integration with other systems, e.g. IBM Watson)
  GET  /about             -> renders project & model-performance info page

Run:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.append(DATA_DIR)  # needed so joblib can unpickle the FeaturePipeline class

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the trained model + preprocessing pipeline once at startup
# ---------------------------------------------------------------------------
model = None
pipeline = None
feature_columns = None
best_model_name = "Unknown"
metrics_report = {}

try:
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    pipeline = joblib.load(os.path.join(MODEL_DIR, "feature_pipeline.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    with open(os.path.join(MODEL_DIR, "best_model_name.txt")) as f:
        best_model_name = f.read().strip()
    import json
    with open(os.path.join(MODEL_DIR, "metrics_report.json")) as f:
        metrics_report = json.load(f)
    print(f"Loaded model: {best_model_name}")
except FileNotFoundError:
    print("[WARNING] Trained model files not found in /model. "
          "Run `python data/train_models.py` first to generate them.")


# ---------------------------------------------------------------------------
# Dropdown choices shown in the form (must match the categories the model
# was trained on; OCCUPATION_TYPE 'Unknown' is what missing data becomes)
# ---------------------------------------------------------------------------
FORM_CHOICES = {
    "CODE_GENDER": ["M", "F"],
    "FLAG_OWN_CAR": ["Y", "N"],
    "FLAG_OWN_REALTY": ["Y", "N"],
    "NAME_INCOME_TYPE": ["Working", "Commercial associate", "Pensioner", "State servant", "Student"],
    "NAME_EDUCATION_TYPE": [
        "Secondary / secondary special", "Higher education", "Incomplete higher",
        "Lower secondary", "Academic degree",
    ],
    "NAME_FAMILY_STATUS": ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
    "NAME_HOUSING_TYPE": [
        "House / apartment", "With parents", "Municipal apartment",
        "Rented apartment", "Office apartment", "Co-op apartment",
    ],
    "OCCUPATION_TYPE": [
        "Unknown", "Laborers", "Core staff", "Sales staff", "Managers", "Drivers",
        "High skill tech staff", "Accountants", "Medicine staff", "Cooking staff",
        "Security staff", "Cleaning staff", "Private service staff",
        "Low-skill Laborers", "Secretaries", "Waiters/barmen staff",
        "Realty agents", "HR staff", "IT staff",
    ],
}


def build_feature_row(form):
    """Convert raw form/JSON input into the single-row DataFrame the
    FeaturePipeline expects (same column names/order as training)."""
    age_years = float(form.get("age_years"))
    years_employed = float(form.get("years_employed"))
    income = float(form.get("annual_income"))
    children = int(form.get("cnt_children"))
    fam_members = float(form.get("cnt_fam_members"))
    emi_paid_off = int(form.get("emi_paid_off", 0) or 0)
    emi_pastdue = int(form.get("emi_pastdue", 0) or 0)
    number_of_loans = int(form.get("number_of_loans", 0) or 0)

    row = {
        "CODE_GENDER": form.get("gender"),
        "FLAG_OWN_CAR": form.get("own_car"),
        "FLAG_OWN_REALTY": form.get("own_realty"),
        "CNT_CHILDREN": children,
        "AMT_INCOME_TOTAL": income,
        "NAME_INCOME_TYPE": form.get("income_type"),
        "NAME_EDUCATION_TYPE": form.get("education_type"),
        "NAME_FAMILY_STATUS": form.get("family_status"),
        "NAME_HOUSING_TYPE": form.get("housing_type"),
        "AGE_YEARS": age_years,
        "YEARS_EMPLOYED": years_employed,
        "FLAG_WORK_PHONE": int(form.get("work_phone", 0)),
        "FLAG_PHONE": int(form.get("phone", 0)),
        "FLAG_EMAIL": int(form.get("email", 0)),
        "OCCUPATION_TYPE": form.get("occupation_type") or "Unknown",
        "CNT_FAM_MEMBERS": fam_members,
        "INCOME_PER_FAMILY_MEMBER": round(income / max(fam_members, 1), 1),
        "CNT_EMI_PAID_OFF": emi_paid_off,
        "CNT_EMI_PASTDUE": emi_pastdue,
        "NO_OF_LOANS": number_of_loans,
    }
    return pd.DataFrame([row])[feature_columns]


def run_prediction(form):
    df_row = build_feature_row(form)
    X = pipeline.transform(df_row)
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else [1 - pred, pred]

    result = {
        "prediction": pred,  # 0 = Approved, 1 = Rejected
        "label": "Rejected" if pred == 1 else "Approved",
        "approval_probability": round(float(proba[0]) * 100, 2),
        "rejection_probability": round(float(proba[1]) * 100, 2),
        "model_used": best_model_name,
    }
    return result


@app.route("/")
def index():
    return render_template("index.html", choices=FORM_CHOICES)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("result.html", error="Model not loaded. Please train the model first.")
    try:
        result = run_prediction(request.form)
        applicant_summary = {
            "Gender": request.form.get("gender"),
            "Age": request.form.get("age_years"),
            "Annual Income": request.form.get("annual_income"),
            "Income Type": request.form.get("income_type"),
            "Education": request.form.get("education_type"),
            "Family Status": request.form.get("family_status"),
            "Housing": request.form.get("housing_type"),
            "Years Employed": request.form.get("years_employed"),
            "Occupation": request.form.get("occupation_type") or "Unknown",
            "EMIs Paid Off": request.form.get("emi_paid_off"),
            "EMIs Past Due": request.form.get("emi_pastdue"),
            "Number of Loans": request.form.get("number_of_loans"),
        }
        return render_template("result.html", result=result, applicant=applicant_summary)
    except Exception as e:
        return render_template("result.html", error=str(e))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint -- used for batch/compliance screening (Scenario 2)
    and for the IBM Watson deployment wrapper to call this exact same logic."""
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_models.py first."}), 503
    try:
        data = request.get_json(force=True)
        result = run_prediction(data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/about")
def about():
    return render_template("about.html", best_model_name=best_model_name, metrics=metrics_report)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
