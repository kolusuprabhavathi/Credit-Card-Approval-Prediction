# Credit Card Approval Prediction — Complete Project Guide

This package contains a full, working implementation of the **Credit Card
Approval Prediction** project: synthetic data generation, EDA, feature
engineering, four trained classification models, a Flask web application,
and an IBM Watson Machine Learning deployment pipeline.

Everything has already been built and tested. This guide explains what's
in the folder and exactly how to run it on your own laptop.

---

## 1. What's inside

```
credit_card_approval/
├── app.py                          # Flask web application
├── requirements.txt                # All Python packages needed
├── data/
│   ├── generate_dataset.py         # Creates a synthetic dataset (run this if you don't have Kaggle data)
│   ├── application_record.csv      # Applicant demographic/financial data (already generated)
│   ├── credit_record.csv           # Monthly repayment history (already generated)
│   ├── preprocessing.py            # Feature engineering + target-label logic (shared by notebook & Flask app)
│   └── train_models.py             # Trains all 4 models, picks the best, saves it
├── notebooks/
│   ├── Credit_Card_Approval_Prediction.ipynb   # Full EDA + training notebook
│   └── build_notebook.py           # (Only needed if you want to regenerate the .ipynb file)
├── model/
│   ├── best_model.pkl              # The saved best-performing model
│   ├── feature_pipeline.pkl        # Saved encoders + scaler (must travel with the model)
│   ├── feature_columns.pkl         # Column order the model expects
│   ├── best_model_name.txt         # Which algorithm won
│   ├── metrics_report.json         # Accuracy/precision/recall/F1/ROC-AUC for all 4 models
│   ├── model_comparison.png        # Bar chart comparing all 4 models
│   └── confusion_matrices.png      # Confusion matrix per model
├── templates/
│   ├── index.html                  # Applicant entry form
│   ├── result.html                 # Approve/Reject result page
│   └── about.html                  # Model performance page
├── static/css/style.css            # All styling
└── deployment/
    ├── deploy_to_watson.py         # Pushes the model to IBM Watson Machine Learning (cloud)
    └── call_watson_endpoint.py     # Calls your deployed Watson endpoint from Python
```

The dataset, the trained model, and the notebook have **already been run
once** so you have working output immediately. You can also regenerate
everything from scratch following the steps below — useful once you swap
in the real Kaggle dataset.

---

## 2. System requirements (per project brief)

- **OS:** Windows / Linux / macOS
- **Python:** 3.8 or above (this was built and tested on 3.12)
- **RAM:** 4 GB minimum, 8 GB recommended
- **Disk:** 2 GB free
- **Internet:** needed once, to install packages (and again only if you
  deploy to IBM Watson Cloud)
- **Tools:** Anaconda Navigator + Jupyter Notebook (for the notebook),
  VS Code or PyCharm (for the Flask app), any modern browser

---

## 3. Step-by-step setup on your own laptop

### Step 1 — Unzip and open a terminal

Unzip the project, then open a terminal/command prompt **inside the
`credit_card_approval` folder**.

### Step 2 — Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

If you prefer Anaconda Navigator instead, create a new environment there
with Python 3.10 or 3.11, then open a terminal from within that
environment.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs Flask, NumPy, pandas, scikit-learn, XGBoost, Matplotlib,
Seaborn, and joblib. (The last two lines in `requirements.txt`,
`ibm-watson-machine-learning` and `requests`, are only required if you plan
to do the cloud deployment step — feel free to skip them for now and
install later.)

### Step 4 — (Optional) Get the real dataset instead of the synthetic one

This package ships with a **synthetic dataset** (`data/application_record.csv`
and `data/credit_record.csv`) that mirrors the exact schema of the real
Kaggle "Credit Card Approval Prediction" dataset, so every script below runs
immediately without needing a Kaggle account.

If you'd rather use the real data:
1. Go to https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction
2. Download `application_record.csv` and `credit_record.csv`
3. Replace the two files inside the `data/` folder with the downloaded ones
   (keep the same filenames)
4. Re-run Step 5 below to retrain on the real data

If you'd like to regenerate a *fresh* synthetic dataset (e.g. with more
rows), run:
```bash
cd data
python generate_dataset.py
cd ..
```

### Step 5 — Train the models

```bash
python data/train_models.py
```

This will:
- Load and merge `application_record.csv` + `credit_record.csv`
- Build the binary approval/rejection target
- Train Logistic Regression, Decision Tree, Random Forest, and XGBoost
- Print accuracy/precision/recall/F1/ROC-AUC for each
- Save the best model (by F1-score) and all supporting files into `model/`
- Save `model/model_comparison.png` and `model/confusion_matrices.png`

You should see console output ending in something like:
```
>>> Best model: Logistic Regression (F1-score = 0.28) <<<
Saved best_model.pkl, feature_pipeline.pkl, feature_columns.pkl -> model/
Training complete.
```

### Step 6 — Explore the Jupyter notebook (optional but recommended)

```bash
jupyter notebook notebooks/Credit_Card_Approval_Prediction.ipynb
```

This notebook walks through the same pipeline interactively, with
visualizations: income/age distributions, the STATUS code breakdown,
the approval/rejection pie chart, a correlation heatmap, model comparison
bar chart, and confusion matrices. Run all cells top to bottom
(`Cell -> Run All` in the Jupyter menu) — it will also (re)save the model
files into `model/`.

### Step 7 — Run the Flask web application

```bash
python app.py
```

You'll see:
```
Loaded model: Logistic Regression
 * Running on http://127.0.0.1:5000
```

Open **http://127.0.0.1:5000** in your browser (Chrome, Edge, or Firefox).

- The home page is the applicant intake form (Scenario 1 / Scenario 4 from
  the brief — an analyst or a prospective customer fills in financial and
  demographic details).
- Submitting the form shows an instant **Approved / Rejected** decision
  with probability bars and a summary of the entered details.
- The **Model Performance** page (`/about`) shows the comparison table
  across all four algorithms.

### Step 8 — (Optional) Use the JSON API for batch/compliance screening

The brief's Scenario 2 describes a compliance officer batch-screening
applicants. The same model is exposed as a JSON API at `/api/predict`, so
you (or another script/system) can screen many applicants programmatically
without using the web form:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "F", "age_years": 29, "family_status": "Married",
    "cnt_children": 1, "cnt_fam_members": 3,
    "education_type": "Higher education",
    "annual_income": 210000, "income_type": "Working",
    "occupation_type": "Accountants", "years_employed": 4,
    "own_car": "N", "own_realty": "Y",
    "housing_type": "House / apartment",
    "work_phone": 0, "phone": 1, "email": 1
  }'
```

Response:
```json
{
  "prediction": 0,
  "label": "Approved",
  "approval_probability": 69.5,
  "rejection_probability": 30.5,
  "model_used": "Logistic Regression"
}
```

You can loop this call over a CSV of applicants to batch-screen an entire
file of past-due or new applications.

### Step 9 — (Optional) Deploy to IBM Watson Machine Learning

The project brief calls for an IBM Watson Machine Learning deployment so
the model can be hosted in the cloud. To do this:

1. **Create an IBM Cloud account** (free tier is fine): https://cloud.ibm.com
2. **Create a Watson Machine Learning service instance:**
   https://cloud.ibm.com/catalog/services/machine-learning
3. **Create a Deployment Space:**
   IBM Cloud console → Watson Machine Learning → Deployment spaces → New
   deployment space. Copy its **Space GUID** from the space's "Manage" tab.
4. **Create an IBM Cloud API key:**
   IBM Cloud console → Manage → Access (IAM) → API keys → Create.
5. **Install the Watson SDK:**
   ```bash
   pip install ibm-watson-machine-learning
   ```
6. **Set your credentials** as environment variables (recommended, so you
   never commit secrets to a file):
   ```bash
   # macOS / Linux
   export WML_API_KEY="your-ibm-cloud-api-key"
   export WML_URL="https://us-south.ml.cloud.ibm.com"   # match your region
   export WML_SPACE_ID="your-deployment-space-guid"

   # Windows (PowerShell)
   $env:WML_API_KEY="your-ibm-cloud-api-key"
   $env:WML_URL="https://us-south.ml.cloud.ibm.com"
   $env:WML_SPACE_ID="your-deployment-space-guid"
   ```
   (Alternatively, edit the four constants directly at the top of
   `deployment/deploy_to_watson.py`.)
7. **Run the deployment script:**
   ```bash
   python deployment/deploy_to_watson.py
   ```
   This stores your trained model as a Watson asset, creates an online
   deployment, and prints a **scoring URL**. Save that URL.
8. **Call your cloud-hosted model** from anywhere using
   `deployment/call_watson_endpoint.py` (set `WML_API_KEY` and
   `WML_SCORING_URL` the same way), or integrate that scoring URL directly
   into the Flask app / any other system.

---

## 4. How the modeling actually works (so you can explain it)

1. **Two raw files, one target.** `application_record.csv` has one row per
   applicant (demographics, income, employment). `credit_record.csv` has
   many rows per applicant — one per month — with a `STATUS` code
   (`0`–`5` for escalating days-past-due tiers, `C` for paid off that
   month, `X` for no active loan that month).
2. **Binary labeling.** For each applicant, if `STATUS` was ever `2`, `3`,
   `4`, or `5` (60+ days past due) in their history, they're labeled
   `TARGET = 1` (high risk / reject). Otherwise `TARGET = 0` (approve).
   This is the "converts multi-class payment status codes into binary
   labels" step mentioned in the brief.
3. **Feature engineering.** Raw `DAYS_BIRTH` / `DAYS_EMPLOYED` (negative
   day counts) are converted into human-readable `AGE_YEARS` and
   `YEARS_EMPLOYED`. Missing occupation becomes `"Unknown"`. An
   `INCOME_PER_FAMILY_MEMBER` ratio is derived. Three additional
   account-history features are aggregated per applicant from
   `credit_record.csv`'s monthly `STATUS` log: `CNT_EMI_PAID_OFF` (months
   paid off in full, `STATUS == 'C'`), `CNT_EMI_PASTDUE` (months past due
   by any amount, `STATUS` in `0`–`5`), and `NO_OF_LOANS` (total active
   loan/credit months on record, `STATUS != 'X'`). These three now appear
   as form fields — "EMI paid off", "EMI of pastdues", "Number of loans" —
   on the application form.
4. **Encoding & scaling.** Categorical fields (gender, education, housing
   type, etc.) are label-encoded; numeric fields are standardized. This
   logic lives in one place (`FeaturePipeline` in `preprocessing.py`) so
   the notebook, `train_models.py`, and `app.py` all transform new data
   identically — this avoids the common bug where a model is trained one
   way and served a different way.
5. **Four models, one winner.** Logistic Regression, Decision Tree, Random
   Forest, and XGBoost are all trained with class-imbalance handling
   (`class_weight="balanced"` or XGBoost's `scale_pos_weight`), since
   rejected applicants are a minority class. The model with the best
   **F1-score** on a held-out 20% test split is saved as the production
   model.
6. **Serving.** `app.py` loads `best_model.pkl` + `feature_pipeline.pkl`
   once at startup. Every form submission or API call runs through the
   exact same `FeaturePipeline.transform()` used in training, then
   `model.predict()` / `model.predict_proba()`.

---

## 5. Using the real Kaggle dataset instead of the synthetic one

The synthetic generator (`data/generate_dataset.py`) was built to match the
real dataset's schema exactly, column-for-column, so swapping in the real
files requires **zero code changes** — just replace the two CSVs in `data/`
and re-run `python data/train_models.py`. You'll likely see different
exact metrics since the real dataset's risk signal differs from the
simulated one.

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'xgboost'` | `pip install xgboost`. The training script will skip XGBoost and still run the other 3 models if it's missing, but install it for the full comparison. |
| `ModuleNotFoundError: No module named 'preprocessing'` when running `app.py` | Make sure you're running `python app.py` from the **project root** (the folder containing `app.py`), not from inside `data/`. |
| Flask page loads but prediction fails with "Model not loaded" | Run `python data/train_models.py` first — the `model/` folder needs the `.pkl` files before `app.py` can serve predictions. |
| `pip install` fails on a package | Make sure you're using Python 3.8+ (`python --version`) and that your virtual environment is activated. |
| IBM Watson deployment script errors with an auth/space error | Double-check `WML_API_KEY`, `WML_URL` (must match the region you created your service in), and `WML_SPACE_ID`. |

---

## 7. Presenting this project

For your team's GitHub/demo links: push this whole folder (minus the
`venv/` virtual environment folder) to a GitHub repository, and if you
deploy the Flask app somewhere reachable (Render, Railway, PythonAnywhere,
or your own server) or deploy to IBM Watson as described above, share that
URL as your live demo link.
