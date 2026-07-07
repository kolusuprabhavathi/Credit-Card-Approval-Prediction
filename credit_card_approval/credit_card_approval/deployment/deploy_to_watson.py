"""
deploy_to_watson.py
---------------------
Deploys the trained credit-card-approval model to IBM Watson Machine
Learning (WML), so predictions can be served from the cloud at scale
instead of (or in addition to) the local Flask app.

PREREQUISITES
  1. An IBM Cloud account with a Watson Machine Learning service instance.
     Create one free at: https://cloud.ibm.com/catalog/services/machine-learning
  2. An IBM Cloud API key:
     IBM Cloud Console -> Manage -> Access (IAM) -> API keys -> Create
  3. Your WML instance's Space ID (a "Deployment Space"):
     IBM Cloud -> Watson Machine Learning -> Deployment spaces -> create one,
     then copy its Space GUID from the space's "Manage" tab.
  4. Install the IBM Watson Machine Learning SDK:
         pip install ibm-watson-machine-learning

CONFIGURATION
  Fill in the four values below (or set them as environment variables of the
  same name) before running this script:
      WML_API_KEY
      WML_URL          (e.g. "https://us-south.ml.cloud.ibm.com" -- pick the
                         region that matches where you created your service)
      WML_SPACE_ID
      WML_DEPLOYMENT_NAME  (optional, just a label)

RUN
    python deployment/deploy_to_watson.py

WHAT IT DOES
  1. Loads your already-trained best_model.pkl + feature_pipeline.pkl
  2. Authenticates to IBM Watson Machine Learning
  3. Stores the scikit-learn model as a WML "model asset"
  4. Creates an online deployment for that model
  5. Prints the scoring endpoint URL you can call from anywhere
     (Flask app, Postman, curl, another notebook, etc.)

NOTE ON THE FEATURE PIPELINE
  Watson Machine Learning's "online deployment" scores a raw scikit-learn
  estimator -- it does not run our custom FeaturePipeline (label encoding +
  scaling) for us. So this script wraps the trained classifier in an
  sklearn Pipeline object that includes the same encoding/scaling steps,
  ensuring the deployed cloud endpoint expects the SAME raw input columns
  as our Flask form, not pre-encoded numbers.
"""

import os
import sys

import joblib
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
sys.path.append(DATA_DIR)

from preprocessing import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS  # noqa: E402

# ---------------------------------------------------------------------------
# 1. CONFIGURATION -- fill these in or set as environment variables
# ---------------------------------------------------------------------------
WML_API_KEY = os.environ.get("WML_API_KEY", "PASTE_YOUR_IBM_CLOUD_API_KEY_HERE")
WML_URL = os.environ.get("WML_URL", "https://us-south.ml.cloud.ibm.com")
WML_SPACE_ID = os.environ.get("WML_SPACE_ID", "PASTE_YOUR_DEPLOYMENT_SPACE_ID_HERE")
WML_DEPLOYMENT_NAME = os.environ.get("WML_DEPLOYMENT_NAME", "credit-card-approval-deployment")


def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    pipeline = joblib.load(os.path.join(MODEL_DIR, "feature_pipeline.pkl"))
    with open(os.path.join(MODEL_DIR, "best_model_name.txt")) as f:
        model_name = f.read().strip()
    print(f"Loaded trained model: {model_name}")
    return model, pipeline, model_name


def build_deployable_pipeline(trained_model, feature_pipeline):
    """
    Wrap the trained classifier + our custom encoders/scaler into a single
    sklearn-compatible object using FunctionTransformer, so Watson ML can
    serialize and score it as one estimator that accepts raw applicant
    fields directly (matching FEATURE_COLUMNS order).
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    def transform_raw(df):
        return feature_pipeline.transform(df)

    full_pipeline = Pipeline(steps=[
        ("feature_engineering", FunctionTransformer(transform_raw)),
        ("classifier", trained_model),
    ])
    return full_pipeline


def deploy():
    try:
        from ibm_watson_machine_learning import APIClient
    except ImportError:
        print("\n[ERROR] The ibm-watson-machine-learning package is not installed.")
        print("Install it with:\n    pip install ibm-watson-machine-learning\n")
        sys.exit(1)

    if "PASTE_YOUR" in WML_API_KEY or "PASTE_YOUR" in WML_SPACE_ID:
        print("\n[ERROR] Please set WML_API_KEY and WML_SPACE_ID before running this script.")
        print("Either edit the constants at the top of this file, or set them as")
        print("environment variables, e.g.:\n")
        print("    export WML_API_KEY='your-ibm-cloud-api-key'")
        print("    export WML_SPACE_ID='your-deployment-space-id'")
        print("    python deployment/deploy_to_watson.py\n")
        sys.exit(1)

    trained_model, feature_pipeline, model_name = load_artifacts()
    deployable_pipeline = build_deployable_pipeline(trained_model, feature_pipeline)

    print("\nAuthenticating with IBM Watson Machine Learning...")
    wml_credentials = {"apikey": WML_API_KEY, "url": WML_URL}
    client = APIClient(wml_credentials)
    client.set.default_space(WML_SPACE_ID)
    print("Authenticated. Using deployment space:", WML_SPACE_ID)

    print("\nStoring model asset in Watson Machine Learning...")
    sklearn_version = __import__("sklearn").__version__
    software_spec_uid = client.software_specifications.get_id_by_name("runtime-23.1-py3.10")

    metadata = {
        client.repository.ModelMetaNames.NAME: f"Credit Card Approval - {model_name}",
        client.repository.ModelMetaNames.TYPE: "scikit-learn_1.1",
        client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: software_spec_uid,
    }

    published_model = client.repository.store_model(
        model=deployable_pipeline,
        meta_props=metadata,
    )
    model_uid = client.repository.get_model_id(published_model)
    print(f"Model stored. Model UID: {model_uid}")

    print("\nCreating online deployment...")
    deployment_metadata = {
        client.deployments.ConfigurationMetaNames.NAME: WML_DEPLOYMENT_NAME,
        client.deployments.ConfigurationMetaNames.ONLINE: {},
    }
    created_deployment = client.deployments.create(model_uid, meta_props=deployment_metadata)
    deployment_uid = client.deployments.get_id(created_deployment)
    scoring_url = client.deployments.get_scoring_href(created_deployment)

    print("\n========================================")
    print("DEPLOYMENT SUCCESSFUL")
    print("========================================")
    print(f"Deployment UID : {deployment_uid}")
    print(f"Scoring URL    : {scoring_url}")
    print("\nSave this scoring URL -- you'll call it from your Flask app,")
    print("Postman, curl, or any HTTP client to get predictions from the cloud.")
    print("\nExample payload to POST to that scoring URL:")
    print("""
{
  "input_data": [
    {
      "fields": %s,
      "values": [["M", "Y", "Y", 0, 250000, "Working", "Higher education",
                   "Married", "House / apartment", 35, 8, 0, 1, 0,
                   "Managers", 3, 83333.3]]
    }
  ]
}
""" % FEATURE_COLUMNS)


if __name__ == "__main__":
    deploy()
