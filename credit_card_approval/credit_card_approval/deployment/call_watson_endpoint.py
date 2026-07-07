"""
call_watson_endpoint.py
-------------------------
Once you've run deploy_to_watson.py and have a scoring URL, use this
helper to send a single applicant's data to the cloud-hosted model and
get back a prediction -- without needing the local Flask app or the
saved .pkl files at all.

Fill in WML_API_KEY and SCORING_URL below (or pass as environment
variables), then run:
    python deployment/call_watson_endpoint.py
"""

import os

import requests

WML_API_KEY = os.environ.get("WML_API_KEY", "PASTE_YOUR_IBM_CLOUD_API_KEY_HERE")
SCORING_URL = os.environ.get("WML_SCORING_URL", "PASTE_YOUR_SCORING_URL_HERE")
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"

FEATURE_COLUMNS = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN",
    "AMT_INCOME_TOTAL", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "AGE_YEARS",
    "YEARS_EMPLOYED", "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL",
    "OCCUPATION_TYPE", "CNT_FAM_MEMBERS", "INCOME_PER_FAMILY_MEMBER",
]


def get_iam_token(api_key):
    resp = requests.post(
        IAM_TOKEN_URL,
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": api_key,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def score_applicant(applicant_row):
    """applicant_row: a list of 17 values in FEATURE_COLUMNS order."""
    if "PASTE_YOUR" in WML_API_KEY or "PASTE_YOUR" in SCORING_URL:
        raise RuntimeError(
            "Set WML_API_KEY and WML_SCORING_URL (env vars or constants above) first."
        )

    token = get_iam_token(WML_API_KEY)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "input_data": [
            {"fields": FEATURE_COLUMNS, "values": [applicant_row]}
        ]
    }
    resp = requests.post(SCORING_URL, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    # Example applicant: gender, own_car, own_realty, children, income,
    # income_type, education, family_status, housing, age, years_employed,
    # work_phone, phone, email, occupation, fam_members, income_per_member
    example_applicant = [
        "M", "Y", "Y", 0, 250000, "Working", "Higher education",
        "Married", "House / apartment", 35, 8, 0, 1, 0,
        "Managers", 3, 83333.3,
    ]
    result = score_applicant(example_applicant)
    print(result)
