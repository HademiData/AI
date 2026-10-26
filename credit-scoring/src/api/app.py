from pathlib import Path
import sys

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.preprocess import engineer_features

MODEL_PATH = ROOT / "results" / "model" / "lightgbm_credit_model.joblib"
DATA_PATH = ROOT / "home-credit-default-risk" / "application_test.csv"

app = FastAPI(title="CreditLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH)
data = data.set_index("SK_ID_CURR", drop=False)


class AssessmentRequest(BaseModel):
    customer_id: int


@app.get("/api/health")
def health():
    return {"status": "healthy", "service": "CreditLens"}


@app.get("/api/stats")
def stats():
    return {
        "customers": int(len(data)),
        "model": "LightGBM",
    }

@app.get("/api/explainability")
def explainability():
    importances = getattr(model, "feature_importances_", None)

    if importances is None:
        raise HTTPException(
            status_code=500,
            detail="The loaded model does not expose feature importance."
        )

    feature_names = getattr(model, "feature_name_", None)

    if feature_names is None or len(feature_names) != len(importances):
        feature_names = [f"Feature {i + 1}" for i in range(len(importances))]

    ranked = sorted(
        [
            {"feature": str(name), "importance": int(value)}
            for name, value in zip(feature_names, importances)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )

    return {"features": ranked[:15]}

@app.get("/api/customers")
def get_customers(
    search: str = "",
    limit: int = 20,
    offset: int = 0,
):
    customer_ids = data.index.astype(int)

    if search.strip():
        customer_ids = customer_ids[
            customer_ids.astype(str).str.contains(search.strip(), regex=False)
        ]

    total = len(customer_ids)
    customer_ids = customer_ids[offset:offset + min(limit, 20)]

    return {
        "customer_ids": customer_ids.tolist(),
        "total": total,
        "limit": min(limit, 20),
        "offset": offset,
    }


@app.post("/api/assess")
def assess(request: AssessmentRequest):
    if request.customer_id not in data.index:
        raise HTTPException(
            status_code=404,
            detail="Customer ID not found in the test dataset.",
        )

    customer = data.loc[[request.customer_id]]
    features = engineer_features(customer).drop(
        columns=["SK_ID_CURR", "TARGET"], errors="ignore"
    )

    probability = float(model.predict_proba(features)[0, 1])

    return {
        "customer_id": request.customer_id,
        "risk_probability": round(probability, 6),
        "risk_percentage": round(probability * 100, 2),
        "risk_level": (
            "High" if probability >= 0.5
            else "Moderate" if probability >= 0.25
            else "Low"
        ),
        "predicted_class": int(probability >= 0.5),
        "disclaimer": (
            "This is a model estimate, not a lending decision. "
            "The 50% threshold is a demonstration threshold, "
            "not a validated lending policy."
        ),
    }


app.mount(
    "/",
    StaticFiles(directory=ROOT / "frontend", html=True),
    name="frontend",
)
