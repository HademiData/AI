
from pathlib import Path
import sys

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "home-credit-default-risk"
MODEL_PATH = PROJECT_ROOT / "results" / "model" / "lightgbm_credit_model.joblib"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from data.preprocess import engineer_features


def load_model():
    return joblib.load(MODEL_PATH)


def predict_customer(model, customer: pd.DataFrame) -> dict:
    customer = engineer_features(customer)
    customer = customer.drop(columns=["SK_ID_CURR", "TARGET"], errors="ignore")

    probability = float(model.predict_proba(customer)[0, 1])

    return {
        "risk_probability": probability,
        "risk_percentage": round(probability * 100, 2),
        "predicted_class": int(probability >= 0.5),
    }


if __name__ == "__main__":
    model = load_model()

    test_path = DATA_DIR / "application_test.csv"
    test = pd.read_csv(test_path)

    customer_ids = test["SK_ID_CURR"].copy()
    test_features = engineer_features(test).drop(
        columns=["SK_ID_CURR", "TARGET"], errors="ignore"
    )

    probabilities = model.predict_proba(test_features)[:, 1]

    submission = pd.DataFrame({
        "SK_ID_CURR": customer_ids,
        "TARGET": probabilities,
    })

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "submission.csv"
    submission.to_csv(output_path, index=False)

    print("Submission saved to:", output_path)
    print("Rows:", len(submission))
    print(submission.head())
