from pathlib import Path
import argparse
import sys

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "home-credit-default-risk"
MODEL_PATH = PROJECT_ROOT / "results/model/lightgbm_credit_model.joblib"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explainability.shap_explainer import explain_customer


def main():
    parser = argparse.ArgumentParser(
        description="Predict and explain credit risk for a customer ID."
    )
    parser.add_argument("customer_id", type=int)
    parser.add_argument(
        "--dataset",
        choices=["train", "test"],
        default="test",
        help="Dataset containing the customer. Defaults to test.",
    )
    args = parser.parse_args()

    filename = (
        "application_train.csv"
        if args.dataset == "train"
        else "application_test.csv"
    )
    data = pd.read_csv(DATA_DIR / filename)
    selected = data.loc[data["SK_ID_CURR"] == args.customer_id]

    if selected.empty:
        raise SystemExit(
            f"Customer ID {args.customer_id} was not found in {filename}."
        )

    model = joblib.load(MODEL_PATH)
    probability, waterfall, force_png, force_html = explain_customer(
        model,
        selected,
        args.customer_id,
        "customer",
    )

    print(f"Customer ID: {args.customer_id}")
    print(f"Predicted payment-difficulty risk: {probability:.2%}")
    print(f"Risk band: {'Higher' if probability >= 0.5 else 'Lower'} model score")
    if "TARGET" in selected.columns:
        print(f"Recorded target: {int(selected.iloc[0]['TARGET'])}")
    print(f"SHAP waterfall: {waterfall}")
    print(f"SHAP force plot (PNG): {force_png}")
    print(f"Interactive SHAP force plot: {force_html}")


if __name__ == "__main__":
    main()
