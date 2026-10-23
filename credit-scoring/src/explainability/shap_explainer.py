from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "home-credit-default-risk"
MODEL_PATH = PROJECT_ROOT / "results/model/lightgbm_credit_model.joblib"
OUTPUT_DIR = PROJECT_ROOT / "results/clients_outputs"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from data.preprocess import engineer_features


def explain_customer(model, raw_customer, customer_id, label):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    customer = engineer_features(raw_customer)
    customer = customer.drop(
        columns=["SK_ID_CURR", "TARGET"], errors="ignore"
    )

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    transformed = preprocessor.transform(customer)
    explainer = shap.TreeExplainer(classifier)
    values = explainer.shap_values(transformed)

    if isinstance(values, list):
        values = values[1]
    elif values.ndim == 3:
        values = values[:, :, 1]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = np.asarray(base_value).reshape(-1)[-1]
    base_value = float(base_value)

    feature_names = preprocessor.get_feature_names_out()
    row = transformed[0]
    if hasattr(row, "toarray"):
        row = row.toarray().ravel()

    explanation = shap.Explanation(
        values=values[0],
        base_values=base_value,
        data=row,
        feature_names=feature_names,
    )

    probability = float(model.predict_proba(customer)[0, 1])
    prefix = OUTPUT_DIR / f"{label}_{customer_id}_shap"

    # Existing waterfall plot
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.tight_layout()
    waterfall_path = Path(f"{prefix}.png")
    plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Static SHAP force plot
    shap.force_plot(
        base_value,
        values[0],
        row,
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    force_png_path = OUTPUT_DIR / f"{label}_{customer_id}_force.png"
    plt.savefig(force_png_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Interactive SHAP force plot
    force_html = shap.force_plot(
        base_value,
        values[0],
        row,
        feature_names=feature_names,
    )
    force_html_path = OUTPUT_DIR / f"{label}_{customer_id}_force.html"
    shap.save_html(str(force_html_path), force_html)

    # Save the most influential feature contributions
    summary = pd.DataFrame({
        "feature": feature_names,
        "shap_value": values[0],
    })
    summary["absolute_shap"] = summary["shap_value"].abs()
    summary = summary.sort_values(
        "absolute_shap", ascending=False
    ).head(15)
    summary.to_csv(f"{prefix}.csv", index=False)

    return probability, waterfall_path, force_png_path, force_html_path


if __name__ == "__main__":
    model = joblib.load(MODEL_PATH)

    train = pd.read_csv(DATA_DIR / "application_train.csv")
    test = pd.read_csv(DATA_DIR / "application_test.csv")

    features = engineer_features(train).drop(
        columns=["SK_ID_CURR", "TARGET"], errors="ignore"
    )
    actual = train["TARGET"].to_numpy()
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    correct = np.flatnonzero(predictions == actual)
    incorrect = np.flatnonzero(predictions != actual)

    examples = [
        ("correct", correct[0], train),
        ("incorrect", incorrect[0], train),
        ("test", 0, test),
    ]

    for label, index, frame in examples:
        customer_id = int(frame.iloc[index]["SK_ID_CURR"])
        probability, waterfall, force_png, force_html = explain_customer(
            model, frame.iloc[[index]], customer_id, label
        )
        print(f"\n{label}: ID={customer_id}, risk={probability:.2%}")
        print(f"  Waterfall: {waterfall}")
        print(f"  Force plot PNG: {force_png}")
        print(f"  Interactive force plot: {force_html}")

    print("\nAll explanations saved to:", OUTPUT_DIR)
