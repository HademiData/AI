from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "home-credit-default-risk"
OUTPUT_DIR = PROJECT_ROOT / "results/clients_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from data.preprocess import engineer_features


def create_client_report(customer_id: int, dataset: str = "train"):
    filename = "application_train.csv" if dataset == "train" else "application_test.csv"
    data = pd.read_csv(DATA_DIR / filename)

    if customer_id not in set(data["SK_ID_CURR"].values):
        raise ValueError(f"Customer {customer_id} was not found in {filename}")

    data = engineer_features(data)
    customer = data.loc[data["SK_ID_CURR"] == customer_id].iloc[0]

    comparison_features = {
        "AGE_YEARS": "Age (years)",
        "AMT_INCOME_TOTAL": "Income",
        "AMT_CREDIT": "Credit amount",
        "AMT_ANNUITY": "Annual annuity",
        "CREDIT_INCOME_RATIO": "Credit-to-income ratio",
    }

    plots = []
    for feature, label in comparison_features.items():
        fig = px.histogram(
            data,
            x=feature,
            nbins=40,
            title=f"{label}: Customer {customer_id} vs training population",
            opacity=0.75,
        )
        fig.add_vline(
            x=float(customer[feature]),
            line_dash="dash",
            line_width=3,
            annotation_text="Selected customer",
        )
        fig.update_layout(
            xaxis_title=label,
            yaxis_title="Number of applicants",
            showlegend=False,
        )
        plots.append(fig)

    # Save the five comparison charts together in one interactive HTML file.
    output_path = OUTPUT_DIR / f"client_{customer_id}_comparison.html"
    with output_path.open("w", encoding="utf-8") as file:
        file.write("<html><head><title>Customer Comparison</title></head><body>")
        file.write(f"<h1>Customer {customer_id} comparison</h1>")
        file.write(
            "<p>Dashed lines indicate this customer's value. "
            "Distributions use the training population.</p>"
        )
        for index, fig in enumerate(plots):
            file.write(fig.to_html(full_html=False, include_plotlyjs="cdn" if index == 0 else False))
        file.write("</body></html>")

    print(f"Saved customer {customer_id} report: {output_path}")


if __name__ == "__main__":
    create_client_report(100002, "train")
    create_client_report(100007, "train")
    create_client_report(100001, "test")
