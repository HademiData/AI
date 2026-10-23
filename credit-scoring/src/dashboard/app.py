from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "home-credit-default-risk"
MODEL_PATH = PROJECT_ROOT / "results/model/lightgbm_credit_model.joblib"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dashboard.styles import apply_styles
from dashboard.pages.overview import render as render_overview
from dashboard.pages.assessment import render as render_assessment
from dashboard.pages.explainability import render as render_explainability

st.set_page_config(
    page_title="CreditLens | Risk Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_styles()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    train = pd.read_csv(DATA_DIR / "application_train.csv")
    test = pd.read_csv(DATA_DIR / "application_test.csv")
    return train, test


def show_sidebar():
    with st.sidebar:
        st.markdown("## CreditLens")
        st.caption("CREDIT RISK INTELLIGENCE")
        st.divider()
        st.markdown("### Model information")
        st.write("**Algorithm:** LightGBM")
        st.write("**Validation ROC-AUC:** 0.7513")
        st.write("**Training sample:** 50,000 applicants")
        st.divider()
        st.caption("Educational decision-support demo")


try:
    model = load_model()
    train, test = load_data()
except Exception as exc:
    st.error(f"Could not load the model or datasets: {exc}")
    st.stop()

def portfolio_overview():
    render_overview(train, test)


def customer_assessment():
    render_assessment(model, train, test)


def model_explainability():
    output_dir = PROJECT_ROOT / "results" / "clients_outputs"
    render_explainability(output_dir)

overview_page = st.Page(
    portfolio_overview,
    title="Portfolio overview",
    icon="📊",
    default=True,
)

assessment_page = st.Page(
    customer_assessment,
    title="Customer assessment",
    icon="👤",
)

explainability_page = st.Page(
    model_explainability,
    title="Model explainability",
    icon="🔍",
)

navigation = st.navigation(
    {
        "CREDIT RISK": [
            overview_page,
            assessment_page,
            explainability_page,
        ]
    },
    position="sidebar",
)

navigation.run()

st.sidebar.divider()
st.sidebar.caption(
    "CreditLens | Educational demonstration. "
    "Scores are estimates, not automated lending decisions."
)
