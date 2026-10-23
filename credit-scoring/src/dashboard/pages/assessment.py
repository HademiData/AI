import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from dashboard.components import page_header, risk_panel
from dashboard.styles import BLUE, RED, MUTED, chart_layout
from data.preprocess import engineer_features
from explainability.shap_explainer import explain_customer


def render(model, train, test):
    page_header(
        "Applicant intelligence / 02",
        "Customer assessment",
        "Inspect an applicant profile and generate an individual risk estimate.",
    )

    source = st.radio(
        "Applicant dataset",
        ["Test", "Training"],
        horizontal=True,
    )
    data = test if source == "Test" else train

    left, right = st.columns([1, 1.7])

    with left:
        customer_id = st.number_input(
            "Customer ID",
            min_value=1,
            value=int(data["SK_ID_CURR"].iloc[0]),
            step=1,
        )
        selected = data.loc[data["SK_ID_CURR"] == customer_id]

    if selected.empty:
        st.warning("Customer ID not found in the selected dataset.")
        return

    raw_customer = selected.iloc[[0]]
    features = engineer_features(raw_customer).drop(
        columns=["SK_ID_CURR", "TARGET"], errors="ignore"
    )
    probability = float(model.predict_proba(features)[0, 1])

    with right:
        st.markdown("### Applicant risk score")
        risk_panel(probability)

    age = -float(raw_customer["DAYS_BIRTH"].iloc[0]) / 365.25
    income = raw_customer["AMT_INCOME_TOTAL"].iloc[0]
    credit = raw_customer["AMT_CREDIT"].iloc[0]
    annuity = raw_customer["AMT_ANNUITY"].iloc[0]
    ratio = credit / income if pd.notna(income) and income else None

    st.markdown("### Applicant profile")
    a, b, c, d = st.columns(4)
    a.metric("Age", f"{age:.1f} years")
    b.metric("Annual income", f"{income:,.0f}" if pd.notna(income) else "N/A")
    c.metric("Credit amount", f"{credit:,.0f}" if pd.notna(credit) else "N/A")
    d.metric("Credit / income", f"{ratio:.2f}×" if ratio is not None else "N/A")

    st.caption(
        f"Annual annuity: {annuity:,.0f}"
        if pd.notna(annuity) else "Annual annuity: N/A"
    )

    if source == "Training":
        actual = int(raw_customer["TARGET"].iloc[0])
        st.write(
            "Recorded outcome:",
            "Payment difficulties" if actual == 1
            else "No recorded payment difficulties",
        )

    st.markdown("---")
    st.markdown("### Explain this prediction")

    if st.button("Generate SHAP explanation", use_container_width=True):
        with st.spinner("Calculating feature contributions..."):
            result = explain_customer(
                model, raw_customer, int(customer_id), "dashboard"
            )
        probability, waterfall, force_png, force_html = result
        st.session_state["assessment_explanation"] = {
            "id": int(customer_id),
            "waterfall": str(waterfall),
            "force_html": str(force_html),
        }

    explanation = st.session_state.get("assessment_explanation")
    if explanation and explanation["id"] == int(customer_id):
        tab1, tab2 = st.tabs(["Waterfall", "Interactive force plot"])
        with tab1:
            st.image(explanation["waterfall"], use_container_width=True)
        with tab2:
            from pathlib import Path
            html_path = Path(explanation["force_html"])
            components.html(
                html_path.read_text(encoding="utf-8"),
                height=450,
                scrolling=True,
            )

    st.caption(
        "The 50% threshold is illustrative, not a validated lending cutoff. "
        "A model score is not a guarantee of future behavior."
    )
