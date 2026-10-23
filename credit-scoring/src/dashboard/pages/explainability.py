import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from dashboard.components import page_header
from dashboard.styles import RED, CYAN, chart_layout


def render(output_dir):
    page_header(
        "Model transparency / 03",
        "Model explainability",
        "Inspect saved SHAP explanations and the features influencing model scores.",
    )

    st.markdown("### How the model works")
    st.write(
        "CreditLens uses LightGBM, a machine learning model that learns patterns "
        "from historical loan application data to estimate an applicant's risk "
        "of experiencing payment difficulties. Its validation ROC-AUC score is "
        "0.7513, measuring how well it distinguishes between applicants with "
        "and without recorded payment difficulties."
    )
    st.write(
        "SHAP helps explain individual predictions by showing which applicant "
        "features pushed the model's estimate higher or lower compared with its "
        "baseline. This helps users understand the model's reasoning, but these "
        "contributions do not prove cause and effect. Risk estimates are not "
        "guarantees of future repayment or automatic lending decisions."
    )

    st.divider()

    if not output_dir.exists():
        st.info("No explanation outputs found yet.")
        return

    waterfalls = sorted(
        p for p in output_dir.glob("*_shap.png")
        if not p.name.startswith("dashboard_")
    )
    force_plots = sorted(
        p for p in output_dir.glob("*_force.html")
        if not p.name.startswith("dashboard_")
    )
    csv_files = sorted(
        p for p in output_dir.glob("*_shap.csv")
        if not p.name.startswith("dashboard_")
    )

    a, b, c = st.columns(3)
    a.metric("Waterfall plots", len(waterfalls))
    b.metric("Force plots", len(force_plots))
    c.metric("Feature reports", len(csv_files))

    tab1, tab2 = st.tabs(["Customer plots", "Feature contributions"])

    with tab1:
        if not waterfalls:
            st.info("Generate an explanation from Customer Assessment.")
        else:
            selected = st.selectbox(
                "Select customer",
                waterfalls,
                format_func=lambda p: p.stem,
            )
            st.image(str(selected), use_container_width=True)

            force_path = output_dir / selected.name.replace(
                "_shap.png", "_force.html"
            )
            if force_path.exists():
                with st.expander("View interactive force plot"):
                    components.html(
                        force_path.read_text(encoding="utf-8"),
                        height=450,
                        scrolling=True,
                    )

    with tab2:
        if not csv_files:
            st.info("No feature contribution reports found.")
        else:
            selected_csv = st.selectbox(
                "Select feature report",
                csv_files,
                format_func=lambda p: p.stem,
            )
            explanation = pd.read_csv(selected_csv).sort_values(
                "absolute_shap", ascending=True
            ).tail(15)

            fig = px.bar(
                explanation,
                x="shap_value",
                y="feature",
                orientation="h",
                color="shap_value",
                color_continuous_scale=[RED, "#253449", CYAN],
                labels={
                    "shap_value": "SHAP value",
                    "feature": "Feature",
                },
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(chart_layout(fig, 480), use_container_width=True)

            st.dataframe(
                explanation.sort_values(
                    "absolute_shap", ascending=False
                ),
                hide_index=True,
                use_container_width=True,
            )

    st.info(
        "SHAP explains how features contribute to the model output relative to "
        "a baseline. These contributions are not causal explanations."
    )
