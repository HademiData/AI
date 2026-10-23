import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.styles import BLUE, RED, TEXT, MUTED, chart_layout


COLORS = {
    "No recorded difficulty": BLUE,
    "Recorded difficulty": RED,
}


def prepare_data(train):
    df = train.copy()

    if "AGE_YEARS" not in df.columns:
        df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25

    if "CREDIT_INCOME_RATIO" not in df.columns:
        income = df["AMT_INCOME_TOTAL"].replace(0, float("nan"))
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / income

    df["OUTCOME"] = df["TARGET"].map({
        0: "No recorded difficulty",
        1: "Recorded difficulty",
    })

    df["AGE_GROUP"] = pd.cut(
        df["AGE_YEARS"],
        bins=[18, 25, 35, 45, 55, 65, 100],
        labels=["18–24", "25–34", "35–44", "45–54", "55–64", "65+"],
        right=False,
    )

    return df


def insight_card(label, value, description):
    st.markdown(
        f"""
        <div style="
            background:#FFFFFF;
            border:1px solid #E1E6ED;
            border-radius:10px;
            padding:20px;
            min-height:132px;
        ">
            <div style="font-size:12px;font-weight:600;color:#64748B;
                        letter-spacing:0.5px;margin-bottom:12px;">
                {label}
            </div>
            <div style="font-size:27px;font-weight:700;color:#172033;
                        margin-bottom:8px;">
                {value}
            </div>
            <div style="font-size:13px;color:#64748B;line-height:1.55;">
                {description}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render(train, test):
    df = prepare_data(train)

    total = len(df)
    difficulties = int(df["TARGET"].sum())
    difficulty_rate = df["TARGET"].mean() * 100
    median_age = df["AGE_YEARS"].median()

    st.markdown(
        """
        <div style="margin-bottom:24px;">
            <div style="font-size:12px;font-weight:700;letter-spacing:1.5px;
                        color:#2563EB;">
                PORTFOLIO ANALYTICS
            </div>
            <h1 style="font-size:32px;margin:8px 0;color:#172033;">
                Credit Risk Overview
            </h1>
            <p style="font-size:15px;color:#64748B;line-height:1.7;">
                A clear view of applicant demographics, income, and historical
                payment difficulties in the Home Credit training dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI row
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Training applicants", f"{total:,}")
    c2.metric("Recorded difficulties", f"{difficulties:,}")
    c3.metric("Difficulty rate", f"{difficulty_rate:.2f}%")
    c4.metric("Median age", f"{median_age:.1f} years")

    st.markdown("")
    st.markdown("## Key findings")
    st.caption("Findings are calculated from the training dataset.")

    age_summary = (
        df.dropna(subset=["AGE_GROUP"])
        .groupby("AGE_GROUP", observed=False)
        .agg(
            applicants=("TARGET", "size"),
            difficulties=("TARGET", "sum"),
            difficulty_rate=("TARGET", "mean"),
        )
        .reset_index()
    )

    eligible = age_summary[age_summary["applicants"] >= 1000]

    if not eligible.empty:
        highest = eligible.loc[eligible["difficulty_rate"].idxmax()]
        lowest = eligible.loc[eligible["difficulty_rate"].idxmin()]
        highest_label = str(highest["AGE_GROUP"])
        highest_rate = highest["difficulty_rate"] * 100
        lowest_label = str(lowest["AGE_GROUP"])
        lowest_rate = lowest["difficulty_rate"] * 100
    else:
        highest_label = "N/A"
        highest_rate = 0
        lowest_label = "N/A"
        lowest_rate = 0

    median_income_difficulties = df.loc[
        df["TARGET"] == 1, "AMT_INCOME_TOTAL"
    ].median()
    median_income_other = df.loc[
        df["TARGET"] == 0, "AMT_INCOME_TOTAL"
    ].median()

    if (
        pd.notna(median_income_difficulties)
        and pd.notna(median_income_other)
        and median_income_other > 0
    ):
        income_ratio = median_income_difficulties / median_income_other
        income_value = f"{income_ratio:.2f}×"
        income_description = (
            "Median income among applicants with difficulties compared "
            "with applicants without recorded difficulties."
        )
    else:
        income_value = "N/A"
        income_description = "Median income comparison unavailable."

    k1, k2, k3 = st.columns(3)

    with k1:
        insight_card(
            "HIGHEST AGE-GROUP RATE",
            highest_label,
            f"{highest_rate:.2f}% recorded difficulty rate. "
            "Only age groups with at least 1,000 applicants are compared.",
        )

    with k2:
        insight_card(
            "MEDIAN INCOME COMPARISON",
            income_value,
            income_description,
        )

    with k3:
        insight_card(
            "LOWEST AGE-GROUP RATE",
            lowest_label,
            f"{lowest_rate:.2f}% recorded difficulty rate among "
            "age groups with at least 1,000 applicants.",
        )

    # Age analysis
    st.markdown("")
    st.markdown("## Applicant age")
    st.caption(
        "The two charts answer different questions: how ages are distributed, "
        "and what percentage in each age group had recorded difficulties."
    )

    left, right = st.columns([1.2, 1], gap="large")

    with left:
        age_df = df.dropna(subset=["AGE_YEARS", "OUTCOME"])

        fig_age = px.histogram(
            age_df,
            x="AGE_YEARS",
            color="OUTCOME",
            histnorm="percent",
            barmode="overlay",
            opacity=0.65,
            nbins=32,
            color_discrete_map=COLORS,
            category_orders={
                "OUTCOME": [
                    "No recorded difficulty",
                    "Recorded difficulty",
                ]
            },
            labels={
                "AGE_YEARS": "Applicant age (years)",
                "percent": "Share within outcome group (%)",
                "OUTCOME": "Historical outcome",
            },
            title="Age distribution by outcome",
        )

        fig_age.update_layout(
            barmode="overlay",
            legend_title_text="",
            bargap=0.08,
            yaxis_title="Share within outcome group (%)",
            xaxis_title="Age (years)",
        )
        chart_layout(fig_age, 430)
        fig_age.update_yaxes(ticksuffix="%")
        st.plotly_chart(
            fig_age,
            use_container_width=True,
            config={"displaylogo": False},
        )

        st.caption(
            "Each outcome group is scaled separately to show its age profile. "
            "Bar heights do not represent the groups' relative population sizes."
        )

    with right:
        rate_data = age_summary.copy()
        rate_data["rate_percent"] = rate_data["difficulty_rate"] * 100
        rate_data["label"] = rate_data["difficulty_rate"].map(
            lambda x: f"{x:.1%}"
        )

        fig_rate = px.bar(
            rate_data,
            x="AGE_GROUP",
            y="rate_percent",
            text="label",
            labels={
                "AGE_GROUP": "Age group",
                "rate_percent": "Applicants with recorded difficulties (%)",
            },
            title="Recorded difficulty rate by age",
            color_discrete_sequence=[BLUE],
        )

        fig_rate.update_traces(
            textposition="outside",
            cliponaxis=False,
            marker_line_width=0,
        )
        fig_rate.update_layout(
            showlegend=False,
            xaxis_title="Age group",
            yaxis_title="Recorded difficulty rate (%)",
            uniformtext_minsize=10,
            uniformtext_mode="hide",
        )
        chart_layout(fig_rate, 430)
        fig_rate.update_yaxes(rangemode="tozero")
        st.plotly_chart(
            fig_rate,
            use_container_width=True,
            config={"displaylogo": False},
        )

        st.caption(
            "Rates show the share of applicants in each age group with TARGET = 1."
        )

    # Income and credit analysis
    st.markdown("")
    st.markdown("## Income and credit")
    st.caption(
        "Compare reported income and requested credit relative to income."
    )

    left, right = st.columns(2, gap="large")

    with left:
        income_df = df.dropna(
            subset=["AMT_INCOME_TOTAL", "OUTCOME"]
        ).copy()
        income_df = income_df[income_df["AMT_INCOME_TOTAL"] > 0]

        if not income_df.empty:
            income_limit = income_df["AMT_INCOME_TOTAL"].quantile(0.99)
            income_df = income_df[
                income_df["AMT_INCOME_TOTAL"] <= income_limit
            ]

            fig_income = px.histogram(
                income_df,
                x="AMT_INCOME_TOTAL",
                color="OUTCOME",
                histnorm="percent",
                barmode="overlay",
                opacity=0.65,
                nbins=30,
                color_discrete_map=COLORS,
                labels={
                    "AMT_INCOME_TOTAL": "Reported income",
                    "percent": "Share within outcome group (%)",
                    "OUTCOME": "Historical outcome",
                },
                title="Income distribution",
            )
            fig_income.update_layout(
                legend_title_text="",
                xaxis_title="Reported income",
                yaxis_title="Share within outcome group (%)",
            )
            chart_layout(fig_income, 390)
            fig_income.update_yaxes(ticksuffix="%")
            st.plotly_chart(
                fig_income,
                use_container_width=True,
                config={"displaylogo": False},
            )
            st.caption(
                "The top 1% of income values are excluded from this chart "
                "to make the main distribution easier to read."
            )
        else:
            st.info("Income data is unavailable.")

    with right:
        ratio_df = df.dropna(
            subset=["CREDIT_INCOME_RATIO", "OUTCOME"]
        ).copy()
        ratio_df = ratio_df[
            ratio_df["CREDIT_INCOME_RATIO"].between(0, 10)
        ]

        if not ratio_df.empty:
            fig_ratio = px.box(
                ratio_df,
                x="OUTCOME",
                y="CREDIT_INCOME_RATIO",
                color="OUTCOME",
                color_discrete_map=COLORS,
                points=False,
                category_orders={
                    "OUTCOME": [
                        "No recorded difficulty",
                        "Recorded difficulty",
                    ]
                },
                labels={
                    "OUTCOME": "Historical outcome",
                    "CREDIT_INCOME_RATIO": "Credit / income ratio",
                },
                title="Credit-to-income ratio",
            )
            fig_ratio.update_layout(
                showlegend=False,
                xaxis_title="",
                yaxis_title="Credit amount / annual income",
            )
            chart_layout(fig_ratio, 390)
            st.plotly_chart(
                fig_ratio,
                use_container_width=True,
                config={"displaylogo": False},
            )
            st.caption(
                "Ratios above 10 and invalid values are omitted from this "
                "chart for readability. Box plots show the median and spread."
            )
        else:
            st.info("Credit-to-income ratio data is unavailable.")

    st.markdown("")
    st.markdown("## Interpretation")

    st.markdown(
        """
        - **Age distribution:** compares the age profiles of applicants with
          and without recorded payment difficulties.
        - **Age-group rate:** uses percentages, not raw counts, so age groups
          of different sizes can be compared more fairly.
        - **Income distribution:** shows the shape of reported income within
          each outcome group.
        - **Credit-to-income ratio:** compares requested credit with reported
          income. A higher ratio means the requested credit is larger relative
          to reported income.
        """
    )

    st.info(
        "These charts describe historical data, not causes of payment "
        "difficulties. TARGET = 1 represents recorded payment difficulties "
        "in this dataset, not certainty about future default."
    )
