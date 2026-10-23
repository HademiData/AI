import streamlit as st

from dashboard.styles import BLUE, CYAN, MUTED, PANEL, RED, TEXT


def brand():
    st.markdown(
        f"""
        <div style="padding:10px 0 24px 0;">
            <div style="font-size:12px;letter-spacing:3px;color:{CYAN};">
                CREDIT INTELLIGENCE
            </div>
            <div style="font-size:25px;font-weight:750;color:{TEXT};">
                SCORING<span style="color:{BLUE};">.AI</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(section, title, subtitle):
    st.caption(section.upper())
    st.title(title)
    st.markdown(
        f'<p style="color:{MUTED};margin-top:-8px;">{subtitle}</p>',
        unsafe_allow_html=True,
    )


def risk_panel(probability):
    color = RED if probability >= 0.5 else CYAN
    label = "HIGHER RISK" if probability >= 0.5 else "LOWER RISK"

    st.markdown(
        f"""
        <div style="background:{PANEL};padding:24px;border-radius:16px;
                    border:1px solid {color}55;">
            <div style="font-size:12px;color:{MUTED};">MODEL RISK SCORE</div>
            <div style="font-size:38px;font-weight:750;color:{color};">
                {probability:.2%}
            </div>
            <div style="font-size:12px;color:{color};">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def footer():
    st.markdown("---")
    st.caption("Credit Scoring | Educational project, not a lending decision system")
