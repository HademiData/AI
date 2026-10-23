import streamlit as st

BLUE = "#155EEF"
CYAN = "#06B6D4"
PANEL = "#FFFFFF"
RED = "#D92D20"
TEXT = "#101828"
MUTED = "#667085"

def apply_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* Global font */
        html, body, [class*="st-"],
        [data-testid="stApp"],
        [data-testid="stMarkdownContainer"],
        input, textarea, button, select {
            font-family: 'Inter', sans-serif !important;
        }

        /* Main dashboard */
        .stApp {
            background-color: #F8FAFC;
            color: #101828;
        }

        /* Headings */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', sans-serif !important;
            color: #101828;
            font-weight: 700;
            letter-spacing: -0.035em;
        }

        /* Sidebar */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child {
            background-color: #155EEF !important;
        }

        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] a,
        [data-testid="stSidebar"] svg {
            color: #FFFFFF !important;
        }

        /* Sidebar input fields */
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] select {
            color: #101828 !important;
            background-color: #FFFFFF !important;
        }

        /* Metric cards */
        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E4E7EC;
            border-radius: 10px;
            padding: 16px;
        }

        [data-testid="stMetricLabel"] {
            font-weight: 500;
            color: #667085;
        }

        [data-testid="stMetricValue"] {
            font-weight: 700;
            letter-spacing: -0.04em;
            color: #101828;
        }

        /* Buttons */
        .stButton > button {
            font-family: 'Inter', sans-serif !important;
            font-weight: 600;
            border-radius: 8px;
        }

        /* Tables and dataframes */
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            font-family: 'Inter', sans-serif !important;
        }

        /* General spacing */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def chart_layout(fig, height=400):
    fig.update_layout(
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Inter, sans-serif",
            color=TEXT,
            size=13,
        ),
        title_font=dict(
            family="Inter, sans-serif",
            size=18,
            color=TEXT,
        ),
        margin=dict(l=20, r=20, t=55, b=35),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E4E7EC",
            borderwidth=1,
            font=dict(family="Inter, sans-serif", size=12),
        ),
        xaxis=dict(
            showgrid=False,
            linecolor="#D0D5DD",
            tickfont=dict(family="Inter, sans-serif", color=TEXT),
        ),
        yaxis=dict(
            gridcolor="#EAECF0",
            linecolor="#D0D5DD",
            tickfont=dict(family="Inter, sans-serif", color=TEXT),
        ),
    )
    return fig
