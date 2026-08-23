import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_URL = (
    "https://raw.githubusercontent.com/dicodingacademy/"
    "dicoding_dataset/refs/heads/main/employee/employee_data.csv"
)

FILTER_COLUMNS = [
    "Department",
    "JobRole",
    "OverTime",
    "Gender",
    "MaritalStatus",
    "BusinessTravel",
]

TENURE_BINS = [-1, 0, 3, 6, 10, 100]

TENURE_LABELS = [
    "Baru (0 Tahun)",
    "1-3 Tahun",
    "4-6 Tahun",
    "7-10 Tahun",
    "Lebih dari 10 Tahun",
]

PRIMARY_COLOR = "#2563EB"
PRIMARY_DARK = "#1D4ED8"
ACCENT_COLOR = "#F97316"
DANGER_COLOR = "#EF4444"
SUCCESS_COLOR = "#10B981"
WARNING_COLOR = "#F59E0B"
NEUTRAL_COLOR = "#94A3B8"

DARK = "#0F172A"
TEXT = "#334155"
MUTED = "#64748B"
BORDER = "#E2E8F0"
SURFACE = "#FFFFFF"
BACKGROUND = "#F8FAFC"


# ============================================================
# PROFESSIONAL UI
# ============================================================

st.markdown(
    f"""
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {{
        background: {BACKGROUND};
        color: {TEXT};
    }}

    .block-container {{
        max-width: 1480px;
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        padding-left: 2.2rem;
        padding-right: 2.2rem;
    }}

    h1, h2, h3, h4 {{
        color: {DARK} !important;
        font-weight: 750 !important;
        letter-spacing: -0.025em;
    }}

    h1 {{
        font-size: 2rem !important;
        margin-bottom: 0.15rem !important;
    }}

    h2 {{
        font-size: 1.3rem !important;
    }}

    h3 {{
        font-size: 1.05rem !important;
    }}

    p {{
        color: {TEXT};
    }}

    hr {{
        border: 0 !important;
        border-top: 1px solid {BORDER} !important;
        margin: 1.35rem 0 !important;
    }}


    /* ========================================================
       TOP PAGE HEADER
       ======================================================== */

    .page-header {{
        background:
            linear-gradient(
                135deg,
                #FFFFFF 0%,
                #F8FAFF 65%,
                #EFF6FF 100%
            );

        border: 1px solid {BORDER};
        border-radius: 18px;

        padding: 24px 28px;

        margin-bottom: 20px;

        box-shadow:
            0 8px 24px rgba(15, 23, 42, 0.045);
    }}

    .page-header-title {{
        color: {DARK};
        font-size: 1.9rem;
        line-height: 1.2;
        font-weight: 800;
        letter-spacing: -0.035em;
    }}

    .page-header-subtitle {{
        color: {MUTED};
        font-size: 0.9rem;
        line-height: 1.55;
        margin-top: 6px;
    }}


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {{
        background:
            linear-gradient(
                180deg,
                #0F172A 0%,
                #111827 100%
            ) !important;

        border-right: 1px solid #1E293B;
    }}

    section[data-testid="stSidebar"] > div {{
        padding-top: 1.2rem;
    }}

    section[data-testid="stSidebar"] * {{
        color: #E2E8F0;
    }}

    section[data-testid="stSidebar"] hr {{
        border-color: #334155 !important;
        margin: 0.9rem 0 !important;
    }}

    section[data-testid="stSidebar"] label {{
        color: #CBD5E1 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
    }}

    section[data-testid="stSidebar"] div[data-baseweb="select"] {{
        background: #FFFFFF !important;
        border-radius: 9px !important;
        border: none !important;
    }}

    section[data-testid="stSidebar"]
    div[data-baseweb="select"] * {{
        color: #0F172A !important;
    }}

    section[data-testid="stSidebar"] input {{
        color: #0F172A !important;
    }}

    .sidebar-brand {{
        padding: 4px 2px 14px 2px;
    }}

    .sidebar-brand-title {{
        color: #FFFFFF;
        font-size: 1.28rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }}

    .sidebar-brand-subtitle {{
        color: #94A3B8;
        font-size: 0.74rem;
        margin-top: 3px;
    }}

    .sidebar-label {{
        color: #64748B;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
        margin-bottom: 8px;
    }}


    /* ========================================================
       KPI CARDS
       ======================================================== */

    div[data-testid="stMetric"] {{
        background: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 14px !important;

        padding: 16px 18px 14px 18px !important;

        min-height: 104px;

        box-shadow:
            0 4px 14px rgba(15, 23, 42, 0.035) !important;

        transition:
            transform 0.15s ease,
            box-shadow 0.15s ease;
    }}

    div[data-testid="stMetric"]:hover {{
        transform: translateY(-2px);

        box-shadow:
            0 8px 22px rgba(15, 23, 42, 0.065) !important;
    }}

    div[data-testid="stMetricLabel"] {{
        color: {MUTED} !important;
        font-size: 0.76rem !important;
        font-weight: 650 !important;
    }}

    div[data-testid="stMetricValue"] {{
        color: {DARK} !important;
        font-size: 1.55rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em;
    }}

    div[data-testid="stMetricDelta"] {{
        font-size: 0.76rem !important;
    }}


    /* ========================================================
       SECTION LABEL
       ======================================================== */

    .section-label {{
        color: {DARK};
        font-size: 1rem;
        font-weight: 750;
        margin: 4px 0 10px 0;
    }}

    .section-caption {{
        color: {MUTED};
        font-size: 0.8rem;
        margin-bottom: 12px;
    }}


    /* ========================================================
       PLOTLY CARDS
       ======================================================== */

    div[data-testid="stPlotlyChart"] {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 6px 8px 2px 8px;

        box-shadow:
            0 4px 14px rgba(15, 23, 42, 0.035);
    }}


    /* ========================================================
       INSIGHT CARDS
       ======================================================== */

    .insight-box {{
        position: relative;

        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 12px;

        padding: 14px 16px 14px 18px;
        margin-bottom: 10px;

        color: {TEXT};
        font-size: 0.86rem;
        line-height: 1.5;

        box-shadow:
            0 3px 10px rgba(15, 23, 42, 0.03);
    }}

    .insight-box::before {{
        content: "";
        position: absolute;
        left: 0;
        top: 12px;
        bottom: 12px;
        width: 3px;

        background: {PRIMARY_COLOR};
        border-radius: 3px;
    }}

    .insight-box b {{
        color: {DARK};
        font-weight: 750;
    }}


    /* ========================================================
       PRIORITY CARDS
       ======================================================== */

    .priority-high {{
        background: #FFF8F5;
        border: 1px solid #FCE0D5;
        border-radius: 12px;

        padding: 14px 16px 14px 18px;
        margin-bottom: 10px;

        color: {TEXT};
        line-height: 1.5;

        box-shadow:
            0 3px 10px rgba(15, 23, 42, 0.025);
    }}

    .priority-high b {{
        color: #C2410C;
    }}

    .priority-medium {{
        background: #FFFDF5;
        border: 1px solid #F7E9B9;
        border-radius: 12px;

        padding: 14px 16px 14px 18px;
        margin-bottom: 10px;

        color: {TEXT};
        line-height: 1.5;

        box-shadow:
            0 3px 10px rgba(15, 23, 42, 0.025);
    }}

    .priority-medium b {{
        color: #A16207;
    }}


    /* ========================================================
       TABS
       ======================================================== */

    div[data-baseweb="tab-list"] {{
        gap: 4px;
    }}

    button[data-baseweb="tab"] {{
        color: {MUTED} !important;
        font-weight: 650 !important;
        font-size: 0.84rem !important;

        padding-left: 14px !important;
        padding-right: 14px !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {DARK} !important;
    }}


    /* ========================================================
       DATAFRAME
       ======================================================== */

    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 12px;
        overflow: hidden;

        box-shadow:
            0 3px 10px rgba(15, 23, 42, 0.03);
    }}


    /* ========================================================
       CAPTION
       ======================================================== */

    div[data-testid="stCaptionContainer"] {{
        color: {MUTED} !important;
        font-size: 0.76rem !important;
    }}


    /* ========================================================
       ALERTS
       ======================================================== */

    div[data-testid="stAlert"] {{
        border-radius: 10px !important;
    }}


    /* ========================================================
       RESPONSIVE
       ======================================================== */

    @media (max-width: 900px) {{

        .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}

        .page-header {{
            padding: 20px;
        }}

        .page-header-title {{
            font-size: 1.55rem;
        }}
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD & CLEAN DATA
# ============================================================

@st.cache_data(show_spinner="Mengambil dan membersihkan data...")
def load_data():

    df = pd.read_csv(DATA_URL)

    string_cols = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for col in string_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
        )

    df = df.drop_duplicates()

    df = df.drop(
        columns=[
            "EmployeeCount",
            "StandardHours",
            "Over18",
        ],
        errors="ignore",
    )

    df["Attrition"] = pd.to_numeric(
        df["Attrition"],
        errors="coerce"
    )

    df["TenureGroup"] = pd.cut(
        df["YearsAtCompany"],
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
    )

    df["IncomeGroup"] = pd.qcut(
        df["MonthlyIncome"],
        q=4,
        labels=[
            "Q1 (Terendah)",
            "Q2",
            "Q3",
            "Q4 (Tertinggi)",
        ],
        duplicates="drop",
    )

    satisfaction_cols = [
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "RelationshipSatisfaction",
    ]

    df["AvgSatisfaction"] = df[
        satisfaction_cols
    ].mean(axis=1)

    df["StatusLabel"] = np.select(
        [
            df["Attrition"] == 0,
            df["Attrition"] == 1,
        ],
        [
            "Bertahan",
            "Keluar",
        ],
        default="Belum Diketahui",
    )

    return df


try:
    data = load_data()

except Exception as e:

    st.error(
        "Gagal mengambil dataset dari sumber online. "
        "Silakan cek koneksi internet atau coba lagi."
    )

    st.exception(e)
    st.stop()


# ============================================================
# HELPERS
# ============================================================

def get_labeled(df):

    return df[
        df["Attrition"].notna()
    ].copy()


def safe_attrition_rate(df):

    if df.empty:
        return np.nan

    return (
        df["Attrition"].mean()
        * 100
    )


def attrition_rate_by(
    df_labeled,
    col,
):

    if (
        df_labeled.empty
        or col not in df_labeled.columns
    ):

        return pd.DataFrame(
            columns=[
                col,
                "AttritionRate",
                "Count",
            ]
        )

    result = (
        df_labeled
        .groupby(
            col,
            observed=True
        )["Attrition"]
        .agg(
            AttritionRate="mean",
            Count="count"
        )
        .reset_index()
    )

    result["AttritionRate"] = (
        result["AttritionRate"]
        * 100
    ).round(1)

    return result.sort_values(
        "AttritionRate",
        ascending=False
    )


def format_thousand(value):

    if pd.isna(value):
        return "-"

    return f"{value:,.0f}"


def empty_state(
    message="Tidak ada data untuk filter ini."
):

    st.info(message)


def style_chart(
    fig,
    height=None,
):

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",

        font=dict(
            family="Arial, sans-serif",
            color=DARK,
        ),

        title=dict(
            font=dict(
                size=15,
                color=DARK,
            ),
            x=0.01,
            xanchor="left",
        ),

        margin=dict(
            l=12,
            r=16,
            t=54,
            b=14,
        ),

        hoverlabel=dict(
            bgcolor="white",
            font_color=DARK,
        ),

        legend=dict(
            bgcolor="rgba(255,255,255,0)",
        ),
    )

    if height:
        fig.update_layout(
            height=height
        )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E5E7EB",
        zeroline=False,
        linecolor="#E5E7EB",
    )

    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        linecolor="#E5E7EB",
    )

    return fig


def page_header(
    title,
    subtitle,
):

    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-header-title">
                {title}
            </div>

            <div class="page-header-subtitle">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def bar_attrition_rate(
    rate_df,
    category_col,
    title,
    order=None,
):

    if rate_df.empty:

        empty_state()
        return

    plot_df = rate_df.copy()

    if order is not None:

        plot_df[category_col] = pd.Categorical(
            plot_df[category_col],
            categories=order,
            ordered=True,
        )

        plot_df = plot_df.sort_values(
            category_col
        )

    else:

        plot_df = plot_df.sort_values(
            "AttritionRate",
            ascending=True,
        )

    fig = px.bar(
        plot_df,
        x="AttritionRate",
        y=category_col,

        orientation="h",

        text=(
            plot_df["AttritionRate"]
            .astype(str)
            + "%"
        ),

        color_discrete_sequence=[
            ACCENT_COLOR
        ],

        title=title,

        hover_data={
            "Count": True
        },

        labels={
            "AttritionRate":
                "Attrition Rate (%)",
            category_col:
                "",
            "Count":
                "Employees",
        },
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    style_chart(
        fig,
        height=max(
            280,
            40 * len(plot_df),
        ),
    )

    fig.update_layout(
        xaxis_title="Attrition Rate (%)",
        yaxis_title=None,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def headcount_chart(
    df,
    column,
    title,
):

    result = (
        df[column]
        .value_counts()
        .reset_index()
    )

    result.columns = [
        column,
        "Employees",
    ]

    result = result.sort_values(
        "Employees"
    )

    fig = px.bar(
        result,

        x="Employees",
        y=column,

        orientation="h",

        text="Employees",

        color_discrete_sequence=[
            PRIMARY_COLOR
        ],

        title=title,
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    style_chart(fig)

    fig.update_layout(
        yaxis_title=None
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div class="sidebar-brand">

        <div class="sidebar-brand-title">
            👥 HR Analytics
        </div>

        <div class="sidebar-brand-subtitle">
            Employee Attrition Dashboard
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div class="sidebar-label">Navigation</div>',
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Attrition Analysis",
        "Employee Profile",
        "HR Action",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()

st.sidebar.markdown(
    '<div class="sidebar-label">Filters</div>',
    unsafe_allow_html=True,
)

filter_values = {}

for col in FILTER_COLUMNS:

    if col in data.columns:

        options = sorted(
            data[col]
            .dropna()
            .unique()
            .tolist()
        )

        selected = st.sidebar.multiselect(
            col,
            options,
            default=[],
        )

        if selected:

            filter_values[col] = selected


def apply_filters(df):

    filtered = df.copy()

    for col, values in filter_values.items():

        filtered = filtered[
            filtered[col].isin(values)
        ]

    return filtered


data_f = apply_filters(data)
labeled_f = get_labeled(data_f)

if data_f.empty:

    st.warning(
        "Tidak ada karyawan yang cocok dengan filter."
    )

    st.stop()


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    page_header(
        "HR Analytics Dashboard",
        "Workforce overview, employee attrition patterns, and HR priorities",
    )

    total_employees = len(data_f)

    n_labeled = len(labeled_f)

    n_unlabeled = (
        total_employees
        - n_labeled
    )

    attrition_rate = safe_attrition_rate(
        labeled_f
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Employees",
        format_thousand(
            total_employees
        ),
    )

    col2.metric(
        "Attrition Rate",
        (
            f"{attrition_rate:.1f}%"
            if not pd.isna(
                attrition_rate
            )
            else "-"
        ),
    )

    col3.metric(
        "Average Age",
        f"{data_f['Age'].mean():.1f} yrs",
    )

    col4.metric(
        "Average Tenure",
        f"{data_f['YearsAtCompany'].mean():.1f} yrs",
    )

    col5.metric(
        "Average Monthly Income",
        format_thousand(
            data_f["MonthlyIncome"].mean()
        ),
    )

    st.caption(
        f"Labeled Employees: "
        f"{format_thousand(n_labeled)}"
        f"  •  "
        f"Unlabeled Employees: "
        f"{format_thousand(n_unlabeled)}"
        f"  •  "
        "Attrition Rate dihitung dari data berlabel."
    )

    st.divider()

    st.markdown(
        '<div class="section-label">Workforce Distribution</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:

        headcount_chart(
            data_f,
            "Department",
            "Employees by Department",
        )

    with c2:

        headcount_chart(
            data_f,
            "JobRole",
            "Employees by Job Role",
        )

    c3, c4 = st.columns(2)

    with c3:

        status_counts = (
            data_f[
                "StatusLabel"
            ]
            .value_counts()
            .reset_index()
        )

        status_counts.columns = [
            "Status",
            "Employees",
        ]

        fig = px.pie(
            status_counts,

            names="Status",
            values="Employees",

            color="Status",

            color_discrete_map={
                "Bertahan":
                    PRIMARY_COLOR,

                "Keluar":
                    ACCENT_COLOR,

                "Belum Diketahui":
                    NEUTRAL_COLOR,
            },

            title="Employee Status",

            hole=0.52,
        )

        fig.update_traces(
            textinfo="percent+label",
            textposition="outside",
            pull=[
                0.02
                if x == "Keluar"
                else 0
                for x in status_counts["Status"]
            ],
        )

        style_chart(fig)

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with c4:

        tenure_rate = attrition_rate_by(
            labeled_f,
            "TenureGroup",
        )

        bar_attrition_rate(
            tenure_rate,
            "TenureGroup",
            "Attrition Rate by Tenure Group",
            order=TENURE_LABELS,
        )

    ot_rate = attrition_rate_by(
        labeled_f,
        "OverTime",
    )

    bar_attrition_rate(
        ot_rate,
        "OverTime",
        "Attrition Rate by OverTime",
    )

    st.divider()

    st.markdown(
        '<div class="section-label">Key Takeaways</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="insight-box">
            <b>OverTime</b><br>
            Karyawan yang sering lembur berkaitan dengan attrition
            rate yang lebih tinggi.
        </div>

        <div class="insight-box">
            <b>Sales Representative</b><br>
            Sales Representative merupakan salah satu job role
            dengan attrition rate tertinggi.
        </div>

        <div class="insight-box">
            <b>Early Tenure</b><br>
            Attrition lebih tinggi pada karyawan dengan masa kerja pendek.
        </div>

        <div class="insight-box">
            <b>Sales Department</b><br>
            Sales menunjukkan attrition rate tertinggi antar department.
        </div>

        <div class="insight-box">
            <b>Work-Life Balance</b><br>
            Work-life balance rendah menunjukkan pola attrition
            yang lebih tinggi.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ATTRITION ANALYSIS
# ============================================================

elif page == "Attrition Analysis":

    page_header(
        "Employee Attrition Analysis",
        "Melihat kelompok karyawan dengan attrition rate lebih tinggi",
    )

    if labeled_f.empty:

        empty_state(
            "Tidak ada karyawan berlabel pada filter ini."
        )

        st.stop()

    overall_rate = safe_attrition_rate(
        labeled_f
    )

    st.metric(
        "Overall Attrition Rate",
        f"{overall_rate:.1f}%",
    )

    st.divider()

    tab1, tab2 = st.tabs(
        [
            "Attrition Rate",
            "Age vs Income",
        ]
    )

    with tab1:

        c1, c2 = st.columns(2)

        with c1:

            bar_attrition_rate(
                attrition_rate_by(
                    labeled_f,
                    "Department",
                ),
                "Department",
                "Attrition Rate by Department",
            )

        with c2:

            bar_attrition_rate(
                attrition_rate_by(
                    labeled_f,
                    "JobRole",
                ),
                "JobRole",
                "Attrition Rate by Job Role",
            )

        c3, c4 = st.columns(2)

        with c3:

            bar_attrition_rate(
                attrition_rate_by(
                    labeled_f,
                    "OverTime",
                ),
                "OverTime",
                "Attrition Rate by OverTime",
            )

        with c4:

            bar_attrition_rate(
                attrition_rate_by(
                    labeled_f,
                    "BusinessTravel",
                ),
                "BusinessTravel",
                "Attrition Rate by Business Travel",
            )

        c5, c6 = st.columns(2)

        with c5:

            bar_attrition_rate(
                attrition_rate_by(
                    labeled_f,
                    "TenureGroup",
                ),
                "TenureGroup",
                "Attrition Rate by Tenure Group",
                order=TENURE_LABELS,
            )

        with c6:

            bar_attrition_rate(
                attrition_rate_by(
                    labeled_f,
                    "WorkLifeBalance",
                ),
                "WorkLifeBalance",
                "Attrition Rate by Work-Life Balance",
            )

        c7, c8 = st.columns(2)

        with c7:

            bar_attrition_rate(
                attrition_rate_by(
                    labeled_f,
                    "IncomeGroup",
                ),
                "IncomeGroup",
                "Attrition Rate by Income Group",
                order=[
                    "Q1 (Terendah)",
                    "Q2",
                    "Q3",
                    "Q4 (Tertinggi)",
                ],
            )

        with c8:

            bar_attrition_rate(
                attrition_rate_by(
                    labeled_f,
                    "MaritalStatus",
                ),
                "MaritalStatus",
                "Attrition Rate by Marital Status",
            )

    with tab2:

        scatter_df = labeled_f.copy()

        scatter_df[
            "Attrition Status"
        ] = (
            scatter_df[
                "Attrition"
            ].map(
                {
                    0: "Bertahan",
                    1: "Keluar",
                }
            )
        )

        fig = px.scatter(
            scatter_df,

            x="Age",
            y="MonthlyIncome",

            color="Attrition Status",

            color_discrete_map={
                "Bertahan":
                    PRIMARY_COLOR,

                "Keluar":
                    ACCENT_COLOR,
            },

            hover_data=[
                "Department",
                "JobRole",
                "OverTime",
                "YearsAtCompany",
            ],

            labels={
                "Age": "Age",
                "MonthlyIncome":
                    "Monthly Income",
            },

            title="Age vs Monthly Income",

            opacity=0.72,
        )

        style_chart(
            fig,
            height=520,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.divider()

    st.markdown(
        '<div class="section-label">Key Insights</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="insight-box">
            OverTime menunjukkan pola attrition yang paling kuat
            di antara kelompok yang dianalisis.
        </div>

        <div class="insight-box">
            Sales Representative dan Laboratory Technician
            menunjukkan attrition rate yang tinggi.
        </div>

        <div class="insight-box">
            Attrition cenderung menurun seiring bertambahnya tenure.
        </div>

        <div class="insight-box">
            Work-Life Balance rendah menunjukkan pola attrition
            yang lebih tinggi, tetapi hubungannya tidak sepenuhnya linear.
        </div>

        <div class="insight-box">
            Semua temuan pada dashboard merupakan asosiasi,
            bukan hubungan sebab-akibat.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# EMPLOYEE PROFILE
# ============================================================

elif page == "Employee Profile":

    page_header(
        "Employee Profile",
        "Eksplorasi karakteristik workforce berdasarkan filter",
    )

    attrition_rate = safe_attrition_rate(
        labeled_f
    )

    c1, c2, c3, c4, c5, c6 = (
        st.columns(6)
    )

    c1.metric(
        "Employees",
        format_thousand(
            len(data_f)
        )
    )

    c2.metric(
        "Attrition Rate",
        (
            f"{attrition_rate:.1f}%"
            if not pd.isna(
                attrition_rate
            )
            else "-"
        )
    )

    c3.metric(
        "Average Age",
        f"{data_f['Age'].mean():.1f}",
    )

    c4.metric(
        "Average Tenure",
        f"{data_f['YearsAtCompany'].mean():.1f}",
    )

    c5.metric(
        "Average Monthly Income",
        format_thousand(
            data_f["MonthlyIncome"].mean()
        ),
    )

    c6.metric(
        "Average Satisfaction",
        f"{data_f['AvgSatisfaction'].mean():.2f} / 4",
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:

        fig = px.histogram(
            data_f,

            x="Age",

            nbins=20,

            color_discrete_sequence=[
                PRIMARY_COLOR
            ],

            title="Age Distribution",
        )

        style_chart(fig)

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with c2:

        fig = px.histogram(
            data_f,

            x="MonthlyIncome",

            nbins=20,

            color_discrete_sequence=[
                PRIMARY_COLOR
            ],

            title="Monthly Income Distribution",
        )

        style_chart(fig)

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with c3:

        fig = px.histogram(
            data_f,

            x="YearsAtCompany",

            nbins=20,

            color_discrete_sequence=[
                PRIMARY_COLOR
            ],

            title="Years at Company Distribution",
        )

        style_chart(fig)

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    c4, c5 = st.columns(2)

    with c4:

        fig = px.box(
            data_f,

            x="Department",

            y="MonthlyIncome",

            color_discrete_sequence=[
                PRIMARY_COLOR
            ],

            title="Monthly Income by Department",
        )

        style_chart(fig)

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with c5:

        sat_by_dept = (
            data_f
            .groupby(
                "Department",
                observed=True
            )["AvgSatisfaction"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            sat_by_dept.sort_values(
                "AvgSatisfaction"
            ),

            x="AvgSatisfaction",

            y="Department",

            orientation="h",

            color_discrete_sequence=[
                PRIMARY_COLOR
            ],

            title="Average Satisfaction by Department",
        )

        fig.update_layout(
            xaxis_range=[
                0,
                4
            ],
            yaxis_title=None,
        )

        style_chart(fig)

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.divider()

    st.markdown(
        '<div class="section-label">Job Role Summary</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Ringkasan karakteristik workforce berdasarkan job role."
    )

    summary_rows = []

    for role, group in data_f.groupby(
        "JobRole",
        observed=True,
    ):

        labeled_role = group[
            group["Attrition"].notna()
        ]

        summary_rows.append(
            {
                "JobRole":
                    role,

                "Employees":
                    len(group),

                "Average Age":
                    round(
                        group["Age"].mean(),
                        1,
                    ),

                "Average Tenure":
                    round(
                        group[
                            "YearsAtCompany"
                        ].mean(),
                        1,
                    ),

                "Average Monthly Income":
                    round(
                        group[
                            "MonthlyIncome"
                        ].mean(),
                        0,
                    ),

                "Attrition Rate (%)":
                    (
                        round(
                            safe_attrition_rate(
                                labeled_role
                            ),
                            1,
                        )
                        if not labeled_role.empty
                        else np.nan
                    ),
            }
        )

    role_summary_df = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            "Attrition Rate (%)",
            ascending=False,
        )
    )

    st.dataframe(
        role_summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Average Monthly Income":
                st.column_config.NumberColumn(
                    format="%,.0f"
                ),

            "Attrition Rate (%)":
                st.column_config.NumberColumn(
                    format="%.1f%%"
                ),
        },
    )


# ============================================================
# HR ACTION
# ============================================================

else:

    page_header(
        "HR Action & Recommendations",
        "Ringkasan prioritas berdasarkan hasil analisis",
    )

    col_a, col_b = st.columns(2)

    with col_a:

        st.markdown("### High Priority")

        st.markdown(
            """
            <div class="priority-high">
                <b>OverTime</b><br>
                Karyawan yang sering lembur berkaitan dengan attrition
                rate yang lebih tinggi.
            </div>

            <div class="priority-high">
                <b>Job Role Berisiko Tinggi</b><br>
                Sales Representative dan Laboratory Technician
                menunjukkan attrition rate tinggi.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:

        st.markdown("### Medium Priority")

        st.markdown(
            """
            <div class="priority-medium">
                <b>Early Tenure</b><br>
                Attrition cenderung lebih tinggi pada karyawan
                dengan masa kerja baru.
            </div>

            <div class="priority-medium">
                <b>Work-Life Balance</b><br>
                Work-life balance rendah menunjukkan pola
                attrition lebih tinggi.
            </div>

            <div class="priority-medium">
                <b>Business Travel</b><br>
                Travel yang lebih sering berkaitan dengan
                attrition yang lebih tinggi.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown(
        '<div class="section-label">Recommended Actions</div>',
        unsafe_allow_html=True,
    )

    recommendations = pd.DataFrame(
        [
            {
                "Priority":
                    "High",

                "Problem":
                    "Attrition tinggi pada karyawan yang sering lembur",

                "Finding":
                    "OverTime = Yes berkaitan dengan attrition rate lebih tinggi",

                "Recommended Action":
                    "Audit distribusi lembur, evaluasi headcount dan redistribusi workload",

                "Target":
                    "Tim dengan proporsi OverTime tinggi",
            },

            {
                "Priority":
                    "High",

                "Problem":
                    "Attrition tinggi pada Sales Representative",

                "Finding":
                    "Sales Representative memiliki attrition rate tertinggi",

                "Recommended Action":
                    "Review kompensasi, insentif, exit interview, dan career path",

                "Target":
                    "Sales Representative",
            },

            {
                "Priority":
                    "Medium",

                "Problem":
                    "Attrition tinggi pada karyawan baru",

                "Finding":
                    "Attrition lebih tinggi pada tenure pendek",

                "Recommended Action":
                    "Perkuat onboarding dan mentoring",

                "Target":
                    "Karyawan dengan tenure pendek",
            },

            {
                "Priority":
                    "Medium",

                "Problem":
                    "Work-life balance rendah",

                "Finding":
                    "WLB rendah menunjukkan pola attrition lebih tinggi",

                "Recommended Action":
                    "Evaluasi workload, business travel, dan fleksibilitas kerja",

                "Target":
                    "Karyawan dengan WLB rendah",
            },

            {
                "Priority":
                    "Medium",

                "Problem":
                    "Attrition perlu dimonitor lebih awal",

                "Finding":
                    "Model notebook dapat memberikan risk score",

                "Recommended Action":
                    "Gunakan risk score sebagai bahan diskusi HR",

                "Target":
                    "Karyawan dengan risk score tinggi",
            },
        ]
    )

    st.dataframe(
        recommendations,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.markdown(
        '<div class="section-label">Important Note</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="insight-box">

        Insight pada dashboard bersifat <b>asosiasi</b>,
        bukan hubungan sebab-akibat.

        Dashboard digunakan sebagai alat bantu analisis
        dan eksplorasi data, bukan alat otomatis untuk
        mengambil keputusan terhadap karyawan.

        Model prediktif tetap berada di notebook.

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "HR Analytics & Employee Attrition Prediction"
)
