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

PRIMARY_COLOR = "#4C72B0"
ACCENT_COLOR = "#DD8452"
NEUTRAL_COLOR = "#8C8C8C"


# ============================================================
# PROFESSIONAL UI
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background: #F8FAFC;
        color: #0F172A;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }

    h1, h2, h3, h4 {
        color: #0F172A !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    p {
        color: #475569;
    }

    hr {
        border: none;
        border-top: 1px solid #E2E8F0;
        margin: 1.5rem 0;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: #0F172A !important;
        border-right: 1px solid #1E293B;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    section[data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }

    section[data-testid="stSidebar"] label {
        color: #CBD5E1 !important;
        font-weight: 500 !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] {
        background: #FFFFFF !important;
        border-radius: 8px !important;
    }

    section[data-testid="stSidebar"]
    div[data-baseweb="select"] * {
        color: #0F172A !important;
    }

    section[data-testid="stSidebar"] input {
        color: #0F172A !important;
    }

    section[data-testid="stSidebar"]
    div[data-testid="stRadio"] label {
        color: #CBD5E1 !important;
        font-weight: 500 !important;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        background: #FFFFFF;
        padding: 28px 32px;
        border-radius: 14px;
        border: 1px solid #E2E8F0;
        margin-bottom: 24px;

        box-shadow:
            0 1px 3px rgba(15, 23, 42, 0.04);
    }

    .hero-title {
        font-size: 2rem;
        line-height: 1.2;
        font-weight: 800;
        color: #0F172A !important;
        margin-bottom: 7px;
        letter-spacing: -0.03em;
    }

    .hero-subtitle {
        font-size: 0.95rem;
        line-height: 1.5;
        color: #64748B !important;
    }


    /* ========================================================
       KPI CARDS
       ======================================================== */

    div[data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;

        padding: 18px 20px !important;

        box-shadow:
            0 1px 3px rgba(15, 23, 42, 0.04) !important;

        min-height: 112px;

        transition:
            transform 0.15s ease,
            box-shadow 0.15s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);

        box-shadow:
            0 5px 15px rgba(15, 23, 42, 0.07) !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-size: 1.55rem !important;
        font-weight: 800 !important;
    }

    div[data-testid="stMetricDelta"] {
        font-size: 0.78rem !important;
    }


    /* ========================================================
       CHART CONTAINERS
       ======================================================== */

    div[data-testid="stPlotlyChart"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 8px 8px 2px 8px;

        box-shadow:
            0 1px 3px rgba(15, 23, 42, 0.035);
    }


    /* ========================================================
       INSIGHT CARDS
       ======================================================== */

    .insight-box {
        background: #FFFFFF !important;
        color: #475569 !important;

        border: 1px solid #E2E8F0;
        border-left: 4px solid #4C72B0;

        padding: 15px 18px;
        border-radius: 9px;

        margin-bottom: 10px;

        line-height: 1.55;

        box-shadow:
            0 1px 3px rgba(15, 23, 42, 0.035);
    }

    .insight-box b {
        color: #0F172A !important;
        font-weight: 700;
    }


    /* ========================================================
       PRIORITY CARDS
       ======================================================== */

    .priority-high {
        background: #FFF7F2 !important;
        color: #475569 !important;

        border: 1px solid #F3D6C7;
        border-left: 4px solid #DD8452;

        padding: 15px 18px;
        border-radius: 9px;

        margin-bottom: 10px;

        line-height: 1.55;
    }

    .priority-high b {
        color: #9A3412 !important;
    }

    .priority-medium {
        background: #FFFDF2 !important;
        color: #475569 !important;

        border: 1px solid #EFE4B8;
        border-left: 4px solid #D9B93F;

        padding: 15px 18px;
        border-radius: 9px;

        margin-bottom: 10px;

        line-height: 1.55;
    }

    .priority-medium b {
        color: #854D0E !important;
    }


    /* ========================================================
       TABS
       ======================================================== */

    button[data-baseweb="tab"] {
        color: #64748B !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0F172A !important;
    }


    /* ========================================================
       DATAFRAME
       ======================================================== */

    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        overflow: hidden;

        box-shadow:
            0 1px 3px rgba(15, 23, 42, 0.035);
    }


    /* ========================================================
       CAPTION
       ======================================================== */

    div[data-testid="stCaptionContainer"] {
        color: #64748B !important;
        font-size: 0.8rem !important;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        border-radius: 8px;
        border: 1px solid #CBD5E1;
        background: #FFFFFF;
        color: #0F172A;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: #94A3B8;
        color: #0F172A;
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 9px;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }

        .hero {
            padding: 22px;
        }

        .hero-title {
            font-size: 1.6rem;
        }

    }

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
                category_col,
        },
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        margin=dict(
            l=10,
            r=10,
            t=55,
            b=10,
        ),
        xaxis_title="Attrition Rate (%)",
        yaxis_title=None,
        height=max(
            280,
            40 * len(plot_df),
        ),
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

    fig = px.bar(
        result.sort_values(
            "Employees"
        ),
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
        textposition="outside"
    )

    fig.update_layout(
        margin=dict(
            l=10,
            r=10,
            t=55,
            b=10,
        ),
        yaxis_title=None,
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
    <div style="
        font-size: 1.4rem;
        font-weight: 800;
        margin-bottom: 4px;
    ">
        👥 HR Analytics
    </div>

    <div style="
        color: #9CA3AF;
        font-size: 0.85rem;
        margin-bottom: 16px;
    ">
        Employee Attrition Dashboard
    </div>
    """,
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
)

st.sidebar.divider()

st.sidebar.markdown(
    "### Filters"
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
        "Tidak ada karyawan yang "
        "cocok dengan filter."
    )

    st.stop()


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                HR Analytics Dashboard
            </div>

            <div class="hero-subtitle">
                Workforce overview, employee attrition patterns,
                and HR priorities
            </div>

        </div>
        """,
        unsafe_allow_html=True,
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
            if not pd.isna(attrition_rate)
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
        f"  |  "
        f"Unlabeled Employees: "
        f"{format_thousand(n_unlabeled)}"
        f"  |  "
        "Attrition Rate dihitung dari data berlabel."
    )

    st.divider()

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
            hole=0.45,
        )

        fig.update_traces(
            textinfo="percent+label"
        )

        fig.update_layout(
            margin=dict(
                l=10,
                r=10,
                t=55,
                b=10,
            )
        )

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

    st.subheader(
        "Key Takeaways"
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

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                Employee Attrition Analysis
            </div>

            <div class="hero-subtitle">
                Melihat kelompok karyawan dengan attrition rate lebih tinggi
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if labeled_f.empty:

        empty_state(
            "Tidak ada karyawan berlabel "
            "pada filter ini."
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

        scatter_df["Attrition Status"] = (
            scatter_df["Attrition"].map(
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

        fig.update_layout(
            height=520,
            margin=dict(
                l=10,
                r=10,
                t=55,
                b=10,
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.divider()

    st.subheader(
        "Key Insights"
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

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                Employee Profile
            </div>

            <div class="hero-subtitle">
                Eksplorasi karakteristik workforce berdasarkan filter
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    attrition_rate = safe_attrition_rate(
        labeled_f
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)

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
            if not pd.isna(attrition_rate)
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

        fig.update_layout(
            margin=dict(
                l=10,
                r=10,
                t=55,
                b=10,
            )
        )

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

        fig.update_layout(
            margin=dict(
                l=10,
                r=10,
                t=55,
                b=10,
            )
        )

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

        fig.update_layout(
            margin=dict(
                l=10,
                r=10,
                t=55,
                b=10,
            )
        )

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

        fig.update_layout(
            margin=dict(
                l=10,
                r=10,
                t=55,
                b=10,
            )
        )

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
            margin=dict(
                l=10,
                r=10,
                t=55,
                b=10,
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.divider()

    st.subheader(
        "Job Role Summary"
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

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                HR Action & Recommendations
            </div>

            <div class="hero-subtitle">
                Ringkasan prioritas berdasarkan hasil analisis
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)

    with col_a:

        st.markdown(
            "### High Priority"
        )

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

        st.markdown(
            "### Medium Priority"
        )

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

    st.subheader(
        "Recommended Actions"
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

    st.subheader(
        "Important Note"
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
