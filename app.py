```python
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HR Analytics | Employee Attrition",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

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

PRIMARY_COLOR = "#4F6FDE"
ACCENT_COLOR = "#E76F51"
SUCCESS_COLOR = "#2A9D8F"
WARNING_COLOR = "#E9C46A"
DARK = "#111827"
MUTED = "#6B7280"
BORDER = "#E5E7EB"
BACKGROUND = "#F8FAFC"
WHITE = "#FFFFFF"


# ============================================================
# GLOBAL UI
# ============================================================

st.markdown(
    f"""
    <style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    .stApp {{
        background: {BACKGROUND};
        color: {DARK};
    }}

    .block-container {{
        max-width: 1500px;
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }}

    h1, h2, h3, h4 {{
        color: {DARK} !important;
        font-weight: 750 !important;
        letter-spacing: -0.02em;
    }}

    p {{
        color: #374151;
    }}

    hr {{
        border: none !important;
        border-top: 1px solid {BORDER} !important;
        margin: 1.5rem 0 !important;
    }}


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {{
        background: #0F172A !important;
        border-right: 1px solid #1E293B;
    }}

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {{
        color: #F8FAFC !important;
    }}

    section[data-testid="stSidebar"] hr {{
        border-top: 1px solid #334155 !important;
        margin: 1rem 0 !important;
    }}

    section[data-testid="stSidebar"] div[data-baseweb="select"] {{
        background: #FFFFFF !important;
        border-radius: 8px !important;
    }}

    section[data-testid="stSidebar"]
    div[data-baseweb="select"] * {{
        color: #111827 !important;
    }}

    section[data-testid="stSidebar"] input {{
        color: #111827 !important;
    }}

    section[data-testid="stSidebar"] button {{
        border-radius: 8px !important;
    }}


    /* =====================================================
       SIDEBAR BRAND
       ===================================================== */

    .sidebar-brand {{
        padding: 4px 0 18px 0;
    }}

    .sidebar-logo {{
        font-size: 1.65rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.03em;
    }}

    .sidebar-subtitle {{
        color: #94A3B8;
        font-size: 0.78rem;
        margin-top: 2px;
    }}

    .sidebar-section {{
        color: #94A3B8;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 18px 0 8px 0;
    }}


    /* =====================================================
       HERO
       ===================================================== */

    .hero {{
        position: relative;
        overflow: hidden;

        background:
            linear-gradient(
                135deg,
                #FFFFFF 0%,
                #F1F5FF 100%
            );

        border: 1px solid {BORDER};
        border-radius: 18px;

        padding: 28px 30px;

        margin-bottom: 22px;

        box-shadow:
            0 8px 30px rgba(15, 23, 42, 0.05);
    }}

    .hero::after {{
        content: "";
        position: absolute;

        width: 180px;
        height: 180px;

        right: -60px;
        top: -70px;

        background: rgba(79, 111, 222, 0.08);

        border-radius: 50%;
    }}

    .hero-title {{
        position: relative;
        z-index: 1;

        font-size: 2.25rem;
        font-weight: 850;

        color: {DARK};

        letter-spacing: -0.04em;
        margin-bottom: 5px;
    }}

    .hero-subtitle {{
        position: relative;
        z-index: 1;

        color: {MUTED};

        font-size: 0.95rem;
        line-height: 1.6;
    }}


    /* =====================================================
       SECTION HEADER
       ===================================================== */

    .section-title {{
        font-size: 1.15rem;
        font-weight: 750;
        color: {DARK};
        margin: 4px 0 12px 0;
    }}

    .section-description {{
        font-size: 0.86rem;
        color: {MUTED};
        margin-bottom: 14px;
    }}


    /* =====================================================
       KPI CARDS
       ===================================================== */

    div[data-testid="stMetric"] {{
        background: {WHITE} !important;

        border: 1px solid {BORDER} !important;
        border-radius: 15px !important;

        padding: 18px 18px 16px 18px !important;

        min-height: 108px;

        box-shadow:
            0 4px 14px rgba(15, 23, 42, 0.045) !important;

        transition:
            transform 0.15s ease,
            box-shadow 0.15s ease;
    }}

    div[data-testid="stMetric"]:hover {{
        transform: translateY(-2px);

        box-shadow:
            0 8px 22px rgba(15, 23, 42, 0.08) !important;
    }}

    div[data-testid="stMetricLabel"] {{
        color: {MUTED} !important;
        font-size: 0.78rem !important;
        font-weight: 650 !important;
    }}

    div[data-testid="stMetricValue"] {{
        color: {DARK} !important;
        font-size: 1.65rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em;
    }}

    div[data-testid="stMetricDelta"] {{
        color: {MUTED} !important;
    }}


    /* =====================================================
       INSIGHT CARDS
       ===================================================== */

    .insight-box {{
        background: #FFFFFF;

        border: 1px solid {BORDER};
        border-left: 4px solid {PRIMARY_COLOR};

        border-radius: 11px;

        padding: 14px 17px;

        margin-bottom: 10px;

        color: #374151;

        box-shadow:
            0 3px 10px rgba(15, 23, 42, 0.035);

        line-height: 1.55;
    }}

    .insight-box b {{
        color: {DARK};
    }}


    /* =====================================================
       PRIORITY CARDS
       ===================================================== */

    .priority-high {{
        background: #FFF7F5;

        border: 1px solid #F5D6CF;
        border-left: 4px solid {ACCENT_COLOR};

        border-radius: 11px;

        padding: 15px 17px;

        margin-bottom: 11px;

        color: #374151;

        line-height: 1.55;
    }}

    .priority-high b {{
        color: #B5472F;
    }}

    .priority-medium {{
        background: #FFFDF5;

        border: 1px solid #F0E4B4;
        border-left: 4px solid {WARNING_COLOR};

        border-radius: 11px;

        padding: 15px 17px;

        margin-bottom: 11px;

        color: #374151;

        line-height: 1.55;
    }}

    .priority-medium b {{
        color: #8A6810;
    }}


    /* =====================================================
       TABS
       ===================================================== */

    button[data-baseweb="tab"] {{
        color: {MUTED} !important;
        font-weight: 650 !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {DARK} !important;
    }}


    /* =====================================================
       DATAFRAME
       ===================================================== */

    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 12px;
        overflow: hidden;
    }}


    /* =====================================================
       CAPTION
       ===================================================== */

    div[data-testid="stCaptionContainer"] {{
        color: {MUTED} !important;
    }}


    /* =====================================================
       BUTTON
       ===================================================== */

    .stButton > button {{
        border-radius: 8px;
        border: 1px solid {BORDER};
        font-weight: 650;
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

    # Preserve missing values instead of converting them
    # into the literal string "nan".
    string_cols = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for col in string_cols:
        df[col] = (
            df[col]
            .astype("string")
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
        errors="coerce",
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
        df["Attrition"].mean() * 100
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
            observed=True,
        )["Attrition"]
        .agg(
            AttritionRate="mean",
            Count="count",
        )
        .reset_index()
    )

    result["AttritionRate"] = (
        result["AttritionRate"] * 100
    ).round(1)

    return result.sort_values(
        "AttritionRate",
        ascending=False,
    )


def format_thousand(value):

    if pd.isna(value):
        return "-"

    return f"{value:,.0f}"


def empty_state(
    message="Tidak ada data untuk filter ini."
):

    st.info(message)


def chart_layout(
    fig,
    height=None,
):

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            family="Inter, Arial, sans-serif",
            color=DARK,
            size=12,
        ),

        title=dict(
            font=dict(
                size=16,
                color=DARK,
            ),
            x=0,
            xanchor="left",
        ),

        margin=dict(
            l=10,
            r=15,
            t=55,
            b=10,
        ),

        hoverlabel=dict(
            bgcolor=WHITE,
            font_color=DARK,
        ),

        legend=dict(
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    if height is not None:
        fig.update_layout(
            height=height
        )

    return fig


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
            "Count": True,
            "AttritionRate": ":.1f",
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

    chart_layout(
        fig,
        height=max(
            280,
            42 * len(plot_df),
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E5E7EB",
        zeroline=False,
    )

    fig.update_yaxes(
        showgrid=False,
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

    chart_layout(fig)

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E5E7EB",
        zeroline=False,
    )

    fig.update_yaxes(
        showgrid=False,
        title=None,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def dynamic_insights(labeled_df):

    if labeled_df.empty:
        st.info(
            "Tidak tersedia cukup data untuk menghasilkan insight."
        )
        return

    # ------------------------------------------
    # Overtime
    # ------------------------------------------

    overtime = attrition_rate_by(
        labeled_df,
        "OverTime",
    )

    if not overtime.empty:

        highest_ot = overtime.iloc[0]

        st.markdown(
            f"""
            <div class="insight-box">
                <b>OverTime</b><br>
                Kelompok <b>{highest_ot["OverTime"]}</b>
                memiliki attrition rate sebesar
                <b>{highest_ot["AttritionRate"]:.1f}%</b>
                dari {int(highest_ot["Count"]):,} karyawan berlabel.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------------------
    # Job Role
    # ------------------------------------------

    roles = attrition_rate_by(
        labeled_df,
        "JobRole",
    )

    if not roles.empty:

        top_role = roles.iloc[0]

        st.markdown(
            f"""
            <div class="insight-box">
                <b>Highest Attrition Job Role</b><br>
                <b>{top_role["JobRole"]}</b>
                memiliki attrition rate tertinggi,
                yaitu <b>{top_role["AttritionRate"]:.1f}%</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------------------
    # Department
    # ------------------------------------------

    departments = attrition_rate_by(
        labeled_df,
        "Department",
    )

    if not departments.empty:

        top_department = departments.iloc[0]

        st.markdown(
            f"""
            <div class="insight-box">
                <b>Highest Attrition Department</b><br>
                <b>{top_department["Department"]}</b>
                memiliki attrition rate sebesar
                <b>{top_department["AttritionRate"]:.1f}%</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------------------
    # Tenure
    # ------------------------------------------

    tenure = attrition_rate_by(
        labeled_df,
        "TenureGroup",
    )

    if not tenure.empty:

        tenure = tenure.dropna(
            subset=["TenureGroup"]
        )

        if not tenure.empty:

            highest_tenure = tenure.iloc[0]

            st.markdown(
                f"""
                <div class="insight-box">
                    <b>Tenure Pattern</b><br>
                    Kelompok <b>{highest_tenure["TenureGroup"]}</b>
                    menunjukkan attrition rate sebesar
                    <b>{highest_tenure["AttritionRate"]:.1f}%</b>.
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="insight-box">
            <b>Interpretation Note</b><br>
            Seluruh temuan pada dashboard merupakan
            <b>asosiasi statistik</b>, bukan hubungan sebab-akibat.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div class="sidebar-brand">

        <div class="sidebar-logo">
            👥 HR Analytics
        </div>

        <div class="sidebar-subtitle">
            Employee Attrition Intelligence
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div class="sidebar-section">Navigation</div>',
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
    '<div class="sidebar-section">Filters</div>',
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

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                HR Analytics Dashboard
            </div>

            <div class="hero-subtitle">
                Workforce overview, employee attrition patterns,
                and HR priorities.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    total_employees = len(data_f)

    n_labeled = len(labeled_f)

    n_unlabeled = (
        total_employees - n_labeled
    )

    attrition_rate = safe_attrition_rate(
        labeled_f
    )

    # ========================================================
    # KPI
    # ========================================================

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
        f"Labeled Employees: {format_thousand(n_labeled)}"
        f"  •  "
        f"Unlabeled Employees: {format_thousand(n_unlabeled)}"
        f"  •  "
        "Attrition Rate dihitung dari data berlabel."
    )

    st.divider()

    # ========================================================
    # WORKFORCE DISTRIBUTION
    # ========================================================

    st.markdown(
        '<div class="section-title">Workforce Distribution</div>',
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

    # ========================================================
    # STATUS + TENURE
    # ========================================================

    c3, c4 = st.columns(2)

    with c3:

        status_counts = (
            data_f["StatusLabel"]
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
                "Bertahan": PRIMARY_COLOR,
                "Keluar": ACCENT_COLOR,
                "Belum Diketahui": NEUTRAL_COLOR,
            },

            title="Employee Status",
            hole=0.55,
        )

        fig.update_traces(
            textinfo="percent+label",
            textposition="outside",
        )

        chart_layout(fig)

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
            "Attrition Rate by Tenure",
            order=TENURE_LABELS,
        )

    # ========================================================
    # OVERTIME
    # ========================================================

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

    # ========================================================
    # KEY TAKEAWAYS
    # ========================================================

    st.markdown(
        '<div class="section-title">Key Takeaways</div>',
        unsafe_allow_html=True,
    )

    dynamic_insights(labeled_f)


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
                Identify employee groups with higher attrition rates
                and explore workforce risk patterns.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if labeled_f.empty:

        empty_state(
            "Tidak ada karyawan berlabel pada filter ini."
        )

        st.stop()

    overall_rate = safe_attrition_rate(
        labeled_f
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Overall Attrition",
        f"{overall_rate:.1f}%",
    )

    col2.metric(
        "Employees Analyzed",
        format_thousand(
            len(labeled_f)
        ),
    )

    col3.metric(
        "Employees Left",
        format_thousand(
            int(labeled_f["Attrition"].sum())
        ),
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
                "Attrition Rate by Tenure",
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
            scatter_df["Attrition"]
            .map(
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
                "Bertahan": PRIMARY_COLOR,
                "Keluar": ACCENT_COLOR,
            },

            hover_data=[
                "Department",
                "JobRole",
                "OverTime",
                "YearsAtCompany",
            ],

            labels={
                "Age": "Age",
                "MonthlyIncome": "Monthly Income",
            },

            title="Age vs Monthly Income",

            opacity=0.72,
        )

        chart_layout(
            fig,
            height=540,
        )

        fig.update_xaxes(
            showgrid=True,
            gridcolor="#E5E7EB",
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor="#E5E7EB",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.divider()

    st.markdown(
        '<div class="section-title">Key Insights</div>',
        unsafe_allow_html=True,
    )

    dynamic_insights(labeled_f)


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
                Explore workforce characteristics based on
                the selected filters.
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
        ),
    )

    c2.metric(
        "Attrition Rate",
        (
            f"{attrition_rate:.1f}%"
            if not pd.isna(attrition_rate)
            else "-"
        ),
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

    # ========================================================
    # DISTRIBUTIONS
    # ========================================================

    st.markdown(
        '<div class="section-title">Workforce Distribution</div>',
        unsafe_allow_html=True,
    )

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

        chart_layout(fig)

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

        chart_layout(fig)

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

        chart_layout(fig)

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ========================================================
    # PROFILE ANALYSIS
    # ========================================================

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

        chart_layout(fig)

        fig.update_xaxes(
            showgrid=False
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor="#E5E7EB",
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
                observed=True,
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

        chart_layout(fig)

        fig.update_layout(
            xaxis_range=[
                0,
                4,
            ]
        )

        fig.update_xaxes(
            showgrid=True,
            gridcolor="#E5E7EB",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.divider()

    # ========================================================
    # JOB ROLE SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">Job Role Summary</div>',
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
                "JobRole": role,

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
        pd.DataFrame(summary_rows)
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

            "Employees":
                st.column_config.NumberColumn(
                    format="%,.0f"
                ),

            "Average Age":
                st.column_config.NumberColumn(
                    format="%.1f"
                ),

            "Average Tenure":
                st.column_config.NumberColumn(
                    format="%.1f"
                ),

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
                Translate analytical findings into practical
                HR actions and monitoring priorities.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)

    # ========================================================
    # HIGH PRIORITY
    # ========================================================

    with col_a:

        st.markdown(
            "### 🔴 High Priority"
        )

        overtime = attrition_rate_by(
            labeled_f,
            "OverTime",
        )

        if not overtime.empty:

            top_ot = overtime.iloc[0]

            st.markdown(
                f"""
                <div class="priority-high">
                    <b>OverTime</b><br>
                    Kelompok <b>{top_ot["OverTime"]}</b>
                    memiliki attrition rate
                    <b>{top_ot["AttritionRate"]:.1f}%</b>.
                    Evaluasi workload dan distribusi lembur.
                </div>
                """,
                unsafe_allow_html=True,
            )

        roles = attrition_rate_by(
            labeled_f,
            "JobRole",
        )

        if not roles.empty:

            top_role = roles.iloc[0]

            st.markdown(
                f"""
                <div class="priority-high">
                    <b>Highest-Risk Job Role</b><br>
                    <b>{top_role["JobRole"]}</b>
                    memiliki attrition rate tertinggi
                    sebesar <b>{top_role["AttritionRate"]:.1f}%</b>.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ========================================================
    # MEDIUM PRIORITY
    # ========================================================

    with col_b:

        st.markdown(
            "### 🟡 Medium Priority"
        )

        tenure = attrition_rate_by(
            labeled_f,
            "TenureGroup",
        )

        if not tenure.empty:

            top_tenure = tenure.iloc[0]

            st.markdown(
                f"""
                <div class="priority-medium">
                    <b>Tenure</b><br>
                    Kelompok <b>{top_tenure["TenureGroup"]}</b>
                    menunjukkan attrition rate
                    <b>{top_tenure["AttritionRate"]:.1f}%</b>.
                    Perkuat onboarding dan mentoring.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="priority-medium">
                <b>Work-Life Balance</b><br>
                Monitor kelompok dengan work-life balance
                rendah dan evaluasi workload serta fleksibilitas kerja.
            </div>

            <div class="priority-medium">
                <b>Business Travel</b><br>
                Evaluasi pola perjalanan kerja dan dampaknya
                terhadap employee experience.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ========================================================
    # RECOMMENDED ACTIONS
    # ========================================================

    st.markdown(
        '<div class="section-title">Recommended Actions</div>',
        unsafe_allow_html=True,
    )

    recommendations = pd.DataFrame(
        [
            {
                "Priority":
                    "High",

                "Problem":
                    "Attrition pada karyawan dengan OverTime tinggi",

                "Finding":
                    "OverTime berkaitan dengan attrition rate yang lebih tinggi",

                "Recommended Action":
                    "Audit workload, distribusi lembur, dan kebutuhan headcount",

                "Target":
                    "Tim dengan proporsi OverTime tinggi",
            },

            {
                "Priority":
                    "High",

                "Problem":
                    "Job role dengan attrition tinggi",

                "Finding":
                    "Beberapa job role menunjukkan attrition rate relatif tinggi",

                "Recommended Action":
                    "Review compensation, career path, workload, dan exit interview",

                "Target":
                    "Job role dengan attrition rate tinggi",
            },

            {
                "Priority":
                    "Medium",

                "Problem":
                    "Attrition pada early tenure",

                "Finding":
                    "Kelompok dengan tenure pendek menunjukkan attrition relatif tinggi",

                "Recommended Action":
                    "Perkuat onboarding, mentoring, dan early employee engagement",

                "Target":
                    "Karyawan dengan tenure pendek",
            },

            {
                "Priority":
                    "Medium",

                "Problem":
                    "Work-life balance",

                "Finding":
                    "WLB rendah menunjukkan pola attrition lebih tinggi",

                "Recommended Action":
                    "Evaluasi workload, fleksibilitas, dan business travel",

                "Target":
                    "Karyawan dengan WLB rendah",
            },

            {
                "Priority":
                    "Medium",

                "Problem":
                    "Early risk monitoring",

                "Finding":
                    "Model prediktif dapat dikembangkan untuk employee risk scoring",

                "Recommended Action":
                    "Gunakan risk score sebagai bahan monitoring dan diskusi HR",

                "Target":
                    "Karyawan dengan risk score tinggi",
            },
        ]
    )

    st.dataframe(
        recommendations,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Priority":
                st.column_config.TextColumn(
                    "Priority"
                ),

            "Problem":
                st.column_config.TextColumn(
                    "Problem",
                    width="medium",
                ),

            "Finding":
                st.column_config.TextColumn(
                    "Finding",
                    width="large",
                ),

            "Recommended Action":
                st.column_config.TextColumn(
                    "Recommended Action",
                    width="large",
                ),

            "Target":
                st.column_config.TextColumn(
                    "Target",
                    width="medium",
                ),
        },
    )

    st.divider()

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.markdown(
        '<div class="section-title">Important Note</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="insight-box">

        Insight pada dashboard bersifat
        <b>asosiasi</b>, bukan hubungan sebab-akibat.

        Dashboard digunakan sebagai alat bantu analisis
        dan eksplorasi data, bukan alat otomatis untuk
        mengambil keputusan terhadap karyawan.

        Model prediktif berada pada tahap notebook
        dan dapat diintegrasikan sebagai tahap lanjutan.

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "HR Analytics & Employee Attrition Prediction • "
    "Built with Python, Pandas, Plotly & Streamlit"
)
```
