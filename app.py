import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/refs/heads/main/employee/employee_data.csv"

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


@st.cache_data(show_spinner="Mengambil dan membersihkan data...")
def load_data():
    df = pd.read_csv(DATA_URL)

    string_cols = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()

    df = df.drop_duplicates()

    drop_cols = [
        c
        for c in [
            "EmployeeCount",
            "StandardHours",
            "Over18",
        ]
        if c in df.columns
    ]

    df = df.drop(columns=drop_cols)

    df["Attrition"] = pd.to_numeric(
        df["Attrition"],
        errors="coerce"
    )

    df["TenureGroup"] = pd.cut(
        df["YearsAtCompany"],
        bins=TENURE_BINS,
        labels=TENURE_LABELS
    )

    df["IncomeGroup"] = pd.qcut(
        df["MonthlyIncome"],
        q=4,
        labels=[
            "Q1 (Terendah)",
            "Q2",
            "Q3",
            "Q4 (Tertinggi)"
        ],
        duplicates="drop"
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
            df["Attrition"] == 1
        ],
        [
            "Bertahan",
            "Keluar"
        ],
        default="Belum Diketahui",
    )

    return df


def get_labeled(df):
    return df[df["Attrition"].notna()].copy()


def safe_attrition_rate(df):
    if df.empty:
        return np.nan

    return df["Attrition"].mean() * 100


def attrition_rate_by(df_labeled, col):
    if df_labeled.empty or col not in df_labeled.columns:
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
        result["AttritionRate"] * 100
    ).round(1)

    return result.sort_values(
        "AttritionRate",
        ascending=False
    )


def format_thousand(value):
    if pd.isna(value):
        return "-"

    return f"{value:,.0f}"


def empty_state(message="Tidak ada data untuk kombinasi filter ini."):
    st.info(message)


def bar_attrition_rate(
    rate_df,
    category_col,
    title,
    order=None
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
            ascending=True
        )

    fig = px.bar(
        plot_df,
        x="AttritionRate",
        y=category_col,
        orientation="h",
        text=plot_df["AttritionRate"].astype(str) + "%",
        color_discrete_sequence=[ACCENT_COLOR],
        labels={
            "AttritionRate": "Attrition Rate (%)",
            category_col: category_col,
        },
        title=title,
        hover_data={"Count": True},
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        margin=dict(
            l=10,
            r=10,
            t=50,
            b=10
        ),
        xaxis_title="Attrition Rate (%)",
        yaxis_title=None,
        height=max(
            280,
            40 * len(plot_df)
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2rem;
        }

        div[data-testid="stMetric"] {
            background-color: #F8F9FB;
            border: 1px solid #E7E9EE;
            border-radius: 10px;
            padding: 14px 16px 8px 16px;
        }

        div[data-testid="stMetricLabel"] {
            font-weight: 500;
            color: #555;
        }

        h1, h2, h3 {
            color: #1F2A44;
        }

        .insight-box {
            background-color: #F4F6FA;
            border-left: 4px solid #4C72B0;
            padding: 10px 16px;
            border-radius: 6px;
            margin-bottom: 10px;
        }

        .priority-high {
            border-left: 4px solid #DD8452;
            padding: 8px 14px;
            background-color: #FFF3EC;
            border-radius: 6px;
            margin-bottom: 8px;
        }

        .priority-medium {
            border-left: 4px solid #E8C547;
            padding: 8px 14px;
            background-color: #FFFBEA;
            border-radius: 6px;
            margin-bottom: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    data = load_data()

except Exception as e:
    st.error(
        "Gagal mengambil dataset dari sumber online. "
        "Silakan cek koneksi internet atau coba lagi."
    )
    st.exception(e)
    st.stop()


st.sidebar.title("HR Analytics")

page = st.sidebar.radio(
    "Navigasi",
    [
        "Overview",
        "Attrition Analysis",
        "Employee Profile",
        "HR Action",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filter")

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


if page == "Overview":

    st.title("HR Analytics Dashboard")
    st.caption(
        "Workforce overview, employee attrition patterns, and HR priorities"
    )

    total_employees = len(data_f)
    n_labeled = len(labeled_f)
    n_unlabeled = total_employees - n_labeled
    attrition_rate = safe_attrition_rate(labeled_f)

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Employees",
        format_thousand(total_employees)
    )

    col2.metric(
        "Attrition Rate",
        f"{attrition_rate:.1f}%"
        if not pd.isna(attrition_rate)
        else "-"
    )

    col3.metric(
        "Average Age",
        f"{data_f['Age'].mean():.1f} th"
    )

    col4.metric(
        "Average Years at Company",
        f"{data_f['YearsAtCompany'].mean():.1f} th"
    )

    col5.metric(
        "Average Monthly Income",
        format_thousand(
            data_f["MonthlyIncome"].mean()
        )
    )

    st.caption(
        f"Labeled Employees: {format_thousand(n_labeled)} | "
        f"Unlabeled Employees: {format_thousand(n_unlabeled)}"
    )

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        dept_counts = (
            data_f["Department"]
            .value_counts()
            .reset_index()
        )

        dept_counts.columns = [
            "Department",
            "Employees"
        ]

        fig = px.bar(
            dept_counts.sort_values("Employees"),
            x="Employees",
            y="Department",
            orientation="h",
            color_discrete_sequence=[PRIMARY_COLOR],
            title="Employees by Department",
        )

        fig.update_layout(
            margin=dict(
                l=10,
                r=10,
                t=50,
                b=10
            ),
            yaxis_title=None
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:
        role_counts = (
            data_f["JobRole"]
            .value_counts()
            .reset_index()
        )

        role_counts.columns = [
            "JobRole",
            "Employees"
        ]

        fig = px.bar(
            role_counts.sort_values("Employees"),
            x="Employees",
            y="JobRole",
            orientation="h",
            color_discrete_sequence=[PRIMARY_COLOR],
            title="Employees by Job Role",
        )

        fig.update_layout(
            margin=dict(
                l=10,
                r=10,
                t=50,
                b=10
            ),
            yaxis_title=None,
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    c3, c4 = st.columns(2)

    with c3:
        status_counts = (
            data_f["StatusLabel"]
            .value_counts()
            .reset_index()
        )

        status_counts.columns = [
            "Status",
            "Employees"
        ]

        color_map = {
            "Bertahan": PRIMARY_COLOR,
            "Keluar": ACCENT_COLOR,
            "Belum Diketahui": NEUTRAL_COLOR,
        }

        fig = px.pie(
            status_counts,
            names="Status",
            values="Employees",
            color="Status",
            color_discrete_map=color_map,
            title="Employee Status",
            hole=0.4,
        )

        fig.update_traces(
            textinfo="percent+label"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c4:
        tenure_rate = attrition_rate_by(
            labeled_f,
            "TenureGroup"
        )

        bar_attrition_rate(
            tenure_rate,
            "TenureGroup",
            "Attrition Rate by Tenure Group",
            order=TENURE_LABELS
        )

    ot_rate = attrition_rate_by(
        labeled_f,
        "OverTime"
    )

    bar_attrition_rate(
        ot_rate,
        "OverTime",
        "Attrition Rate by OverTime"
    )

    st.markdown("---")
    st.subheader("Key Takeaways")

    st.markdown(
        """
        <div class="insight-box">
        OverTime adalah sinyal attrition paling kuat di dataset.
        </div>

        <div class="insight-box">
        Sales Representative memiliki attrition rate tertinggi.
        </div>

        <div class="insight-box">
        Tahun pertama bekerja merupakan periode yang paling rawan.
        </div>

        <div class="insight-box">
        Sales memiliki attrition rate tertinggi antar department.
        </div>

        <div class="insight-box">
        Work-life balance rendah berkaitan dengan attrition yang lebih tinggi.
        </div>
        """,
        unsafe_allow_html=True,
    )


elif page == "Attrition Analysis":

    st.title("Employee Attrition Analysis")
    st.caption(
        "Kelompok karyawan mana yang memiliki attrition rate lebih tinggi?"
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
        f"{overall_rate:.1f}%"
    )

    tab1, tab2 = st.tabs(
        [
            "Attrition Rate",
            "Age vs Income"
        ]
    )

    with tab1:

        c1, c2 = st.columns(2)

        with c1:
            bar_attrition_rate(
                attrition_rate_by(
                    labeled_f,
                    "Department"
                ),
                "Department",
                "Attrition Rate by Department"
            )

        with c2:
            bar_attrition_rate(
                attrition_rate_by(
                    labeled_f,
                    "JobRole"
                ),
                "JobRole",
                "Attrition Rate by Job Role"
            )

        c3, c4 = st.columns(2)

        with c3:
            bar_attrition_rate(
                attrition_rate_by(
                    labeled_f,
                    "OverTime"
                ),
                "OverTime",
                "Attrition Rate by OverTime"
            )

        with c4:
            bar_attrition_rate(
                attrition_rate_by(
                    labeled_f,
                    "BusinessTravel"
                ),
                "BusinessTravel",
                "Attrition Rate by Business Travel"
            )

        c5, c6 = st.columns(2)

        with c5:
            bar_attrition_rate(
                attrition_rate_by(
                    labeled_f,
                    "TenureGroup"
                ),
                "TenureGroup",
                "Attrition Rate by Tenure Group",
                order=TENURE_LABELS
            )

        with c6:
            bar_attrition_rate(
                attrition_rate_by(
                    labeled_f,
                    "WorkLifeBalance"
                ),
                "WorkLifeBalance",
                "Attrition Rate by Work-Life Balance"
            )

        c7, c8 = st.columns(2)

        with c7:
            bar_attrition_rate(
                attrition_rate_by(
                    labeled_f,
                    "IncomeGroup"
                ),
                "IncomeGroup",
                "Attrition Rate by Income Group",
                order=[
                    "Q1 (Terendah)",
                    "Q2",
                    "Q3",
                    "Q4 (Tertinggi)"
                ]
            )

        with c8:
            bar_attrition_rate(
                attrition_rate_by(
                    labeled_f,
                    "MaritalStatus"
                ),
                "MaritalStatus",
                "Attrition Rate by Marital Status"
            )

    with tab2:

        scatter_df = labeled_f.copy()

        scatter_df["Attrition Status"] = (
            scatter_df["Attrition"]
            .map({
                0: "Bertahan",
                1: "Keluar"
            })
        )

        fig = px.scatter(
            scatter_df,
            x="Age",
            y="MonthlyIncome",
            color="Attrition Status",
            color_discrete_map={
                "Bertahan": PRIMARY_COLOR,
                "Keluar": ACCENT_COLOR
            },
            hover_data=[
                "Department",
                "JobRole",
                "OverTime",
                "YearsAtCompany"
            ],
            labels={
                "Age": "Usia",
                "MonthlyIncome": "Monthly Income"
            },
            title="Age vs Monthly Income",
            opacity=0.75,
        )

        fig.update_layout(
            height=520,
            margin=dict(
                l=10,
                r=10,
                t=50,
                b=10
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("Insight")

    st.markdown(
        """
        <div class="insight-box">
        Karyawan yang sering lembur memiliki attrition rate yang lebih tinggi.
        </div>

        <div class="insight-box">
        Sales Representative dan Laboratory Technician menunjukkan attrition yang tinggi.
        </div>

        <div class="insight-box">
        Attrition menurun seiring bertambahnya tenure.
        </div>

        <div class="insight-box">
        Work-Life Balance rendah menunjukkan pola attrition yang lebih tinggi.
        </div>

        <div class="insight-box">
        Semua temuan menunjukkan asosiasi, bukan hubungan sebab-akibat.
        </div>
        """,
        unsafe_allow_html=True,
    )


elif page == "Employee Profile":

    st.title("Employee Profile")
    st.caption(
        "Eksplorasi karakteristik workforce berdasarkan filter."
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
        f"{attrition_rate:.1f}%"
        if not pd.isna(attrition_rate)
        else "-"
    )

    c3.metric(
        "Average Age",
        f"{data_f['Age'].mean():.1f}"
    )

    c4.metric(
        "Average Tenure",
        f"{data_f['YearsAtCompany'].mean():.1f}"
    )

    c5.metric(
        "Average Monthly Income",
        format_thousand(
            data_f["MonthlyIncome"].mean()
        )
    )

    c6.metric(
        "Average Satisfaction",
        f"{data_f['AvgSatisfaction'].mean():.2f} / 4"
    )

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        fig = px.histogram(
            data_f,
            x="Age",
            nbins=20,
            color_discrete_sequence=[
                PRIMARY_COLOR
            ],
            title="Age Distribution"
        )

        fig.update_layout(
            margin=dict(
                l=10,
                r=10,
                t=50,
                b=10
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:
        fig = px.histogram(
            data_f,
            x="MonthlyIncome",
            nbins=20,
            color_discrete_sequence=[
                PRIMARY_COLOR
            ],
            title="Monthly Income Distribution"
        )

        fig.update_layout(
            margin=dict(
                l=10,
                r=10,
                t=50,
                b=10
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c3:
        fig = px.histogram(
            data_f,
            x="YearsAtCompany",
            nbins=20,
            color_discrete_sequence=[
                PRIMARY_COLOR
            ],
            title="Years at Company Distribution"
        )

        fig.update_layout(
            margin=dict(
                l=10,
                r=10,
                t=50,
                b=10
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
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
            title="Monthly Income by Department"
        )

        fig.update_layout(
            margin=dict(
                l=10,
                r=10,
                t=50,
                b=10
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
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
            title="Average Satisfaction by Department"
        )

        fig.update_layout(
            xaxis_range=[0, 4],
            yaxis_title=None,
            margin=dict(
                l=10,
                r=10,
                t=50,
                b=10
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("Job Role Summary")

    summary_rows = []

    for role, group in data_f.groupby(
        "JobRole",
        observed=True
    ):
        labeled_role = group[
            group["Attrition"].notna()
        ]

        summary_rows.append(
            {
                "JobRole": role,
                "Employees": len(group),
                "Average Age": round(
                    group["Age"].mean(),
                    1
                ),
                "Average Tenure": round(
                    group["YearsAtCompany"].mean(),
                    1
                ),
                "Average Monthly Income": round(
                    group["MonthlyIncome"].mean(),
                    0
                ),
                "Attrition Rate (%)":
                    round(
                        safe_attrition_rate(
                            labeled_role
                        ),
                        1
                    )
                    if not labeled_role.empty
                    else np.nan,
            }
        )

    role_summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(
            "Attrition Rate (%)",
            ascending=False
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
        }
    )


else:

    st.title("HR Action & Recommendations")
    st.caption(
        "Ringkasan prioritas dan rekomendasi berdasarkan hasil analisis."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### High Priority")

        st.markdown(
            """
            <div class="priority-high">
            <b>OverTime</b><br>
            Karyawan yang sering lembur berkaitan dengan attrition rate yang lebih tinggi.
            </div>

            <div class="priority-high">
            <b>Job Role berisiko tinggi</b><br>
            Sales Representative dan Laboratory Technician memiliki attrition rate tinggi.
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
            Attrition rate lebih tinggi pada karyawan dengan masa kerja baru.
            </div>

            <div class="priority-medium">
            <b>Work-Life Balance</b><br>
            Work-life balance rendah menunjukkan pola attrition lebih tinggi.
            </div>

            <div class="priority-medium">
            <b>Business Travel</b><br>
            Perjalanan dinas yang lebih sering berkaitan dengan attrition yang lebih tinggi.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Recommended Actions")

    recommendations = pd.DataFrame(
        [
            {
                "Priority": "High",
                "Problem": "Attrition tinggi pada karyawan yang sering lembur",
                "Finding": "OverTime = Yes berkaitan dengan attrition rate lebih tinggi",
                "Recommended Action": "Audit distribusi jam lembur, evaluasi headcount dan redistribusi beban kerja",
                "Target": "Tim dengan proporsi OverTime tinggi",
            },
            {
                "Priority": "High",
                "Problem": "Attrition tinggi pada Sales Representative",
                "Finding": "Sales Representative memiliki attrition rate tertinggi",
                "Recommended Action": "Exit interview, review kompensasi, insentif, dan career path",
                "Target": "Sales Representative",
            },
            {
                "Priority": "Medium",
                "Problem": "Attrition tinggi pada karyawan baru",
                "Finding": "Attrition lebih tinggi pada tenure pendek",
                "Recommended Action": "Perkuat onboarding dan mentoring",
                "Target": "Karyawan tenure kurang dari 1 tahun",
            },
            {
                "Priority": "Medium",
                "Problem": "Work-life balance rendah",
                "Finding": "WLB rendah menunjukkan pola attrition lebih tinggi",
                "Recommended Action": "Evaluasi workload, business travel, dan opsi kerja fleksibel",
                "Target": "Karyawan dengan WLB rendah",
            },
            {
                "Priority": "Medium",
                "Problem": "Attrition perlu dimonitor lebih awal",
                "Finding": "Model di notebook dapat memberi risk score",
                "Recommended Action": "Gunakan risk score sebagai bahan diskusi HR",
                "Target": "Karyawan dengan risk score tinggi",
            },
        ]
    )

    st.dataframe(
        recommendations,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.subheader("Important Note")

    st.markdown(
        """
        <div class="insight-box">
        Insight pada dashboard bersifat asosiasi, bukan hubungan sebab-akibat.
        Dashboard digunakan sebagai alat bantu analisis dan eksplorasi data,
        bukan sebagai alat otomatis untuk mengambil keputusan terhadap karyawan.
        Model prediktif tetap berada di notebook.
        </div>
        """,
        unsafe_allow_html=True,
    )


st.divider()

st.caption(
    "HR Analytics & Employee Attrition Prediction"
)
