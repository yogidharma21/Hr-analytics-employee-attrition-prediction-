"""
HR Analytics & Employee Attrition Dashboard
Portfolio project — companion Streamlit app for the analysis notebook.
"""

import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

DATA_URL = (
    "https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/"
    "refs/heads/main/employee/employee_data.csv"
)
RANDOM_STATE = 42

# ------------------------------------------------------------------------------------
# Palette — kept deliberately small & non-clashing.
# Navy/slate for structure, one blue (retain) + one amber (risk) as the only accent pair.
# ------------------------------------------------------------------------------------
C_BG = "#F4F6FA"
C_SURFACE = "#FFFFFF"
C_SIDEBAR = "#111827"
C_SIDEBAR_MUTED = "#9CA3AF"
C_TEXT = "#1F2933"
C_TEXT_MUTED = "#6B7280"
C_BORDER = "#E5E7EB"
C_STAY = "#3B6FA0"     # muted blue  — "bertahan"
C_LEAVE = "#E08A3C"    # muted amber — "keluar / risiko"
C_STAY_SOFT = "#DCE6F1"
C_LEAVE_SOFT = "#FBE6D2"
C_ACCENT_GOOD = "#2F9E64"
C_ACCENT_BAD = "#D65A5A"

# Custom Plotly template — makes sure axis titles, tick labels, and chart titles
# always render in dark, readable colors regardless of the host page's theme.
_hr_template = go.layout.Template()
_hr_template.layout = go.Layout(
    font=dict(family="Inter, Segoe UI, sans-serif", color=C_TEXT, size=13),
    title=dict(font=dict(color=C_TEXT, size=16)),
    paper_bgcolor=C_SURFACE,
    plot_bgcolor=C_SURFACE,
    xaxis=dict(
        title_font=dict(color=C_TEXT, size=13),
        tickfont=dict(color=C_TEXT_MUTED, size=12),
        gridcolor=C_BORDER,
        linecolor=C_BORDER,
        zerolinecolor=C_BORDER,
    ),
    yaxis=dict(
        title_font=dict(color=C_TEXT, size=13),
        tickfont=dict(color=C_TEXT_MUTED, size=12),
        gridcolor=C_BORDER,
        linecolor=C_BORDER,
        zerolinecolor=C_BORDER,
    ),
    legend=dict(font=dict(color=C_TEXT)),
    coloraxis_colorbar=dict(tickfont=dict(color=C_TEXT_MUTED)),
)
pio.templates["hr_dashboard"] = _hr_template

PLOTLY_TEMPLATE = "plotly_white+hr_dashboard"


# ======================================================================================
# PAGE CONFIG + GLOBAL CSS
# ======================================================================================
st.set_page_config(
    page_title="HR Attrition Intelligence",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
        html, body, [class*="css"] {{
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }}
        .stApp {{
            background-color: {C_BG};
        }}
        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {C_SIDEBAR};
        }}
        section[data-testid="stSidebar"] * {{
            color: {C_SIDEBAR_MUTED} !important;
        }}
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            color: #FFFFFF !important;
        }}
        section[data-testid="stSidebar"] label {{
            color: #E5E7EB !important;
        }}
        div[role="radiogroup"] label {{
            padding: 6px 10px;
            border-radius: 8px;
        }}
        /* Headings */
        h1, h2, h3, h4 {{
            color: {C_TEXT};
            font-weight: 700;
        }}
        p, li, span, label {{
            color: {C_TEXT};
        }}
        /* KPI card */
        .kpi-card {{
            background-color: {C_SURFACE};
            border: 1px solid {C_BORDER};
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 1px 3px rgba(16, 24, 40, 0.04);
        }}
        .kpi-label {{
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            color: {C_TEXT_MUTED};
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 1.9rem;
            font-weight: 800;
            color: {C_TEXT};
            line-height: 1.1;
        }}
        .kpi-sub {{
            font-size: 0.78rem;
            color: {C_TEXT_MUTED};
            margin-top: 4px;
        }}
        /* Section card wrapper */
        .section-card {{
            background-color: {C_SURFACE};
            border: 1px solid {C_BORDER};
            border-radius: 14px;
            padding: 20px 22px;
            margin-bottom: 18px;
            box-shadow: 0 1px 3px rgba(16, 24, 40, 0.04);
        }}
        /* Badges */
        .badge-risk {{
            display: inline-block;
            background-color: {C_LEAVE_SOFT};
            color: #8A4A16;
            font-weight: 700;
            font-size: 0.75rem;
            padding: 3px 10px;
            border-radius: 999px;
        }}
        .badge-safe {{
            display: inline-block;
            background-color: {C_STAY_SOFT};
            color: #1E3A5F;
            font-weight: 700;
            font-size: 0.75rem;
            padding: 3px 10px;
            border-radius: 999px;
        }}
        .insight-title {{
            font-weight: 700;
            font-size: 1rem;
            color: {C_TEXT};
            margin-bottom: 2px;
        }}
        .divider-line {{
            border: none;
            border-top: 1px solid {C_BORDER};
            margin: 10px 0 18px 0;
        }}
        header[data-testid="stHeader"] {{
            background-color: {C_BG};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================================================================
# DATA LOADING + FEATURE ENGINEERING  (mirrors the notebook exactly)
# ======================================================================================
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_URL)
    df_clean = df.drop(columns=["EmployeeCount", "StandardHours", "Over18"])

    labeled = df_clean.dropna(subset=["Attrition"]).copy()
    labeled["Attrition"] = labeled["Attrition"].astype(int)

    unlabeled = df_clean[df_clean["Attrition"].isna()].copy()

    return df, labeled, unlabeled


def add_engineered_features(data):
    data = data.copy()
    bins = [-1, 0, 3, 6, 10, 100]
    labels = ["Baru (0 Tahun)", "1-3 Tahun", "4-6 Tahun", "7-10 Tahun", "Lebih dari 10 Tahun"]
    data["TenureGroup"] = pd.cut(data["YearsAtCompany"], bins=bins, labels=labels)
    data["IncomeGroup"] = pd.qcut(
        data["MonthlyIncome"], q=4, labels=["Q1 (Terendah)", "Q2", "Q3", "Q4 (Tertinggi)"]
    )
    data["AvgSatisfaction"] = data[
        ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction"]
    ].mean(axis=1)
    return data


@st.cache_resource(show_spinner="Melatih model prediksi attrition...")
def train_model(labeled: pd.DataFrame):
    identifier_col, target_col = "EmployeeId", "Attrition"
    feature_cols = [c for c in labeled.columns if c not in [identifier_col, target_col]]

    X = labeled[feature_cols]
    y = labeled[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    categorical_features = X.select_dtypes(include="object").columns.tolist()
    numerical_features = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        pipe, {"classifier__C": [0.01, 0.1, 1, 10]}, cv=cv, scoring="f1", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    final_model = grid.best_estimator_

    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }
    cm = confusion_matrix(y_test, y_pred)

    # Feature coefficients for interpretation
    cat_names = list(
        final_model.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .get_feature_names_out(categorical_features)
    )
    all_feature_names = numerical_features + cat_names
    coefs = final_model.named_steps["classifier"].coef_[0]
    coef_df = pd.DataFrame({"Feature": all_feature_names, "Coefficient": coefs})
    coef_df["AbsCoefficient"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("AbsCoefficient", ascending=False)

    return {
        "model": final_model,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "confusion_matrix": cm,
        "coef_df": coef_df,
        "best_params": grid.best_params_,
    }


@st.cache_data(show_spinner=False)
def score_unlabeled(_model, feature_cols, unlabeled: pd.DataFrame):
    X_unlabeled = unlabeled[feature_cols]
    scores = _model.predict_proba(X_unlabeled)[:, 1]
    risk_df = unlabeled[
        ["EmployeeId", "Department", "JobRole", "OverTime", "YearsAtCompany", "MonthlyIncome"]
    ].copy()
    risk_df["AttritionRiskScore"] = (scores * 100).round(1)
    risk_df = risk_df.sort_values("AttritionRiskScore", ascending=False).reset_index(drop=True)
    return risk_df


# ======================================================================================
# SMALL UI HELPERS
# ======================================================================================
def kpi_card(label, value, sub=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title, subtitle=None):
    st.markdown(f"### {title}")
    if subtitle:
        st.markdown(f"<span style='color:{C_TEXT_MUTED}; font-size:0.9rem;'>{subtitle}</span>", unsafe_allow_html=True)
    st.markdown("<hr class='divider-line'/>", unsafe_allow_html=True)


def attrition_rate_by(df, col, order=None):
    t = df.groupby(col, observed=True)["Attrition"].agg(rate="mean", count="count")
    t["rate"] = (t["rate"] * 100).round(1)
    if order is not None:
        t = t.reindex(order)
    else:
        t = t.sort_values("rate", ascending=False)
    return t.reset_index()


def bar_rate_chart(data, x_col, overall_rate, title, horizontal=False):
    orientation = "h" if horizontal else "v"
    x_arg, y_arg = ("rate", x_col) if horizontal else (x_col, "rate")
    fig = px.bar(
        data,
        x=x_arg,
        y=y_arg,
        orientation=orientation,
        color="rate",
        color_continuous_scale=[C_STAY, C_LEAVE],
        text="rate",
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    if horizontal:
        fig.add_vline(x=overall_rate, line_dash="dash", line_color=C_TEXT_MUTED)
    else:
        fig.add_hline(y=overall_rate, line_dash="dash", line_color=C_TEXT_MUTED)
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=title,
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=50, b=10),
        height=380,
        plot_bgcolor=C_SURFACE,
        paper_bgcolor=C_SURFACE,
    )
    return fig


# ======================================================================================
# LOAD DATA + MODEL
# ======================================================================================
raw_df, labeled, unlabeled = load_data()
labeled_fe = add_engineered_features(labeled)
overall_rate = labeled_fe["Attrition"].mean() * 100

model_bundle = train_model(labeled)
risk_df = score_unlabeled(model_bundle["model"], model_bundle["feature_cols"], unlabeled)


# ======================================================================================
# SIDEBAR NAVIGATION
# ======================================================================================
with st.sidebar:
    st.markdown("## 👥 HR Attrition\n### Intelligence Dashboard")
    st.markdown(
        "<span style='font-size:0.85rem;'>Analisis workforce & prediksi risiko attrition karyawan.</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "Navigasi",
        [
            "📊 Workforce Overview",
            "📉 Attrition Analysis",
            "🧑‍💼 Employee Profile",
            "🎯 HR Action Center",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        f"""
        <span style='font-size:0.78rem;'>
        Total data: {raw_df.shape[0]} karyawan<br/>
        Berlabel: {labeled.shape[0]} • Belum berlabel: {unlabeled.shape[0]}<br/>
        Model: Logistic Regression (tuned)<br/>
        ROC-AUC test: {model_bundle['metrics']['ROC-AUC']:.3f}
        </span>
        """,
        unsafe_allow_html=True,
    )


# ======================================================================================
# PAGE 1 — WORKFORCE OVERVIEW
# ======================================================================================
if page == "📊 Workforce Overview":
    st.title("Workforce Overview")
    st.caption("Gambaran umum komposisi karyawan berdasarkan data yang berlabel.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Employees", f"{labeled_fe.shape[0]:,}", "Karyawan berlabel")
    with c2:
        kpi_card("Attrition Rate", f"{overall_rate:.1f}%", "Rata-rata perusahaan")
    with c3:
        kpi_card("Avg. Tenure", f"{labeled_fe['YearsAtCompany'].mean():.1f} thn", "Masa kerja rata-rata")
    with c4:
        kpi_card("Avg. Monthly Income", f"${labeled_fe['MonthlyIncome'].mean():,.0f}", "Pendapatan rata-rata")

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        dept_counts = labeled_fe["Department"].value_counts().reset_index()
        dept_counts.columns = ["Department", "Count"]
        fig = px.bar(
            dept_counts, x="Count", y="Department", orientation="h",
            color_discrete_sequence=[C_STAY], text="Count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(template=PLOTLY_TEMPLATE, title="Jumlah Karyawan per Departemen",
                           margin=dict(l=10, r=10, t=50, b=10), height=340,
                           plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        role_counts = labeled_fe["JobRole"].value_counts().reset_index()
        role_counts.columns = ["JobRole", "Count"]
        fig = px.bar(
            role_counts, x="Count", y="JobRole", orientation="h",
            color_discrete_sequence=[C_STAY], text="Count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(template=PLOTLY_TEMPLATE, title="Jumlah Karyawan per Job Role",
                           margin=dict(l=10, r=10, t=50, b=10), height=340, yaxis={"categoryorder": "total ascending"},
                           plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        fig = px.histogram(labeled_fe, x="Age", nbins=20, color_discrete_sequence=[C_STAY])
        fig.update_layout(template=PLOTLY_TEMPLATE, title="Distribusi Usia Karyawan", bargap=0.05,
                           margin=dict(l=10, r=10, t=50, b=10), height=320,
                           plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        fig = px.histogram(labeled_fe, x="YearsAtCompany", nbins=20, color_discrete_sequence=[C_STAY])
        fig.update_layout(template=PLOTLY_TEMPLATE, title="Distribusi Tenure (Years At Company)", bargap=0.05,
                           margin=dict(l=10, r=10, t=50, b=10), height=320,
                           plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ======================================================================================
# PAGE 2 — ATTRITION ANALYSIS
# ======================================================================================
elif page == "📉 Attrition Analysis":
    st.title("Attrition Analysis")
    st.caption(f"Garis putus-putus abu-abu menandakan rata-rata attrition rate perusahaan ({overall_rate:.1f}%).")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        t = attrition_rate_by(labeled_fe, "Department")
        st.plotly_chart(bar_rate_chart(t, "Department", overall_rate, "Attrition Rate by Department", horizontal=True), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        t = attrition_rate_by(labeled_fe, "JobRole")
        st.plotly_chart(bar_rate_chart(t, "JobRole", overall_rate, "Attrition Rate by Job Role", horizontal=True), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        tenure_order = ["Baru (0 Tahun)", "1-3 Tahun", "4-6 Tahun", "7-10 Tahun", "Lebih dari 10 Tahun"]
        t = attrition_rate_by(labeled_fe, "TenureGroup", order=tenure_order)
        st.plotly_chart(bar_rate_chart(t, "TenureGroup", overall_rate, "Attrition Rate by Tenure Group"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        t = attrition_rate_by(labeled_fe, "OverTime")
        st.plotly_chart(bar_rate_chart(t, "OverTime", overall_rate, "Attrition Rate by OverTime"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    col5, col6 = st.columns(2)
    with col5:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        t = attrition_rate_by(labeled_fe, "JobSatisfaction", order=[1, 2, 3, 4])
        t["JobSatisfaction"] = t["JobSatisfaction"].astype(str)
        st.plotly_chart(bar_rate_chart(t, "JobSatisfaction", overall_rate, "Attrition Rate by Job Satisfaction"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col6:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        t = attrition_rate_by(labeled_fe, "WorkLifeBalance", order=[1, 2, 3, 4])
        t["WorkLifeBalance"] = t["WorkLifeBalance"].astype(str)
        st.plotly_chart(bar_rate_chart(t, "WorkLifeBalance", overall_rate, "Attrition Rate by Work-Life Balance"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ======================================================================================
# PAGE 3 — EMPLOYEE PROFILE
# ======================================================================================
elif page == "🧑‍💼 Employee Profile":
    st.title("Employee Profile Explorer")
    st.caption("Gunakan filter di bawah untuk menjelajahi karakteristik karyawan secara interaktif.")

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        dept_sel = st.multiselect("Department", sorted(labeled_fe["Department"].unique()))
    with f2:
        role_sel = st.multiselect("Job Role", sorted(labeled_fe["JobRole"].unique()))
    with f3:
        gender_sel = st.multiselect("Gender", sorted(labeled_fe["Gender"].unique()))
    with f4:
        marital_sel = st.multiselect("Marital Status", sorted(labeled_fe["MaritalStatus"].unique()))
    st.markdown("</div>", unsafe_allow_html=True)

    filtered = labeled_fe.copy()
    if dept_sel:
        filtered = filtered[filtered["Department"].isin(dept_sel)]
    if role_sel:
        filtered = filtered[filtered["JobRole"].isin(role_sel)]
    if gender_sel:
        filtered = filtered[filtered["Gender"].isin(gender_sel)]
    if marital_sel:
        filtered = filtered[filtered["MaritalStatus"].isin(marital_sel)]

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Jumlah Karyawan (filter)", f"{filtered.shape[0]:,}")
    with c2:
        rate = filtered["Attrition"].mean() * 100 if len(filtered) else 0
        kpi_card("Attrition Rate (filter)", f"{rate:.1f}%")
    with c3:
        kpi_card("Rata-rata Income", f"${filtered['MonthlyIncome'].mean():,.0f}" if len(filtered) else "-")

    st.write("")
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    plot_df = filtered.copy()
    plot_df["Status"] = plot_df["Attrition"].map({0: "Bertahan", 1: "Keluar"})
    fig = px.scatter(
        plot_df, x="YearsAtCompany", y="MonthlyIncome", color="Status",
        color_discrete_map={"Bertahan": C_STAY, "Keluar": C_LEAVE},
        opacity=0.75, hover_data=["Department", "JobRole", "Age"],
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, title="Income vs Tenure (berdasarkan status attrition)",
                       margin=dict(l=10, r=10, t=50, b=10), height=420,
                       plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        fig = px.histogram(plot_df, x="Age", color="Status", barmode="overlay",
                            color_discrete_map={"Bertahan": C_STAY, "Keluar": C_LEAVE}, opacity=0.7)
        fig.update_layout(template=PLOTLY_TEMPLATE, title="Distribusi Usia",
                           margin=dict(l=10, r=10, t=50, b=10), height=340,
                           plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        edu_counts = plot_df["EducationField"].value_counts().reset_index()
        edu_counts.columns = ["EducationField", "Count"]
        fig = px.bar(edu_counts, x="Count", y="EducationField", orientation="h",
                     color_discrete_sequence=[C_STAY])
        fig.update_layout(template=PLOTLY_TEMPLATE, title="Karyawan per Education Field",
                           margin=dict(l=10, r=10, t=50, b=10), height=340,
                           yaxis={"categoryorder": "total ascending"},
                           plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Lihat data karyawan (tabel)"):
        st.dataframe(filtered, use_container_width=True, height=350)


# ======================================================================================
# PAGE 4 — HR ACTION CENTER
# ======================================================================================
elif page == "🎯 HR Action Center":
    st.title("HR Action Center")
    st.caption("Daftar karyawan berisiko tinggi (dari data yang belum berlabel) hasil prediksi model, plus ringkasan insight & rekomendasi.")

    m = model_bundle["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (label, key) in zip(
        [c1, c2, c3, c4, c5],
        [("Accuracy", "Accuracy"), ("Precision", "Precision"), ("Recall", "Recall"),
         ("F1-Score", "F1"), ("ROC-AUC", "ROC-AUC")],
    ):
        with col:
            kpi_card(label, f"{m[key]:.3f}", "Test set")

    st.write("")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        section_title("🔴 Top 15 Karyawan Berisiko Tertinggi", "Berdasarkan skor prediksi attrition model")
        top_risk = risk_df.head(15).copy()
        top_risk["AttritionRiskScore"] = top_risk["AttritionRiskScore"].astype(str) + "%"
        st.dataframe(top_risk, use_container_width=True, height=430, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        section_title("Distribusi Skor Risiko")
        fig = px.histogram(risk_df, x="AttritionRiskScore", nbins=25, color_discrete_sequence=[C_LEAVE])
        fig.add_vline(x=50, line_dash="dash", line_color=C_TEXT_MUTED, annotation_text="Skor 50%")
        fig.update_layout(template=PLOTLY_TEMPLATE, margin=dict(l=10, r=10, t=10, b=10), height=430,
                           plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    section_title("⚠️ High-Risk Groups", "Kelompok karyawan dengan attrition rate jauh di atas rata-rata perusahaan")
    g1, g2, g3, g4 = st.columns(4)
    groups = [
        ("Sales Representative", "43.1% attrition rate", g1),
        ("Sering Lembur (OverTime = Yes)", "31.9% attrition rate", g2),
        ("Tenure Baru (0 tahun)", "Attrition tertinggi di kelompok tenure", g3),
        ("Work-Life Balance Rendah", "Skor 1 — attrition tertinggi", g4),
    ]
    for name, desc, col in groups:
        with col:
            st.markdown(
                f"""
                <div style="border:1px solid {C_BORDER}; border-radius:12px; padding:14px; background:{C_LEAVE_SOFT}22;">
                    <span class="badge-risk">RISIKO TINGGI</span>
                    <div style="font-weight:700; margin-top:8px;">{name}</div>
                    <div style="color:{C_TEXT_MUTED}; font-size:0.85rem; margin-top:2px;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        section_title("📌 Key Insights")
        insights = [
            "Lembur (OverTime) adalah red flag terbesar — attrition rate 31.9% vs 10.8%.",
            "Sales Representative punya attrition rate tertinggi di antara semua job role (43.1%).",
            "Karyawan dengan tenure baru (0 tahun) paling rentan keluar; risiko menurun seiring masa kerja.",
            "Job satisfaction & work-life balance rendah berkorelasi kuat dengan attrition.",
            "TotalWorkingYears, YearsWithCurrManager, dan Age adalah prediktor numerik terkuat.",
            "OverTime, JobRole, dan MaritalStatus signifikan secara statistik (Chi-Square, p < 0.05).",
            "Model Logistic Regression (tuned) mencapai ROC-AUC 0.822 di test set.",
        ]
        for ins in insights:
            st.markdown(f"- {ins}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        section_title("✅ Recommended Actions")
        actions = [
            "Audit & batasi jam lembur, terutama di tim/departemen dengan beban kerja tinggi.",
            "Perkuat program onboarding & mentoring untuk karyawan baru (tenure 0-1 tahun).",
            "Evaluasi ulang kompensasi & jenjang karier untuk role Sales Representative.",
            "Lakukan 1-on-1 check-in rutin untuk karyawan dengan skor risiko attrition tinggi.",
            "Pantau skor work-life balance & job satisfaction secara berkala sebagai early warning.",
        ]
        for i, act in enumerate(actions, 1):
            st.markdown(f"**{i}.** {act}")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Lihat parameter & detail model"):
        st.write("Best params (GridSearchCV):", model_bundle["best_params"])
        st.write("Confusion Matrix (test set):")
        cm = model_bundle["confusion_matrix"]
        cm_df = pd.DataFrame(cm, index=["Aktual: Bertahan", "Aktual: Keluar"],
                              columns=["Prediksi: Bertahan", "Prediksi: Keluar"])
        st.dataframe(cm_df, use_container_width=True)
