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
# Palette — Clean Modern Enterprise / SaaS Theme
# ------------------------------------------------------------------------------------
C_BG = "#F8FAFC"             # Slate 50 (Soft light background)
C_SURFACE = "#FFFFFF"        # Pure white for cards
C_SIDEBAR = "#0F172A"        # Slate 900 (Sleek deep dark blue)
C_SIDEBAR_TEXT = "#F1F5F9"   # Light text for sidebar
C_SIDEBAR_MUTED = "#94A3B8"  # Slate 400
C_TEXT = "#0F172A"           # Slate 900
C_TEXT_MUTED = "#64748B"     # Slate 500
C_BORDER = "#E2E8F0"         # Slate 200

# Primary Pairs (Vibrant & Distinct)
C_STAY = "#2563EB"           # Royal Blue (Bertahan)
C_LEAVE = "#EA580C"          # Warm Orange/Coral (Keluar/Risiko)
C_STAY_SOFT = "#EFF6FF"      # Light Blue tint
C_LEAVE_SOFT = "#FFF7ED"     # Light Orange tint

C_ACCENT_GOOD = "#10B981"    # Emerald Green
C_ACCENT_BAD = "#EF4444"     # Red

# Custom Plotly template — Makes charts crisp & modern
_hr_template = go.layout.Template()
_hr_template.layout = go.Layout(
    font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", color=C_TEXT, size=12),
    title=dict(font=dict(color=C_TEXT, size=15, family="Inter, sans-serif"), x=0, xanchor="left"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(
        title_font=dict(color=C_TEXT_MUTED, size=12),
        tickfont=dict(color=C_TEXT_MUTED, size=11),
        gridcolor="#F1F5F9",
        linecolor=C_BORDER,
        zerolinecolor=C_BORDER,
    ),
    yaxis=dict(
        title_font=dict(color=C_TEXT_MUTED, size=12),
        tickfont=dict(color=C_TEXT_MUTED, size=11),
        gridcolor="#F1F5F9",
        linecolor=C_BORDER,
        zerolinecolor=C_BORDER,
    ),
    legend=dict(font=dict(color=C_TEXT, size=11)),
    coloraxis_colorbar=dict(tickfont=dict(color=C_TEXT_MUTED, size=11)),
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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, sans-serif;
        }}
        .stApp {{
            background-color: {C_BG};
        }}
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background-color: {C_SIDEBAR} !important;
            border-right: 1px solid #1E293B;
        }}
        section[data-testid="stSidebar"] * {{
            color: {C_SIDEBAR_TEXT} !important;
        }}
        section[data-testid="stSidebar"] .stRadio label {{
            background-color: transparent;
            padding: 10px 14px;
            border-radius: 8px;
            transition: all 0.2s ease;
            font-weight: 500;
        }}
        section[data-testid="stSidebar"] .stRadio label:hover {{
            background-color: rgba(255, 255, 255, 0.08);
        }}
        section[data-testid="stSidebar"] [data-aria-selected="true"] {{
            background-color: {C_STAY} !important;
            color: #FFFFFF !important;
            font-weight: 600;
        }}
        
        /* Headings */
        h1 {{
            color: {C_TEXT};
            font-weight: 800;
            letter-spacing: -0.02em;
            font-size: 1.85rem !important;
        }}
        h2, h3, h4 {{
            color: {C_TEXT};
            font-weight: 700;
            letter-spacing: -0.01em;
        }}
        p, li, span, label {{
            color: {C_TEXT};
        }}

        /* Modern KPI Card */
        .kpi-card {{
            background-color: {C_SURFACE};
            border: 1px solid {C_BORDER};
            border-radius: 12px;
            padding: 18px 20px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05);
            position: relative;
            overflow: hidden;
        }}
        .kpi-card::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, {C_STAY}, #60A5FA);
        }}
        .kpi-card-risk::before {{
            background: linear-gradient(90deg, {C_LEAVE}, #FDBA74) !important;
        }}
        .kpi-label {{
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: {C_TEXT_MUTED};
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 1.8rem;
            font-weight: 800;
            color: {C_TEXT};
            line-height: 1.2;
            letter-spacing: -0.02em;
        }}
        .kpi-sub {{
            font-size: 0.78rem;
            color: {C_TEXT_MUTED};
            margin-top: 4px;
            font-weight: 400;
        }}

        /* Section Card Wrapper */
        .section-card {{
            background-color: {C_SURFACE};
            border: 1px solid {C_BORDER};
            border-radius: 12px;
            padding: 22px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.03);
        }}

        /* Badges */
        .badge-risk {{
            display: inline-block;
            background-color: {C_LEAVE_SOFT};
            color: #C2410C;
            border: 1px solid #FFEDD5;
            font-weight: 700;
            font-size: 0.7rem;
            padding: 2px 10px;
            border-radius: 999px;
            letter-spacing: 0.03em;
        }}
        .badge-safe {{
            display: inline-block;
            background-color: {C_STAY_SOFT};
            color: #1E40AF;
            border: 1px solid #DBEAFE;
            font-weight: 700;
            font-size: 0.7rem;
            padding: 2px 10px;
            border-radius: 999px;
            letter-spacing: 0.03em;
        }}
        
        .divider-line {{
            border: none;
            border-top: 1px solid {C_BORDER};
            margin: 12px 0 18px 0;
        }}
        header[data-testid="stHeader"] {{
            background-color: {C_BG};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================================================================
# DATA LOADING + FEATURE ENGINEERING (mirrors notebook)
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
def kpi_card(label, value, sub="", is_risk=False):
    card_class = "kpi-card kpi-card-risk" if is_risk else "kpi-card"
    st.markdown(
        f"""
        <div class="{card_class}">
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
        st.markdown(f"<div style='color:{C_TEXT_MUTED}; font-size:0.88rem; margin-top:-6px; margin-bottom:10px;'>{subtitle}</div>", unsafe_allow_html=True)
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
    fig.update_traces(
        texttemplate="<b>%{text}%</b>",
        textposition="outside",
        textfont=dict(color=C_TEXT, size=11),
        cliponaxis=False,
    )
    if horizontal:
        fig.add_vline(x=overall_rate, line_dash="dash", line_color=C_TEXT_MUTED, line_width=1.5)
    else:
        fig.add_hline(y=overall_rate, line_dash="dash", line_color=C_TEXT_MUTED, line_width=1.5)
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=title, font=dict(color=C_TEXT, size=15, family="Inter")),
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=45, b=10),
        height=360,
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
    st.markdown(
        f"""
        <div style="padding: 10px 0 20px 0;">
            <div style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">👥 HR Attrition</div>
            <div style="font-size: 0.8rem; color: {C_SIDEBAR_MUTED}; margin-top: 2px;">Intelligence & Analytics Dashboard</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
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
    
    st.markdown("<div style='margin: 20px 0; border-top: 1px solid #1E293B;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="background-color: #1E293B; padding: 14px; border-radius: 10px; font-size: 0.78rem; line-height: 1.5; color: {C_SIDEBAR_MUTED};">
            <div style="font-weight: 600; color: #F8FAFC; margin-bottom: 6px;">📌 System Summary</div>
            Total Karyawan: <b style="color:#FFF;">{raw_df.shape[0]}</b><br/>
            Berlabel: <b style="color:#FFF;">{labeled.shape[0]}</b> • Unlabeled: <b style="color:#FFF;">{unlabeled.shape[0]}</b><br/>
            Model: <b style="color:#FFF;">Logistic Regression</b><br/>
            ROC-AUC Score: <b style="color:#10B981;">{model_bundle['metrics']['ROC-AUC']:.3f}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ======================================================================================
# PAGE 1 — WORKFORCE OVERVIEW
# ======================================================================================
if page == "📊 Workforce Overview":
    st.title("Workforce Overview")
    st.caption("Gambaran umum komposisi demografi dan demografi kerja karyawan berlabel.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Employees", f"{labeled_fe.shape[0]:,}", "Karyawan berlabel")
    with c2:
        kpi_card("Attrition Rate", f"{overall_rate:.1f}%", "Rata-rata perusahaan", is_risk=True)
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
        fig.update_traces(textposition="outside", textfont=dict(color=C_TEXT, size=11), cliponaxis=False)
        fig.update_layout(template=PLOTLY_TEMPLATE, title=dict(text="Jumlah Karyawan per Departemen"),
                          margin=dict(l=10, r=10, t=45, b=10), height=320)
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
        fig.update_traces(textposition="outside", textfont=dict(color=C_TEXT, size=11), cliponaxis=False)
        fig.update_layout(template=PLOTLY_TEMPLATE, title=dict(text="Jumlah Karyawan per Job Role"),
                          margin=dict(l=10, r=10, t=45, b=10), height=320, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        fig = px.histogram(labeled_fe, x="Age", nbins=20, color_discrete_sequence=[C_STAY])
        fig.update_layout(template=PLOTLY_TEMPLATE, title=dict(text="Distribusi Usia Karyawan"), bargap=0.08,
                          margin=dict(l=10, r=10, t=45, b=10), height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        fig = px.histogram(labeled_fe, x="YearsAtCompany", nbins=20, color_discrete_sequence=[C_STAY])
        fig.update_layout(template=PLOTLY_TEMPLATE, title=dict(text="Distribusi Masa Kerja (Years At Company)"), bargap=0.08,
                          margin=dict(l=10, r=10, t=45, b=10), height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ======================================================================================
# PAGE 2 — ATTRITION ANALYSIS
# ======================================================================================
elif page == "📉 Attrition Analysis":
    st.title("Attrition Analysis")
    st.caption(f"Analisis rasio turnover. Garis putus-putus menunjukkan rata-rata attrition rate ({overall_rate:.1f}%).")

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
    st.caption("Eksplorasi interaktif karakteristik demografi dan segmen karyawan.")

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
        kpi_card("Jumlah Karyawan", f"{filtered.shape[0]:,}", "Hasil Filter")
    with c2:
        rate = filtered["Attrition"].mean() * 100 if len(filtered) else 0
        kpi_card("Attrition Rate", f"{rate:.1f}%", "Hasil Filter", is_risk=(rate > overall_rate))
    with c3:
        kpi_card("Rata-rata Income", f"${filtered['MonthlyIncome'].mean():,.0f}" if len(filtered) else "-")

    st.write("")
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    plot_df = filtered.copy()
    plot_df["Status"] = plot_df["Attrition"].map({0: "Bertahan", 1: "Keluar"})
    fig = px.scatter(
        plot_df, x="YearsAtCompany", y="MonthlyIncome", color="Status",
        color_discrete_map={"Bertahan": C_STAY, "Keluar": C_LEAVE},
        opacity=0.85, hover_data=["Department", "JobRole", "Age"],
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, title=dict(text="Monthly Income vs Tenure"),
                      margin=dict(l=10, r=10, t=45, b=10), height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        fig = px.histogram(plot_df, x="Age", color="Status", barmode="overlay",
                            color_discrete_map={"Bertahan": C_STAY, "Keluar": C_LEAVE}, opacity=0.7)
        fig.update_layout(template=PLOTLY_TEMPLATE, title=dict(text="Distribusi Usia berdasarkan Status"),
                          margin=dict(l=10, r=10, t=45, b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        edu_counts = plot_df["EducationField"].value_counts().reset_index()
        edu_counts.columns = ["EducationField", "Count"]
        fig = px.bar(edu_counts, x="Count", y="EducationField", orientation="h",
                     color_discrete_sequence=[C_STAY])
        fig.update_layout(template=PLOTLY_TEMPLATE, title=dict(text="Karyawan per Education Field"),
                          margin=dict(l=10, r=10, t=45, b=10), height=320, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Lihat Data Detail Karyawan (Tabel)"):
        st.dataframe(filtered, use_container_width=True, height=350)


# ======================================================================================
# PAGE 4 — HR ACTION CENTER
# ======================================================================================
elif page == "🎯 HR Action Center":
    st.title("HR Action Center")
    st.caption("Prediksi risiko attrition untuk karyawan unlabeled serta rincian rekomendasi strategis HR.")

    m = model_bundle["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (label, key) in zip(
        [c1, c2, c3, c4, c5],
        [("Accuracy", "Accuracy"), ("Precision", "Precision"), ("Recall", "Recall"),
         ("F1-Score", "F1"), ("ROC-AUC", "ROC-AUC")],
    ):
        with col:
            kpi_card(label, f"{m[key]:.3f}", "Test Set Evaluation")

    st.write("")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        section_title("🔴 Top 15 Karyawan Berisiko Tertinggi", "Hasil kalkulasi probabilitas model Logistic Regression")
        top_risk = risk_df.head(15).copy()
        top_risk["AttritionRiskScore"] = top_risk["AttritionRiskScore"].astype(str) + "%"
        st.dataframe(top_risk, use_container_width=True, height=410, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        section_title("Distribusi Skor Risiko Attrition")
        fig = px.histogram(risk_df, x="AttritionRiskScore", nbins=25, color_discrete_sequence=[C_LEAVE])
        fig.add_vline(x=50, line_dash="dash", line_color=C_TEXT_MUTED, annotation_text="Threshold 50%")
        fig.update_layout(template=PLOTLY_TEMPLATE, margin=dict(l=10, r=10, t=10, b=10), height=410)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    section_title("⚠️ Kelompok Rentan (High-Risk Groups)", "Segmen karyawan dengan attrition rate jauh melampaui rata-rata")
    g1, g2, g3, g4 = st.columns(4)
    groups = [
        ("Sales Representative", "43.1% attrition rate", g1),
        ("Sering Lembur (OverTime = Yes)", "31.9% attrition rate", g2),
        ("Tenure Baru (0 tahun)", "Attrition tertinggi di tenure group", g3),
        ("Work-Life Balance Rendah", "Skor 1 — Attrition tertinggi", g4),
    ]
    for name, desc, col in groups:
        with col:
            st.markdown(
                f"""
                <div style="border:1px solid {C_BORDER}; border-radius:10px; padding:14px; background:{C_LEAVE_SOFT};">
                    <span class="badge-risk">RISIKO TINGGI</span>
                    <div style="font-weight:700; margin-top:8px; color:{C_TEXT};">{name}</div>
                    <div style="color:{C_TEXT_MUTED}; font-size:0.82rem; margin-top:2px;">{desc}</div>
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
            "<b>OverTime (Lembur):</b> Indikator paling signifikan (31.9% attrition vs 10.8%).",
            "<b>Job Role:</b> Sales Representative memiliki tingkat attrition paling tinggi (43.1%).",
            "<b>Tenure Rawan:</b> Karyawan baru (0 tahun) paling rentan keluar; kestabilan meningkat setelah 3 tahun.",
            "<b>Kepuasan & Balance:</b> Skor kepuasan kerja dan WLB rendah berkorelasi langsung dengan pengunduran diri.",
            "<b>Model Performance:</b> Model Logistic Regression terkalibrasi baik dengan ROC-AUC <b>0.822</b> pada test set.",
        ]
        for ins in insights:
            st.markdown(f"<div style='margin-bottom:8px; font-size:0.9rem; line-height:1.4;'>• {ins}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        section_title("✅ Recommended Actions")
        actions = [
            "Evaluasi & batasi kebijakan jam lembur, khususnya di divisi dengan beban kerja ekstrim.",
            "Perkuat program onboarding & mentorship intensif untuk karyawan baru (tahun ke-0).",
            "Kaji ulang skema insentif dan kepuasan kerja untuk posisi Sales Representative.",
            "Jadwalkan sesi 1-on-1 intervensi khusus bagi 15 karyawan dengan skor risiko tertinggi.",
            "Pantau berkala indikator Work-Life Balance sebagai early warning system HR.",
        ]
        for i, act in enumerate(actions, 1):
            st.markdown(f"<div style='margin-bottom:8px; font-size:0.9rem; line-height:1.4;'><b>{i}.</b> {act}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Lihat Parameter & Detail Evaluasi Model"):
        st.write("**Best Parameters (GridSearchCV):**", model_bundle["best_params"])
        st.write("**Confusion Matrix (Test Set):**")
        cm = model_bundle["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["Aktual: Bertahan", "Aktual: Keluar"],
            columns=["Prediksi: Bertahan", "Prediksi: Keluar"]
        )
        st.dataframe(cm_df, use_container_width=True)
