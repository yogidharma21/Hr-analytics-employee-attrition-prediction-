import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_URL = "https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/refs/heads/main/employee/employee_data.csv"
BASE_DIR = Path(__file__).resolve().parent
LOCAL_PATHS = [BASE_DIR / "data" / "employee_data.csv", BASE_DIR / "employee_data.csv"]

st.markdown("""
<style>
.main {background:#f7f8fa;}
[data-testid="stSidebar"] {background:#111827;}
[data-testid="stSidebar"] * {color:#f9fafb;}
.hero h1 {font-size:2.2rem;margin-bottom:.1rem;color:#111827;}
.hero p {color:#6b7280;margin-top:0;}
div[data-testid="stMetric"] {background:white;border:1px solid #e5e7eb;padding:14px 16px;border-radius:12px;}
.card {background:white;border:1px solid #e5e7eb;border-radius:12px;padding:16px;margin-bottom:12px;}
.high {border-left:5px solid #dc2626;}
.medium {border-left:5px solid #d97706;}
</style>
""", unsafe_allow_html=True)

@st.cache_data

def load_data():
    for path in LOCAL_PATHS:
        if path.exists():
            return pd.read_csv(path)
    return pd.read_csv(DATA_URL)

@st.cache_data

def prepare_data(data):
    df = data.copy()
    if "Attrition" in df.columns:
        df["AttritionLabel"] = df["Attrition"].map({0: "Stay", 1: "Leave"})
    if "YearsAtCompany" in df.columns:
        df["TenureGroup"] = pd.cut(
            df["YearsAtCompany"],
            bins=[-np.inf, 1, 3, 5, np.inf],
            labels=["≤1 year", "2–3 years", "4–5 years", ">5 years"],
        )
    return df

try:
    df = prepare_data(load_data())
except Exception as exc:
    st.error("Dataset gagal dimuat.")
    st.exception(exc)
    st.stop()

st.sidebar.markdown("## 👥 HR Analytics")
st.sidebar.caption("Employee Attrition Dashboard")
page = st.sidebar.radio("Navigation", ["Overview", "Attrition Analysis", "Employee Profile", "HR Action"])
st.sidebar.divider()
st.sidebar.markdown("### Filters")

filtered = df.copy()
for col, label in [("Department", "Department"), ("JobRole", "Job Role"), ("OverTime", "OverTime"), ("Gender", "Gender")]:
    if col in df.columns:
        values = sorted(df[col].dropna().unique().tolist())
        selected = st.sidebar.multiselect(label, values, default=values)
        filtered = filtered[filtered[col].isin(selected)]
st.sidebar.caption(f"{len(filtered):,} employees shown")

def attrition_rate(data):
    if "Attrition" not in data.columns:
        return np.nan
    x = data.dropna(subset=["Attrition"])
    return np.nan if x.empty else x["Attrition"].mean() * 100

def rate_chart(data, category, title):
    if category not in data.columns or "Attrition" not in data.columns:
        return None
    x = data.dropna(subset=["Attrition"]).copy()
    if x.empty:
        return None
    r = x.groupby(category)["Attrition"].mean().mul(100).reset_index(name="Attrition Rate").sort_values("Attrition Rate", ascending=False)
    fig = px.bar(r, x=category, y="Attrition Rate", text="Attrition Rate", title=title)
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(margin=dict(l=20,r=20,t=55,b=20), yaxis_title="Attrition Rate (%)", xaxis_title="")
    return fig

# Overview
if page == "Overview":
    st.markdown('<div class="hero"><h1>HR Analytics Dashboard</h1><p>Workforce overview, employee attrition patterns, and HR priorities</p></div>', unsafe_allow_html=True)
    avg_age = filtered["Age"].mean() if "Age" in filtered else np.nan
    avg_tenure = filtered["YearsAtCompany"].mean() if "YearsAtCompany" in filtered else np.nan
    avg_income = filtered["MonthlyIncome"].mean() if "MonthlyIncome" in filtered else np.nan
    unlabeled = int(filtered["Attrition"].isna().sum()) if "Attrition" in filtered else 0
    a,b,c,d,e = st.columns(5)
    a.metric("Total Employees", f"{len(filtered):,}")
    b.metric("Attrition Rate", f"{attrition_rate(filtered):.1f}%" if not np.isnan(attrition_rate(filtered)) else "N/A")
    c.metric("Avg Age", f"{avg_age:.1f}" if not np.isnan(avg_age) else "N/A")
    d.metric("Avg Tenure", f"{avg_tenure:.1f} yrs" if not np.isnan(avg_tenure) else "N/A")
    e.metric("Avg Monthly Income", f"{avg_income:,.0f}" if not np.isnan(avg_income) else "N/A")
    st.divider()
    l,r = st.columns(2)
    with l:
        if "Department" in filtered.columns:
            counts = filtered["Department"].value_counts().reset_index()
            counts.columns = ["Department","Employees"]
            fig = px.bar(counts, x="Department", y="Employees", text="Employees", title="Employees by Department")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
    with r:
        if "JobRole" in filtered.columns:
            counts = filtered["JobRole"].value_counts().head(8).sort_values().reset_index()
            counts.columns = ["JobRole","Employees"]
            fig = px.bar(counts, x="Employees", y="JobRole", orientation="h", text="Employees", title="Top Job Roles by Headcount")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
    l,r = st.columns(2)
    with l:
        if "AttritionLabel" in filtered.columns:
            dist = filtered["AttritionLabel"].value_counts().reset_index()
            dist.columns = ["Status","Employees"]
            fig = px.pie(dist, names="Status", values="Employees", hole=.55, title="Employee Status")
            st.plotly_chart(fig, use_container_width=True)
    with r:
        fig = rate_chart(filtered.assign(TenureGroup=filtered.get("TenureGroup")), "TenureGroup", "Attrition Rate by Tenure") if "TenureGroup" in filtered else None
        if fig: st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Key Takeaways")
    ot_rate = np.nan
    if "OverTime" in filtered.columns and "Attrition" in filtered.columns:
        x=filtered.dropna(subset=["Attrition"]); rr=x.groupby("OverTime")["Attrition"].mean().mul(100)
        ot_rate = rr.get("Yes", np.nan)
    role_rate = np.nan
    if "JobRole" in filtered.columns and "Attrition" in filtered.columns:
        x=filtered.dropna(subset=["Attrition"]); rr=x.groupby("JobRole")["Attrition"].mean().mul(100)
        role_rate = rr.get("Sales Representative", np.nan)
    cards=[("OverTime", f"Attrition pada karyawan yang lembur sekitar {ot_rate:.1f}%." if not np.isnan(ot_rate) else "Belum ada data OverTime."),
           ("High-risk Role", f"Sales Representative sekitar {role_rate:.1f}% attrition." if not np.isnan(role_rate) else "Belum ada data role ini."),
           ("Data Coverage", f"{unlabeled:,} karyawan belum memiliki label Attrition." if unlabeled else "Semua karyawan yang tampil memiliki label.")]
    cols=st.columns(3)
    for col,(title,text) in zip(cols,cards):
        with col: st.markdown(f'<div class="card"><b>{title}</b><br>{text}</div>', unsafe_allow_html=True)

# Attrition Analysis
elif page == "Attrition Analysis":
    st.markdown('<div class="hero"><h1>Attrition Analysis</h1><p>Kelompok karyawan dengan attrition rate yang lebih tinggi</p></div>', unsafe_allow_html=True)
    labeled=filtered.dropna(subset=["Attrition"]) if "Attrition" in filtered.columns else pd.DataFrame()
    if labeled.empty:
        st.warning("Tidak ada data berlabel Attrition pada filter saat ini.")
    else:
        l,r=st.columns(2)
        with l:
            for col,title in [("Department","Attrition Rate by Department"),("JobRole","Attrition Rate by Job Role")]:
                fig=rate_chart(labeled,col,title)
                if fig: st.plotly_chart(fig,use_container_width=True)
        with r:
            for col,title in [("OverTime","Attrition Rate by Overtime"),("BusinessTravel","Attrition Rate by Business Travel")]:
                fig=rate_chart(labeled,col,title)
                if fig: st.plotly_chart(fig,use_container_width=True)
        l,r=st.columns(2)
        with l:
            if "TenureGroup" in labeled:
                fig=rate_chart(labeled,"TenureGroup","Attrition Rate by Tenure Group")
                if fig: st.plotly_chart(fig,use_container_width=True)
        with r:
            if "WorkLifeBalance" in labeled:
                tmp=labeled.assign(WLB=labeled["WorkLifeBalance"].astype(str))
                fig=rate_chart(tmp,"WLB","Attrition Rate by Work-Life Balance")
                if fig: st.plotly_chart(fig,use_container_width=True)
        if "Age" in labeled.columns and "MonthlyIncome" in labeled.columns:
            sample=labeled.sample(min(len(labeled),1200), random_state=42)
            fig=px.scatter(sample,x="Age",y="MonthlyIncome",color="AttritionLabel",size="YearsAtCompany",hover_data=["Department","JobRole","OverTime"],opacity=.65,title="Age vs Monthly Income")
            st.plotly_chart(fig,use_container_width=True)

# Employee Profile
elif page == "Employee Profile":
    st.markdown('<div class="hero"><h1>Employee Profile</h1><p>Karakteristik workforce berdasarkan filter yang dipilih</p></div>', unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    a.metric("Employees",f"{len(filtered):,}")
    a2=attrition_rate(filtered); b.metric("Attrition Rate",f"{a2:.1f}%" if not np.isnan(a2) else "N/A")
    c.metric("Avg Years at Company",f"{filtered['YearsAtCompany'].mean():.1f}" if "YearsAtCompany" in filtered else "N/A")
    d.metric("Avg Monthly Income",f"{filtered['MonthlyIncome'].mean():,.0f}" if "MonthlyIncome" in filtered else "N/A")
    l,r=st.columns(2)
    with l:
        if "Age" in filtered:
            st.plotly_chart(px.histogram(filtered,x="Age",nbins=20,title="Age Distribution"),use_container_width=True)
    with r:
        if "Department" in filtered and "MonthlyIncome" in filtered:
            fig=px.box(filtered,x="Department",y="MonthlyIncome",title="Monthly Income by Department")
            fig.update_layout(xaxis_title="")
            st.plotly_chart(fig,use_container_width=True)
    if "JobRole" in filtered:
        role_table=filtered.groupby("JobRole").agg(Employees=("EmployeeId","count"),Avg_Age=("Age","mean"),Avg_Tenure=("YearsAtCompany","mean"),Avg_Income=("MonthlyIncome","mean")).reset_index().sort_values("Employees",ascending=False)
        for col in ["Avg_Age","Avg_Tenure"]: role_table[col]=role_table[col].round(1)
        role_table["Avg_Income"]=role_table["Avg_Income"].round(0)
        st.markdown("### Job Role Summary")
        st.dataframe(role_table,use_container_width=True,hide_index=True)

# HR Action
else:
    st.markdown('<div class="hero"><h1>HR Action</h1><p>Temuan utama dan area yang layak menjadi prioritas HR</p></div>', unsafe_allow_html=True)
    labeled=filtered.dropna(subset=["Attrition"]) if "Attrition" in filtered.columns else pd.DataFrame()
    if labeled.empty:
        st.warning("Tidak ada data berlabel pada filter saat ini.")
    else:
        overall=labeled["Attrition"].mean()*100
        role_rates=labeled.groupby("JobRole")["Attrition"].mean().mul(100).sort_values(ascending=False) if "JobRole" in labeled else pd.Series(dtype=float)
        ot_rates=labeled.groupby("OverTime")["Attrition"].mean().mul(100) if "OverTime" in labeled else pd.Series(dtype=float)
        top_role=role_rates.index[0] if len(role_rates) else "N/A"
        top_role_rate=role_rates.iloc[0] if len(role_rates) else np.nan
        ot_rate=ot_rates.get("Yes",np.nan)
        a,b,c=st.columns(3)
        a.metric("Top Attrition Role",top_role,f"{top_role_rate:.1f}%" if not np.isnan(top_role_rate) else None)
        b.metric("OverTime Attrition",f"{ot_rate:.1f}%" if not np.isnan(ot_rate) else "N/A")
        c.metric("Overall Attrition",f"{overall:.1f}%")
        if not np.isnan(ot_rate):
            st.markdown(f'<div class="card high"><b>High Priority — Overtime</b><br>Karyawan yang lembur memiliki attrition rate sekitar <b>{ot_rate:.1f}%</b>. Evaluasi workload, staffing, dan pembagian resource.</div>',unsafe_allow_html=True)
        if not np.isnan(top_role_rate):
            st.markdown(f'<div class="card high"><b>High Priority — {top_role}</b><br>Role ini memiliki attrition rate tertinggi pada data yang dipilih, sekitar <b>{top_role_rate:.1f}%</b>. Review retention dan career path.</div>',unsafe_allow_html=True)
        st.markdown('<div class="card medium"><b>Medium Priority — Early Tenure</b><br>Attrition cenderung lebih tinggi pada karyawan dengan masa kerja pendek. Perkuat onboarding dan mentoring.</div>',unsafe_allow_html=True)
        st.markdown('<div class="card medium"><b>Medium Priority — Work-Life Balance</b><br>Kelompok dengan work-life balance rendah menunjukkan attrition yang lebih tinggi, tetapi sinyalnya perlu dibaca hati-hati.</div>',unsafe_allow_html=True)
        actions=pd.DataFrame({"Priority":["High","High","Medium","Medium"],"Action":["Review overtime dan workload pada tim dengan beban tinggi","Program retensi untuk job role dengan attrition tinggi","Perkuat onboarding dan mentoring karyawan baru","Evaluasi work-life balance dan business travel"],"Target":["Tim dengan overtime tinggi",top_role,"Karyawan tenure ≤ 1 tahun","Kelompok dengan WLB rendah / travel tinggi"]})
        st.markdown("### Recommended Actions")
        st.dataframe(actions,use_container_width=True,hide_index=True)
        st.caption("Dashboard menggunakan data historis. Insight menunjukkan asosiasi dalam dataset, bukan hubungan sebab-akibat.")

st.divider()
st.caption("HR Analytics & Employee Attrition Prediction • Streamlit dashboard")
