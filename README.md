# HR Analytics & Employee Attrition Prediction

End-to-end **Data Analytics & Machine Learning project** untuk menganalisis pola employee attrition dan membangun model yang dapat membantu HR mengidentifikasi karyawan yang berisiko keluar dari perusahaan.

Project ini menggabungkan **HR Analytics, Exploratory Data Analysis, Statistical Analysis, Machine Learning, Model Interpretation, Risk Scoring, dan Business Recommendation**.

---

## 📌 Project Overview

Employee attrition adalah kondisi ketika karyawan meninggalkan perusahaan. Attrition yang tinggi dapat berdampak pada biaya rekrutmen, onboarding, produktivitas tim, dan keberlangsungan workforce.

Project ini mencoba menjawab dua hal:

1. **Data Analytics:** kelompok karyawan mana yang memiliki tingkat attrition lebih tinggi dan faktor apa yang berkaitan dengan attrition?
2. **Data Science:** apakah machine learning dapat digunakan sebagai early warning system untuk memberikan skor risiko attrition?

Hasil akhirnya tidak hanya berupa model, tetapi juga diterjemahkan menjadi insight dan rekomendasi yang bisa digunakan sebagai bahan pertimbangan HR.

---

## 🎯 Business Problem

Perusahaan ingin memahami pola employee attrition dan mengetahui kelompok karyawan yang perlu mendapat perhatian lebih.

Pertanyaan utama:

- Bagaimana karakteristik workforce perusahaan?
- Bagaimana pola attrition berdasarkan department, job role, overtime, tenure, dan karakteristik lainnya?
- Faktor apa yang paling berkaitan dengan attrition?
- Model machine learning mana yang paling sesuai untuk mendeteksi karyawan berisiko attrition?
- Bagaimana hasil model dapat membantu strategi retensi HR?

---
## Dashboard
[Streamlit](https://mouvpexgklhcr8vfwyutda.streamlit.app/).

## 📊 Dataset

**Source:** [Dicoding Employee Dataset](https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/refs/heads/main/employee/employee_data.csv)

Dataset terdiri dari:

- **1,470 karyawan**
- **35 kolom**
- Target: `Attrition`
  - `0` = karyawan bertahan
  - `1` = karyawan keluar
- **1,058 data memiliki label Attrition**
- **412 data tidak memiliki label Attrition**

Data berlabel digunakan untuk training dan evaluation, sedangkan 412 data tanpa label digunakan sebagai simulasi penerapan model untuk menghasilkan attrition risk score.

---

## 🔄 Project Workflow

```text
Business Understanding
        ↓
Data Understanding
        ↓
Data Quality Check
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Statistical Analysis
        ↓
Feature Engineering
        ↓
Data Preprocessing
        ↓
Machine Learning
        ↓
5-Fold Cross Validation
        ↓
Hyperparameter Tuning
        ↓
Final Test Evaluation
        ↓
Model Interpretation
        ↓
Attrition Risk Scoring
        ↓
Business Insights
        ↓
Business Recommendations
