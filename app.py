import streamlit as st
import pandas as pd
import numpy as np
import chardet
import io
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Data Quality Monitoring System", layout="wide")

st.title("📊 Automated Data Quality Monitoring & Anomaly Detection System")

# ---------- Universal Loader ----------
def load_dataset(uploaded_file):
    try:
        name = uploaded_file.name.lower()

        if name.endswith(".csv"):
            try:
                return pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                enc = chardet.detect(uploaded_file.read(100000))["encoding"]
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding=enc)

        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)

    except Exception as e:
        st.error(f"Error: {e}")
        return None


# ---------- Quality Score ----------
def data_quality_score(df):
    missing = df.isna().sum().sum()
    duplicates = df.duplicated().sum()

    score = 100
    score -= (missing / (df.shape[0]*df.shape[1])) * 50
    score -= (duplicates / df.shape[0]) * 50

    return round(max(score, 0), 2)


# ---------- File Upload ----------
file = st.file_uploader("Upload Dataset", type=["csv", "xlsx", "xls"])

if file:
    df = load_dataset(file)

    if df is not None:

        st.success("Dataset Loaded Successfully!")

        # Preview
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isna().sum().sum())

        # ---------- Quality Score ----------
        st.subheader("Data Quality Score")

        score = data_quality_score(df)
        st.metric("Quality Score", f"{score}/100")

        if score > 80:
            st.success("✅ Dataset is Good — Ready for ML/Analytics")
        elif score > 50:
            st.warning("⚠ Dataset is Average — Needs Cleaning")
        else:
            st.error("❌ Dataset Poor — Cleaning Required")

        # ---------- Suggestions ----------
        st.subheader("Improvement Suggestions")

        if df.isna().sum().sum() > 0:
            st.write("• Handle missing values:")
            st.write("  - Fill numeric with mean/median")
            st.write("  - Fill categorical with mode")

        if df.duplicated().sum() > 0:
            st.write("• Remove duplicate records")

        # ---------- Outlier Detection ----------
        st.subheader("Outlier Detection")

        numeric = df.select_dtypes(include=np.number)

        if not numeric.empty:
            outlier_count = 0
            for col in numeric.columns:
                Q1 = numeric[col].quantile(0.25)
                Q3 = numeric[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((numeric[col] < Q1-1.5*IQR) |
                            (numeric[col] > Q3+1.5*IQR)).sum()
                outlier_count += outliers

            st.write(f"Total Possible Outliers: {outlier_count}")
            st.write("Suggestion: Consider normalization or capping.")

        # ---------- ML Anomaly Detection ----------
        st.subheader("ML-Based Anomaly Detection")

        if not numeric.empty:
            model = IsolationForest(contamination=0.05)
            preds = model.fit_predict(numeric.fillna(0))

            df["Anomaly"] = preds
            st.write(df[df["Anomaly"] == -1].head())

        # ---------- Column Search ----------
        st.subheader("Search Data")

        column = st.selectbox("Select Column", df.columns)
        keyword = st.text_input("Search Value")

        if keyword:
            result = df[
                df[column].astype(str)
                .str.contains(keyword, case=False, na=False)
            ]
            st.dataframe(result)

        # ---------- Auto Report ----------
        st.subheader("Generate Report")

        report = f"""
        DATA QUALITY REPORT

        Rows: {df.shape[0]}
        Columns: {df.shape[1]}
        Missing Values: {df.isna().sum().sum()}
        Duplicate Rows: {df.duplicated().sum()}
        Quality Score: {score}/100
        """

        st.download_button(
            "Download Report",
            report,
            file_name="data_quality_report.txt"
        )



