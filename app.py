import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Quality Monitor", layout="wide")

st.title("📊 Automated Data Quality Monitoring & Anomaly Detection System")

# =========================
# Upload Dataset
# =========================
file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file:

    df = pd.read_csv(file)

    # =========================
    # GLOBAL SEARCH BAR
    # =========================
    st.sidebar.header("🔎 Search Dataset")

    search_term = st.sidebar.text_input(
        "Search any value (name, number, etc.)"
    )

    if search_term:
        mask = df.astype(str).apply(
            lambda row: row.str.contains(search_term, case=False).any(),
            axis=1
        )
        df = df[mask]
        st.success(f"Showing results for: {search_term}")

    # =========================
    # DATASET OVERVIEW
    # =========================
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Cells", df.isnull().sum().sum())

    st.dataframe(df.head())

    # =========================
    # DATA TYPES
    # =========================
    st.header("Column Data Types")
    st.write(df.dtypes)

    # =========================
    # MISSING VALUES
    # =========================
    st.header("Missing Values Analysis")

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:
        st.bar_chart(missing)
    else:
        st.success("No Missing Values")

    # =========================
    # DUPLICATES
    # =========================
    st.header("Duplicate Records")

    dup_count = df.duplicated().sum()
    st.metric("Duplicate Rows", dup_count)

    # =========================
    # CONSTANT COLUMNS
    # =========================
    st.header("Constant Columns")

    const_cols = [c for c in df.columns if df[c].nunique() == 1]

    if const_cols:
        st.write(const_cols)
    else:
        st.success("No Constant Columns")

    # =========================
    # NUMERIC ANALYSIS
    # =========================
    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:

        # OUTLIERS
        st.header("Outlier Detection (IQR)")

        outliers = {}

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            count = len(df[
                (df[col] < Q1 - 1.5 * IQR) |
                (df[col] > Q3 + 1.5 * IQR)
            ])

            outliers[col] = count

        st.write(outliers)

        # ANOMALY DETECTION
        st.header("Anomaly Detection (ML)")

        model = IsolationForest(contamination=0.05)
        df["Anomaly"] = model.fit_predict(
            df[numeric_cols].fillna(0)
        )

        anomaly_count = (df["Anomaly"] == -1).sum()
        st.metric("Anomalies Found", anomaly_count)

        st.bar_chart(df["Anomaly"].value_counts())

        # DISTRIBUTION PLOT
        st.header("Data Distribution")

        col_selected = st.selectbox("Select Numeric Column", numeric_cols)

        fig, ax = plt.subplots()
        df[col_selected].hist(ax=ax)
        st.pyplot(fig)

    # =========================
    # SUMMARY STATS
    # =========================
    st.header("Statistical Summary")
    st.dataframe(df.describe(include="all"))

    # =========================
    # DOWNLOAD DATA
    # =========================
    st.header("Download Processed Dataset")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV",
        csv,
        "processed_data.csv",
        "text/csv"
    )

else:
    st.info("Upload a CSV file to start analysis.")
