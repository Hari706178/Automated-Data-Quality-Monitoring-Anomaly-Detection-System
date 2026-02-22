import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="RL Data Quality System", layout="wide")

st.title("🤖 Automated Data Quality Monitoring + RL Recommendation System")

# ================= DATA LOADER =================
def load_dataset(uploaded_file):
    try:
        name = uploaded_file.name.lower()

        if name.endswith(".csv"):
            try:
                return pd.read_csv(uploaded_file)
            except Exception:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding="latin1")

        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)

    except Exception as e:
        st.error(f"Loading error: {e}")
        return None

# ================= DATA QUALITY SCORE =================
def data_quality_score(df):
    missing = df.isna().sum().sum()
    duplicates = df.duplicated().sum()

    score = 100
    score -= (missing / (df.shape[0]*df.shape[1])) * 50
    score -= (duplicates / df.shape[0]) * 50

    return round(max(score, 0), 2)

# ================= RL MODEL (Q-LEARNING STYLE SIMULATION) =================
Q_table = {
    "good": [0, 0, 0],
    "average": [0, 0, 0],
    "poor": [0, 0, 0]
}

actions = [
    "Fill Missing Values",
    "Remove Duplicates",
    "Normalize / Scale Data"
]

def rl_recommendation(score):
    if score > 80:
        state = "good"
    elif score > 50:
        state = "average"
    else:
        state = "poor"

    reward = np.random.rand(3)
    Q_table[state] = list(reward)

    best_action = actions[np.argmax(Q_table[state])]
    return state, best_action

# ================= STREAMLIT APP =================
file = st.file_uploader("Upload Dataset", type=["csv", "xlsx", "xls"])

if file:
    df = load_dataset(file)

    if df is not None:
        st.success("Dataset Loaded Successfully")

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isna().sum().sum())

        score = data_quality_score(df)
        st.subheader("Data Quality Score")
        st.metric("Score", f"{score}/100")

        state, recommendation = rl_recommendation(score)

        st.subheader("Reinforcement Learning Recommendation")
        st.write(f"Dataset State: **{state.upper()}**")
        st.success(f"Recommended Action: {recommendation}")

        # ================= Statistical Anomaly Detection (IQR) =================
        numeric = df.select_dtypes(include=np.number)

        if not numeric.empty:
            st.subheader("Anomaly Detection (IQR Method)")
            anomaly_count = 0
            anomaly_index = set()

            for col in numeric.columns:
                Q1 = numeric[col].quantile(0.25)
                Q3 = numeric[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                outliers = numeric[(numeric[col] < lower) | (numeric[col] > upper)]
                anomaly_count += len(outliers)
                anomaly_index.update(outliers.index)

            st.write(f"Total Potential Anomalies Detected: {anomaly_count}")

            if anomaly_index:
                st.dataframe(df.loc[list(anomaly_index)].head())

        # ================= Search Feature =================
        st.subheader("Search Dataset")
        column = st.selectbox("Select Column", df.columns)
        keyword = st.text_input("Search Value")

        if keyword:
            filtered = df[
                df[column].astype(str)
                .str.contains(keyword, case=False, na=False)
            ]
            st.dataframe(filtered)

        # ================= Report =================
        report = f"""
        DATA QUALITY REPORT

        Rows: {df.shape[0]}
        Columns: {df.shape[1]}
        Missing Values: {df.isna().sum().sum()}
        Duplicate Rows: {df.duplicated().sum()}
        Quality Score: {score}/100
        RL Recommendation: {recommendation}
        """

        st.download_button(
            "Download Report",
            report,
            file_name="data_quality_report.txt"
        )



