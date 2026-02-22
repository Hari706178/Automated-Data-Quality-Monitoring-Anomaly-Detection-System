import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Data Monitoring RL System", layout="wide")

st.title("🤖 AI Powered Data Monitoring & RL Cleaning Agent")

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

# ================= QUALITY SCORE =================
def quality_score(df):
    missing = df.isna().sum().sum()
    duplicates = df.duplicated().sum()
    score = 100
    score -= (missing / (df.shape[0]*df.shape[1])) * 50
    score -= (duplicates / df.shape[0]) * 50
    return round(max(score,0),2)

# ================= RL Q-LEARNING =================
actions = ["fill_missing","remove_duplicates","normalize"]
Q_table = {"good":np.zeros(3),"average":np.zeros(3),"poor":np.zeros(3)}
alpha = 0.5
gamma = 0.9

def get_state(score):
    if score > 80:
        return "good"
    elif score > 50:
        return "average"
    else:
        return "poor"


def q_learning_update(state, action_idx, reward, next_state):
    Q_table[state][action_idx] = Q_table[state][action_idx] + alpha * (
        reward + gamma * np.max(Q_table[next_state]) - Q_table[state][action_idx]
    )


def choose_action(state):
    return np.argmax(Q_table[state])

# ================= CLEANING FUNCTIONS =================
def auto_clean(df, action):
    df_clean = df.copy()
    if action == "fill_missing":
        for col in df_clean.columns:
            if df_clean[col].dtype in ["float64","int64"]:
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown", inplace=True)
    elif action == "remove_duplicates":
        df_clean = df_clean.drop_duplicates()
    elif action == "normalize":
        numeric = df_clean.select_dtypes(include=np.number)
        for col in numeric.columns:
            min_val = numeric[col].min()
            max_val = numeric[col].max()
            if max_val != min_val:
                df_clean[col] = (numeric[col] - min_val)/(max_val - min_val)
    return df_clean

# ================= FILE UPLOAD =================
file = st.file_uploader("Upload Dataset", type=["csv","xlsx","xls"])

if file:
    df = load_dataset(file)
    if df is not None:
        st.success("Dataset Loaded Successfully")
        st.dataframe(df.head())

        score_before = quality_score(df)
        state = get_state(score_before)

        st.subheader("📊 Data Quality Score")
        st.metric("Score", f"{score_before}/100")

        # Confidence Meter
        if score_before > 80:
            st.success("🟢 Ready for ML")
        elif score_before > 50:
            st.warning("🟡 Needs Cleaning")
        else:
            st.error("🔴 Not Ready")

        # RL Action
        action_idx = choose_action(state)
        recommended_action = actions[action_idx]
        st.subheader("🤖 RL Recommended Action")
        st.write(recommended_action)

        # Auto Clean Button
        if st.button("🚀 Auto Clean Dataset Using RL Recommendation"):
            df_clean = auto_clean(df, recommended_action)
            score_after = quality_score(df_clean)
            next_state = get_state(score_after)
            reward = score_after - score_before
            q_learning_update(state, action_idx, reward, next_state)

            st.subheader("✅ Cleaned Dataset Preview")
            st.dataframe(df_clean.head())

            st.subheader("📈 Quality Improvement")
            st.write(f"Before: {score_before}")
            st.write(f"After: {score_after}")
            st.write(f"Reward: {reward}")

            # Data Drift Detection
            st.subheader("📉 Data Drift Detection")
            numeric_before = df.select_dtypes(include=np.number)
            numeric_after = df_clean.select_dtypes(include=np.number)
            for col in numeric_before.columns:
                if col in numeric_after.columns:
                    mean_shift = numeric_after[col].mean() - numeric_before[col].mean()
                    var_shift = numeric_after[col].var() - numeric_before[col].var()
                    st.write(f"{col} → Mean Shift: {round(mean_shift,4)}, Variance Shift: {round(var_shift,4)}")

        # ================= DASHBOARD VISUALS =================
        st.subheader("📊 Missing Value Heatmap")
        fig, ax = plt.subplots()
        ax.imshow(df.isna(), aspect='auto')
        ax.set_title("Missing Value Map")
        st.pyplot(fig)

        st.subheader("📊 Correlation Heatmap")
        corr = df.select_dtypes(include=np.number).corr()
        fig2, ax2 = plt.subplots()
        cax = ax2.imshow(corr, cmap="coolwarm")
        fig2.colorbar(cax)
        ax2.set_xticks(range(len(corr.columns)))
        ax2.set_xticklabels(corr.columns, rotation=90)
        ax2.set_yticks(range(len(corr.columns)))
        ax2.set_yticklabels(corr.columns)
        st.pyplot(fig2)

        # Distribution Plot
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            st.subheader("📊 Distribution Plot")
            col_choice = st.selectbox("Select Column", numeric_cols)
            fig3, ax3 = plt.subplots()
            ax3.hist(df[col_choice].dropna(), bins=20)
            st.pyplot(fig3)

        # ================= DATA TYPE INTELLIGENCE =================
        st.subheader("🧠 Data Type Intelligence")
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    pd.to_numeric(df[col])
                    st.write(f"{col} → Numeric stored as Object (convert recommended)")
                except:
                    try:
                        pd.to_datetime(df[col])
                        st.write(f"{col} → Date column detected")
                    except:
                        pass

        # ================= FEATURE IMPORTANCE =================
        st.subheader("⭐ Feature Importance (Linear Approximation)")
        numeric = df.select_dtypes(include=np.number)
        if numeric.shape[1] > 1:
            target = numeric.columns[-1]
            X = numeric.drop(columns=[target]).values
            y = numeric[target].values
            if len(X) > 0:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                importance = pd.Series(np.abs(coef), index=numeric.columns[:-1])
                st.bar_chart(importance)

        # ================= DATA PROFILING =================
        st.subheader("📑 Data Profiling Summary")
        profile = pd.DataFrame({
            "Mean": numeric.mean(),
            "Median": numeric.median(),
            "Skewness": numeric.skew(),
            "Kurtosis": numeric.kurt(),
            "Unique Values": df.nunique()
        })
        st.dataframe(profile)

        # ================= SEARCH =================
        st.subheader("🔎 Search Dataset")
        column = st.selectbox("Search Column", df.columns)
        keyword = st.text_input("Search Keyword")
        if keyword:
            result = df[df[column].astype(str).str.contains(keyword, case=False, na=False)]
            st.dataframe(result)

        # ================= REPORT DOWNLOAD =================
        report = f"AI DATA QUALITY REPORT\nScore Before Cleaning: {score_before}\nState: {state}"
        st.download_button("Download Report", report, file_name="AI_Data_Report.txt")




