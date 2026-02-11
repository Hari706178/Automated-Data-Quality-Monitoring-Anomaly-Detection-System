import streamlit as st
import pandas as pd
import chardet
import io

st.set_page_config(page_title="Universal Data Analyzer", layout="wide")

st.title("📊 Universal Data Analysis App")
st.write("Upload ANY CSV or Excel dataset safely — no encoding errors!")

# ---------- Universal Dataset Loader ----------
def load_dataset(uploaded_file):
    try:
        file_name = uploaded_file.name.lower()

        # CSV Handling with encoding detection
        if file_name.endswith(".csv"):
            try:
                return pd.read_csv(uploaded_file)

            except UnicodeDecodeError:
                uploaded_file.seek(0)
                rawdata = uploaded_file.read(100000)
                encoding = chardet.detect(rawdata)["encoding"]
                uploaded_file.seek(0)

                return pd.read_csv(
                    io.BytesIO(uploaded_file.read()),
                    encoding=encoding,
                    low_memory=False
                )

        # Excel Handling
        elif file_name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)

        else:
            st.error("Unsupported file format.")
            return None

    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


# ---------- File Upload ----------
file = st.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx", "xls"]
)

if file:
    df = load_dataset(file)

    if df is not None:

        st.success("✅ Dataset Loaded Successfully!")

        # ---------- Dataset Info ----------
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isna().sum().sum())

        # ---------- Column Search ----------
        st.subheader("🔎 Search Column")
        column = st.selectbox("Select Column", df.columns)

        keyword = st.text_input("Search Value")

        if keyword:
            filtered = df[
                df[column]
                .astype(str)
                .str.contains(keyword, case=False, na=False)
            ]
            st.dataframe(filtered)

        # ---------- Basic Stats ----------
        st.subheader("Statistical Summary")
        st.write(df.describe(include="all"))

        # ---------- Missing Values ----------
        st.subheader("Missing Values Analysis")
        st.write(df.isnull().sum())

        # ---------- Correlation ----------
        numeric_df = df.select_dtypes(include="number")
        if not numeric_df.empty:
            st.subheader("Correlation Matrix")
            st.dataframe(numeric_df.corr())


