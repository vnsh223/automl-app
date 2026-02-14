import streamlit as st
import pandas as pd
from supervised.automl import AutoML

st.title("🚀 AutoML Web Application")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("### Dataset Preview")
    st.write(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    if st.button("Train Model"):

        X = df.drop(columns=[target])
        y = df[target]

        st.write("Training started... Please wait ⏳")

        automl = AutoML(mode="Explain")
        automl.fit(X, y)

        st.success("Model Training Completed ✅")

        leaderboard = automl.get_leaderboard()
        st.write("### Model Leaderboard")
        st.write(leaderboard)
