import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

        st.success("Model Training Completed ✅")

if st.button("Train Model"):

    X = df.drop(columns=[target])
    y = df[target]

    st.write("Training started... Please wait ⏳")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    st.success("Model Training Completed ✅")
    st.write("Model Accuracy:", accuracy)
