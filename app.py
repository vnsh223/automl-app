import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title("🚀 AutoML Web Application")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Select target column
    target = st.selectbox("Select Target Column", df.columns)

    # Train button (ONLY ONE BUTTON)
    if st.button("Train Model", key="train_model_btn"):

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
        st.write(f"Model Accuracy: {accuracy:.2f}")
