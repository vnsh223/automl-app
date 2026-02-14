import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("🚀 AutoML Web Application")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:

    # Read dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Select target column
    target = st.selectbox("Select Target Column", df.columns)

    # Train Button
    if st.button("Train Model"):

        # Split features and target
        X = df.drop(columns=[target])
        y = df[target]

        # Convert categorical columns in X
        X = pd.get_dummies(X)

        # Convert categorical target column
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train Model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Accuracy
        accuracy = model.score(X_test, y_test)

        st.success("✅ Model Trained Successfully")
        st.write(f"🎯 Accuracy: {accuracy * 100:.2f}%")
