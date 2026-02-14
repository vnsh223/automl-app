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
    if st.button("Train Model"):
        X = df.drop(columns=[target])
        y = df[target]
    if y.dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)

    model = RandomForestClassifier()
    model.fit(X, y)

    st.success("Model Trained Successfully")

    # Convert categorical columns into numbers
        X = pd.get_dummies(X)

    # If target is text → convert it also
    if st.button("Train Model"):

    X = df.drop(columns=[target])
    y = df[target]

    # Convert categorical columns into numbers

    st.success("Model Trained Successfully")
    st.write("Training started... Please wait ⏳")

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    st.success("Model Training Completed ✅")
    st.write(f"Accuracy: {accuracy:.2f}")
