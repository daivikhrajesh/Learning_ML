import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.title("Titanic Dataset")

uploaded_file = st.file_uploader('Upload CSV', type='csv')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)

    target = st.selectbox("Select the target column", df.columns)
    features = st.multiselect("Select feature columns", df.columns, default=df.columns.difference([target]).tolist())

    if target and features:
        X = df[features]
        y = df[target]

        # Handle missing values
        X = X.fillna(X.mode().iloc[0])
        y = y.fillna(y.mode().iloc[0])

        # Encode categorical variables
        le = LabelEncoder()
        for column in X.select_dtypes(include=['object']).columns:
            X[column] = le.fit_transform(X[column])

        if y.dtype == 'object':
            y = le.fit_transform(y)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train the model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
