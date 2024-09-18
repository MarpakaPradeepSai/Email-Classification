import streamlit as st
import joblib
import pandas as pd

# Load the saved model and vectorizer
model = joblib.load('logistic_regression_CV(0.985650).pkl')
vectorizer = joblib.load('complete_cv_vectorizer.pkl')

# Set up the Streamlit app
st.title("Spam/Ham Classifier")
st.write("Enter a message to classify it as spam or ham.")

# Text input for user message
user_input = st.text_area("Message")

if st.button("Classify"):
    # Preprocess and transform the input
    input_data = vectorizer.transform([user_input])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display result
    if prediction[0] == 0:  # Check if the prediction is 0 (ham)
        st.success("This message is classified as: **Ham**")
    else:  # Otherwise, it must be 1 (spam)
        st.error("This message is classified as: **Spam**")

# Optional: Add more information or features
st.sidebar.header("About")
st.sidebar.text("This app classifies messages as spam or ham using a Logistic Regression model.")
