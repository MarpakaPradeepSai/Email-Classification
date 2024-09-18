import streamlit as st
import joblib
import pandas as pd

# Load the saved model and vectorizer
model = joblib.load('limited_bernoulli_NB_TFIDF(0.991928).pkl')
vectorizer = joblib.load('limited_tfidf_vectorizer.pkl')

# Set up the Streamlit app
st.markdown("<h1 style='text-align: center;'>Spam/Ham Classifier</h1>", unsafe_allow_html=True)
st.write("Enter a message to classify it as spam or ham.")

# Text input for user message
user_input = st.text_area("Message")

# Add custom CSS for button styling
st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #007BFF;  /* Blue color */
        color: white !important;     /* Text color */
        border: none;                /* Remove border */
        padding: 10px 20px;         /* Padding */
        border-radius: 5px;         /* Rounded corners */
        cursor: pointer;             /* Pointer cursor on hover */
    }
    .stButton > button:hover {
        background-color: #0056b3;   /* Darker blue on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
st.sidebar.text("This app classifies messages as spam or ham using a Bernoulli Naive Bayes model.")
