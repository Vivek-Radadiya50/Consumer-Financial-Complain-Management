import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model, tokenizer, and class names
model = load_model('model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_len = 500  # Same max_len used during model training
class_names_dict = {0: 'credit_card', 1: 'retail_banking', 2: 'credit_reporting', 3: 'mortgages_and_loans', 4: 'debt_collection'}

def predict_class(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    pred_probs = model.predict(padded_sequence)
    pred_label = np.argmax(pred_probs, axis=1)[0]
    return class_names_dict.get(pred_label, "Unknown class")

# Streamlit app layout
st.markdown("""
    <style>
    .text-box {
        width: 100%;
        max-width: 700px;
        height: 300px;
        margin: 0 auto;
    }
    .button {
        font-size: 18px;
        padding: 10px 20px;
        margin: 10px 0;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .display-text {
        font-size: 24px; /* Change this value to adjust the text size */
        font-weight: bold;
        color: #333;
    }
    .header-text {
        font-size: 32px; /* Change this value to adjust the text size */
        font-weight: bold;
        color: #007bff;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="header-text">Consumer Financial Protection Bureau</div>', unsafe_allow_html=True)

st.markdown('<div class="display-text">Please enter your complaint here:</div>', unsafe_allow_html=True)

input_text = st.text_area(
    "",
    height=300,
    placeholder="Enter text here...",
)

if st.button("Submit"):
    if input_text:
        with st.spinner("Processing..."):
            class_name = predict_class(input_text)
            st.markdown(f'<div class="display-text">Your concern department is: {class_name}</div>', unsafe_allow_html=True)
            st.button("Enter another query", use_container_width=True)
    else:
        st.markdown('<div class="display-text">Please enter some text for prediction.</div>', unsafe_allow_html=True)
