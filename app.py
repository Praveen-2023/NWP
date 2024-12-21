import numpy as np
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============================
# Load the trained model and vocabulary
# ============================

model = load_model('next_word_model.h5')

# Load word_to_id and id_to_word from saved pickle files
with open('word_to_id.pkl', 'rb') as f:
    word_to_id = pickle.load(f)

with open('id_to_word.pkl', 'rb') as f:
    id_to_word = pickle.load(f)

# ============================
# Preprocessing Functions
# ============================

def preprocess_text_input(text):
    tokens = text.lower().split()
    return tokens

def handle_oov(tokens, word_to_id):
    return [word_to_id.get(word, word_to_id.get('<UNK>', 0)) for word in tokens]

# ============================
# Streamlit Application Code
# ============================

# Streamlit Application
st.title("Next-Word Prediction using MLP")

# User inputs
user_input = st.text_input("Enter your text:")
k = st.slider("Number of words to predict:", 1, 10, 5)
context_length = st.slider("Context length:", 2, 10, 5)
embedding_dim = st.slider("Embedding dimension:", 16, 128, 64)
activation_function = st.selectbox("Activation function:", ['relu', 'tanh', 'sigmoid'])

if st.button("Predict"):
    if user_input.strip():
        # Process input and predict
        tokens = preprocess_text_input(user_input)
        input_seq = handle_oov(tokens[-context_length:], word_to_id)
        input_seq = pad_sequences([input_seq], maxlen=context_length)

        # Predict next words
        predictions = model.predict(input_seq)
        next_words = [id_to_word[idx] for idx in np.argsort(predictions[0])[-k:]]
        st.write("Predicted words:", next_words)
    else:
        st.warning("Please enter some text to predict.")
