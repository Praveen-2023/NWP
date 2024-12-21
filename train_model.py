import re
import numpy as np
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# ============================
# Step 1: Data Preprocessing
# ============================

# Load dataset
def load_dataset(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def preprocess_text(text):
    text = re.sub('[^a-zA-Z0-9 \.]', '', text).lower()
    words = text.split()
    return words

def create_vocabulary(words):
    word_counts = Counter(words)
    vocab = sorted(word_counts.keys())
    word_to_id = {word: idx for idx, word in enumerate(vocab)}
    id_to_word = {idx: word for word, idx in word_to_id.items()}
    return vocab, word_to_id, id_to_word

def prepare_sequences(words, word_to_id, context_length):
    X, y = [], []
    for i in range(len(words) - context_length):
        X.append([word_to_id[word] for word in words[i:i + context_length]])
        y.append(word_to_id[words[i + context_length]])
    return np.array(X), np.array(y)

# Load and preprocess
text = load_dataset('trainingfile.txt')  # Ensure this path is correct
words = preprocess_text(text)
vocab, word_to_id, id_to_word = create_vocabulary(words)
context_length = 5
X, y = prepare_sequences(words, word_to_id, context_length)

# ============================
# Step 2: Build the MLP Model
# ============================

embedding_dim = 64
hidden_units = 1024

model = Sequential([
    Embedding(input_dim=len(vocab), output_dim=embedding_dim, input_length=context_length),
    Flatten(),
    Dense(hidden_units, activation='relu'),
    Dense(len(vocab), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=128, validation_split=0.1)  # Adjust epochs as needed
model.save('next_word_model.h5')  # Save model

# ============================
# Step 3: Save the vocabulary
# ============================

# Save word_to_id and id_to_word dictionaries using pickle
with open('word_to_id.pkl', 'wb') as f:
    pickle.dump(word_to_id, f)

with open('id_to_word.pkl', 'wb') as f:
    pickle.dump(id_to_word, f)

print("Model training completed and saved.")
