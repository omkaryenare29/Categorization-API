# Import necessary libraries
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import numpy as np
import re
import pickle
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Download NLTK data
nltk.download("stopwords")
nltk.download("punkt_tab")
nltk.download("wordnet")

# Load trained models
lstm_model = tf.keras.models.load_model(r"models/lstm_model.h5")  # LSTM model

with open(r"models/dt_model.pkl", "rb") as f:
    decision_tree_model = pickle.load(f)  # Decision Tree model

with open(r"models/rf_model.pkl", "rb") as f:
    random_forest_model = pickle.load(f)  # Random Forest model

# Load tokenizers & label encoders
with open(r"tokenizer_lstm.pkl", "rb") as handle:
    tokenizer_lstm = pickle.load(handle)

with open(r"label_encoder_lstm.pkl", "rb") as handle:
    label_encoder_lstm = pickle.load(handle)

with open(r"vectorizer_dtrf.pkl", "rb") as handle:
    vectorizer_dtrf = pickle.load(handle)   


with open(r"label_encoder_dtrf.pkl", "rb") as handle:
    label_encoder_dtrf = pickle.load(handle)

# Initialize FastAPI
app = FastAPI()

# Define input format
class EmailInput(BaseModel):
    text: str  # Combined subject and body

# Preprocessing function
def preprocess_sample(text):
    text = text.lower()  
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    tokens = word_tokenize(text)  
    tokens = [word for word in tokens if word not in stopwords.words("english")]  
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  
    clean_text = " ".join(tokens)
    return clean_text

# Majority Voting Function
def majority_vote(predictions):
    counter = Counter(predictions)
    return counter.most_common(1)[0][0]  

# API Endpoint for Predictions
@app.post("/predict")
def predict(email: EmailInput):
    clean_text = preprocess_sample(email.text)

    # 🔹 LSTM Prediction
    sample_seq = pad_sequences(tokenizer_lstm.texts_to_sequences([clean_text]), maxlen=150)  
    pred_lstm = np.argmax(lstm_model.predict(sample_seq), axis=-1)[0]  # Ensure proper shape
    category_lstm = label_encoder_lstm.inverse_transform([pred_lstm])[0]

    print(vectorizer_dtrf)
    print(f"Cleaned text: {clean_text}")

    sample_tfidf = vectorizer_dtrf.transform([clean_text])  
    pred_dtr = decision_tree_model.predict(sample_tfidf.toarray())[0]  # Convert sparse to dense
    category_dtr = label_encoder_dtrf.inverse_transform([pred_dtr])[0]  

    pred_rf = random_forest_model.predict(sample_tfidf.toarray())[0]  # Convert sparse to dense
    category_rf = label_encoder_dtrf.inverse_transform([pred_rf])[0]  

    # 🔹 Majority Voting
    final_category = majority_vote([category_lstm, category_dtr, category_rf])

    return {
        "LSTM": category_lstm,
        "DecisionTree": category_dtr,
        "RandomForest": category_rf,
        "Final_Category": final_category,
    }

# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
