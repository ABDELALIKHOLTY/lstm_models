import os

# MUST BE SET BEFORE ANY KERAS IMPORT!
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import re
import string
import numpy as np
import tensorflow as tf
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
# Note: SpaCy and Negspacy are kept if LIME needs them for custom splitting, but simplified here.

# --- ATTENTION LAYER DEFINITION (Required for loading Attention Model) ---
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class BahdanauAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                               initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                               initializer="zeros")
        super().build(input_shape)

    def call(self, x, mask=None):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask, axis=-1)
            a = a * mask
            a = a / (K.sum(a, axis=1, keepdims=True) + K.epsilon())
            
        output = x * a
        return K.sum(output, axis=1)

    def compute_mask(self, input, mask):
        return None

    def get_config(self):
        config = super().get_config()
        return config

class TextPreprocessor:
    def __init__(self):
        # We assume the models were trained using basic NLTK processes 
        self.negation_list = {
            'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'none',
            "don't", 'dont', "doesn't", 'doesnt', "didn't", 'didnt',
            "can't", 'cant', "couldn't", 'couldnt', "shouldn't", 'shouldnt',
            "isn't", 'isnt', "aren't", 'arent', "wasn't", 'wasnt', "weren't", 'werent',
            "hasn't", 'hasnt', "haven't", 'havent', "hadn't", 'hadnt',
            "won't", 'wont', "wouldn't", 'wouldnt', 'aint', 'never'
        }
        try:
            # Protect negation words from being stripped during preprocessing
            self.stop_words = set(stopwords.words('english')) - self.negation_list
        except:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english')) - self.negation_list
            
        self.lm = WordNetLemmatizer()
        self.punctuation = string.punctuation
        
    def clean_text(self, text):
        # 1. Lowercase & Punctuation
        text = "".join([char.lower() for char in text if char not in self.punctuation])
        
        # 2. Tokenize and handle negations (e.g. "not good" -> "not_good")
        tokens = text.split()
        new_tokens = []
        skip_next = False
        
        for i in range(len(tokens)):
            if skip_next:
                skip_next = False
                continue
            
            if tokens[i] in self.negation_list and i + 1 < len(tokens):
                new_tokens.append(f"{tokens[i]}_{tokens[i+1]}")
                skip_next = True
            else:
                new_tokens.append(tokens[i])
                
        return " ".join(new_tokens)

class DualLSTMAnalyzer:
    def __init__(self, 
                 simple_lstm_path="./Sentiment_Classifier_lstm",
                 attention_lstm_path="./Sentiment_Classifier_lstm_attention"):
        
        self.preprocessor = TextPreprocessor()
        self.simple_model = None
        self.attention_model = None
        
        # Load Models using tf.keras.models.load_model (with Legacy Keras support)
        try:
            print("[->] Loading LSTM (Simple) from Keras SavedModel...")
            self.simple_model = tf.keras.models.load_model(simple_lstm_path)
            print("[OK] LSTM (Simple) Loaded")
        except Exception as e:
            print(f"[X] Failed to load Simple LSTM: {e}")
            
        try:
            print("[->] Loading LSTM + Attention from Keras SavedModel...")
            self.attention_model = tf.keras.models.load_model(
                attention_lstm_path,
                custom_objects={'BahdanauAttention': BahdanauAttention}
            )
            print("[OK] LSTM + Attention Loaded")
        except Exception as e:
            print(f"[X] Failed to load Attention LSTM: {e}")
    
    def _predict_with_model(self, text, model):
        """Make prediction with a Keras model"""
        if model is None:
            return 0.5
        
        try:
            # Use model.predict() with numpy array (like the notebook)
            prediction = model.predict(np.array([text]), verbose=0)
            
            # Extract scalar value from output
            if isinstance(prediction, np.ndarray):
                value = float(prediction[0][0]) if prediction.ndim > 1 else float(prediction[0])
            else:
                value = float(prediction)
            
            # Clamp to [0, 1] range
            return max(0.0, min(1.0, value))
        except Exception as e:
            print(f"[!] Prediction error: {e}")
            return 0.5

    def analyze(self, text, model_type="lstm"):
        """Analyze text with selected LSTM model and return detailed LIME explanations
        
        Args:
            text: Input text to analyze
            model_type: 'lstm' for simple LSTM or 'attention' for LSTM with Attention
        """
        # Clean text for neural models
        neural_input = self.preprocessor.clean_text(text)
        
        # Select model based on model_type
        if model_type == "attention":
            selected_model = self.attention_model
            model_name = "LSTM + Attention"
        else:  # default to lstm
            selected_model = self.simple_model
            model_name = "LSTM"
        
        # Get prediction and LIME explanation
        score = self._predict_with_model(neural_input, selected_model)
        lime_explanation = self.explain_single(text, selected_model, model_name)
        
        return {
            "text": text,
            "model_used": model_name,
            "score": score,
            "sentiment": "POSITIVE" if score > 0.5 else "NEGATIVE",
            "confidence": max(score, 1 - score),
            "lime_data": lime_explanation
        }

    def explain_single(self, text, model, model_name):
        """Generate LIME explanation for a single model"""
        
        # Return empty explanation if model is None
        if model is None:
            print(f"[!] Model {model_name} is None, returning empty explanation")
            return {
                "model_name": model_name,
                "error": "Model failed to load",
                "word_weights": [],
                "top_words_positive": [],
                "top_words_negative": [],
                "confidence": 0
            }
        
        from lime.lime_text import LimeTextExplainer
        explainer = LimeTextExplainer(
            class_names=['NEGATIVE', 'POSITIVE'],
            split_expression=r'\W+',
            bow=True
        )
        
        # Closure to capture model in predict_proba
        model_ref = model  # Ensure we capture the model reference
        
        def predict_proba(texts):
            """Probability prediction for LIME"""
            if model_ref is None:
                # If model is None, return neutral scores
                return np.array([[0.5, 0.5]] * len(texts))
            
            probas = []
            for t in texts:
                c = self.preprocessor.clean_text(t)
                try:
                    score = self._predict_with_model(c, model_ref)
                except Exception as e:
                    print(f"[!] Prediction error in LIME: {e}")
                    score = 0.5
                probas.append([1-score, score])
            return np.array(probas)
        
        try:
            # Run LIME with reduced samples for speed
            exp = explainer.explain_instance(text, predict_proba, num_features=10, num_samples=50)
            
            return {
                "model_name": model_name,
                "word_weights": exp.as_list(),
                "top_words_positive": [w for w, score in exp.as_list() if score > 0],
                "top_words_negative": [w for w, score in exp.as_list() if score < 0],
                "confidence": exp.score}
        except Exception as e:
            print(f"[!] LIME explanation error: {e}")
            return {
                "model_name": model_name,
                "error": str(e),
                "word_weights": [],
                "top_words_positive": [],
                "top_words_negative": [],
                "confidence": 0
            }

app = FastAPI()
analyzer = DualLSTMAnalyzer()

class Req(BaseModel):
    text: str
    model_type: str = "lstm"  # 'lstm' or 'attention'

@app.get("/", response_class=HTMLResponse)
def index():
    return open("index.html", encoding="utf-8").read()

@app.post("/api/predict")
def predict(req: Req):
    return analyzer.analyze(req.text, model_type=req.model_type)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
