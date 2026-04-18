<div align="center">
  <h1>🧠 SentimentAI: Deep Neural Sentiment Analysis</h1>
  <p><i>Comparing Basic LSTM and Bahdanau Attention for Complex Linguistic Constructs</i></p>
  
  [![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-Legacy_Keras-orange.svg)](https://tensorflow.org)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
  [![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://docker.com)
</div>

---

## 📖 Project Overview

This repository hosts a Deep Learning-based **Sentiment Analysis API and Interactive UI Dashboard**. It provides an environment to explore, test, and compare the performance of two distinct neural network architectures:
1. **Standard LSTM**
2. **LSTM enhanced with Bahdanau Attention**

The project focuses specifically on evaluating how well these models handle **difficult linguistic phenomena**, such as *negation*, by leveraging **LIME (Local Interpretable Model-agnostic Explanations)** to visually break down and explain the inner mathematical workings of the models.

---

## ✨ Key Features

- 🧠 **Dual Inference Engine**: Perform real-time text analysis using either a classic sequential LSTM or an Attention-based architecture.
- 🔍 **LIME Interpretability**: Understand *exactly why* the model made its decision. Visualizes the positive and negative contributions of each individual word to the final score.
- ⚡ **Modern FastAPI Backend**: High-performance, asynchronous API serving the TensorFlow models.
- 🐳 **Docker Ready**: Fully containerized environment ensuring compatibility with older TensorFlow/Keras `.pb` formats without dependency management headaches.

---

## 🔬 Architectural Comparison: The "Negation" Challenge

Standard LSTMs process text sequentially and compress the entire context into a single hidden state vector. This informational bottleneck often causes the model to "forget" earlier words or fail to effectively link dependent modifiers (e.g., "not") to their targets (e.g., "good").

**Bahdanau Attention** fundamentally solves this problem by allowing the model to dynamically "focus" on specific, relevant parts of the sentence when making its final prediction, instead of relying on a single static state.

### 🖼️ Case Study: The phrase `"this match is not good"`

We tested both models on a strict negation phrase to observe how structural memory influences predictive logic.

#### ❌ Model 1: Basic LSTM (Failure Instance)
The standard LSTM struggles to maintain the sequential influence of the negation. It observes the highly polarized word **"good"** and incorrectly predicts a positive outcome, as the weight of "good" overpowers the preceding "not".
- **Prediction:** 🟢 `POSITIVE` (Score: 0.7021 | 70% Confidence)
- **LIME Explanation:**
<p align="center">
  <img width="826" height="704" alt="lstm" src="https://github.com/user-attachments/assets/4e13ce2f-a561-4580-a5a4-ab29f02891d1" />
</p>

#### ✅ Model 2: LSTM + Bahdanau Attention (Success Instance)
The Attention mechanism successfully identifies the linguistic relationship. It focuses intensely on **"not"**, recognizing structurally that it reverses the polarity of the sentence, regardless of the word "good".
- **Prediction:** 🔴 `NEGATIVE` (Score: 0.0271 | 97% Confidence)
- **LIME Explanation:**
<p align="center">
<img width="824" height="689" alt="lstm_att" src="https://github.com/user-attachments/assets/9cc3dbcf-3290-4e3a-81ad-7b44957d9386" />

</p>

---

## 🚀 Installation & Usage

### 🐳 Option 1: Docker (Highly Recommended)

The simplest way to run this application is by downloading the pre-built Docker image from Docker Hub. This resolves any TensorFlow versioning conflicts automatically.

```bash
docker run -p 8000:8000 abdelalikholty/sentiment-app:latest
```
*(If you wish to build the image manually from source: `docker build -t sentiment-app .` and then run it).*

Navigate to **`http://localhost:8000`** in any web browser to access the interactive dashboard.

### 🐍 Option 2: Local Python Environment

1. Ensure you have **Python 3.12** installed.
2. Initialize and activate a virtual environment.
3. Install the required dependencies (ensure `tf-keras` is installed for legacy model support):
```bash
pip install -r requirements.txt
```
4. Start the FastAPI Uvicorn server:
```bash
python app.py
```
Navigate to **`http://localhost:8000`**.

---
*Powered by Deep Learning & Interpretability Research.*
