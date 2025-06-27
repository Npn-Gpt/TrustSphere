#   Table of Contents

1.  main.py: FastAPI Application Entry Point
    * [Explanation](#mainpy-fastapi-application-entry-point)
    * [Code Location](README.md#fastapi-backend-mainpy)
2.  app.py: Streamlit Web Application
    * [Explanation](#apppy-streamlit-web-application)
    * [Code Location](README.md#streamlit-frontend-apppy)
3.  schemas.py: Data Schemas
    * [Explanation](#schemaspy-data-schemas)
    * [Code Location](README.md#data-schemas)
4.  services/trust_engine.py: Trust Engine Service
    * [Explanation](#servicestrust_enginepy-trust-engine-service)
    * [Code Location](README.md#trust-engine)
5.  models/nlp_models.py: NLP Models
    * [Explanation](#modelsnlp_modelspy-nlp-models)
    * [Code Location](README.md#nlp-models)
6.  emotion_video.py: Real-Time Emotion Monitoring
    * [Explanation](#emotion_videopy-real-time-emotion-monitoring)
    * [Code Location](README.md#emotion-monitoring)
7.  Models Used in TrustSphere
    * [Explanation](#models-used-in-trustsphere)
    * [Code Location](README.md#nlp-models)
## main.py: FastAPI Application Entry Point

This file is the backend entry point for TrustSphere. It uses FastAPI to provide a REST API for text and emotion analysis.

**Key Functionality:**

* Initializes a FastAPI app and exposes endpoints:
    * `/` — Health check endpoint.
    * `/analyze` — Accepts POST requests with text and optional facial emotion, returns sentiment, trust score, spam score, and embeddings.
* Uses Pydantic schemas for request/response validation.
* Delegates analysis to the trust engine (services/trust_engine.py).

## app.py: Streamlit Web Application

This file is the main frontend for TrustSphere, built with Streamlit.

**Key Functionality:**

* Provides a user interface for entering text, viewing results, and interacting with emotion monitoring.
* Sends analysis requests to the FastAPI backend.
* Displays sentiment, trust score, spam score, facial emotion, and suggested responses.
* Integrates real-time emotion monitoring and multiple face detection (via emotion_video.py).
* Visualizes results using Plotly charts and dashboards.
* Allows feedback logging and journal entry.

## schemas.py: Data Schemas

Defines the structure of data exchanged between frontend and backend using Pydantic models:

* `TrustRequest`: Input text and optional facial emotion.
* `TrustResponse`: Sentiment, sentiment score, trust score, spam score, embedding, and facial emotion.

## services/trust_engine.py: Trust Engine Service

Contains the core logic for analyzing text and calculating trustworthiness.

**Key Functionality:**

* `analyze_sentiment(text)`: Uses RoBERTa to determine sentiment and confidence.
* `detect_spam(text)`: Uses BART zero-shot classification to detect spam and assign a spam confidence score.
* `compute_embedding(text)`: Uses DistilBERT to generate a semantic embedding for the text.
* `explain_trust(...)`: Calculates a trust score based on sentiment, spam, and emotion, and provides a human-readable reason.
* `evaluate_trust(text, facial_emotion)`: Orchestrates the above, integrates facial emotion (if provided), and logs the analysis.
* Feedback and analysis logging to SQLite and CSV.

## models/nlp_models.py: NLP Models

Loads and initializes the NLP models used for sentiment analysis and embeddings:

* **Sentiment Analysis**: RoBERTa ("cardiffnlp/twitter-roberta-base-sentiment-latest")
* **Embeddings**: DistilBERT ("distilbert-base-uncased")

## emotion_video.py: Real-Time Emotion Monitoring

Implements real-time facial emotion detection and monitoring using DeepFace and OpenCV.

**Key Functionality:**

* `EmotionMonitor` class: Handles webcam capture, emotion detection, and maintains an emotion timeline.
* Allows adjusting emotion weights for trust score calculation.
* Supports multiple face detection and emotion assignment.
* Provides methods for capturing emotion snapshots and generating visualizations.
* Integrates with the frontend for real-time feedback.

## Models Used in TrustSphere

TrustSphere uses the following pre-trained models:

* **RoBERTa** (for Sentiment Analysis): "cardiffnlp/twitter-roberta-base-sentiment-latest"
* **DistilBERT** (for Embeddings): "distilbert-base-uncased"
* **BART** (for Spam Detection): "facebook/bart-large-mnli"
* **DeepFace** (for Emotion Detection): Uses TensorFlow backend, supports multiple emotions and faces

## Summary

TrustSphere is a modular, AI-powered platform for analyzing the trustworthiness of text by combining:
- NLP-based sentiment and spam detection
- Semantic embeddings
- Real-time facial emotion analysis
- Interactive dashboards and feedback logging

The architecture separates frontend (Streamlit), backend (FastAPI), NLP models, emotion monitoring, and trust logic for maintainability and extensibility.
