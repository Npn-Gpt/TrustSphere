from models.nlp_models import sentiment_pipe, tokenizer, embedding_model
from transformers import pipeline
import torch
import os
import csv
import sqlite3
from datetime import datetime, timedelta
from typing import Optional
import security_utils
import random


# Spam detection model (prefer GPU if available)
zero_shot_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if torch.cuda.is_available() else -1
)

# Logging paths
LOG_FILE = os.path.join("logs", "analysis_log.csv")
DB_PATH = os.path.join("logs", "feedback.db")

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

def analyze_sentiment(text: str):
    result = sentiment_pipe(text)[0]
    return result['label'].lower(), float(result['score'])

def detect_spam(text: str):
    labels = ["spam", "not spam", "ai-generated", "human-written"]
    result = zero_shot_classifier(text, candidate_labels=labels)
    spam_score = 0.0
    is_spam = False

    for label, score in zip(result['labels'], result['scores']):
        if label == "spam":
            spam_score = score
            if score >= 0.1:
                is_spam = True

    return is_spam, round(spam_score, 3)

def compute_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embedding if isinstance(embedding, list) else embedding.tolist()

def explain_trust(sentiment: str, score: float, is_spam: bool, spam_score: float) -> (float, str):
    # Fuzzy trust base
    trust_score = 50.0
    reason = []

    # Spam penalty (soft, not hard cutoff)
    trust_score -= 30 * spam_score  # Instead of fixed 25.0 for spam
    if spam_score > 0.1:
        reason.append(f"⚠️ Spam-like content detected (confidence: {spam_score:.2f}).")

    # Sentiment influence (soft, not hard categories)
    if sentiment == "positive":
        trust_score += 20 * score
        reason.append("😊 Positive tone.")
    elif sentiment == "neutral":
        trust_score += 10 * score
        reason.append("😐 Neutral tone.")
    elif sentiment == "negative":
        trust_score -= 20 * score
        reason.append("😠 Negative tone.")

    # Clamp and add a little random fuzziness
    trust_score = max(0, min(100, trust_score + random.uniform(-3, 3)))
    return round(trust_score, 2), " ".join(reason)

def log_analysis(text, result: dict):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "timestamp", "text", "sentiment", "sentiment_score",
                "trust_score", "is_spam", "spam_confidence", "trust_reason", "facial_emotion", "audio_emotion"
            ])
        writer.writerow([
            datetime.now().isoformat(),
            security_utils.encrypt_data(text[:100]),
            result["sentiment"],
            result["sentiment_score"],
            result["trust_score"],
            result["is_spam"],
            result["spam_confidence"],
            result.get("trust_reason", ""),
            security_utils.encrypt_data(result.get("facial_emotion", "N/A")),
            result.get("audio_emotion", "N/A")
        ])

def init_feedback_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            text TEXT,
            sentiment TEXT,
            trust_score REAL,
            rating INTEGER,
            comment TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_feedback(text: str, sentiment: str, trust_score: float, rating: int, comment: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO feedback (timestamp, text, sentiment, trust_score, rating, comment)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        text[:100],
        sentiment,
        trust_score,
        rating,
        comment
    ))
    conn.commit()
    conn.close()

# Initialize feedback DB at import
init_feedback_db()

# --- Auto-deletion of logs older than 24 hours ---
def cleanup_old_logs(log_path=LOG_FILE, hours=24):
    if not os.path.exists(log_path):
        return
    import pandas as pd
    df = pd.read_csv(log_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = datetime.now() - timedelta(hours=hours)
    df = df[df["timestamp"] > cutoff]
    df.to_csv(log_path, index=False)

# Call cleanup at import
def _init_cleanup():
    try:
        cleanup_old_logs()
    except Exception:
        pass
_init_cleanup()

def evaluate_trust(text: str, facial_emotion: Optional[str] = None, audio_emotion: Optional[str] = None, audio_weight: Optional[float] = None):
    sentiment, sentiment_score = analyze_sentiment(text)
    is_spam, spam_confidence = detect_spam(text)
    embedding = compute_embedding(text)
    trust_score, trust_reason = explain_trust(sentiment, sentiment_score, is_spam, spam_confidence)

    # Fuzzy facial emotion adjustment
    if facial_emotion:
        if facial_emotion == "happy" and sentiment == "positive":
            trust_score += random.uniform(3, 7)
            trust_reason += " 😊 Facial emotion aligns with sentiment."
        elif facial_emotion in ["angry", "sad"] and sentiment == "positive":
            trust_score -= random.uniform(7, 13)
            trust_reason += " ⚠️ Facial emotion may contradict sentiment."
        elif facial_emotion == "neutral":
            trust_score += random.uniform(-2, 2)
            trust_reason += " 😐 Neutral facial emotion."
        trust_score = max(0, min(100, trust_score))

    # Fuzzy audio emotion adjustment with weight
    if audio_emotion:
        weight = audio_weight if audio_weight is not None else 0.3
        if audio_emotion == "calm" and sentiment == "positive":
            trust_score += random.uniform(3, 7) * weight
            trust_reason += f" 🎵 Calm audio tone supports positive sentiment (weight {weight:.2f})."
        elif audio_emotion in ["angry", "fear", "sad"] and sentiment == "positive":
            trust_score -= random.uniform(7, 13) * weight
            trust_reason += f" ⚠️ Audio tone may contradict positive sentiment (weight {weight:.2f})."
        elif audio_emotion == "angry":
            trust_score -= random.uniform(7, 13) * weight
            trust_reason += f" ⚠️ Angry tone detected in audio (weight {weight:.2f})."
        trust_score = max(0, min(100, trust_score))

    result = {
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "embedding": embedding,
        "trust_score": trust_score,
        "trust_reason": trust_reason,
        "is_spam": is_spam,
        "spam_confidence": spam_confidence,
        "facial_emotion": facial_emotion or "N/A",
        "audio_emotion": audio_emotion or "N/A",
        "audio_weight": audio_weight
    }
    log_analysis(text, result)
    return result