# TrustSphere: AI-Powered Trust & Sentiment Analysis with Emotion Integration

**TrustSphere** is an advanced AI web platform that analyzes text for sentiment, spam, and trustworthiness, and integrates real-time facial emotion detection via webcam. It combines state-of-the-art NLP models, emotion AI, and interactive dashboards to provide a holistic **trust score** for any message or journal entry.

---

## 🚀 Features

* **Sentiment Analysis** using RoBERTa (Twitter-based model)
* **Text Embeddings** via DistilBERT
* **Spam Classification** via Zero-Shot BART
* **Facial Emotion Detection** using DeepFace (via webcam)
* **Trust Score Calculation** based on NLP + emotion + spam signal
* **Feedback Logging** with SQLite
* **Dynamic Visualization Dashboard** using Plotly and Streamlit
* **Multiple Face Detection** and emotion assignment
* **Real-time Emotion Monitoring** and timeline
* **Audio Emotion Analysis** and transcription

---

## 🧠 Technology Stack

* **Frontend**: Streamlit
* **Backend**: FastAPI + Uvicorn
* **NLP**: HuggingFace Transformers (RoBERTa, DistilBERT, BART)
* **Emotion**: OpenCV + DeepFace
* **Audio**: sounddevice, wavio, soundfile, librosa
* **Database**: SQLite for logs and feedback
* **Visualization**: Plotly, Pandas

---

## 📁 Project Structure

```
TrustSphereV2/
├── app.py                   # Streamlit frontend logic
├── main.py                  # Streamlit entry point
├── api.py                   # FastAPI backend
├── schemas.py               # Pydantic data models
├── emotion_video.py         # Real-time emotion monitoring
├── audio_analysis.py        # Audio analysis and emotion detection
├── services/
│   └── trust_engine.py      # Sentiment, spam, trust logic
├── models/
│   └── nlp_models.py        # Transformers setup
├── requirements.txt         # Dependencies
├── README.md                # Project intro
├── EXPLANATION.md           # Module-by-module breakdown
└── logs/                    # Logs and feedback storage
```

---

## ⚙️ Setup and Installation

1. **Clone the repository**:

```bash
git clone <repo_url>
cd TrustSphereV2
```

2. **Create a virtual environment**:

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

> 💡 First-time model loads may take a few minutes due to HuggingFace model downloads.

---

## 🚦 Running the Application

1. **Start the backend (FastAPI)**:

```bash
uvicorn api:app --reload
# Runs at: http://127.0.0.1:8000
```

2. **Start the frontend (Streamlit)**:

```bash
streamlit run main.py
```

---

## ✨ Usage Flow

1. Launch the app in your browser.
2. Use the sidebar to set sentiment and emotion weights.
3. Enter a message or journal entry.
4. Optionally activate **real-time emotion monitoring** or detect emotions from a static snapshot.
5. Optionally upload or record audio for emotion analysis.
6. Click **Analyze** to view:
   * Sentiment + confidence
   * Spam score
   * Facial emotion (if applicable)
   * Audio emotion (if applicable)
   * Trust score
   * GPT-style feedback & suggested response
7. Log entries to journal or send feedback.

---

## 📊 Visual Insights

* Real-time **emotion monitoring timeline**
* Emotion distribution bars
* Multiple face detection with selectable emotions
* Sentiment vs Trust scatter plots
* Time-series trust score tracking
* Audio emotion and transcription display

---

## 🛠️ Troubleshooting

* Ensure webcam access for DeepFace features.
* Use Python 3.8–3.10 (avoid 3.11+ due to TensorFlow incompatibility).
* Install Visual C++ Redistributables on Windows if TensorFlow or DeepFace DLL errors occur.
* If you encounter GPU/driver issues, set DeepFace to use CPU by default.
* For Apple Silicon, DeepFace may not be fully supported.
* For audio features, ensure your microphone is accessible and the following packages are installed: `sounddevice`, `wavio`, `soundfile`, `librosa`.

---

## 📚 See Also

* [EXPLANATION.md](EXPLANATION.md) — Full breakdown of each file and model

---

## 📃 License

MIT License — Feel free to use, modify, and share with credit.

---

## 🙌 Acknowledgments

* HuggingFace Transformers
* DeepFace
* Streamlit & FastAPI teams
* Plotly, OpenCV, SQLite
* sounddevice, wavio, soundfile, librosa
