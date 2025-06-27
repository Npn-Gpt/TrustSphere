import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # This will suppress TF warnings
os.environ['PYTHONWARNINGS'] = 'ignore'  # Suppress Python warnings
import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings

# Add current directory to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure asyncio
import asyncio
import nest_asyncio
nest_asyncio.apply()
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Import other dependencies
import sqlite3
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import time
import streamlit as st
import audio_analysis
import translation_utils
import security_utils
import sounddevice as sd
import tempfile
import soundfile as sf
import numpy as np
import requests
import cv2
import wavio

# Initialize event loop
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Import local modules after environment setup
from emotion_video import get_emotion_monitor, capture_emotion_snapshot
from services.trust_engine import save_feedback

FASTAPI_URL = "http://127.0.0.1:8001/analyze"

# --- Journal Persistence ---
JOURNAL_CSV = os.path.join("logs", "journal.csv")

def load_journal():
    if os.path.exists(JOURNAL_CSV):
        try:
            df = pd.read_csv(JOURNAL_CSV)
            return df.to_dict(orient="records")
        except Exception:
            return []
    return []

def append_journal_entry(entry):
    import csv
    file_exists = os.path.isfile(JOURNAL_CSV)
    with open(JOURNAL_CSV, mode="a", newline='', encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp", "text", "trust", "emotion", "audio_emotion", "detected_language"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(entry)

def show_trust_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Trust Score", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "red"},
                {'range': [40, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "green"}
            ],
        }))
    st.plotly_chart(fig, use_container_width=True)

def display_trust_factors(sentiment_score, spam_score, emotion_weight):
    st.subheader("📊 Trust Score Breakdown")
    st.bar_chart({
        "Sentiment Confidence": sentiment_score * 100,
        "Spam Likelihood": spam_score * 100,
        "Emotion Influence": emotion_weight * 100
    })

def suggest_response(sentiment, emotion):
    if sentiment == "negative" or emotion in ["sad", "angry"]:
        return "It seems like you're going through something. Remember, you're not alone. 💙"
    elif sentiment == "positive" and emotion == "happy":
        return "That's great to hear! Keep up the positivity! 🌟"
    return "Thanks for sharing. Would you like to tell us more?"

def sentiment_emoji(sentiment):
    return {"positive": "😊", "neutral": "😐", "negative": "😠"}.get(sentiment.lower(), "❓")


def emotion_emoji(emotion):
    mapping = {
        "happy": "😊",
        "sad": "😢",
        "angry": "😠",
        "neutral": "😐",
        "fear": "😨",
        "surprise": "😮",
        "disgust": "🤢",
        "unknown": "❓"
    }
    return mapping.get(emotion.lower(), "❓")


def interpret_trust(trust_score):
    if trust_score >= 75:
        return "🟢 High Trust: Very reliable and emotionally positive."
    elif trust_score >= 40:
        return "🟡 Moderate Trust: Neutral or balanced tone."
    else:
        return "🔴 Low Trust: Negative or emotionally charged."


def load_custom_css():

    try:
        with open("css/styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        # Fallback CSS if file not found
        st.markdown("""
        <style>
        h1 {
            color: #4CAF50;
            font-family: 'Segue UI', sans-serif;
            text-align: center;
        }

        div.stButton > button {
            background: linear-gradient(to right, #007acc, #000000);
            color: white;
            font-size: 16px;
            border-radius: 10px;
            padding: 0.5em 1em;
            transition: 0.2s ease-in-out;
        }

        div.stButton > button:hover {
            background-color: #005f99;
            transform: scale(1.02);
        }
        </style>
        """, unsafe_allow_html=True)


def display_results(result):
    sentiment = result['sentiment']
    sentiment_score = result['sentiment_score']
    trust_score = result['trust_score']
    embedding = result['embedding']
    emotion = result.get('facial_emotion', 'N/A')  # ✅ fixed to avoid tuple error
    spam_score = result['spam_confidence']

    st.success("✅ Analysis Completed")

    st.markdown("---")
    st.markdown(f"**🧠 Sentiment:** `{sentiment}`")
    st.markdown(f"**📊 Confidence:** `{sentiment_score:.2f}`")
    st.markdown(f"**🛡️ Trust Score:** `{trust_score:.2f}`")
    st.markdown(f"**🔍 Reason:** _{result.get('trust_reason', 'N/A')}_")
    st.markdown("---")

    show_trust_gauge(trust_score)
    display_trust_factors(sentiment_score, spam_score, st.session_state.get("emotion_weight", 0.3))

    if result['is_spam']:
        st.warning(f"🚨 Spam detected (Confidence: `{spam_score:.2f}`)")
    else:
        st.success("✅ Not detected as spam")

    # Show a GPT-like suggested response
    if sentiment or emotion:
        suggestion = suggest_response(sentiment, emotion)
        st.markdown(f"### 💬 Suggested Response\n> {suggestion}")


def display_emotion_monitoring():
    st.subheader("🎭 Real-time Emotion Monitoring")

    # Get the singleton monitor instance
    monitor = get_emotion_monitor()

    # UI elements for controlling the monitor
    monitoring_col1, monitoring_col2 = st.columns([3, 1])

    with monitoring_col2:
        if not monitor.is_running:
            if st.button("▶️ Start Monitoring"):
                monitor.start_monitoring()
                st.rerun()
        else:
            if st.button("⏹️ Stop Monitoring"):
                monitor.stop_monitoring()
                st.rerun()

    with monitoring_col1:
        if monitor.is_running:
            st.success("✅ Emotion monitoring active")
        else:
            st.info("⏸️ Emotion monitoring inactive")

    # Add button to use current detected emotion for analysis
    if monitor.is_running:
        current_emotion, _ = monitor.get_current_emotion()
        if current_emotion:
            if st.button("Use this video emotion for trust analysis"):
                st.session_state.selected_facial_emotion = current_emotion
                st.success(f"Video emotion '{current_emotion}' set for trust analysis. Switch to the Analysis tab to proceed.")

    # Show emotion weights sliders if monitoring is active
    if monitor.is_running:
        st.write("##### Emotion Weight Adjustment")
        st.write("Adjust how much each emotion affects the trust score:")

        col1, col2 = st.columns(2)

        with col1:
            happy_weight = st.slider("😊 Happy weight", 0.0, 2.0, monitor.emotion_weights['happy'], 0.1)
            neutral_weight = st.slider("😐 Neutral weight", 0.0, 2.0, monitor.emotion_weights['neutral'], 0.1)
            surprise_weight = st.slider("😮 Surprise weight", 0.0, 2.0, monitor.emotion_weights['surprise'], 0.1)

        with col2:
            sad_weight = st.slider("😢 Sad weight", 0.0, 2.0, monitor.emotion_weights['sad'], 0.1)
            angry_weight = st.slider("😠 Angry weight", 0.0, 2.0, monitor.emotion_weights['angry'], 0.1)
            fear_weight = st.slider("😨 Fear weight", 0.0, 2.0, monitor.emotion_weights['fear'], 0.1)

        # Update weights
        monitor.set_emotion_weight('happy', happy_weight)
        monitor.set_emotion_weight('neutral', neutral_weight)
        monitor.set_emotion_weight('surprise', surprise_weight)
        monitor.set_emotion_weight('sad', sad_weight)
        monitor.set_emotion_weight('angry', angry_weight)
        monitor.set_emotion_weight('fear', fear_weight)

        # Display current emotion with frame
        current_emotion, current_frame = monitor.get_current_emotion()

        emotion_col1, emotion_col2 = st.columns([2, 1])

        with emotion_col1:
            if current_frame is not None:
                frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption=f"Current Emotion: {current_emotion}", use_container_width=True)

        with emotion_col2:
            if current_emotion:
                st.markdown(f"## {emotion_emoji(current_emotion)}")
                st.markdown(f"### Current: {current_emotion.capitalize()}")

                # Show emotion distribution
                emotion_dist = monitor.get_emotion_distribution(seconds=10)
                if emotion_dist:
                    # Create a horizontal bar chart for emotions
                    emotions = list(emotion_dist.keys())
                    percentages = list(emotion_dist.values())

                    fig = go.Figure(go.Bar(
                        x=percentages,
                        y=emotions,
                        orientation='h',
                        marker_color=['green' if e == 'happy' else
                                    'blue' if e == 'sad' else
                                    'red' if e == 'angry' else
                                    'gray' if e == 'neutral' else
                                    'purple' if e == 'fear' else
                                    'orange' if e == 'surprise' else
                                    'brown' for e in emotions]
                    ))

                    fig.update_layout(
                        title='Last 10 seconds emotion distribution',
                        xaxis_title='Percentage (%)',
                        height=300,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )

                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("### Waiting for face...")

        # Emotion timeline
        timeline_img = monitor.generate_emotion_timeline_image()
        if timeline_img is not None:
            pil_img = Image.fromarray(timeline_img)
            st.image(pil_img, caption="Emotion Timeline (Last 60 seconds)", use_container_width=True)

        # Emotion score for text analysis
        emotion_score = monitor.get_weighted_emotion_score()
        if emotion_score is not None:
            st.metric("Emotion Score", f"{emotion_score:.2f} (-1 to +1)")

    # Add a small delay to prevent overwhelming the system
    time.sleep(0.1)


def display_multi_face_detection():
    st.subheader("👥 Multiple Face Detection")

    if st.button("📸 Detect Multiple Faces"):
        try:
            with st.spinner("Detecting faces..."):
                faces_with_emotions, frame = capture_emotion_snapshot()

                # Display the resulting frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption=f"Detected {len(faces_with_emotions)} faces", use_container_width=True)

                # List detected emotions for selection
                if faces_with_emotions:
                    st.write("##### Select a face to use for analysis:")

                    # Create selector for multiple faces
                    face_options = [f"Face {i + 1}: {emotion}" for i, (_, _, _, _, emotion) in
                                    enumerate(faces_with_emotions)]
                    selected_face = st.radio("", face_options, index=0)

                    selected_index = int(selected_face.split(':')[0].replace('Face ', '')) - 1
                    selected_emotion = faces_with_emotions[selected_index][4]

                    if st.button("Use this video emotion for trust analysis (multi-face)"):
                        st.session_state.selected_facial_emotion = selected_emotion
                        st.success(f"Video emotion '{selected_emotion}' set for trust analysis. Switch to the Analysis tab to proceed.")
                    else:
                        st.session_state.selected_facial_emotion = selected_emotion
                        st.success(f"Selected emotion: {selected_emotion}")
                else:
                    st.warning("No faces detected")
        except Exception as e:
            st.error(f"Error detecting faces: {e}")


def display_visual_dashboard():
    st.subheader("📈 Sentiment & Emotion Analysis Dashboard")

    log_path = "logs/analysis_log.csv"
    if not os.path.exists(log_path):
        st.info("No analysis log found yet.")
        return

    try:
        # Optional: Remove rows with inconsistent columns (one-time cleanup)
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        header = lines[0]
        valid_lines = [line for line in lines if line.count(",") == header.count(",")]

        with open(log_path, "w", encoding="utf-8") as f:
            f.writelines(valid_lines)

        df = pd.read_csv(log_path, on_bad_lines="skip")  # Skip malformed lines

        if "facial_emotion" not in df.columns:
            st.warning("Facial emotion data not found in logs.")
            return

        df = df[df["facial_emotion"] != "N/A"]

        if df.empty:
            st.info("No emotion-augmented records yet.")
            return

        # Scatter plot
        st.markdown("### 🎯 Sentiment vs Trust Score by Facial Emotion")
        fig_scatter = px.scatter(
            df,
            x="sentiment_score",
            y="trust_score",
            color="facial_emotion",
            hover_data=["text", "trust_reason"],
            title="Sentiment vs Trust Score",
            labels={"sentiment_score": "Sentiment Confidence", "trust_score": "Trust Score"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Bar chart
        st.markdown("### 📊 Frequency of Facial Emotions by Sentiment")
        emotion_counts = df.groupby(["sentiment", "facial_emotion"]).size().reset_index(name="count")
        fig_bar = px.bar(
            emotion_counts,
            x="sentiment",
            y="count",
            color="facial_emotion",
            barmode="group",
            title="Facial Emotion Frequency"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Line chart
        st.markdown("### 📈 Trust Score Over Time")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
        df = df.dropna(subset=["timestamp"])
        fig_line = px.line(
            df.sort_values("timestamp"),
            x="timestamp",
            y="trust_score",
            color="sentiment",
            markers=True,
            title="Trust Score Over Time"
        )
        st.plotly_chart(fig_line, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Failed to load or process log file: {str(e)}")


def display_analysis_log():
    st.markdown("---")
    pass  # Moved to dashboard


# --- Streamlit Sidebar Login ---
def api_login():
    st.sidebar.header("🔒 API Login")
    if 'api_token' not in st.session_state:
        st.session_state['api_token'] = None
    if 'api_user' not in st.session_state:
        st.session_state['api_user'] = None
    if st.session_state['api_token']:
        st.sidebar.success(f"Logged in as {st.session_state['api_user']}")
        if st.sidebar.button("Logout"):
            st.session_state['api_token'] = None
            st.session_state['api_user'] = None
            st.rerun()
        return True
    else:
        with st.sidebar.form("login_form"):
            username = st.text_input("User ID", value="user")
            password = st.text_input("Password", type="password", value="1234")
            submit = st.form_submit_button("Login")
            if submit:
                try:
                    resp = requests.post("http://127.0.0.1:8001/token", data={"username": username, "password": password})
                    if resp.status_code == 200:
                        token = resp.json()["access_token"]
                        st.session_state['api_token'] = token
                        st.session_state['api_user'] = username
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Login failed. Check credentials.")
                except Exception as e:
                    st.error(f"Login error: {e}")
        return False


def record_audio(duration=5, fs=44100):
    st.info(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    wavio.write(temp_wav.name, recording, fs, sampwidth=2)
    return temp_wav.name


def main():
    # Initialize session state
    if 'selected_facial_emotion' not in st.session_state:
        st.session_state.selected_facial_emotion = None
    if 'journal' not in st.session_state:
        st.session_state.journal = load_journal()
    if 'monitor' not in st.session_state:
        st.session_state.monitor = get_emotion_monitor()
    if 'audio_weight' not in st.session_state:
        st.session_state.audio_weight = 0.3

    load_custom_css()

    # --- API Login ---
    if not api_login():
        st.stop()

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Trust Settings")
        st.markdown("Adjust influence weights below:")
        text_weight = st.slider("📖 Text Sentiment Weight", 0.0, 1.0, 0.7, 0.05)
        emotion_weight = st.slider("🎭 Facial Emotion Weight", 0.0, 1.0, 0.3, 0.05)
        audio_weight = st.slider("🎤 Audio Influence Weight", 0.0, 1.0, st.session_state.audio_weight, 0.05)
        st.session_state["text_weight"] = text_weight
        st.session_state["emotion_weight"] = emotion_weight
        st.session_state["audio_weight"] = audio_weight

    # --- Header Section ---
    st.markdown("<h1 style='text-align: center;'>🤝 TrustSphere</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; font-size: 16px;">
            Analyze text sentiment and trust score using advanced AI (RoBERTa + DistilBERT).
        </div>
    """, unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Analysis", "🎭 Facial Emotions", "🎤 Audio Analysis", "📊 Dashboard"])

    # ---------- Tab 1: Analysis ----------
    with tab1:
        text_input = st.text_area("📝 Enter your text here:", height=150, max_chars=1000)

        # Remove audio upload and related logic
        audio_transcription = None
        audio_emotions = None
        detected_lang = 'en'
        original_text_input = text_input

        # --- Translation for non-English text ---
        if text_input.strip():
            translated_text, detected_lang = translation_utils.translate_to_english(text_input)
            if detected_lang != 'en':
                st.info(f"Detected language: {detected_lang.upper()}. Translated to English for analysis.")
                st.markdown(f"**Translated Text:** {translated_text}")
                text_input = translated_text  # Use translated text for analysis

        # Show selected emotion
        if st.session_state.selected_facial_emotion:
            st.info(f"Using facial emotion: {st.session_state.selected_facial_emotion} {emotion_emoji(st.session_state.selected_facial_emotion)}")

        if st.button("🚀 Analyze Now"):
            if text_input.strip() or st.session_state.selected_facial_emotion:
                with st.spinner("Analyzing with AI..."):
                    try:
                        # Determine emotion source
                        facial_emotion = None
                        monitor = st.session_state.monitor

                        if monitor.is_running:
                            current_emotion, _ = monitor.get_current_emotion()
                            if current_emotion:
                                facial_emotion = current_emotion
                                st.info(f"Using real-time emotion: {facial_emotion}")
                        elif st.session_state.selected_facial_emotion:
                            facial_emotion = st.session_state.selected_facial_emotion

                        # Add audio emotion to payload if available
                        audio_emotion_label = None
                        if audio_emotions:
                            # Use the top detected audio emotion
                            audio_emotion_label = audio_emotions[0]["label"]

                        # Call backend
                        headers = {"Authorization": f"Bearer {st.session_state['api_token']}"}
                        response = requests.post(FASTAPI_URL, json={
                            "text": text_input,
                            "facial_emotion": facial_emotion,
                            "audio_emotion": audio_emotion_label,
                            "audio_weight": st.session_state.get("audio_weight", 0.3)
                        }, headers=headers)
                        response.raise_for_status()
                        result = response.json()

                        # Display result
                        display_results(result)
                        # Store result and text in session state for later use
                        st.session_state['last_analysis_result'] = result
                        st.session_state['last_analysis_text'] = original_text_input
                    except Exception as e:
                        st.error(f"❌ An error occurred: {str(e)}")
            else:
                st.warning("⚠️ Please enter some text or select a facial emotion before analyzing.")

        # Feedback form and Save to Journal (only show if analysis result is available)
        if 'last_analysis_result' in st.session_state:
            st.subheader("🗣️ Submit Feedback")
            with st.form("feedback_form"):
                rating = st.slider("Rate this analysis:", 1, 5, 3)
                comment = st.text_area("Additional comments (optional):", max_chars=300)
                submit_feedback = st.form_submit_button("📩 Submit Feedback")
                if submit_feedback:
                    try:
                        save_feedback(
                            st.session_state['last_analysis_text'][:100],
                            st.session_state['last_analysis_result']["sentiment"],
                            st.session_state['last_analysis_result']["trust_score"],
                            rating, comment
                        )
                        st.success("✅ Feedback submitted. Thank you!")
                    except Exception as e:
                        st.error(f"Error saving feedback: {str(e)}")

            if st.button("📝 Save to Journal"):
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "text": st.session_state['last_analysis_text'],
                    "trust": st.session_state['last_analysis_result']['trust_score'],
                    "emotion": st.session_state['last_analysis_result'].get('facial_emotion', 'N/A'),
                    "audio_emotion": st.session_state['last_analysis_result'].get('audio_emotion', 'N/A'),
                    "detected_language": detected_lang
                }
                st.session_state.journal.append(entry)
                st.success("Saved to journal!")
                append_journal_entry(entry)

    # ---------- Tab 2: Facial Emotions ----------
    with tab2:
        st.header("Facial Emotion Detection Options")
        emotion_tab1, emotion_tab2 = st.tabs(["👤 Real-time Monitoring", "👥 Multiple Face Detection"])

        with emotion_tab1:
            display_emotion_monitoring()
        with emotion_tab2:
            display_multi_face_detection()

    # ---------- Tab 3: Audio Analysis ----------
    with tab3:
        st.header("🎤 Audio Analysis")
        st.write("Upload an audio file or record your voice in real time for emotion and trust analysis.")
        audio_tab1, audio_tab2 = st.tabs(["📤 Upload Audio", "🎙️ Record Audio"])

        with audio_tab1:
            audio_file = st.file_uploader("Upload audio", type=["wav", "mp3"], key="audio_upload_tab")
            audio_path = None
            if audio_file is not None:
                temp_audio_path = f"temp_{audio_file.name}"
                with open(temp_audio_path, "wb") as f:
                    f.write(audio_file.read())
                # Add UI for selecting start and end time for truncation
                data, samplerate = sf.read(temp_audio_path)
                total_seconds = len(data) / samplerate
                st.info(f"Audio length: {total_seconds:.2f} seconds")
                start_sec, end_sec = st.slider(
                    "Select segment to analyze (seconds)",
                    min_value=0.0, max_value=float(total_seconds),
                    value=(0.0, float(min(30.0, total_seconds))), step=0.1,
                    key="upload_segment_slider"
                )
                if st.button("Analyze Selected Segment", key="analyze_uploaded_segment"):
                    # Save truncated segment to a new file
                    start_sample = int(start_sec * samplerate)
                    end_sample = int(end_sec * samplerate)
                    segment = data[start_sample:end_sample]
                    truncated_path = f"truncated_{audio_file.name}"
                    sf.write(truncated_path, segment, samplerate)
                    audio_path = truncated_path
                else:
                    audio_path = temp_audio_path
            if audio_path:
                with st.spinner("Processing audio..."):
                    audio_result = audio_analysis.analyze_audio(audio_path)
                    audio_transcription = audio_result["transcription"]
                    audio_emotions = audio_result["emotions"]
                st.success("Audio processed!")
                st.markdown(f"**📝 Detected Speech:** {audio_transcription}")
                st.markdown("**🎵 Detected Audio Emotions:**")
                for emo in audio_emotions:
                    st.markdown(f"- {emo['label']}: {emo['score']:.2f}")
                if st.button("Use this audio for trust analysis", key="use_uploaded_audio"):
                    st.session_state["audio_transcription"] = audio_transcription
                    st.session_state["audio_emotion_label"] = audio_emotions[0]["label"] if audio_emotions else None
                    st.success("Audio data set for trust analysis. Switch to the Analysis tab to proceed.")

        with audio_tab2:
            record_col, duration_col = st.columns([2, 1])
            with duration_col:
                duration = st.slider("Recording Duration (seconds)", 1, 60, 5, key="record_duration")
            with record_col:
                if st.button("🎙️ Record Audio", key="record_audio_btn"):
                    audio_path = record_audio(duration)
                    st.session_state["recorded_audio_path"] = audio_path
                    st.success(f"Audio recorded: {audio_path}")
            audio_path = st.session_state.get("recorded_audio_path", None)
            if audio_path:
                with st.spinner("Processing audio..."):
                    audio_result = audio_analysis.analyze_audio(audio_path)
                    audio_transcription = audio_result["transcription"]
                    audio_emotions = audio_result["emotions"]
                st.success("Audio processed!")
                st.markdown(f"**📝 Detected Speech:** {audio_transcription}")
                st.markdown("**🎵 Detected Audio Emotions:**")
                for emo in audio_emotions:
                    st.markdown(f"- {emo['label']}: {emo['score']:.2f}")
                if st.button("Use this audio for trust analysis", key="use_recorded_audio"):
                    st.session_state["audio_transcription"] = audio_transcription
                    st.session_state["audio_emotion_label"] = audio_emotions[0]["label"] if audio_emotions else None
                    st.success("Audio data set for trust analysis. Switch to the Analysis tab to proceed.")

    # ---------- Tab 4: Dashboard ----------
    with tab4:
        display_visual_dashboard()
        display_analysis_log()

        st.subheader("🗃️ Collected Feedback")
        try:
            conn = sqlite3.connect("logs/feedback.db")
            df_fb = pd.read_sql_query("SELECT * FROM feedback ORDER BY timestamp DESC", conn)
            conn.close()
            st.dataframe(df_fb)
        except Exception as e:
            st.error(f"Failed to load feedback: {e}")

        st.subheader("📘 Your Trust Journal")
        if os.path.exists(JOURNAL_CSV):
            df_journal = pd.read_csv(JOURNAL_CSV)
        else:
            df_journal = pd.DataFrame(st.session_state.journal)
        st.dataframe(df_journal)

        st.subheader("📂 View Past Analysis Logs")
        log_path = "logs/analysis_log.csv"
        if os.path.exists(log_path):
            try:
                df = pd.read_csv(log_path)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                # Decrypt text and facial_emotion columns
                if "text" in df.columns:
                    df["text"] = df["text"].apply(lambda x: security_utils.decrypt_data(x) if isinstance(x, str) else x)
                if "facial_emotion" in df.columns:
                    df["facial_emotion"] = df["facial_emotion"].apply(lambda x: security_utils.decrypt_data(x) if isinstance(x, str) else x)
                df = df.sort_values("timestamp", ascending=False).head(20)
                st.dataframe(df)
            except Exception as e:
                st.error(f"Failed to load logs: {str(e)}")
        else:
            st.info("No logs found.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.stop()