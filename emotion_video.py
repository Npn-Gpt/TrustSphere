import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # This will suppress TF warnings
os.environ['PYTHONWARNINGS'] = 'ignore'  # Suppress Python warnings
import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings

# Suppress specific PyTorch warnings and fix path issues
import torch
torch.set_warn_always(False)
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import time
import threading
import queue
from datetime import datetime, timedelta

# Lazy import of DeepFace
DeepFace = None

def lazy_import_deepface():
    global DeepFace
    if DeepFace is None:
        try:
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging
            from deepface import DeepFace
        except Exception as e:
            print(f"Error importing DeepFace: {e}")
            return None

class EmotionMonitor:
    def __init__(self):
        self.is_running = False
        self.emotion_history = []
        self.current_frame = None
        self.current_emotion = None
        self.emotion_queue = queue.Queue(maxsize=100)
        self.thread = None
        self.lock = threading.Lock()
        self.emotion_weights = {
            'happy': 1.0, 'sad': 1.0, 'angry': 1.0, 'neutral': 1.0,
            'fear': 1.0, 'surprise': 1.0, 'disgust': 1.0
        }
        self._last_frame_time = time.time()
        self._frame_interval = 0.1  # 100ms between frames

    def set_emotion_weight(self, emotion, weight):
        """Set the weight for a specific emotion."""
        if emotion in self.emotion_weights:
            self.emotion_weights[emotion] = weight
        else:
            raise ValueError(f"Unknown emotion: {emotion}")

    def get_current_emotion(self):
        """Get the current emotion and frame."""
        with self.lock:
            return self.current_emotion, self.current_frame

    def get_emotion_distribution(self, seconds=10):
        """Get emotion distribution over the last N seconds."""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(seconds=seconds)
            recent_emotions = [rec[1] for rec in self.emotion_history if rec[0] >= cutoff_time]
            if not recent_emotions:
                return None
            total = len(recent_emotions)
            return {emotion: (recent_emotions.count(emotion) / total) * 100 
                   for emotion in set(recent_emotions)}

    def get_weighted_emotion_score(self):
        """Get weighted emotion score."""
        with self.lock:
            if not self.current_emotion:
                return None
            return self.emotion_weights.get(self.current_emotion, 1.0)

    def generate_emotion_timeline_image(self):
        """Generate emotion timeline image."""
        with self.lock:
            if not self.emotion_history:
                return None
            return self.current_frame

    def start_monitoring(self):
        """Start emotion monitoring in a separate thread."""
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._monitor_emotions)
            self.thread.daemon = True
            self.thread.start()
            return self.thread

    def stop_monitoring(self):
        """Stop emotion monitoring."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None

    def _monitor_emotions(self):
        """Monitor emotions in a separate thread."""
        lazy_import_deepface()
        if DeepFace is None:
            self.is_running = False
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.is_running = False
            return

        last_analysis_time = time.time() - 1

        try:
            while self.is_running:
                current_time = time.time()
                if current_time - self._last_frame_time < self._frame_interval:
                    time.sleep(0.01)
                    continue

                ret, frame = cap.read()
                if not ret:
                    continue

                with self.lock:
                    self.current_frame = frame.copy()
                    self._last_frame_time = current_time

                if current_time - last_analysis_time >= 0.5:
                    try:
                        result = DeepFace.analyze(
                            frame, actions=['emotion'], enforce_detection=False, silent=True
                        )
                        emotion_data = result[0]['emotion']
                        dominant_emotion = result[0]['dominant_emotion']
                        confidence = emotion_data[dominant_emotion]

                        timestamp = datetime.now()
                        emotion_record = (timestamp, dominant_emotion, confidence, emotion_data)

                        with self.lock:
                            self.current_emotion = dominant_emotion
                            self.emotion_history.append(emotion_record)
                            cutoff_time = timestamp - timedelta(seconds=60)
                            self.emotion_history = [rec for rec in self.emotion_history if rec[0] >= cutoff_time]

                        if not self.emotion_queue.full():
                            self.emotion_queue.put(emotion_record)

                        last_analysis_time = current_time

                    except Exception as e:
                        print(f"Emotion detection error: {e}")

                time.sleep(0.01)

        finally:
            cap.release()
            self.is_running = False

# Create a singleton instance
emotion_monitor = EmotionMonitor()

def get_emotion_monitor():
    """Get the singleton emotion monitor instance."""
    return emotion_monitor

def capture_emotion_snapshot():
    lazy_import_deepface()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open webcam")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise Exception("Failed to capture frame")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    results = []
    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        try:
            emotion_result = DeepFace.analyze(
                face_img, actions=['emotion'], enforce_detection=False, silent=True
            )
            dominant_emotion = emotion_result[0]['dominant_emotion']
            results.append((x, y, w, h, dominant_emotion))
        except Exception as e:
            print(f"Face emotion detection error: {e}")
            results.append((x, y, w, h, "unknown"))

    for (x, y, w, h, emotion) in results:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return results, frame
