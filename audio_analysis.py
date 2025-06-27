import speech_recognition as sr
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import soundfile as sf
import os

# Load models once at module level
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

MAX_AUDIO_SECONDS = 660  # Allow up to 660 seconds

def truncate_audio(audio_path, max_seconds=MAX_AUDIO_SECONDS, output_path=None):
    data, samplerate = sf.read(audio_path)
    max_samples = int(max_seconds * samplerate)
    if len(data) > max_samples:
        data = data[:max_samples]
        if output_path is None:
            output_path = f"truncated_{os.path.basename(audio_path)}"
        sf.write(output_path, data, samplerate)
        return output_path
    return audio_path

def transcribe_audio(audio_path):
    audio_path = truncate_audio(audio_path)
    speech, rate = sf.read(audio_path)
    if len(speech.shape) > 1:
        speech = speech.mean(axis=1)  # Convert to mono if stereo
    input_values = tokenizer(speech, return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    return transcription.strip()

def detect_audio_emotion(audio_path):
    from transformers import pipeline
    audio_path = truncate_audio(audio_path)
    emotion_pipeline = pipeline(
        "audio-classification",
        model="superb/wav2vec2-base-superb-er",
        device=-1  # Force CPU
    )
    results = emotion_pipeline(audio_path)
    return results

def analyze_audio(audio_path):
    text = transcribe_audio(audio_path)
    emotions = detect_audio_emotion(audio_path)
    return {"transcription": text, "emotions": emotions} 