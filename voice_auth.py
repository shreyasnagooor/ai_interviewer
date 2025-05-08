# voice_auth.py

import torch
import torchaudio
from speechbrain.inference import SpeakerRecognition
from pathlib import Path
import sounddevice as sd
import soundfile as sf
import time

model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp_voice_auth"
)

EMBEDDING_PATH = "enroll_embedding.pt"

def enroll_voice():
    print("Recording 5 seconds for enrollment...")
    duration = 5
    fs = 16000
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write("enroll.wav", recording, fs)
    signal, fs = torchaudio.load("enroll.wav")
    embedding = model.encode_batch(signal)
    torch.save(embedding, EMBEDDING_PATH)
    print("Voice enrolled and saved.")

def load_enrollment():
    if Path(EMBEDDING_PATH).exists():
        return torch.load(EMBEDDING_PATH)
    return None

def authenticate(enrolled_embedding):
    if enrolled_embedding is None:
        print("[ERROR] No enrolled voice found.")
        return False
    duration = 5
    fs = 16000
    test_file = f"test_{int(time.time())}.wav"
    print("Recording 5-second test sample...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write(test_file, recording, fs)
    signal, fs = torchaudio.load(test_file)
    test_embedding = model.encode_batch(signal)
    score, _ = model.similarity(enrolled_embedding, test_embedding)
    print(f"Voice similarity score: {score.item():.4f}")
    return score.item() > 0.8  # You can tune this threshold
