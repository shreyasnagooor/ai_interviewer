import sounddevice as sd
import scipy.io.wavfile
import random
import time
import os

def record_sample(file_path, duration=4, fs=16000):
    print("[INFO] Recording voice sample...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    audio = audio.squeeze()
    scipy.io.wavfile.write(file_path, fs, audio)
    print(f"[INFO] Sample saved to {file_path}")
