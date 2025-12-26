import whisper
import sounddevice as sd
import numpy as np

#loading model
model = whisper.load_model("small")

SAMPLE_RATE = 16000
DURATION = 5

def record(duration,sample_rate):
    print("Recording...")
    audio = sd.rec(int(duration*sample_rate),samplerate = sample_rate, channels = 1, dtype = 'float32')
    sd.wait()
    return audio.flatten()

while True:
    result = model.transcribe(record(DURATION,SAMPLE_RATE),fp16 = False)
    text = result["text"].strip()

    if text:
        print(f"You said: {text}")

