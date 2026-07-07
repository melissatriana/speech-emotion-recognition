import librosa
import numpy as np
import torch

from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification
)

MODEL_PATH = "melissatriana/speech-emotion-recognition"

TARGET_SR = 16000
CHUNK_SEC = 3
CHUNK_SIZE = TARGET_SR * CHUNK_SEC

id2label = {
    0:"Neutral",
    1:"Happy",
    2:"Surprise",
    3:"Disgust",
    4:"Disappointed"
}

processor = Wav2Vec2Processor.from_pretrained(
    "melissatriana/speech-emotion-recognition",
    subfolder="model"
)

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "melissatriana/speech-emotion-recognition",
    subfolder="model"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

def normalize_audio(audio):

    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak

    return audio


def get_chunk(audio):

    if len(audio) <= CHUNK_SIZE:

        return np.pad(
            audio,
            (0, CHUNK_SIZE-len(audio))
        )

    mid = len(audio)//2

    start = max(
        0,
        mid-CHUNK_SIZE//2
    )

    return audio[start:start+CHUNK_SIZE]


def preprocess_audio(audio_path):

    audio,_ = librosa.load(
        audio_path,
        sr=TARGET_SR
    )

    audio = np.nan_to_num(audio)

    audio = normalize_audio(audio)

    audio = get_chunk(audio)

    return audio

def predict_emotion(audio_path):

    audio = preprocess_audio(audio_path)

    inputs = processor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt"
    )

    input_values = inputs.input_values.to(device)

    with torch.no_grad():

        outputs = model(
            input_values=input_values
        )

    probs = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    pred = torch.argmax(probs).item()

    confidence = probs[pred].item()

    probabilities = {
        id2label[i]: float(probs[i])
        for i in range(5)
    }

    return (
        id2label[pred],
        confidence,
        probabilities
    )
