from utils import predict_emotion

emotion, confidence, probs = predict_emotion(
    "./sample_audio/02-03-01-02.wav"
)
print("Emotion :", emotion)
print("Confidence :", confidence)
print(probs)