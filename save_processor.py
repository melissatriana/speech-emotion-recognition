from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base"
)

processor.save_pretrained("./model")

print("Processor berhasil disimpan.")