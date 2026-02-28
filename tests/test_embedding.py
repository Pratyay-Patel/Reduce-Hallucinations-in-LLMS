from src.models import load_embedding_model

print("Loading embedding model...")
model = load_embedding_model()

sentences = [
    "The sky is blue.",
    "The sky has a blue color."
]

embeddings = model.encode(sentences)

print("Embedding shape:", embeddings.shape)