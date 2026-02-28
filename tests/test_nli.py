from src.models import load_nli_model

print("Loading NLI model...")
model, tokenizer = load_nli_model()
print("NLI model loaded successfully")