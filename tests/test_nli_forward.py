from src.models import load_nli_model
import torch

print("Loading NLI...")
model, tokenizer = load_nli_model()

premise = "The Eiffel Tower is located in Paris."
hypothesis = "The Eiffel Tower is in France."

inputs = tokenizer(
    premise,
    hypothesis,
    return_tensors="pt",
    truncation=True
)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)

print("Entailment probability:", probs[0, 2].item())