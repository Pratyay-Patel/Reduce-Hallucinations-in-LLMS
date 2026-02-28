from src.models import load_llm, generate_answers

print("Loading model...")
model, tokenizer = load_llm("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

print("Generating...")
prompt = "Question: What is 2 + 2?\nAnswer:"
answers = generate_answers(model, tokenizer, prompt, num_return_sequences=1)

print("Output:")
print(answers)