from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
token = os.getenv("HF_TOKEN")
if not token:
    # If not in env, we might rely on 'hf auth login' cache
    pass

# Llama 2 13B Chat
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

print(f"Loading model {model_id}...")
start_time = time.time()

# Optimize for NVIDIA GPU or Apple Silicon (M-series chips)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16, # Use float16 to reduce memory usage
    device_map="auto",
)


print(f"Model loaded in {time.time() - start_time:.2f} seconds.")

prompt = "Explain quantum computing in simple terms."

print(f"Generating response for prompt: '{prompt}'")
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    gen_start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    gen_end = time.time()

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n--- Response ---")
print(response)
print("----------------")
print(f"Generation took {gen_end - gen_start:.2f} seconds.")
