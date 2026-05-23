from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.3"

print(f"Loading model: {MODEL_PATH}")
start_time = time.time()

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

device = "cuda"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    token=os.getenv("HF_TOKEN")
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    token=os.getenv("HF_TOKEN")
)

print(f"Model loaded in {time.time() - start_time:.2f} seconds.")

prompt = "Explain quantum computing in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))