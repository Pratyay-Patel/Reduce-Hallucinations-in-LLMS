from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
token = os.getenv("HF_TOKEN")
if not token:
    print("Warning: HF_TOKEN not found in environment. Please add it to your .env file if accessing gated models.")

# Llama 3 8B Instruct
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

print(f"Loading model {model_id}...")
start_time = time.time()

# Optimize for Apple Silicon (M-series chips)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16, # Use float16 to reduce memory usage
    device_map=device,
    attn_implementation="eager"
)

print(f"Model loaded in {time.time() - start_time:.2f} seconds.")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing in simple terms."},
]

print(f"Generating response for messages: {messages}")

# Correct usage of apply_chat_template matching user reference
inputs = tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True, 
    return_tensors="pt"
).to(device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

with torch.no_grad():
    gen_start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )
    gen_end = time.time()

# Only decode the newly generated tokens
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

print("\n--- Response ---")
print(response)
print("----------------")
print(f"Generation took {gen_end - gen_start:.2f} seconds.")
