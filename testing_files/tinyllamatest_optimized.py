from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from codecarbon import EmissionsTracker

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Initialize codecarbon emissions tracker
tracker = EmissionsTracker(
    project_name="tinyllama_optimized_test",
)



print(f"Loading model {model_id}...")
start_time = time.time()

# Optimize for Apple Silicon (M-series chips)
# Use 'mps' device if available, otherwise 'cpu'
device = "mps"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16, # Use float16 to reduce memory usage
    device_map=device,
    # trust_remote_code=True, # Usually not needed for TinyLlama native support
    # attn_implementation="eager" # Often helps with compatibility
)

print(f"Model loaded in {time.time() - start_time:.2f} seconds.")

prompt = "Explain transformers in simple terms."

print(f"Generating response for prompt: '{prompt}'")
inputs = tokenizer(prompt, return_tensors="pt").to(device)
print("Starting emissions tracking...")
tracker.start()
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
print("\nStopping emissions tracking...")
emissions = tracker.stop()
print(emissions)
print(f"Generation took {gen_end - gen_start:.2f} seconds.")

# Stop tracking and get emissions data
