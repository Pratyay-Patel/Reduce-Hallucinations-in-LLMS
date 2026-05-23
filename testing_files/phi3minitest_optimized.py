import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from codecarbon import EmissionsTracker

model_id = "microsoft/Phi-3-mini-4k-instruct"

# Initialize codecarbon emissions tracker
tracker = EmissionsTracker(
    project_name="final3",
    measure_power_secs=1,
    save_to_file=True,
    log_level="info"
)

print(f"Loading model {model_id}...")
start_time = time.time()

# Optimize for Apple Silicon (M-series chips)
# Use 'mps' device if available, otherwise 'cpu'
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16, # Use float16 to reduce memory usage
    device_map=device,
    # trust_remote_code=True, # Disabled to use native transformers implementation
    attn_implementation="eager"
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

# Get detailed metrics
if hasattr(tracker, '_last_measured_time') and hasattr(tracker, '_start_time'):
    duration = tracker._last_measured_time - tracker._start_time
else:
    duration = gen_end - gen_start

energy_consumed = 0.0
cpu_power = 0.0
gpu_power = 0.0
ram_power = 0.0
cpu_energy = 0.0
gpu_energy = 0.0
ram_energy = 0.0

if hasattr(tracker, 'final_emissions'):
    final_data = tracker.final_emissions
    energy_consumed = getattr(final_data, 'energy_consumed', 0.0)
    cpu_power = getattr(final_data, 'cpu_power', 0.0)
    gpu_power = getattr(final_data, 'gpu_power', 0.0)
    ram_power = getattr(final_data, 'ram_power', 0.0)
    cpu_energy = getattr(final_data, 'cpu_energy', 0.0)
    gpu_energy = getattr(final_data, 'gpu_energy', 0.0)
    ram_energy = getattr(final_data, 'ram_energy', 0.0)

# Log all metrics
print("\n" + "="*60)
print("CODECARBON METRICS")
print("="*60)
print(f"Emissions: {emissions:.10f} kg CO2")
print(f"Energy consumed: {energy_consumed:.10f} kWh")
print(f"Duration: {duration:.2f}s")
print(f"CPU Power: {cpu_power:.4f}W, Energy: {cpu_energy:.10f} kWh")
print(f"GPU Power: {gpu_power:.4f}W, Energy: {gpu_energy:.10f} kWh")
print(f"RAM Power: {ram_power:.4f}W, Energy: {ram_energy:.10f} kWh")
print("="*60)

print(f"\nGeneration took {gen_end - gen_start:.2f} seconds.")
