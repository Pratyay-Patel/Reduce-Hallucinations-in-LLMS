import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

load_dotenv()

def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not found.")
        print("Please set it before running the script.")
        print("Example: export HF_TOKEN='your_hf_token'")
        return

    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    print(f"Using model: {model_id}")
    print("Token found. Downloading/Loading tokenizer...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        print("Tokenizer loaded successfully!")
        
        print("Downloading/Loading model... This might take a while if not already cached.")
        # Load the model with automatic device placement and half precision to save memory
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            token=token,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Model loaded successfully!")
        
        # Quick generation test
        print("Running a simple generation test...")
        messages = [
            {"role": "user", "content": "What is the capital of France? Answer in one word."}
        ]
        
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False
        ).to(model.device)
        
        # Standard terminators for Llama 3 models
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        outputs = model.generate(
            input_ids,
            max_new_tokens=10,
            eos_token_id=terminators,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = outputs[0][input_ids.shape[-1]:]
        print("\nResponse:")
        print(tokenizer.decode(response, skip_special_tokens=True).strip())
        print("\nTest completed successfully.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
