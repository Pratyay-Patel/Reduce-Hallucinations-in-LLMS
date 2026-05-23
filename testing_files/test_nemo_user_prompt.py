import sys
import os
import time
import json

# Adjust path to find the Nvidia prompt classifier in the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prompt_class_dir = os.path.join(parent_dir, "Nvidia prompt class")

if prompt_class_dir not in sys.path:
    sys.path.append(prompt_class_dir)

try:
    import nvidia_classifier
except ImportError as e:
    print(f"Error importing nvidia classifier: {e}")
    print(f"Checked path: {prompt_class_dir}")
    print(f"Contents of parent dir: {os.listdir(parent_dir)}")
    sys.exit(1)

# Global cache
NEMO_MODEL = None
NEMO_TOKENIZER = None

def get_nemo_model():
    global NEMO_MODEL, NEMO_TOKENIZER
    if NEMO_MODEL is None or NEMO_TOKENIZER is None:
        print("Loading NeMo Curator model for classification...")
        try:
            NEMO_MODEL, NEMO_TOKENIZER = nvidia_classifier.load_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    return NEMO_MODEL, NEMO_TOKENIZER

def test_user_prompt(prompt):
    try:
        model, tokenizer = get_nemo_model()
        result = nvidia_classifier.analyze_prompt(model, tokenizer, prompt)
        
        prompt_tokens = tokenizer(prompt)["input_ids"]
        num_tokens = len(prompt_tokens)
        
        return {
            "nemo_raw_json": result,
            "num_tokens": num_tokens
        }
        
    except Exception as e:
        print(f"Error classifying prompt: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
    else:
        # Ask for input if not provided
        print("Please enter the prompt you want to test:")
        user_prompt = input("> ")
    
    if user_prompt:
        output = test_user_prompt(user_prompt)
        if output:
            print(json.dumps(output, indent=2))
    else:
        print("No prompt provided.")
