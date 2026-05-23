from llmlingua import PromptCompressor
import time
import torch
import sys

def test_compression(prompt=None):
    print("Initializing LLM Lingua Prompt Compressor...")
    start_time = time.time()
    
    # Optimization for Apple Silicon
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Using a smaller model tailored for efficiency in this test environment
        llm_lingua = PromptCompressor(
           model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map="cpu"
        )
    except Exception as e:
        print(f"Error initializing PromptCompressor: {e}")
        return

    print(f"Compressor initialized in {time.time() - start_time:.2f} seconds.")

    if not prompt:
        # Default sample prompt
        prompt = """
        The Apollo program, also known as Project Apollo, was the third United States human spaceflight program carried out by the National Aeronautics and Space Administration (NASA), which succeeded in preparing and landing the first humans on the Moon from 1968 to 1972. It was first conceived during 1960 during the Eisenhower administration as a three-man spacecraft to follow the one-man Project Mercury which put the first Americans in space. Apollo was later dedicated to President John F. Kennedy's national goal of "landing a man on the Moon and returning him safely to the Earth" by the end of the 1960s, which he proposed in an address to Congress on May 25, 1961.
        
        Question: What was the main goal of the Apollo program?
        """

    print("\n--- Original Prompt ---")
    print(prompt)
    print(f"Original Length: {len(prompt)} characters")

    print("\n--- Compressing Prompt ---")
    try:
        # Compression matching final.py usage
        # Calculate a safe rate or use a fixed one
        token_count = len(llm_lingua.tokenizer.encode(prompt))
        print(f"Token count: {token_count}")
        
        # Target compression
        rate = 0.5 # 50% compression
        
        compressed_prompt = llm_lingua.compress_prompt(
            prompt, 
            rate=rate, 
            force_tokens=['\n', '?']
        )
        
        final_prompt = compressed_prompt['compressed_prompt']
        print(f"\n--- Compressed Prompt ---")
        print(f"{final_prompt}")
        print(f"Compressed Length: {len(final_prompt)} characters")
        print(f"Compression Ratio: {len(final_prompt)/len(prompt):.2f}")
        
    except Exception as e:
        print(f"Error during compression: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])    
        test_compression(user_prompt)
    else:
        print("Using default prompt. You can provide a prompt as an argument.")
        test_compression()
