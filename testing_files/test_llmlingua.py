from llmlingua import PromptCompressor
import time
import torch

def test_llmlingua():
    print("Initializing LLM Lingua Prompt Compressor...")
    start_time = time.time()
    
    # Using a small model for the compressor to be efficient
    # The default is often llama-2-7b-chat-hf, which might be too big for quick test.
    # We'll try to use the user's previously mentioned phi-3 if compatible, 
    # but LLM Lingua works best with its default or specific supported models.
    # Let's stick to a safe, small default or explicitly request a smaller one if needed.
    # We will use 'microsoft/Phi-3-mini-4k-instruct' as the compressor model 
    # if supported, otherwise let it default or pick a known small one.
    # For now, let's use the standard initialization which usually downloads a small model (like llama2-7b or similar)
    # However, to avoid large downloads, we can specify a smaller model if we know one, 
    # but LLM Lingua relies on the perplexity of the model. 
    # Let's use 'gpt2' for a very fast, lightweight test, or 'microsoft/Phi-3-mini-4k-instruct' to match environment.
    
    # Important: Optimization for Apple Silicon
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        llm_lingua = PromptCompressor(
            model_name="facebook/opt-125m",
            device_map=device,
        )
    except Exception as e:
        print(f"Failed to load Phi-3 for compression. Fallback to default (llama-2-7b typically) or lightweight.")
        print(f"Error: {e}")
        return

    print(f"Compressor initialized in {time.time() - start_time:.2f} seconds.")

    # Example Prompt (long context)
    prompt = """
    Instruction: summarize the following text.
    
    Context:
    The Apollo program, also known as Project Apollo, was the third United States human spaceflight program carried out by the National Aeronautics and Space Administration (NASA), which succeeded in preparing and landing the first humans on the Moon from 1968 to 1972. It was first conceived during 1960 during the Eisenhower administration as a three-man spacecraft to follow the one-man Project Mercury which put the first Americans in space. Apollo was later dedicated to President John F. Kennedy's national goal of "landing a man on the Moon and returning him safely to the Earth" by the end of the 1960s, which he proposed in an address to Congress on May 25, 1961.
    
    The program was the third US human spaceflight program to fly, preceded by the two-man Project Gemini conceived in 1961 to validate space docking and extra-vehicular activity (EVA) capabilities needed for Apollo, and which it followed in 1968. Apollo ran from 1961 to 1972, with the first crewed flight in 1968. It achieved its goal with the Apollo 11 mission in July 1969, when Neil Armstrong and Buzz Aldrin landed on the Moon while Michael Collins remained in lunar orbit. Five subsequent Apollo missions also landed astronauts on the Moon, the last in December 1972. Throughout these six spaceflights, twelve people walked on the Moon.
    
    Apollo was a major milestone in human history and remains the only time that humans have operated beyond low Earth orbit. The program spurred advances in many areas of technology incidental to rocketry and human spaceflight, including avionics, telecommunications, and computers. Apollo sparked interest in many fields of engineering and is considered by many to be the greatest achievement in the history of mankind.
    
    Question: What was the main goal of the Apollo program?
    """

    print("\n--- Original Prompt ---")
    print(prompt)
    print(f"Original Length: {len(prompt)} characters")

    print("\n--- Compressing Prompt ---")
    # Using rate=0.6 and disabling dynamic windows to be safer with small models
    compressed_prompt = llm_lingua.compress_prompt(
        prompt.split("\n"),
        instruction="What was the main goal of the Apollo program?",
        question="What was the main goal of the Apollo program?",
        rate=0.7, 
        target_token=300
    )
    
    print("\n--- Compressed Prompt Data ---")
    # compressed_prompt is usually a dictionary containing 'compressed_prompt' string
    print(compressed_prompt.keys())
    
    final_prompt = compressed_prompt['compressed_prompt']
    print(f"\nText: {final_prompt}")
    print(f"Compressed Length: {len(final_prompt)} characters")
    print(f"Compression Ratio: {len(final_prompt)/len(prompt):.2f}")


if __name__ == "__main__":
    test_llmlingua()
