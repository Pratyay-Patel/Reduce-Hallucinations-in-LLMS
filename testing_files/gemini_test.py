import os
import time
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

def test_gemini_api():
    print("--- Testing Gemini API Integration ---")
    
    # Check for API Key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment variables.")
        print("Please set it in your .env file or environment.")
        return

    print("API Key found.")

    # Configure Gemini
    # Configure Gemini
    print("Configuring Google Generative AI...")
    try:
        genai.configure(api_key=api_key)
        
        print("Model initialized. Listing available models to verify:")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
        
        model = genai.GenerativeModel('gemini-3-flash-preview')
        print("Using 'gemini-3-flash-preview'...")
    except Exception as e:
        print(f"Error configuring Gemini: {e}")
        return

    # Test Prompt
    prompt = "Explain quantum computing in one sentence."
    print(f"\nSending prompt: '{prompt}'")

    # Generate Response
    start_time = time.time()
    try:
        response = model.generate_content(prompt)
        end_time = time.time()
        
        print("\n--- Response ---")
        print(response.text)
        print("----------------")
        print(f"Generation took {end_time - start_time:.2f} seconds.")
        
    except Exception as e:
        print(f"\nError during generation: {e}")

if __name__ == "__main__":
    test_gemini_api()
