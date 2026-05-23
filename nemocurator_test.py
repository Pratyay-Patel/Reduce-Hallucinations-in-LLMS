import sys
import os
import time

# Add the directory containing the Nvidia prompt classifier code to sys.path
# Using absolute path or relative to be safe.
prompt_class_dir = os.path.abspath("./Nvidia prompt class")
if prompt_class_dir not in sys.path:
    sys.path.append(prompt_class_dir)

try:
    import temp as nvidia_classifier
except ImportError as e:
    print(f"Error importing nvidia classifier: {e}")
    print(f"Checked path: {prompt_class_dir}")
    print(f"Contents: {os.listdir(prompt_class_dir)}")
    sys.exit(1)

# Global cache for the specific model used by get_prompt_category
NEMO_MODEL = None
NEMO_TOKENIZER = None

def get_prompt_category(prompt):
    """
    Analyzes the prompt and returns its complexity category: 'Easy', 'Medium', or 'Hard'.
    Loads the model lazily if not already loaded.
    """
    global NEMO_MODEL, NEMO_TOKENIZER
    
    if NEMO_MODEL is None or NEMO_TOKENIZER is None:
        print("Loading NeMo Curator model for classification...")
        try:
            NEMO_MODEL, NEMO_TOKENIZER = nvidia_classifier.load_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to Medium if model fails? Or raise?
            return "Medium" 

    try:
        # analyze_prompt returns a dict with lists inside
        result = nvidia_classifier.analyze_prompt(NEMO_MODEL, NEMO_TOKENIZER, prompt)
        
        complexity_score = 0.0
        if "prompt_complexity_score" in result:
             try:
                complexity_score = float(result["prompt_complexity_score"][0])
             except:
                complexity_score = 0.0
        
        return classify_complexity(complexity_score)
        
    except Exception as e:
        print(f"Error classifying prompt: {e}")
        return "Medium" # Fallback

def classify_complexity(score):
    """
    Classifies complexity score into Easy, Medium, Hard.
    Thresholds are illustrative and can be tuned.
    """
    # Assuming score is between 0 and 1, which the weighted sum suggests.
    if score < 0.35:
        return "Easy"
    elif score < 0.65:
        return "Medium"
    else:
        return "Hard"

def analyze_text(prompt):
    print("Initializing Nvidia Prompt Classifier...")
    start_time = time.time()
    
    # Load the model using the imported module
    # temp.py has load_model() returning (model, tokenizer)
    try:
        model, tokenizer = nvidia_classifier.load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print(f"Model loaded in {time.time() - start_time:.2f} seconds.")

    print("\n--- Analyzing Prompt ---")
    print(f"Prompt Length: {len(prompt)}")

    # analyze_prompt takes (model, tokenizer, prompt_text)
    # prompt_text can be list or str
    try:
        result = nvidia_classifier.analyze_prompt(model, tokenizer, prompt)
        
        print("\n--- Analysis Results ---")
        # Pretty print the dictionary
        complexity_score = 0.0
        
        for key, value in result.items():
            print(f"{key}: {value}")
            if key == "prompt_complexity_score":
                # Value is likely a list containing a numpy float/tensor
                try:
                    complexity_score = float(value[0])
                except:
                    complexity_score = 0.0
        
        category = classify_complexity(complexity_score)
        print(f"\n--- Classification ---")
        print(f"Complexity Score: {complexity_score:.4f}")
        print(f"Category: {category}")
            
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    # Test Prompt
    sample_prompt = """
    Instruction: summarize the following text.
    
    Context:
    The Apollo program, also known as Project Apollo, was the third United States human spaceflight program carried out by the National Aeronautics and Space Administration (NASA), which succeeded in preparing and landing the first humans on the Moon from 1968 to 1972. It was first conceived during 1960 during the Eisenhower administration as a three-man spacecraft to follow the one-man Project Mercury which put the first Americans in space. 
    
    The program was the third US human spaceflight program to fly, preceded by the two-man Project Gemini conceived in 1961 to validate space docking and extra-vehicular activity (EVA) capabilities needed for Apollo, and which it followed in 1968. Apollo ran from 1961 to 1972, with the first crewed flight in 1968. It achieved its goal with the Apollo 11 mission in July 1969, when Neil Armstrong and Buzz Aldrin landed on the Moon while Michael Collins remained in lunar orbit.
    
    Apollo was a major milestone in human history and remains the only time that humans have operated beyond low Earth orbit. The program spurred advances in many areas of technology incidental to rocketry and human spaceflight, including avionics, telecommunications, and computers. Apollo sparked interest in many fields of engineering and is considered by many to be the greatest achievement in the history of mankind.
    
    Question: What was the main goal of the Apollo program?
    """
    
    # Check if a prompt was provided as a command line argument
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
        analyze_text(user_prompt)
    else:
        analyze_text(sample_prompt)