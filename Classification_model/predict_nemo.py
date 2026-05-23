import json
import joblib
import pandas as pd
import numpy as np
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prompt_class_dir = os.path.join(parent_dir, "Nvidia prompt class")

if prompt_class_dir not in sys.path:
    sys.path.insert(0, prompt_class_dir)

try:
    import nvidia_classifier  # type: ignore
except ImportError as e:
    print(f"Error importing nvidia classifier: {e}")
    print(f"Checked path: {prompt_class_dir}")
    sys.exit(1)


class NemoFeatureExtractor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        if self.model is None or self.tokenizer is None:
            print("Loading NeMo Curator model for classification...")
            try:
                self.model, self.tokenizer = nvidia_classifier.load_model()
            except Exception as e:
                print(f"Error loading model: {e}")
                raise e
                
    def get_features(self, prompt):
        """
        Fetches raw NeMo output JSON and token count for a prompt.
        """
        self.load_model()
        try:
            result = nvidia_classifier.analyze_prompt(self.model, self.tokenizer, prompt)
            
            prompt_tokens = self.tokenizer(prompt)["input_ids"]
            num_tokens = len(prompt_tokens)
            
            return {
                "nemo_raw_json": result,
                "num_tokens": num_tokens
            }
        except Exception as e:
            print(f"Error classifying prompt: {e}")
            return None

def predict_label(nemo_raw_output_json, compressed_prompt_len, scaler_path='advanced_scaler.pkl', model_path='best_advanced_model.pkl'):
    """
    Predicts the label for a given prompt using the trained advanced model.
    
    Args:
        nemo_raw_output_json (str or dict): The NeMo raw output containing prompt features.
        compressed_prompt_len (int/float): The length of the compressed prompt.
        scaler_path (str): Path to the saved StandardScaler.
        model_path (str): Path to the saved ensemble model.
        
    Returns:
        int: The predicted label (0 or 1).
    """
    # Load model and scaler
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except FileNotFoundError as e:
        raise RuntimeError(f"Required model or scaler file not found. Ensure {model_path} and {scaler_path} exist.") from e
    
    # Parse json if it's a string
    if isinstance(nemo_raw_output_json, str):
        data = json.loads(nemo_raw_output_json)
    else:
        data = nemo_raw_output_json
        
    # Helper to extract values from lists if present
    def extract_val(k):
        v = data.get(k, 0.0)
        return v[0] if isinstance(v, list) and len(v) > 0 else v

    # Build feature dictionary in precise order matching training
    # ['compressed_prompt_len', 'task_type_prob', 'creativity_scope', 'reasoning', 
    # 'contextual_knowledge', 'number_of_few_shots', 'domain_knowledge', 'constraint_ct']
    features = {
        'compressed_prompt_len': compressed_prompt_len,
        'task_type_prob': extract_val('task_type_prob'),
        'creativity_scope': extract_val('creativity_scope'),
        'reasoning': extract_val('reasoning'),
        'contextual_knowledge': extract_val('contextual_knowledge'),
        'number_of_few_shots': extract_val('number_of_few_shots'),
        'domain_knowledge': extract_val('domain_knowledge'),
        'constraint_ct': extract_val('constraint_ct'),
    }
    
    # Create DataFrame (1 row) to maintain column names 
    df_features = pd.DataFrame([features])
    
    # Scale features
    X_scaled = scaler.transform(df_features)
    
    # Predict
    pred = model.predict(X_scaled)
    
    return int(pred[0])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
    else:
        print("Please enter the prompt you want to predict:")
        user_prompt = input("> ")

    if user_prompt:
        extractor = NemoFeatureExtractor()
        print("Extracting features from NeMo...")
        features = extractor.get_features(user_prompt)
        
        if features:
            nemo_raw_json = features["nemo_raw_json"]
            num_tokens = features["num_tokens"]
            
            print(f"\nExtracted Features:")
            print(f"Tokens: {num_tokens}")
            print(f"Raw JSON: {json.dumps(nemo_raw_json)}")
            
            # Use current directory paths for the model and scaler to ensure they can be loaded
            current_dir = os.path.dirname(os.path.abspath(__file__))
            scaler_path = os.path.join(current_dir, 'advanced_scaler.pkl')
            model_path = os.path.join(current_dir, 'best_advanced_model.pkl')
            
            print(f"\nRunning prediction...")
            try:
                label = predict_label(
                    nemo_raw_output_json=nemo_raw_json, 
                    compressed_prompt_len=num_tokens,
                    scaler_path=scaler_path,
                    model_path=model_path
                )
                print(f"------------")
                print(f"✅ Predicted Label: {label}")
            except Exception as e:
                print(f"❌ Error during prediction: {e}")
    else:
        print("No prompt provided.")
