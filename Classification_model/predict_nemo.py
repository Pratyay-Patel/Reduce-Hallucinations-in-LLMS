import json
import joblib
import pandas as pd
import numpy as np
import sys
import os

from sklearn.pipeline import Pipeline

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prompt_class_dir = os.path.join(parent_dir, "Nvidia prompt class")

if prompt_class_dir not in sys.path:
    sys.path.insert(0, prompt_class_dir)

nvidia_classifier = None

FEATURE_COLS = [
    "compressed_prompt_len",
    "task_type_prob",
    "creativity_scope",
    "reasoning",
    "contextual_knowledge",
    "number_of_few_shots",
    "domain_knowledge",
    "constraint_ct",
]


def _load_nvidia_classifier():
    global nvidia_classifier
    if nvidia_classifier is None:
        import nvidia_classifier as _nc  # type: ignore
        nvidia_classifier = _nc
    return nvidia_classifier


class NemoFeatureExtractor:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model(self):
        if self.model is None or self.tokenizer is None:
            print("Loading NeMo Curator model for classification...")
            nc = _load_nvidia_classifier()
            try:
                self.model, self.tokenizer = nc.load_model()
            except Exception as e:
                print(f"Error loading model: {e}")
                raise e

    def get_features(self, prompt):
        """
        Fetches raw NeMo output JSON and token count for a prompt.
        """
        self.load_model()
        nc = _load_nvidia_classifier()
        try:
            result = nc.analyze_prompt(self.model, self.tokenizer, prompt)

            prompt_tokens = self.tokenizer(prompt)["input_ids"]
            num_tokens = len(prompt_tokens)

            return {
                "nemo_raw_json": result,
                "num_tokens": num_tokens
            }
        except Exception as e:
            print(f"Error classifying prompt: {e}")
            return None


def predict_from_frame(X, model, scaler=None):
    """
    Predict labels from the 8 raw training features.

    If `model` is a sklearn Pipeline (scaler [+ poly + selector] + clf),
    it is applied directly. Otherwise features are scaled then passed to the clf.
    """
    if isinstance(X, pd.DataFrame):
        missing = [c for c in FEATURE_COLS if c not in X.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        X_in = X[FEATURE_COLS]
    else:
        X_in = np.asarray(X, dtype=float)
        if X_in.ndim == 1:
            X_in = X_in.reshape(1, -1)
        if X_in.shape[1] != len(FEATURE_COLS):
            raise ValueError(
                f"Expected {len(FEATURE_COLS)} features, got {X_in.shape[1]}"
            )
        X_in = pd.DataFrame(X_in, columns=FEATURE_COLS)

    if isinstance(model, Pipeline):
        return model.predict(X_in)

    if scaler is None:
        raise ValueError("scaler is required when the saved model is not a Pipeline")

    X_scaled = scaler.transform(X_in)
    n_in = getattr(model, "n_features_in_", None)
    if n_in is not None and int(n_in) != X_scaled.shape[1]:
        raise ValueError(
            f"Model expects {n_in} features after scaling, got {X_scaled.shape[1]}. "
            "Re-save best_advanced_model.pkl as a Pipeline from train_advanced.ipynb."
        )
    return model.predict(X_scaled)


def predict_label(nemo_raw_output_json, compressed_prompt_len, scaler_path='advanced_scaler.pkl', model_path='best_advanced_model.pkl'):
    """
    Predicts the label for a given prompt using the trained advanced model.

    Args:
        nemo_raw_output_json (str or dict): The NeMo raw output containing prompt features.
        compressed_prompt_len (int/float): The length of the compressed prompt.
        scaler_path (str): Path to the saved StandardScaler (used if the model is not a Pipeline).
        model_path (str): Path to the saved model or sklearn Pipeline.

    Returns:
        int: The predicted label (0 or 1).
    """
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    except FileNotFoundError as e:
        raise RuntimeError(f"Required model or scaler file not found. Ensure {model_path} and {scaler_path} exist.") from e

    if isinstance(nemo_raw_output_json, str):
        data = json.loads(nemo_raw_output_json)
    else:
        data = nemo_raw_output_json

    def extract_val(k):
        v = data.get(k, 0.0)
        return v[0] if isinstance(v, list) and len(v) > 0 else v

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

    df_features = pd.DataFrame([features])
    pred = predict_from_frame(df_features, model, scaler=scaler)
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
