"""Environment and Hugging Face access verification script.

This script verifies:
1) .env loading and HF_TOKEN presence
2) Access to each configured model via Hugging Face API

Run directly:
    python tests/verify_setup.py
"""

import os
import sys
from dotenv import load_dotenv
from huggingface_hub import HfApi

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import HF_TOKEN, LLM_MODELS


def main() -> None:
    load_dotenv()
    print("[INFO] Verifying environment and model access...")

    if HF_TOKEN:
        print("✅ HF_TOKEN found")
    else:
        print("❌ HF_TOKEN missing")

    api = HfApi()

    for model in LLM_MODELS:
        try:
            api.model_info(model, token=HF_TOKEN)
            print(f"✅ ACCESS OK: {model}")
        except Exception as err:
            print(f"❌ ACCESS FAILED: {model} -> {err}")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"[ERROR] verify_setup failed unexpectedly: {err}")
