"""Orchestrator for setup and lightweight pipeline tests.

Run:
    python tests/test_all.py
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import verify_setup
import test_pipeline


def main() -> None:
    print("===== VERIFYING SETUP =====")
    try:
        verify_setup.main()
    except Exception as err:
        print(f"[ERROR] Setup verification failed: {err}")

    print("===== TESTING PIPELINE =====")
    try:
        test_pipeline.main()
    except Exception as err:
        print(f"[ERROR] Pipeline test failed: {err}")

    print("===== ALL TESTS COMPLETED =====")


if __name__ == "__main__":
    main()
