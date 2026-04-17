"""
Configuration module for LLM Hallucination + Prompt Compression Pipeline.

This module centralizes all configuration parameters including model names,
paths, hyperparameters, and reproducibility settings.
"""

from typing import List
import os
import torch
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# ============================================================================
# Model Configurations
# ============================================================================

# LLM models to test in the experiment.
# Ordered to run sequentially with explicit cleanup between models in runner.py.
# LLaMA-2 is included as a classical 7B baseline.
# LLaMA-3 is included as a modern architecture baseline.
LLM_MODELS: List[str] = [
    "microsoft/Phi-3-mini-4k-instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct"
]

# NLI model for entailment checking
NLI_MODEL_NAME: str = "roberta-large-mnli"

# Embedding model for self-consistency measurement
EMB_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# ============================================================================
# Hardware Configuration
# ============================================================================

# Device for model inference.
# Auto-select CUDA when available; all model loaders can still fall back to CPU.
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# Compression Settings
# ============================================================================

# Enable/disable prompt compression
COMPRESSION_ENABLED: bool = True

# Token threshold for compression (compress if prompt exceeds this).
# Raised to reduce over-compression on long-context prompts while still controlling GPU memory.
COMPRESSION_THRESHOLD_TOKENS: int = 2000

NLI_HALLUCINATION_THRESHOLD = 0.5

FORCE_COMPRESSION = True

# LLMLingua-2 model for compression
LLMLINGUA_MODEL: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

# ============================================================================
# Data Paths
# ============================================================================

# Paths to dataset JSONL files
DATA_PATHS: List[str] = [
    "data/gsm8k_subset.jsonl",
    "data/squad_v2_subset.jsonl",
    "data/hotpotqa_subset.jsonl",
    "data/triviaqa_subset.jsonl"
]

# Path for results CSV output
# Default points to the long-running file so checkpoint resume continues correctly.
RESULTS_PATH: str = "results/experiment_results_final.csv"

# ============================================================================
# Generation Parameters
# ============================================================================

# Maximum number of new tokens to generate.
# Reduced to cap decode-time memory and latency for 7B/8B models on 32GB VRAM.
MAX_NEW_TOKENS: int = 64

# Large-model loading safety toggles.
# FP16 halves activation/weight memory vs FP32 on supported GPUs.
# 4-bit can be enabled as an emergency path when OOM occurs.
USE_FP16: bool = True
USE_4BIT: bool = False

# Default temperature for sampling (0 = deterministic)
TEMPERATURE_DEFAULT: float = 0.7

# Default top-p (nucleus sampling) parameter
TOP_P_DEFAULT: float = 0.9

# Number of answer samples to generate for self-consistency measurement
SELF_CONSISTENCY_SAMPLES: int = 1

# ============================================================================
# Reproducibility
# ============================================================================

# Random seed for reproducibility
SEED: int = 42
