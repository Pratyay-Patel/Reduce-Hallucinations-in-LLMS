"""
Configuration module for LLM Hallucination + Prompt Compression Pipeline.

This module centralizes all configuration parameters including model names,
paths, hyperparameters, and reproducibility settings.
"""

from typing import List

# ============================================================================
# Model Configurations
# ============================================================================

# LLM models to test in the experiment
LLM_MODELS: List[str] = [
   "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#,
   #"microsoft/Phi-3-mini-4k-instruct"
]

# NLI model for entailment checking
NLI_MODEL_NAME: str = "roberta-large-mnli"

# Embedding model for self-consistency measurement
EMB_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# ============================================================================
# Hardware Configuration
# ============================================================================

# Device for model inference
DEVICE: str = "cpu"  # Use "cpu" if GPU not available

# ============================================================================
# Compression Settings
# ============================================================================

# Enable/disable prompt compression
COMPRESSION_ENABLED: bool = True

# Token threshold for compression (compress if prompt exceeds this)
COMPRESSION_THRESHOLD_TOKENS: int = 100

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
RESULTS_PATH: str = "results/experiment_results.csv"

# ============================================================================
# Generation Parameters
# ============================================================================

# Maximum number of new tokens to generate
MAX_NEW_TOKENS: int = 128

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
