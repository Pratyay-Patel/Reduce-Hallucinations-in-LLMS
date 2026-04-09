"""
Models module for LLM Hallucination + Prompt Compression Pipeline.

This module handles loading and inference for all neural network models:
- LLM (Phi-3-mini-4k-instruct) for answer generation
- NLI model (roberta-large-mnli) for entailment checking
- Embedding model (sentence-transformers) for self-consistency measurement
"""

from typing import List, Tuple
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

from src.config import (
    DEVICE,
    MAX_NEW_TOKENS,
    TEMPERATURE_DEFAULT,
    TOP_P_DEFAULT,
    USE_FP16,
    USE_4BIT,
    HF_TOKEN,
    NLI_MODEL_NAME,
    EMB_MODEL_NAME
)


def load_with_optional_token(model_name, loader_fn, **kwargs):
    if HF_TOKEN:
        try:
            print("[INFO] Using Hugging Face token for model access")
            return loader_fn(model_name, token=HF_TOKEN, **kwargs)
        except Exception as e:
            print(f"[WARN] Token-based load failed: {e}")

    if "meta-llama" in model_name:
        raise ValueError("HF_TOKEN is required for gated models like LLaMA")

    print("[INFO] Loading public model without token")
    return loader_fn(model_name, **kwargs)


def load_llm(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load LLM with FP16 precision and automatic device mapping.
    
    Uses FP16 to reduce memory usage and device_map="auto" to automatically
    distribute model layers across available GPU memory.
    
    Args:
        model_name: HuggingFace model identifier (e.g., "microsoft/Phi-3-mini-4k-instruct")
    
    Returns:
        Tuple of (model, tokenizer) where model is in eval mode
    
    Requirements: 2.1
    """
    print(f"Loading model: {model_name}")

    if "Llama-2-7b-chat-hf" in model_name:
        print("Loading gated model (LLaMA-2)... checking HF_TOKEN in environment")

    tokenizer = load_with_optional_token(
        model_name,
        AutoTokenizer.from_pretrained,
        trust_remote_code=True
    )

    # Safety: ensure pad_token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # device_map="auto" lets Transformers place weights on available GPU memory.
    # This is critical for fitting 7B/8B models reliably on 32GB cards.
    preferred_dtype = torch.float16 if USE_FP16 and torch.cuda.is_available() else torch.float32
    model = None

    try:
        model = load_with_optional_token(
            model_name,
            AutoModelForCausalLM.from_pretrained,
            torch_dtype=preferred_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        if preferred_dtype == torch.float16 and torch.cuda.is_available():
            print("[INFO] Loaded using FP16 with automatic GPU mapping.")
        else:
            print("[INFO] Loaded using automatic device mapping.")
    except RuntimeError as err:
        # If OOM or CUDA initialization fails, try optional 4-bit loading first.
        print(f"[WARN] Primary GPU load failed: {err}")
        if USE_4BIT and torch.cuda.is_available():
            print("[INFO] Retrying with 4-bit quantization...")
            try:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                model = load_with_optional_token(
                    model_name,
                    AutoModelForCausalLM.from_pretrained,
                    quantization_config=quant_config,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                print("[INFO] Loaded using 4-bit quantization with auto device map.")
            except Exception as quant_err:
                print(f"[WARN] 4-bit load failed: {quant_err}")

        # Final fallback path: force CPU so runs do not crash in production scripts.
        if model is None:
            print("[WARN] Falling back to CPU inference for this model.")
            model = load_with_optional_token(
                model_name,
                AutoModelForCausalLM.from_pretrained,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model.eval()

    return model, tokenizer


def generate_answers(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    num_return_sequences: int = 1,
    temperature: float = TEMPERATURE_DEFAULT,
    top_p: float = TOP_P_DEFAULT,
    max_new_tokens: int = MAX_NEW_TOKENS
) -> List[str]:
    """
    Generate one or more answers from the LLM.
    
    Uses sampling when num_return_sequences > 1 to enable diverse outputs.
    Returns decoded text without special tokens.
    
    Args:
        model: Loaded LLM model
        tokenizer: Corresponding tokenizer
        prompt: Input prompt text
        num_return_sequences: Number of answers to generate (default 1)
        temperature: Sampling temperature (default from config)
        top_p: Nucleus sampling parameter (default from config)
        max_new_tokens: Maximum tokens to generate (default from config)
    
    Returns:
        List of decoded answer strings
    
    Requirements: 2.2, 2.4
    """
    # Tokenize input with an explicit cap so long-context datasets do not exceed
    # the model's maximum sequence length and trigger attention OOMs.
    model_max_length = getattr(model.config, "max_position_embeddings", None)
    if model_max_length is None:
        model_max_length = getattr(tokenizer, "model_max_length", 4096)
    if not model_max_length or model_max_length > 100000:
        model_max_length = 4096

    max_input_length = max(1, int(model_max_length) - int(max_new_tokens) - 8)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )

    input_token_count = inputs["input_ids"].shape[1]
    if input_token_count >= max_input_length:
        print(
            f"[WARN] Prompt truncated to {input_token_count} tokens "
            f"to fit within the model context window ({model_max_length})."
        )

    # Move to model device safely (works with device_map="auto")
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    
    # Enable sampling ONLY when generating multiple sequences
    do_sample = num_return_sequences > 1

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode outputs without special tokens
    answers = []
    input_length = inputs['input_ids'].shape[1]

    for output in outputs:
        generated_tokens = output[input_length:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        answers.append(answer.strip())
    
    return answers


def load_nli_model() -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Load roberta-large-mnli for entailment checking.
    
    Returns model in eval mode on configured device.
    
    Returns:
        Tuple of (model, tokenizer) for NLI inference
    
    Requirements: 2.3, 7.4
    """
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    
    return model, tokenizer


def load_embedding_model() -> SentenceTransformer:
    """
    Load sentence transformer for semantic similarity.
    
    Used for computing self-consistency scores via cosine similarity
    between answer embeddings.
    
    Returns:
        SentenceTransformer model ready for encoding
    
    Requirements: 2.3
    """
    model = SentenceTransformer(EMB_MODEL_NAME, device=DEVICE)
    
    return model


def clear_gpu() -> None:
    """
    Best-effort GPU memory cleanup between large-model runs.

    This helps avoid fragmentation/OOM when loading multiple 7B/8B models
    sequentially in one process.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()