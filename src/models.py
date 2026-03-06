"""
Models module for LLM Hallucination + Prompt Compression Pipeline.

This module handles loading and inference for all neural network models:
- LLM (Phi-3-mini-4k-instruct) for answer generation
- NLI model (roberta-large-mnli) for entailment checking
- Embedding model (sentence-transformers) for self-consistency measurement
"""

from typing import List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

from src.config import (
    DEVICE,
    MAX_NEW_TOKENS,
    TEMPERATURE_DEFAULT,
    TOP_P_DEFAULT,
    NLI_MODEL_NAME,
    EMB_MODEL_NAME
)


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
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # Safety: ensure pad_token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model =AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
    )

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    #model.to(DEVICE)
    
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
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")

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