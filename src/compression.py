"""
Compression module for prompt compression using LLMLingua-2.

This module provides functions to compress prompts that exceed a token threshold,
track token counts, and manage the LLMLingua-2 compression model.
"""

from typing import Tuple
from src.config import COMPRESSION_THRESHOLD_TOKENS, LLMLINGUA_MODEL
from src.config import FORCE_COMPRESSION

# Global compressor instance
_compressor = None
_llm_tokenizer = None


def init_compression(llm_tokenizer, llmlingua_model: str = LLMLINGUA_MODEL):
    """
    Initialize the prompt compressor with LLM tokenizer.
    Must be called before using compression functions.
    
    Args:
        llm_tokenizer: The tokenizer from the LLM model
        llmlingua_model: Name of the LLMLingua-2 model to use
    """
    global _compressor, _llm_tokenizer

    # Lazy import to avoid heavy model loading at module import time
    from llmlingua import PromptCompressor
    
    _llm_tokenizer = llm_tokenizer

    # device_map="auto" allows compressor placement on available accelerators.
    # If initialization fails, we disable compression instead of crashing runs.
    try:
        _compressor = PromptCompressor(
            model_name=llmlingua_model,
            use_llmlingua2=True,
            device_map="auto"
        )
        print("[INFO] Prompt compression initialized.")
    except Exception as err:
        print(f"[WARN] Compression model failed ({err}); disabling compression.")
        _compressor = None


def count_tokens(text: str) -> int:
    """
    Count tokens using the LLM's tokenizer.
    
    Args:
        text: The text to tokenize
        
    Returns:
        Number of tokens in the text
        
    Safe Behavior:
        Returns 0 if tokenizer has not been initialized (MVP-safe).
    """
    if _llm_tokenizer is None:
        # Safe fallback during staged development
        print("[WARN] Tokenizer not initialized. Skipping compression.")
        return 0
    
    tokens = _llm_tokenizer.encode(text)
    return len(tokens)


def maybe_compress_prompt(prompt: str) -> Tuple[str, int, int, bool]:
    """
    Compress prompt if it exceeds threshold.
    
    Args:
        prompt: The prompt text to potentially compress
        
    Returns:
        Tuple containing:
            - compressed_prompt: The (possibly compressed) prompt
            - orig_tokens: Original token count
            - compressed_tokens: Final token count
            - was_compressed: Whether compression was applied
            
    Safe Behavior:
        If compression is not initialized, returns original prompt without crashing.
    """
    if _compressor is None or _llm_tokenizer is None:
        # Safe fallback: no compression applied
        print("[WARN] Tokenizer not initialized. Skipping compression.")
        return prompt, 0, 0, False
    
    # Count original tokens
    orig_tokens = count_tokens(prompt)
    
    # Check if compression is needed
    if not FORCE_COMPRESSION and orig_tokens <= COMPRESSION_THRESHOLD_TOKENS:
        # No compression needed
        return prompt, orig_tokens, orig_tokens, False
    
    # Apply compression based on target token threshold
    compressed_result = _compressor.compress_prompt(
        prompt,
        target_token=COMPRESSION_THRESHOLD_TOKENS
    )
    
    compressed_prompt = compressed_result['compressed_prompt']
    compressed_tokens = count_tokens(compressed_prompt)
    
    return compressed_prompt, orig_tokens, compressed_tokens, True