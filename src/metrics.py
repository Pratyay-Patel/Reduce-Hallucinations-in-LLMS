import re
import string
import numpy as np
import torch
import torch.nn.functional as F
import nltk
from nltk.corpus import stopwords


# ==============================
# Helper Functions for EM
# ==============================

def _normalize_text(text: str) -> str:
    """
    Normalize text for string comparison:
    - Lowercase
    - Remove punctuation
    - Remove extra whitespace
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_last_number(text: str):
    """
    Extract the last occurring number from text.
    Handles:
    - integers
    - decimals
    - numbers with commas
    """
    if not text:
        return None

    text = text.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", text)

    if numbers:
        return numbers[-1]

    return None


STOPWORDS = set(stopwords.words('english'))

# ==============================
# Exact Match (Upgraded)
# ==============================

def exact_match(prediction: str, gold: str) -> float:
    """
    Research-grade Exact Match:

    1. If gold contains a number → compare numeric answers.
    2. Otherwise → compare normalized text.
    """

    if not prediction or not gold:
        return 0.0

    # ---- Numeric comparison (for GSM8K-like datasets) ----
    gold_number = _extract_last_number(gold)
    pred_number = _extract_last_number(prediction)

    if gold_number is not None:
        return float(gold_number == pred_number)

    # ---- Fallback: normalized string comparison ----
    norm_pred = _normalize_text(prediction)
    norm_gold = _normalize_text(gold)

    return float(norm_pred == norm_gold)


# ==============================
# Keyword Match (Semantic Lenient)
# ==============================

def keyword_match_score(prediction: str, gold: str) -> float:
    """
    Compute keyword overlap score.
    Returns 1.0 if all gold keywords appear in prediction.
    """

    if not prediction or not gold:
        return 0.0

    norm_pred = _normalize_text(prediction)
    norm_gold = _normalize_text(gold)

    gold_tokens = [
        t for t in norm_gold.split()
        if t not in STOPWORDS
    ]

    pred_tokens = set(norm_pred.split())

    if not gold_tokens:
        return 0.0

    match_count = sum(1 for t in gold_tokens if t in pred_tokens)

    return float(match_count == len(gold_tokens))

# ==============================
# Self-Consistency (unchanged)
# ==============================

def self_consistency_score(responses, embedding_model):
    if not responses:
        return 0.0

    embeddings = embedding_model.encode(responses, convert_to_tensor=True)
    similarity_matrix = F.cosine_similarity(
        embeddings.unsqueeze(1),
        embeddings.unsqueeze(0),
        dim=2
    )

    # Exclude diagonal
    n = len(responses)
    if n <= 1:
        return 1.0

    score = (similarity_matrix.sum() - n) / (n * (n - 1))
    return float(score)


# ==============================
# NLI Support Score (unchanged)
# ==============================

def nli_support_score(model, tokenizer, premise: str, hypothesis: str) -> float:
    if not premise or not hypothesis:
        return 0.0

    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)

    # Entailment index = 2 for roberta-large-mnli
    entailment_prob = probs[0][2].item()

    return float(entailment_prob)