"""
Dataset loading module for JSONL files.

This module provides functions to load and parse JSONL (JSON Lines) files
containing benchmark datasets like GSM8K and SQuAD v2.
"""

import json
import logging
from typing import Dict, Iterable, List

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> Iterable[Dict]:
    """
    Load a single JSONL file, yielding one dict per line.
    Skips malformed lines with a warning.
    
    Args:
        path: Path to the JSONL file
        
    Yields:
        Dict containing parsed JSON object from each valid line
        
    Requirements: 1.1, 1.3
    """
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                yield data
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON at {path}:{line_num} - {e}")
                continue


def load_all_datasets(paths: List[str]) -> Iterable[Dict]:
    """
    Load multiple JSONL files and yield all samples.
    Each sample contains: id, dataset, context, question, answer
    
    Args:
        paths: List of paths to JSONL files
        
    Yields:
        Dict containing sample data with fields: id, dataset, context, question, answer
        
    Requirements: 1.2
    """
    for path in paths:
        yield from load_jsonl(path)
