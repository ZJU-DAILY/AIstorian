import jieba
import evaluate
from datasets import load_metric

def compute_rouge(predictions, references, language="zh"):
    """
    Function to compute ROUGE scores.

    Parameters:
    - predictions (list): List of predicted texts.
    - references (list): List of reference texts.

    Returns:
    - dict: Dictionary containing ROUGE scores (e.g., ROUGE-1, ROUGE-2, ROUGE-L).
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")

    rouge = evaluate.load('./evaluate/metrics/rouge')  # Adjust to local path to avoid downloading from hub
    results = rouge.compute(predictions=predictions, references=references, tokenizer=lambda x: jieba.lcut(x) if language == "zh" else None)
    return results

def compute_rag_metric(predictions, references):
    """
    Function to compute RAG metrics.
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")

    metric = load_metric("./summary/metrics/rag/f1.py")
    # threshold and similarity_func can be changed later
    results = metric.compute(predictions=predictions, references=references)
    return results

