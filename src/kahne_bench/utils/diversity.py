"""
Dataset diversity validation utilities.

Implements Self-BLEU and ROUGE metrics to ensure generated test cases
have low n-gram overlap and high semantic variety.
"""

from collections import Counter
from typing import Sequence
import math


def _get_ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    """Extract n-grams from text."""
    words = text.lower().split()
    return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]


def _count_ngrams(text: str, n: int) -> Counter:
    """Count n-grams in text."""
    return Counter(_get_ngrams(text, n))


def calculate_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """
    Calculate BLEU score between reference and hypothesis.

    Args:
        reference: Reference text
        hypothesis: Hypothesis text to score
        max_n: Maximum n-gram size

    Returns:
        BLEU score (0-1)
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not hyp_words:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_words) / len(hyp_words))) if hyp_words else 0.0

    # Calculate precision for each n
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = _count_ngrams(reference, n)
        hyp_ngrams = _count_ngrams(hypothesis, n)

        if not hyp_ngrams:
            precisions.append(0.0)
            continue

        # Clipped counts
        clipped = sum(
            min(count, ref_ngrams.get(ngram, 0))
            for ngram, count in hyp_ngrams.items()
        )
        total = sum(hyp_ngrams.values())

        precisions.append(clipped / total if total > 0 else 0.0)

    # Geometric mean of precisions
    if all(p > 0 for p in precisions):
        log_precisions = sum(math.log(p) for p in precisions)
        geo_mean = math.exp(log_precisions / len(precisions))
    else:
        geo_mean = 0.0

    return bp * geo_mean


def calculate_self_bleu(texts: Sequence[str], sample_size: int = 100) -> float:
    """
    Calculate Self-BLEU score for a corpus of texts.

    Lower Self-BLEU indicates higher diversity (less similarity between texts).

    Args:
        texts: List of text samples
        sample_size: Maximum number of pairs to evaluate

    Returns:
        Average Self-BLEU score (0-1, lower = more diverse)
    """
    if len(texts) < 2:
        return 0.0

    import random
    texts_list = list(texts)

    # Sample pairs if corpus is large
    if len(texts) > sample_size:
        indices = random.sample(range(len(texts)), min(sample_size, len(texts)))
        texts_list = [texts_list[i] for i in indices]

    total_bleu = 0.0
    count = 0

    for i, hyp in enumerate(texts_list):
        for j, ref in enumerate(texts_list):
            if i != j:
                total_bleu += calculate_bleu(ref, hyp)
                count += 1

    return total_bleu / count if count > 0 else 0.0


def calculate_rouge_similarity(text1: str, text2: str) -> dict[str, float]:
    """
    Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Dictionary with rouge-1, rouge-2, rouge-l scores
    """
    words1 = text1.lower().split()
    words2 = text2.lower().split()

    if not words1 or not words2:
        return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}

    # ROUGE-1 (unigrams)
    set1 = set(words1)
    set2 = set(words2)
    overlap = set1 & set2
    rouge1 = 2 * len(overlap) / (len(set1) + len(set2)) if (set1 or set2) else 0.0

    # ROUGE-2 (bigrams)
    bigrams1 = set(_get_ngrams(text1, 2))
    bigrams2 = set(_get_ngrams(text2, 2))
    overlap2 = bigrams1 & bigrams2
    rouge2 = 2 * len(overlap2) / (len(bigrams1) + len(bigrams2)) if (bigrams1 or bigrams2) else 0.0

    # ROUGE-L (longest common subsequence) - O(min(m,n)) space optimized
    def lcs_length(x: list, y: list) -> int:
        # Optimize: use shorter sequence for column dimension
        if len(x) < len(y):
            x, y = y, x
        m, n = len(x), len(y)
        if n == 0:
            return 0
        # Two-row approach instead of full table
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(m):
            for j in range(n):
                if x[i] == y[j]:
                    curr[j + 1] = prev[j] + 1
                else:
                    curr[j + 1] = max(prev[j + 1], curr[j])
            prev, curr = curr, prev
        return prev[n]

    lcs = lcs_length(words1, words2)
    precision = lcs / len(words2) if words2 else 0.0
    recall = lcs / len(words1) if words1 else 0.0
    rouge_l = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "rouge-1": rouge1,
        "rouge-2": rouge2,
        "rouge-l": rouge_l,
    }


def validate_dataset_diversity(
    prompts: Sequence[str],
    max_self_bleu: float = 0.4,
    max_rouge: float = 0.5,
    sample_size: int = 100,
) -> dict:
    """
    Validate that a dataset of prompts has sufficient diversity.

    Args:
        prompts: List of prompt texts
        max_self_bleu: Maximum acceptable Self-BLEU (lower = more diverse)
        max_rouge: Maximum acceptable average ROUGE
        sample_size: Sample size for metrics

    Returns:
        Validation report with metrics and pass/fail status
    """
    self_bleu = calculate_self_bleu(prompts, sample_size)

    # Sample ROUGE calculations
    import random
    if len(prompts) > sample_size:
        sample_indices = random.sample(range(len(prompts)), sample_size)
        sample = [prompts[i] for i in sample_indices]
    else:
        sample = list(prompts)

    rouge_scores = []
    for i in range(min(len(sample) - 1, 50)):
        for j in range(i + 1, min(len(sample), i + 10)):
            scores = calculate_rouge_similarity(sample[i], sample[j])
            rouge_scores.append(scores["rouge-1"])

    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

    passed = self_bleu <= max_self_bleu and avg_rouge <= max_rouge

    return {
        "self_bleu": self_bleu,
        "average_rouge": avg_rouge,
        "num_prompts": len(prompts),
        "diversity_passed": passed,
        "thresholds": {
            "max_self_bleu": max_self_bleu,
            "max_rouge": max_rouge,
        },
    }
