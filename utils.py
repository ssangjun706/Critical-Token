import re
import numpy as np
import random
import torch

from typing import Iterable
from collections import Counter


def set_random_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_pass_at_k(samples, k):
    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    num_samples = len(samples)
    num_correct = np.sum(samples)
    return estimator(int(num_samples), int(num_correct), k)


def normalize_number(expr):
    expr = str(expr).strip()

    expr = expr.replace(",", "")
    expr = expr.replace("_", "")
    expr = expr.replace(" ", "")

    try:
        return float(expr)
    except:
        pass

    expr = expr.replace("^", "**")

    if re.match(r"^[\d\+\-\*\/\(\)\.\*\*\s]+$", expr):
        try:
            return eval(expr, {"__builtins__": {}}, {})
        except:
            pass

    return expr


def is_correct(generated_answer, true_answer):
    gen_str = str(generated_answer).strip()
    true_str = str(true_answer).strip()

    if gen_str == true_str:
        return True

    try:
        gen_norm = normalize_number(generated_answer)
        true_norm = normalize_number(true_answer)

        if gen_norm == true_norm:
            return True
        elif isinstance(gen_norm, (int, float)) and isinstance(true_norm, (int, float)):
            return abs(gen_norm - true_norm) < 1e-10

        return False
    except:
        pass

    return False


def majority_vote(predictions: Iterable) -> str | None:
    if not predictions:
        return None

    counter = Counter(predictions)
    return counter.most_common(1)[0][0]
