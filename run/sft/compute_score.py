"""Custom reward function for verl.trainer.main_eval.

Parses an integer `answer` from a JSON-wrapped response and returns hard
accuracy against the rounded ground truth. MAE is computed separately in
run/sft/aggregate_results.py (main_eval aggregates scores via np.mean and
thresholding, so the scorer must return a scalar float).
"""

from __future__ import annotations

import json
import re

_JSON_RE = re.compile(
    r'\{[^{}]*"answer"\s*:\s*([-+]?\d+(?:\.\d+)?)[^{}]*\}', re.DOTALL
)
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def extract_answer(response: str):
    """Return a float prediction or None if no number could be parsed."""
    if not response:
        return None
    m = _JSON_RE.search(response)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    try:
        obj = json.loads(response)
        if isinstance(obj, dict) and "answer" in obj:
            return float(obj["answer"])
    except Exception:
        pass
    m = _NUM_RE.search(response)
    return float(m.group(0)) if m else None


def compute_score_data_source(data_source, response, ground_truth, extra_info=None):
    pred = extract_answer(response)
    if pred is None:
        return 0.0
    try:
        gt = float(ground_truth)
    except (TypeError, ValueError):
        return 0.0
    return 1.0 if round(pred) == round(gt) else 0.0
