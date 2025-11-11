# training/weights.py
from __future__ import annotations
import json
from typing import Dict, Tuple

from engine.schema import FIELD_ORDER

DEFAULT_MIN_W = 0.6
DEFAULT_MAX_W = 1.6

def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def compute_weights_from_accuracy(
    per_field_accuracy: Dict[str, int],
    min_w: float = DEFAULT_MIN_W,
    max_w: float = DEFAULT_MAX_W
) -> Dict[str, float]:
    """
    Map per-field accuracy (%) -> weight in [min_w, max_w], linearly.
    50% -> min_w, 100% -> max_w (below 50% just clamps to min_w).
    Unknown fields get 1.0 (neutral) so we don't penalize missing metrics.
    """
    weights: Dict[str, float] = {}
    for field in FIELD_ORDER:
        acc = per_field_accuracy.get(field)
        if acc is None:
            weights[field] = 1.0
            continue
        try:
            a = float(acc)
        except Exception:
            a = 50.0
        a = _clip(a, 50.0, 100.0)
        # linear map: 50%->min_w, 100%->max_w
        w = min_w + (a - 50.0) * (max_w - min_w) / 50.0
        weights[field] = _clip(w, min_w, max_w)
    return weights

def sanitize_weights(weights: Dict[str, float],
                     min_w: float = DEFAULT_MIN_W,
                     max_w: float = DEFAULT_MAX_W) -> Tuple[bool, str]:
    """
    Ensure only known fields and safe numeric ranges.
    """
    unknown_keys = [k for k in weights.keys() if k not in FIELD_ORDER]
    if unknown_keys:
        return False, f"Unknown weight keys: {unknown_keys}"
    for k, v in weights.items():
        try:
            fv = float(v)
        except Exception:
            return False, f"Weight for '{k}' is not a number: {v}"
        if not (min_w <= fv <= max_w):
            return False, f"Weight for '{k}' out of range [{min_w},{max_w}]: {fv}"
    return True, "ok"

def to_pretty_json(weights: Dict[str, float]) -> str:
    return json.dumps(weights, indent=2, sort_keys=True, ensure_ascii=False)

def load_weights(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_weights(path: str, weights: Dict[str, float]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_pretty_json(weights))
        f.write("\n")
