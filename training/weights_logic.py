import json
import os

# -------------------------------------------------------------
# Compute weights from accuracy data
# -------------------------------------------------------------
def compute_weights_from_accuracy(per_field_accuracy, min_w=0.4, max_w=1.6):
    """
    Compute dynamic field weights based on accuracy.
    Fields with low accuracy get higher weights (more importance).
    Fields with high accuracy get lower weights (less need for learning).

    Args:
        per_field_accuracy (dict): {"Field": accuracy_percent, ...}
        min_w (float): minimum possible weight
        max_w (float): maximum possible weight

    Returns:
        dict: {"Field": weight_value, ...}
    """
    weights = {}
    if not per_field_accuracy:
        return weights

    for field, acc in per_field_accuracy.items():
        try:
            acc_val = float(acc) / 100.0
            # Invert accuracy: lower acc → higher weight
            weight = 1.0 / max(0.1, acc_val + 0.05)
            # Normalize and clamp
            weight = max(min_w, min(max_w, weight))
            weights[field] = round(weight, 3)
        except Exception:
            weights[field] = 1.0

    return weights


# -------------------------------------------------------------
# Sanitize weights
# -------------------------------------------------------------
def sanitize_weights(weights, min_w=0.4, max_w=1.6):
    """
    Ensure weights are numeric and within bounds.
    """
    if not weights:
        return False, "No weights provided"

    cleaned = {}
    for field, w in weights.items():
        try:
            wv = float(w)
            if wv < min_w or wv > max_w:
                return False, f"Weight for {field} out of range ({wv})"
            cleaned[field] = round(wv, 3)
        except Exception:
            return False, f"Invalid weight value for {field}"

    return True, "All weights valid"


# -------------------------------------------------------------
# Save weights file
# -------------------------------------------------------------
def save_weights_file(weights, path="training/field_weights.json"):
    """
    Save weights dictionary to file for persistent use.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)
    print(f"[INFO] Saved {len(weights)} weights to {path}")
