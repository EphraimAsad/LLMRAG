import re
import json
from datetime import datetime
from typing import List, Dict, Tuple, Any
from engine.parser_rules_runtime import normalize_field, normalize_value


# ------------------------------
# Field + value validation
# ------------------------------
def _get_allowed_fields() -> List[str]:
    """Load all allowed field names from schema.py (fallback to default list)."""
    try:
        from engine.schema import ALL_FIELDS
        return ALL_FIELDS
    except Exception:
        return [
            "Gram Stain", "Shape", "Catalase", "Oxidase", "Coagulase", "Urease",
            "Indole", "Citrate", "Methyl Red", "VP", "Haemolysis", "Haemolysis Type",
            "Motility", "Spore Formation", "Capsule", "Oxygen Requirement",
            "Growth Temperature", "Dnase", "ONPG", "Nitrate Reduction",
            "Gelatin Hydrolysis", "Esculin Hydrolysis", "Lysine Decarboxylase",
            "Ornitihine Decarboxylase", "Arginine dihydrolase", "H2S",
            "Glucose Fermentation", "Lactose Fermentation", "Sucrose Fermentation",
            "Maltose Fermentation", "Mannitol Fermentation", "Sorbitol Fermentation",
            "Xylose Fermentation", "Rhamnose Fermentation", "Arabinose Fermentation",
            "Raffinose Fermentation", "Trehalose Fermentation", "Inositol Fermentation",
            "NaCl Tolerant (>=6%)", "Media Grown On", "Colony Morphology"
        ]


def _get_allowed_values(field: str) -> List[str]:
    """Allowed canonical values per field."""
    simple = ["Positive", "Negative", "Variable", "Unknown"]
    field = normalize_field(field)
    if field in {"Gram Stain"}:
        return ["Positive", "Negative", "Variable"]
    if field == "Haemolysis Type":
        return ["None", "Alpha", "Beta", "Gamma"]
    if field == "Oxygen Requirement":
        return ["Aerobic", "Anaerobic", "Microaerophilic", "Capnophilic", "Facultative Anaerobe", "Intracellular"]
    return simple


# ------------------------------
# Regex safety check
# ------------------------------
def _safe_regex(pattern: str) -> bool:
    """Return True if regex compiles and is safe."""
    try:
        if not pattern or len(pattern) > 256:
            return False
        if "(.*)+" in pattern or "(.*){2,}" in pattern:
            return False
        re.compile(pattern, re.IGNORECASE)
        return True
    except re.error:
        return False


# ------------------------------
# Main sanitization
# ------------------------------
def sanitize_rules(rules: List[Dict[str, Any]]) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """
    Validates and cleans rule suggestions.
    Returns: (ok, message, sanitized_rules)
    """
    if not rules:
        return False, "No rules provided", []

    allowed_fields = _get_allowed_fields()
    cleaned = []
    errors = []

    for r in rules:
        try:
            field = normalize_field(r.get("field", ""))
            value = normalize_value(field, r.get("value", ""))
            pattern = r.get("pattern", "")
            ptype = r.get("pattern_type", "phrase").lower()
            case_sensitive = bool(r.get("case_sensitive", False))

            # Validate field
            if field not in allowed_fields:
                errors.append(f"Invalid field '{field}'")
                continue

            # Validate value
            allowed_vals = _get_allowed_values(field)
            if value not in allowed_vals:
                errors.append(f"Invalid value '{value}' for {field}")
                continue

            # Validate pattern
            if not pattern or len(pattern) < 2:
                errors.append(f"Empty or invalid pattern for {field}")
                continue

            if ptype == "regex":
                if not _safe_regex(pattern):
                    errors.append(f"Unsafe regex: {pattern}")
                    continue
            elif ptype != "phrase":
                errors.append(f"Unknown pattern_type: {ptype}")
                continue

            # Passed — build cleaned entry
            cleaned.append({
                "field": field,
                "value": value,
                "pattern_type": ptype,
                "pattern": pattern.strip(),
                "case_sensitive": case_sensitive,
                "priority": int(r.get("priority", 50)),
                "source": r.get("source", "gold"),
                "created_at": r.get("created_at", datetime.now().isoformat()),
                "status": "active"
            })

        except Exception as e:
            errors.append(f"Error in rule: {r} -> {e}")

    if errors:
        return False, f"Some rules invalid: {len(errors)} problems", cleaned

    return True, f"All {len(cleaned)} rules validated successfully.", cleaned


# ------------------------------
# CLI for quick test
# ------------------------------
if __name__ == "__main__":
    import os

    path = os.path.join("training", "rule_candidates.json")
    if not os.path.exists(path):
        raise FileNotFoundError("Missing training/rule_candidates.json")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ok, msg, cleaned = sanitize_rules(data.get("rules", []))
    print(msg)
    print(json.dumps(cleaned[:3], indent=2))
