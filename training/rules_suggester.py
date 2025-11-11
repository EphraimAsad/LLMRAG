import re
import json
from datetime import datetime
from typing import Dict, List, Any
from engine.parser_llm import evaluate_gold_tests
from engine.parser_rules_runtime import normalize_field, normalize_value


# ------------------------------
# Helper: find indicative phrases
# ------------------------------
def extract_candidate_phrases(text: str, field: str, expected_value: str) -> List[str]:
    """
    Try to extract potential clue phrases from the text that could indicate the expected value.
    This is rule-based and simple for now — we'll enhance later with NLP/LLM assistance.
    """
    text = text.lower()
    expected_value = expected_value.lower()
    phrases = []

    # Basic patterns for biochemical wording
    core_words = [
        "positive", "negative", "variable", "ferments", "non-", "oxidase", "catalase",
        "coagulase", "haemolytic", "motile", "non-motile", "aerobic", "anaerobic",
        "microaerophilic", "capnophilic", "facultative", "urease", "indole", "dnase",
        "vp", "mr", "h2s", "beta", "alpha", "gamma", "tolerant", "resistant",
        "pigment", "colonies", "agar", "acid", "rod", "cocci", "sugar", "growth"
    ]

    # Search for small context windows around key words and expected values
    for word in core_words:
        pattern = rf".{{0,25}}{word}.{{0,25}}"
        for m in re.finditer(pattern, text):
            snippet = m.group(0).strip(",. ")
            if len(snippet) >= 4 and expected_value in snippet:
                phrases.append(snippet)
            elif any(ev in snippet for ev in ["positive", "negative", "variable", expected_value]):
                phrases.append(snippet)

    # Deduplicate short phrases
    phrases = list(dict.fromkeys(phrases))
    return phrases[:3]  # keep top 3 max


# ------------------------------
# Suggest new rules
# ------------------------------
def suggest_rules_from_gold(gold_data: List[Dict[str, Any]], model_name: str, use_llm: bool = True) -> List[Dict[str, Any]]:
    """
    Runs evaluation and proposes rule candidates for any misparsed fields.
    Returns a list of candidate rule dicts ready for sanitization.
    """
    summary = evaluate_gold_tests(gold_data, model_name=model_name, use_llm=use_llm)
    failed_cases = summary["failed_cases"]

    suggestions = []
    for fail in failed_cases:
        field = normalize_field(fail["field"])
        exp_value = normalize_value(field, fail["expected"])
        text = fail["text"]

        phrases = extract_candidate_phrases(text, field, exp_value)
        for phrase in phrases:
            # Heuristic for phrase vs regex
            if re.search(r"\b[-\s]\b", phrase) or len(phrase.split()) > 2:
                pattern_type = "regex"
                pattern = re.escape(phrase)
            else:
                pattern_type = "phrase"
                pattern = phrase

            suggestions.append({
                "field": field,
                "value": exp_value,
                "pattern_type": pattern_type,
                "pattern": pattern,
                "case_sensitive": False,
                "priority": 50,
                "source": "gold",
                "created_at": datetime.now().isoformat(),
                "status": "active"
            })

    # Deduplicate by (field, pattern)
    seen = set()
    unique_suggestions = []
    for s in suggestions:
        sig = (s["field"].lower(), s["pattern"].lower())
        if sig not in seen:
            seen.add(sig)
            unique_suggestions.append(s)

    print(f"[INFO] Suggested {len(unique_suggestions)} new candidate rules from {len(failed_cases)} failed fields.")
    return unique_suggestions


# ------------------------------
# Save candidates for review
# ------------------------------
def save_rule_suggestions(suggestions: List[Dict[str, Any]], out_path: str = "training/rule_candidates.json"):
    """
    Save all candidate rules to file for human review (and later sanitization).
    """
    os.makedirs("training", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "rules": suggestions}, f, indent=2)
    print(f"[INFO] Saved {len(suggestions)} rule candidates to {out_path}")


# ------------------------------
# Main quick test
# ------------------------------
if __name__ == "__main__":
    import os

    gold_path = os.path.join("training", "gold_tests.json")
    if not os.path.exists(gold_path):
        raise FileNotFoundError("Missing training/gold_tests.json")

    with open(gold_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    model_name = os.getenv("OLLAMA_MODEL", "deepseek-coder:6.7b")
    candidates = suggest_rules_from_gold(gold_data, model_name, use_llm=True)
    save_rule_suggestions(candidates)
