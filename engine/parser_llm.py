import os
import json
import re
from typing import Dict, Any, List
import ollama  # Using Ollama for LLM access
from datetime import datetime

from engine.parser_rules_runtime import apply_rules, merge_into_record, normalize_field, normalize_value

# --------------------------------------------
# Config
# --------------------------------------------
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-coder:6.7b")  # or any available Ollama model
TEMPERATURE = 0.2

# --------------------------------------------
# Prompt Template for LLM Parsing
# --------------------------------------------
LLM_PROMPT_TEMPLATE = """
You are an expert microbiologist AI.
Extract key biochemical and morphological results from the following text into a JSON object
with exactly these fields (fill with "Unknown" if not mentioned):

{fields_list}

Rules:
- Only use "Positive", "Negative", or "Variable" for test results unless otherwise noted.
- For oxygen requirement, use one of: Aerobic, Anaerobic, Microaerophilic, Capnophilic, Facultative Anaerobe, Intracellular.
- Growth Temperature should be numeric (single number or "low//high" range).
- Use proper capitalization (e.g. "Positive", not "positive").
- Do not infer extra results not mentioned in the text.

Text:
\"\"\"{input_text}\"\"\"
"""

# --------------------------------------------
# Helper: get all schema fields
# --------------------------------------------
def get_schema_fields() -> List[str]:
    """
    Read the full field list from schema.py (if available) or fallback to defaults.
    """
    try:
        from engine.schema import ALL_FIELDS
        return ALL_FIELDS
    except Exception:
        # fallback (minimal)
        return [
            "Gram Stain", "Shape", "Catalase", "Oxidase", "Coagulase", "Urease", "Indole", "Citrate",
            "Methyl Red", "VP", "Haemolysis", "Haemolysis Type", "Motility", "Spore Formation",
            "Capsule", "Oxygen Requirement", "Growth Temperature", "Dnase", "ONPG",
            "Nitrate Reduction", "Gelatin Hydrolysis", "Esculin Hydrolysis", "Lysine Decarboxylase",
            "Ornitihine Decarboxylase", "Arginine dihydrolase", "H2S",
            "Glucose Fermentation", "Lactose Fermentation", "Sucrose Fermentation", "Maltose Fermentation",
            "Mannitol Fermentation", "Sorbitol Fermentation", "Xylose Fermentation", "Rhamnose Fermentation",
            "Arabinose Fermentation", "Raffinose Fermentation", "Trehalose Fermentation", "Inositol Fermentation",
            "NaCl Tolerant (>=6%)", "Media Grown On", "Colony Morphology"
        ]


# --------------------------------------------
# Core Parser Logic
# --------------------------------------------
def smart_parse(text: str, model_name: str = DEFAULT_MODEL, use_llm: bool = True) -> Dict[str, str]:
    """
    Smart hybrid parser:
    1. Apply learned + base rules.
    2. Optionally use the LLM for unknown fields.
    3. Return clean, normalized dict.
    """
    if not text or not text.strip():
        return {}

    # Step 1 — Rules-only parse
    rule_results = apply_rules(text)

    # Step 2 — Start record with all fields
    record = {f: "Unknown" for f in get_schema_fields()}
    record = merge_into_record(record, rule_results)

    # Step 3 — If not using LLM, return what rules found
    if not use_llm:
        return record

    # Step 4 — Build dynamic field list for prompt
    fields_str = "\n".join(f"- {f}" for f in get_schema_fields())

    # Step 5 — Send prompt to LLM
    prompt = LLM_PROMPT_TEMPLATE.format(
        fields_list=fields_str,
        input_text=text
    )

    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": TEMPERATURE}
        )
        raw = response["message"]["content"].strip()

        # Extract JSON block
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in LLM response")

        parsed = json.loads(json_match.group(0))
    except Exception as e:
        print(f"[WARN] LLM parsing failed: {e}")
        parsed = {}

    # Step 6 — Normalize LLM output and merge
    normalized = {}
    for k, v in parsed.items():
        field = normalize_field(k)
        value = normalize_value(field, v)
        normalized[field] = value

    record = merge_into_record(record, normalized)
    return record


# --------------------------------------------
# Utility for Batch Gold Test Evaluation
# --------------------------------------------
def evaluate_gold_tests(gold_data: List[Dict[str, Any]], model_name: str = DEFAULT_MODEL, use_llm: bool = True):
    """
    Run smart_parse on every gold test, compare with expected, and calculate field accuracy.
    Returns a summary dict with per-field and overall stats.
    """
    all_fields = get_schema_fields()
    results = {f: {"total": 0, "correct": 0} for f in all_fields}
    failed_cases = []

    for case in gold_data:
        name = case.get("name")
        input_text = case.get("input", "")
        expected = case.get("expected", {})

        parsed = smart_parse(input_text, model_name=model_name, use_llm=use_llm)

        for field, exp_val in expected.items():
            norm_field = normalize_field(field)
            norm_exp = normalize_value(norm_field, exp_val)
            got = parsed.get(norm_field, "Unknown")

            results[norm_field]["total"] += 1
            if got == norm_exp:
                results[norm_field]["correct"] += 1
            else:
                failed_cases.append({
                    "name": name,
                    "field": norm_field,
                    "expected": norm_exp,
                    "got": got,
                    "text": input_text
                })

    # Compute accuracy per field
    per_field_acc = {f: (v["correct"] / v["total"]) if v["total"] else 0.0 for f, v in results.items()}
    overall_acc = sum(v["correct"] for v in results.values()) / max(1, sum(v["total"] for v in results.values()))

    summary = {
        "timestamp": datetime.now().isoformat(),
        "overall_accuracy": round(overall_acc * 100, 2),
        "per_field_accuracy": {f: round(a * 100, 2) for f, a in per_field_acc.items()},
        "failed_cases": failed_cases
    }
    return summary
