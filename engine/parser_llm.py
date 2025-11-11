"""
parser_llm.py — Semantic (LLM) parser for microbiology text using Ollama.

Usage:
    from engine.parser_llm import smart_parse
    rec = smart_parse("Paste your report text here")

What it does:
- Runs fast rule parser first (engine.parser_rules).
- If any fields remain Unknown, asks Ollama (DeepSeek) to infer them.
- Returns a fully canonicalized dict per engine.schema (including 'Other').

Notes:
- Default model: 'deepseek-r1:latest' (change via env OLLAMA_MODEL or function arg).
- Works even if Ollama is not available: it will return the rule-only result.
"""

from __future__ import annotations
import json
import os
import re
from typing import Dict, Any, Optional, Tuple

try:
    import ollama  # pip install ollama
    _OLLAMA_AVAILABLE = True
except Exception:
    _OLLAMA_AVAILABLE = False

from .schema import (
    FIELD_ORDER, UNKNOWN, normalize_value, canonicalize_record
)
from .parser_rules import parse_and_canonicalize as rules_parse_full, parse_text_rules

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:latest")
MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "800"))  # generous but safe
TIMEOUT_S  = int(os.getenv("OLLAMA_TIMEOUT", "30"))

# Only these Oxygen values are considered valid for normalization (project rule)
OXY_ALLOWED = {
    "Aerobic", "Anaerobic", "Intracellular",
    "Microaerophilic", "Capnophilic", "Facultative Anaerobe"
}

# -----------------------------------------------------------------------------
# Prompt builder
# -----------------------------------------------------------------------------

def _fields_instructions() -> str:
    # Give the model a compact list of fields and allowed values it must obey.
    lines = []
    lines.append("- Use EXACT field names below. If unknown/unsure, output 'Unknown'.")
    lines.append("- Multi-value fields use '; ' as a separator (example: 'Blood Agar; Chocolate Agar').")
    lines.append("- Growth Temperature MUST be 'low//high' in °C (e.g., '30//37').")
    lines.append("- For Oxygen Requirement, use exactly one of: " + ", ".join(sorted(OXY_ALLOWED)) + ".")
    lines.append("- Categorical tests must be one of: Positive, Negative, Variable, or Unknown.")
    lines.append("")
    lines.append("Fields (output keys):")
    for f in FIELD_ORDER:
        lines.append(f"  - {f}")
    return "\n".join(lines)

SYSTEM_PROMPT = (
    "You are a careful microbiology information extractor. "
    "Read a short lab-style description and produce a STRICT JSON object keyed by the given schema. "
    "If a value is not stated or cannot be inferred with high confidence, output 'Unknown'. "
    "Do not invent media or tests that are not mentioned."
)

USER_TEMPLATE = """Extract fields from this text.

Text:
\"\"\"
{TEXT}
\"\"\"

Rules:
{RULES}

Output:
- A single JSON object ONLY (no prose, no markdown, no code fences).
- Include ALL fields listed (fill missing with 'Unknown').
- Do NOT include any extra keys.
"""

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _strip_code_fences(s: str) -> str:
    # Remove ```json ... ``` or ``` ... ``` wrappers if present
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        return None

def _enforce_oxygen(value: str) -> str:
    # Keep schema normalization but ensure this exact closed set when applicable
    v = (value or "").strip()
    if v in OXY_ALLOWED:
        return v
    # allow common synonyms to map to your allowed set
    low = v.lower()
    if "facult" in low:
        return "Facultative Anaerobe"
    if "microaero" in low:
        return "Microaerophilic"
    if "capno" in low or "co2" in low:
        return "Capnophilic"
    if "intracell" in low:
        return "Intracellular"
    if "aerob" in low and "anaerob" not in low:
        return "Aerobic"
    if "anaerob" in low:
        return "Anaerobic"
    return UNKNOWN

def _canonicalize_full(obj: Dict[str, Any]) -> Dict[str, str]:
    # Normalize each provided value, then canonicalize to ensure all fields present.
    pre = {}
    for k, v in obj.items():
        if k == "Oxygen Requirement":
            pre[k] = _enforce_oxygen(str(v))
        else:
            pre[k] = normalize_value(k, str(v))
    return canonicalize_record(pre)

def _merge_records(rule_rec: Dict[str, str], llm_rec: Dict[str, str]) -> Dict[str, str]:
    """Prefer rule parser values; fill Unknowns from LLM. Keep 'Other' merged."""
    out = dict(rule_rec)
    for k, v in llm_rec.items():
        if k == "Other":
            # merge extra info
            if out.get("Other", UNKNOWN) in (None, "", UNKNOWN):
                out["Other"] = v
            elif v not in (None, "", UNKNOWN):
                if out["Other"].strip() != v.strip():
                    out["Other"] = "; ".join([out["Other"], v]).strip("; ")
            continue

        if k not in out or out[k] in (None, "", UNKNOWN):
            out[k] = v
    return out

# -----------------------------------------------------------------------------
# Core call to Ollama
# -----------------------------------------------------------------------------

def _ollama_chat(model: str, text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not _OLLAMA_AVAILABLE:
        return None, "Ollama Python package not available."

    try:
        prompt_user = USER_TEMPLATE.format(TEXT=text, RULES=_fields_instructions())
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_user},
            ],
            options={"num_predict": MAX_TOKENS, "temperature": 0.2},
        )
        content = resp["message"]["content"]
    except Exception as e:
        return None, f"Ollama error: {e}"

    content = _strip_code_fences(content)
    data = _safe_json_load(content)
    if data is None:
        return None, "Failed to parse JSON from LLM response."
    if not isinstance(data, dict):
        return None, "LLM returned non-object JSON."

    return data, None

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def parse_with_llm(text: str, model: str = DEFAULT_MODEL, retries: int = 1) -> Dict[str, str]:
    """
    Ask the LLM to produce a full JSON record (all schema fields present).
    Returns a canonicalized record (Unknown for anything missing).
    """
    if not text or not text.strip():
        return canonicalize_record({})

    # First attempt
    data, err = _ollama_chat(model, text)

    # Retry once with a stronger reminder if needed
    if data is None and retries > 0:
        retry_note = (
            "\nIMPORTANT: Your previous output was not valid JSON. "
            "Now output ONLY a single JSON object following the rules, with double quotes."
        )
        data, err = _ollama_chat(model, text + retry_note)

    if data is None:
        # Fallback to empty canonical record if LLM is unreachable
        return canonicalize_record({})

    # Enforce canonical values + full schema coverage
    return _canonicalize_full(data)

def smart_parse(text: str, model: str = DEFAULT_MODEL) -> Dict[str, str]:
    """
    Hybrid parse:
    1) Use rule parser to extract high-confidence fields.
    2) Ask LLM for a full record.
    3) Merge: keep rule parser values; fill remaining Unknowns from LLM.
    Returns a canonical record per schema (includes 'Other').
    """
    # Rule-only partial canonical (already fills Unknown + Other)
    rule_full = rules_parse_full(text)

    # If rule parser did well (few Unknowns), you can early-return.
    # Here we always ask LLM, but you can short-circuit if desired.
    llm_full = parse_with_llm(text, model=model)

    merged = _merge_records(rule_full, llm_full)
    # Final pass to ensure normalization is strictly applied
    return canonicalize_record(merged)
