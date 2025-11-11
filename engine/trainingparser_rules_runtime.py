import json
import os
import re
from typing import Dict, List, Any, Tuple

# --- Optional: use your schema normalizer if present ---
def _try_schema_normalizers():
    try:
        # You named it schema.py earlier; adjust if your path differs:
        from engine.schema import normalize_field, normalize_value
        return normalize_field, normalize_value
    except Exception:
        # Fallback no-op normalizers (still useful)
        def normalize_field(field: str) -> str:
            # Common canonical fixes
            m = field.strip()
            # Unify DNase/Dnase capitalization
            if m.lower() == "dnase": return "Dnase"
            if m.lower() == "onpg": return "ONPG"
            if m.lower() == "oxidase": return "Oxidase"
            if m.lower() == "catalase": return "Catalase"
            if m.lower() == "coagulase": return "Coagulase"
            if m.lower() == "gram stain": return "Gram Stain"
            if m.lower() == "haemolysis type": return "Haemolysis Type"
            if m.lower() == "oxygen requirement": return "Oxygen Requirement"
            return m

        # Clamp values to common sets when obvious
        POSNEG = {"positive": "Positive", "neg": "Negative", "negative": "Negative", "variable": "Variable"}
        OXY = {
            "aerobic": "Aerobic",
            "anaerobic": "Anaerobic",
            "microaerophilic": "Microaerophilic",
            "capnophilic": "Capnophilic",
            "facultative anaerobe": "Facultative Anaerobe",
            "intracellular": "Intracellular",
        }

        def normalize_value(field: str, value: str) -> str:
            v = (value or "").strip()
            f = normalize_field(field)
            low = v.lower()
            if f in {"Catalase","Oxidase","Coagulase","Urease","Indole","Citrate","VP","Methyl Red",
                     "Dnase","ONPG","Gelatin Hydrolysis","Esculin Hydrolysis","H2S",
                     "Motility","Capsule","Spore Formation",
                     "Lactose Fermentation","Glucose Fermentation","Sucrose Fermentation","Maltose Fermentation",
                     "Mannitol Fermentation","Sorbitol Fermentation","Xylose Fermentation","Rhamnose Fermentation",
                     "Arabinose Fermentation","Raffinose Fermentation","Trehalose Fermentation","Inositol Fermentation",
                     "Nitrate Reduction","Lysine Decarboxylase","Ornitihine Decarboxylase","Arginine dihydrolase"}:
                return POSNEG.get(low, v or "Unknown")
            if f == "Haemolysis":
                return POSNEG.get(low, v or "Unknown")
            if f == "Haemolysis Type":
                if low in {"alpha","beta","gamma","none"}:
                    return v.capitalize() if low != "none" else "None"
                return v or "Unknown"
            if f == "Oxygen Requirement":
                return OXY.get(low, v or "Unknown")
            # Leave morphology/media/etc. as-is
            return v or "Unknown"

        return normalize_field, normalize_value

normalize_field, normalize_value = _try_schema_normalizers()


# ------------------------------
# Rule data model (runtime)
# ------------------------------
class Rule:
    __slots__ = ("field","value","pattern_type","pattern","case_sensitive","priority","status")

    def __init__(self, d: Dict[str, Any]):
        self.field = normalize_field(d.get("field","").strip())
        self.value = normalize_value(self.field, d.get("value","").strip())
        self.pattern_type = d.get("pattern_type","phrase").strip().lower()
        self.pattern = d.get("pattern","")
        self.case_sensitive = bool(d.get("case_sensitive", False))
        self.priority = int(d.get("priority", 50))
        self.status = d.get("status","active")

        # Pre-compile regex safely if applicable
        self._regex = None
        if self.pattern_type == "regex" and self.pattern:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            try:
                # Guardrails: reject very long or dangerous patterns
                if len(self.pattern) > 256:
                    raise ValueError("Pattern too long")
                if "(.*)+" in self.pattern or "(.*){2,}" in self.pattern:
                    raise ValueError("Potential catastrophic pattern")
                self._regex = re.compile(self.pattern, flags)
            except Exception:
                self._regex = None

    def matches(self, text: str) -> bool:
        if self.status != "active":
            return False
        if not text:
            return False
        if self.pattern_type == "phrase":
            hay = text if self.case_sensitive else text.lower()
            needle = self.pattern if self.case_sensitive else self.pattern.lower()
            return needle in hay
        elif self.pattern_type == "regex":
            if not self._regex:
                return False
            return bool(self._regex.search(text))
        return False


# ------------------------------
# Loading rules
# ------------------------------
def _load_json_safely(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"version": 1, "rules": []}

def load_rules() -> List[Rule]:
    """
    Load base rules (optional) and learned rules (training/learned_rules.json),
    merge, and return a single ordered list by priority (low first).
    """
    rules: List[Rule] = []

    # 1) Base rules (optional): engine/parser_rules.py -> provide a function list_base_rules()
    #    If you don’t have it yet, skip silently.
    try:
        from engine.parser_rules import list_base_rules  # type: ignore
        for d in list_base_rules():
            rules.append(Rule(d))
    except Exception:
        pass

    # 2) Learned rules
    learned_path = os.path.join("training","learned_rules.json")
    data = _load_json_safely(learned_path)
    for d in data.get("rules", []):
        try:
            r = Rule(d)
            if r.status == "active" and r.field and r.value and r.pattern:
                rules.append(r)
        except Exception:
            continue

    # Order: lowest priority first (applied first), then by pattern length (more specific first)
    rules.sort(key=lambda r: (r.priority, -len(r.pattern or "")))
    return rules


# ------------------------------
# Apply rules to text
# ------------------------------
def apply_rules(text: str) -> Dict[str, str]:
    """
    Returns a partial record: {field: value} extracted only by rules.
    You can feed this into your smart_parse() to fill missing fields
    before LLM fallback.
    """
    result: Dict[str, str] = {}
    if not text:
        return result

    rules = load_rules()
    for r in rules:
        # If rule matches and the field not already set, set it
        if r.matches(text) and r.field not in result:
            result[r.field] = r.value
    return result


# ------------------------------
# Utility: merge rule results into a record
# ------------------------------
def merge_into_record(base: Dict[str, str], add: Dict[str, str]) -> Dict[str, str]:
    """
    Keep existing base values; only fill empty/Unknown from add.
    """
    out = dict(base or {})
    for k, v in (add or {}).items():
        if k not in out or not out[k] or str(out[k]).lower() == "unknown":
            out[k] = v
    return out
