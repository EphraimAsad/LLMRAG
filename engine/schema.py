"""
schema.py — Canonical field definitions, enums, and normalization/validation helpers
for BactAI-D. This module is the single source of truth for ALL field names and
allowed values used by the parser, validator, identifier, gold tests, and UI.

Conventions:
- Unknown MUST be exactly "Unknown".
- For categorical tests: allowed values are {"Positive", "Negative", "Variable"}.
- Multi-entry fields are semicolon-separated, e.g., "Small; Yellow".
- Media names end with "Agar" (enforced by normalization map elsewhere).
- Shapes can include up to 2 values from the SHAPE enum (parser should enforce).
- Growth Temperature format: "low//high" in °C, e.g., "10//40" (strings).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

# -----------------------------
# Canonical enums / constants
# -----------------------------

UNKNOWN = "Unknown"
PNV = ["Positive", "Negative", "Variable"]

SHAPES = ["Cocci", "Rods", "Bacilli", "Spiral", "Short Rods"]  # <=2 allowed in final records

HAEMOLYSIS_TYPES = ["None", "Beta", "Gamma", "Alpha"]

# Fields that accept many free-text entries (but still normalized)
# and may have multiple values separated by semicolons.
MULTI_ENTRY_FIELDS = {
    "Colony Morphology",
    "Media Grown On",
    "Haemolysis Type",       # e.g., "Beta; Alpha" if mixed reports
    "Oxygen Requirement",    # allow multiple like "Aerobic; Facultative Anaerobe"
}

# Categorical “Positive/Negative/Variable” fields
PNV_FIELDS = [
    "Gram Stain",
    "Catalase",
    "Oxidase",
    "Haemolysis",
    "Indole",
    "Motility",
    "Capsule",
    "Spore Formation",
    "Methyl Red",
    "VP",
    "Citrate",
    "Urease",
    "H2S",
    "Lactose Fermentation",
    "Glucose Fermentation",
    "Sucrose Fermentation",
    "Nitrate Reduction",
    "Lysine Decarboxylase",
    "Ornithine Decarboxylase",
    "Arginine dihydrolase",
    "Gelatin Hydrolysis",
    "Esculin Hydrolysis",
    "Dnase",
    "ONPG",
    "NaCl Tolerant (>=6%)",
    "Lipase Test",
    "Xylose Fermentation",
    "Rhamnose Fermentation",
    "Mannitol Fermentation",
    "Sorbitol Fermentation",
    "Maltose Fermentation",
    "Arabinose Fermentation",
    "Raffinose Fermentation",
    "Inositol Fermentation",
    "Trehalose Fermentation",
    "Coagulase",
]

# Text/free fields (single value expected)
TEXT_FIELDS = [
    "Genus",
    "Species",
]

# Constrained-enum fields (non-PNV)
ENUM_FIELDS: Dict[str, List[str]] = {
    "Shape": SHAPES,
    "Haemolysis Type": HAEMOLYSIS_TYPES,  # also listed as multi-entry in MULTI_ENTRY_FIELDS
}

# Special “range as string” field
RANGE_FIELDS = [
    "Growth Temperature",  # "10//40"
]

# Full ordered field list (good for UI, CSV order, etc.)
FIELD_ORDER: List[str] = (
    TEXT_FIELDS
    + [
        "Gram Stain",
        "Shape",
        "Colony Morphology",
        "Haemolysis",
        "Haemolysis Type",
        "Indole",
        "Growth Temperature",
        "Media Grown On",
        "Motility",
        "Capsule",
        "Spore Formation",
        "Oxygen Requirement",
        "Methyl Red",
        "VP",
        "Citrate",
        "Urease",
        "H2S",
        "Lactose Fermentation",
        "Glucose Fermentation",
        "Sucrose Fermentation",
        "Nitrate Reduction",
        "Lysine Decarboxylase",
        "Ornithine Decarboxylase",
        "Arginine dihydrolase",
        "Gelatin Hydrolysis",
        "Esculin Hydrolysis",
        "Dnase",
        "ONPG",
        "NaCl Tolerant (>=6%)",
        "Lipase Test",
        "Xylose Fermentation",
        "Rhamnose Fermentation",
        "Mannitol Fermentation",
        "Sorbitol Fermentation",
        "Maltose Fermentation",
        "Arabinose Fermentation",
        "Raffinose Fermentation",
        "Inositol Fermentation",
        "Trehalose Fermentation",
        "Coagulase",
        "Oxidase",     # enzymes often grouped together; kept here explicitly
        "Catalase",
    ]
)

# Quick membership lookups
ALL_FIELDS = set(FIELD_ORDER)
PNV_FIELD_SET = set(PNV_FIELDS)
ENUM_FIELD_SET = set(ENUM_FIELDS.keys())
RANGE_FIELD_SET = set(RANGE_FIELDS)
MULTI_FIELD_SET = set(MULTI_ENTRY_FIELDS)
TEXT_FIELD_SET = set(TEXT_FIELDS)

# -----------------------------
# Synonym normalization maps
# (kept small here; extend via data/normalization_map.json later)
# -----------------------------

VALUE_SYNONYMS = {
    # P/N/V
    "positive": "Positive",
    "+": "Positive",
    "+ve": "Positive",
    "pos": "Positive",

    "negative": "Negative",
    "-": "Negative",
    "-ve": "Negative",
    "neg": "Negative",

    "variable": "Variable",
    "var": "Variable",

    "unknown": UNKNOWN,
    "n/a": UNKNOWN,
    "na": UNKNOWN,
    "not reported": UNKNOWN,
}

SHAPE_SYNONYMS = {
    "bacillus": "Bacilli",
    "bacilli": "Bacilli",
    "rod": "Rods",
    "rods": "Rods",
    "short rods": "Short Rods",
    "coccus": "Cocci",
    "cocci": "Cocci",
    "spiral": "Spiral",
    "spirillum": "Spiral",
}

HAEM_TYPE_SYNONYMS = {
    "beta-haemolytic": "Beta",
    "beta-hemolytic": "Beta",
    "alpha-haemolytic": "Alpha",
    "alpha-hemolytic": "Alpha",
    "gamma-haemolytic": "Gamma",
    "gamma-hemolytic": "Gamma",
    "no haemolysis": "None",
    "no hemolysis": "None",
    "none": "None",
}

# -----------------------------
# Helper API
# -----------------------------

def is_multi_entry(field: str) -> bool:
    return field in MULTI_FIELD_SET

def allowed_values(field: str) -> Optional[List[str]]:
    if field in PNV_FIELD_SET:
        return PNV
    if field in ENUM_FIELD_SET:
        return ENUM_FIELDS[field]
    return None  # free text or special

def all_fields_ordered() -> List[str]:
    return list(FIELD_ORDER)

# -----------------------------
# Normalization
# -----------------------------

def _normalize_basic_token(tok: str) -> str:
    t = tok.strip()
    if not t:
        return t
    low = t.lower()
    # P/N/V & Unknown
    if low in VALUE_SYNONYMS:
        return VALUE_SYNONYMS[low]
    # Shapes
    if low in SHAPE_SYNONYMS:
        return SHAPE_SYNONYMS[low]
    # Haemolysis Type
    if low in HAEM_TYPE_SYNONYMS:
        return HAEM_TYPE_SYNONYMS[low]
    # Capitalize first letter of each word by default
    return " ".join(w.capitalize() if w else w for w in t.split())

def normalize_value(field: str, value: str) -> str:
    """
    Normalize a single field value into canonical labels.
    - For multi-entry fields, caller should split/join by '; ' externally,
      but this function can be applied to each token.
    - For P/N/V fields, map synonyms to canonical.
    - For enums like Shape/Haemolysis Type, map synonyms where possible.
    """
    if value is None:
        return UNKNOWN
    v = str(value).strip()
    if not v:
        return UNKNOWN

    # Growth Temperature: keep "low//high" as-is after trimming spaces
    if field == "Growth Temperature":
        return v.replace(" ", "")

    # General token normalization
    if is_multi_entry(field):
        parts = [p for p in (s.strip() for s in v.split(";")) if p]
        return "; ".join(_normalize_basic_token(p) for p in parts) if parts else UNKNOWN

    # PNV / Enum / Text
    av = allowed_values(field)
    norm = _normalize_basic_token(v)

    if av:
        # If it matches allowed list after normalization, accept
        if norm in av:
            return norm
        # If norm is UNKNOWN and UNKNOWN not in av, still return UNKNOWN (caller can decide)
        return norm

    # Text fields — just title-case words (except Genus/Species which may have specific casing)
    if field in TEXT_FIELD_SET:
        return v.strip()

    # Oxygen Requirement is free text but might be multi; if user passed semicolons, normalize as multi
    if field == "Oxygen Requirement":
        parts = [p for p in (s.strip() for s in v.split(";")) if p]
        return "; ".join(_normalize_basic_token(p) for p in parts) if parts else UNKNOWN

    # Otherwise generic normalization
    return norm

# -----------------------------
# Validation
# -----------------------------

def validate_record(record: Dict[str, str]) -> List[str]:
    """
    Validate a single bacterium record (dict of field->value).
    Returns a list of human-readable issues (empty if valid).
    """
    issues: List[str] = []

    # Missing required text fields
    for f in TEXT_FIELDS:
        if f not in record or not str(record[f]).strip():
            issues.append(f"Missing required field: {f}")

    # Field-by-field validation
    for field in FIELD_ORDER:
        if field not in record:
            # Not strictly an error, but note for completeness
            continue

        val = str(record[field]).strip()

        if field in PNV_FIELD_SET:
            if val not in (PNV + [UNKNOWN]):
                issues.append(f"{field}: '{val}' is not one of {PNV} or '{UNKNOWN}'")

        elif field in ENUM_FIELD_SET:
            # allow multi for Haemolysis Type if separated by ';'
            if field in MULTI_FIELD_SET and ";" in val:
                parts = [p.strip() for p in val.split(";") if p.strip()]
                bad = [p for p in parts if p not in ENUM_FIELDS[field]]
                if bad:
                    issues.append(f"{field}: invalid values {bad}; allowed {ENUM_FIELDS[field]}")
            else:
                if val not in (ENUM_FIELDS[field] + [UNKNOWN]):
                    issues.append(f"{field}: '{val}' is not one of {ENUM_FIELDS[field]} or '{UNKNOWN}'")

        elif field in RANGE_FIELD_SET:
            # Expect "low//high" numeric strings
            if val == UNKNOWN:
                pass
            else:
                if "//" not in val:
                    issues.append(f"{field}: expected format 'low//high' (°C), got '{val}'")
                else:
                    try:
                        low_s, high_s = val.split("//", 1)
                        low, high = float(low_s), float(high_s)
                        if low > high:
                            issues.append(f"{field}: low>{high} ('{val}')")
                    except Exception:
                        issues.append(f"{field}: could not parse numbers in '{val}'")

        else:
            # MULTI_ENTRY_FIELDS or free text — nothing strict to validate
            pass

    # Shape: at most 2 values (enforced as guidance)
    shape = record.get("Shape", "").strip()
    if shape and shape != UNKNOWN:
        parts = [p.strip() for p in shape.split(";") if p.strip()]
        if len(parts) > 2:
            issues.append("Shape: at most two shapes allowed (e.g., 'Cocci; Rods').")

        # ensure each in SHAPES
        bad = [p for p in parts if p not in SHAPES]
        if bad:
            issues.append(f"Shape: invalid values {bad}; allowed {SHAPES}")

    # Ensure ONPG & NaCl tolerance fields exist (project requirement)
    for must in ("ONPG", "NaCl Tolerant (>=6%)"):
        if must not in record:
            issues.append(f"Missing required field in schema: {must}")

    return issues

# -----------------------------
# Canonicalization helpers
# -----------------------------

# -----------------------------
# Support for unknown / extra tests
# -----------------------------

OTHER_FIELD = "Other"

def canonicalize_record(record: Dict[str, str]) -> Dict[str, str]:
    """
    Return a new dict with normalized values and any missing fields filled with 'Unknown'.
    Any unrecognized fields are moved into the 'Other' field as 'Name: Value' pairs.
    """
    out: Dict[str, str] = {}
    extras: List[str] = []

    for field, val in record.items():
        if field in FIELD_ORDER:
            out[field] = normalize_value(field, val)
        else:
            # preserve unknown tests in a semicolon-separated "Other" list
            extras.append(f"{field}: {val}")

    # Fill in any missing known fields
    for field in FIELD_ORDER:
        if field not in out:
            out[field] = UNKNOWN

    # Merge extras into 'Other'
    out[OTHER_FIELD] = "; ".join(extras) if extras else UNKNOWN
    return out

# Make sure 'Other' is officially known to the schema
FIELD_ORDER.append(OTHER_FIELD)
ALL_FIELDS.add(OTHER_FIELD)

__all__ = [
    "UNKNOWN",
    "PNV",
    "SHAPES",
    "HAEMOLYSIS_TYPES",
    "PNV_FIELDS",
    "TEXT_FIELDS",
    "ENUM_FIELDS",
    "RANGE_FIELDS",
    "FIELD_ORDER",
    "all_fields_ordered",
    "allowed_values",
    "is_multi_entry",
    "normalize_value",
    "validate_record",
    "canonicalize_record",
]

