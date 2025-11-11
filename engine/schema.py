"""
Schema definitions and normalizers for BactAI-D
Ensures all fields and values are canonical and validated consistently.
"""

# -------------------------------------------------------------
# Canonical field list
# -------------------------------------------------------------
ALL_FIELDS = [
    # Morphology
    "Genus", "Species", "Gram Stain", "Shape", "Colony Morphology",
    "Media Grown On", "Motility", "Capsule", "Spore Formation",

    # Core enzyme/biochemical
    "Catalase", "Oxidase", "Coagulase", "Urease", "Indole", "Citrate",
    "Methyl Red", "VP", "Dnase", "ONPG", "Lipase Test", "Lecithinase",

    # Haemolysis
    "Haemolysis", "Haemolysis Type",

    # Fermentation tests
    "Glucose Fermentation", "Lactose Fermentation", "Sucrose Fermentation",
    "Maltose Fermentation", "Mannitol Fermentation", "Sorbitol Fermentation",
    "Xylose Fermentation", "Rhamnose Fermentation", "Arabinose Fermentation",
    "Raffinose Fermentation", "Trehalose Fermentation", "Inositol Fermentation",
    "Fructose Fermentation", "Mannose Fermentation", "Inulin Fermentation",

    # Decarboxylase and reduction tests
    "Nitrate Reduction", "Lysine Decarboxylase", "Ornitihine Decarboxylase",
    "Arginine dihydrolase", "H2S", "Gelatin Hydrolysis", "Esculin Hydrolysis",

    # Oxygen and temperature
    "Oxygen Requirement", "Growth Temperature", "NaCl Tolerant (>=6%)",

    # Specialty or species-specific assays
    "CAMP Test", "Hippurate Hydrolysis", "Bile Solubility",
    "Optochin Sensitivity", "Casein Hydrolysis", "Tyrosine Hydrolysis",

    # Resistance / other
    "Antibiotic Resistance", "Gas Production", "Metabolic Product",
    "Other Products", "Growth Factors"
]

# -------------------------------------------------------------
# Field normalization
# -------------------------------------------------------------
def normalize_field(field: str) -> str:
    """Normalize field names into canonical schema labels."""
    if not field:
        return "Unknown"

    f = field.strip().lower()
    mapping = {
        # General
        "gram": "Gram Stain", "gram reaction": "Gram Stain",
        "morphology": "Shape", "shape/morphology": "Shape",
        "colony": "Colony Morphology", "media": "Media Grown On",

        # Common biochemical short forms
        "vp": "VP", "mr": "Methyl Red", "oxidase test": "Oxidase",
        "catalase test": "Catalase", "coagulase test": "Coagulase",
        "urease test": "Urease", "indole test": "Indole",
        "citrate test": "Citrate", "dnase test": "Dnase",
        "onpg test": "ONPG", "lipase": "Lipase Test",

        # Haemolysis
        "hemolysis": "Haemolysis", "hemolysis type": "Haemolysis Type",
        "haemolysis type": "Haemolysis Type",

        # Sugar fermentation
        "glucose": "Glucose Fermentation", "lactose": "Lactose Fermentation",
        "sucrose": "Sucrose Fermentation", "maltose": "Maltose Fermentation",
        "mannitol": "Mannitol Fermentation", "sorbitol": "Sorbitol Fermentation",
        "xylose": "Xylose Fermentation", "rhamnose": "Rhamnose Fermentation",
        "arabinose": "Arabinose Fermentation", "raffinose": "Raffinose Fermentation",
        "trehalose": "Trehalose Fermentation", "inositol": "Inositol Fermentation",
        "fructose": "Fructose Fermentation", "mannose": "Mannose Fermentation",
        "inulin": "Inulin Fermentation",

        # Decarboxylase / reduction
        "lysine": "Lysine Decarboxylase", "ornithine": "Ornitihine Decarboxylase",
        "arginine": "Arginine dihydrolase", "gelatin": "Gelatin Hydrolysis",
        "esculin": "Esculin Hydrolysis", "nitrate": "Nitrate Reduction",
        "h2s": "H2S",

        # Oxygen / temperature / salt
        "oxygen": "Oxygen Requirement", "temperature": "Growth Temperature",
        "nacl": "NaCl Tolerant (>=6%)", "salt tolerance": "NaCl Tolerant (>=6%)",

        # Special tests
        "camp": "CAMP Test", "camp test": "CAMP Test",
        "hippurate": "Hippurate Hydrolysis", "hippurate hydrolysis": "Hippurate Hydrolysis",
        "bile solubility": "Bile Solubility", "optochin": "Optochin Sensitivity",
        "lecithinase test": "Lecithinase", "lecithinase": "Lecithinase",
        "casein": "Casein Hydrolysis", "tyrosine": "Tyrosine Hydrolysis",

        # Misc
        "antibiotic": "Antibiotic Resistance", "gas": "Gas Production",
        "product": "Metabolic Product", "other": "Other Products",
        "growth factor": "Growth Factors"
    }

    if f in mapping:
        return mapping[f]

    # Try to match partial name fragments
    for key, val in mapping.items():
        if key in f:
            return val

    # Fallback to title case
    return field.strip().title()


# -------------------------------------------------------------
# Value normalization
# -------------------------------------------------------------
def normalize_value(field: str, value: str) -> str:
    """Normalize biochemical results to canonical values."""
    if not value:
        return "Unknown"

    val = str(value).strip().lower()
    field = normalize_field(field)

    # Generic positives/negatives
    if val in {"pos", "positive", "+", "yes"}:
        return "Positive"
    if val in {"neg", "negative", "-", "no"}:
        return "Negative"
    if val in {"var", "variable"}:
        return "Variable"
    if val in {"unk", "unknown"}:
        return "Unknown"

    # Field-specific logic
    if field == "Gram Stain":
        if "neg" in val:
            return "Negative"
        if "pos" in val:
            return "Positive"
        return "Variable"

    if field == "Haemolysis Type":
        if "alpha" in val:
            return "Alpha"
        if "beta" in val:
            return "Beta"
        if "gamma" in val:
            return "Gamma"
        if "none" in val:
            return "None"
        return "Unknown"

    if field == "Oxygen Requirement":
        oxy_map = {
            "aerobic": "Aerobic",
            "anaerobic": "Anaerobic",
            "microaerophilic": "Microaerophilic",
            "capnophilic": "Capnophilic",
            "facultative": "Facultative Anaerobe",
            "intracellular": "Intracellular",
        }
        for k, v in oxy_map.items():
            if k in val:
                return v
        return "Unknown"

    if field == "Growth Temperature":
        # Keep numeric / range formats as-is
        val = val.replace("°c", "").replace("c", "").replace(" ", "")
        return val if val else "Unknown"

    # Default
    return value.strip().title()


# -------------------------------------------------------------
# Quick validation helpers
# -------------------------------------------------------------
def is_valid_field(field: str) -> bool:
    return normalize_field(field) in ALL_FIELDS


def get_all_fields() -> list:
    return ALL_FIELDS
