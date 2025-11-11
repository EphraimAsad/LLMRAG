"""
engine/schema.py
Defines the canonical schema for BactAI-D and normalization helpers
used across the LLM parser, rule engine, and sanitizers.
"""

# =========================================================
#                    FIELD DEFINITIONS
# =========================================================
ALL_FIELDS = [
    # Core identity
    "Genus",
    "Species",

    # Morphology
    "Gram Stain",
    "Shape",
    "Colony Morphology",
    "Motility",
    "Capsule",
    "Spore Formation",
    "Oxygen Requirement",
    "Growth Temperature",
    "Media Grown On",

    # Enzyme & biochemical tests
    "Catalase",
    "Oxidase",
    "Coagulase",
    "Lipase Test",
    "Urease",
    "Indole",
    "Citrate",
    "Methyl Red",
    "VP",
    "H2S",
    "Dnase",
    "ONPG",
    "Nitrate Reduction",
    "Gelatin Hydrolysis",
    "Esculin Hydrolysis",
    "Lysine Decarboxylase",
    "Ornitihine Decarboxylase",
    "Arginine dihydrolase",

    # Fermentation tests
    "Glucose Fermentation",
    "Lactose Fermentation",
    "Sucrose Fermentation",
    "Maltose Fermentation",
    "Mannitol Fermentation",
    "Sorbitol Fermentation",
    "Xylose Fermentation",
    "Rhamnose Fermentation",
    "Arabinose Fermentation",
    "Raffinose Fermentation",
    "Trehalose Fermentation",
    "Inositol Fermentation",

    # Other differential tests
    "Haemolysis",
    "Haemolysis Type",
    "NaCl Tolerant (>=6%)",

    # Reserved for extension
    "Other"
]

# =========================================================
#                  NORMALIZATION HELPERS
# =========================================================
def normalize_field(field: str) -> str:
    """Normalize field names to canonical schema keys."""
    if not field:
        return "Unknown"
    f = field.strip().lower().replace("_", " ")

    mappings = {
        "gram": "Gram Stain",
        "gram stain": "Gram Stain",
        "shape": "Shape",
        "morphology": "Colony Morphology",
        "motility": "Motility",
        "capsule": "Capsule",
        "spore": "Spore Formation",
        "spore formation": "Spore Formation",
        "oxygen": "Oxygen Requirement",
        "oxygen requirement": "Oxygen Requirement",
        "growth temp": "Growth Temperature",
        "growth temperature": "Growth Temperature",
        "media": "Media Grown On",
        "media grown on": "Media Grown On",
        "haemolysis": "Haemolysis",
        "haemolysis type": "Haemolysis Type",
        "oxidase": "Oxidase",
        "catalase": "Catalase",
        "coagulase": "Coagulase",
        "urease": "Urease",
        "indole": "Indole",
        "citrate": "Citrate",
        "methyl red": "Methyl Red",
        "vp": "VP",
        "dnase": "Dnase",
        "onpg": "ONPG",
        "h2s": "H2S",
        "nitrate": "Nitrate Reduction",
        "nitrate reduction": "Nitrate Reduction",
        "gelatin": "Gelatin Hydrolysis",
        "gelatin hydrolysis": "Gelatin Hydrolysis",
        "esculin": "Esculin Hydrolysis",
        "esculin hydrolysis": "Esculin Hydrolysis",
        "lysine": "Lysine Decarboxylase",
        "lysine decarboxylase": "Lysine Decarboxylase",
        "ornitihine": "Ornitihine Decarboxylase",
        "ornithine": "Ornitihine Decarboxylase",
        "arginine": "Arginine dihydrolase",
        "arginine dihydrolase": "Arginine dihydrolase",
        "lipase": "Lipase Test",
        "lipase test": "Lipase Test",
        "nacl": "NaCl Tolerant (>=6%)",
        "salt": "NaCl Tolerant (>=6%)",
        "sodium chloride": "NaCl Tolerant (>=6%)",
        "other": "Other",
    }

    # Fermentation pattern
    sugars = [
        "glucose", "lactose", "sucrose", "maltose", "mannitol",
        "sorbitol", "xylose", "rhamnose", "arabinose", "raffinose",
        "trehalose", "inositol"
    ]
    for sugar in sugars:
        if sugar in f and "ferment" in f:
            return f"{sugar.capitalize()} Fermentation"

    return mappings.get(f, field.strip())


def normalize_value(field: str, value: str) -> str:
    """Normalize value text into canonical representation."""
    if not value:
        return "Unknown"
    v = value.strip().lower()
    f = normalize_field(field)

    # Standard biochemical
    if v in {"pos", "positive", "+", "p"}:
        return "Positive"
    if v in {"neg", "negative", "-", "n"}:
        return "Negative"
    if v in {"var", "variable", "v"}:
        return "Variable"
    if v == "unknown":
        return "Unknown"

    # Haemolysis type
    if f == "Haemolysis Type":
        if "alpha" in v:
            return "Alpha"
        if "beta" in v:
            return "Beta"
        if "gamma" in v:
            return "Gamma"
        if "none" in v:
            return "None"

    # Oxygen requirement
    if f == "Oxygen Requirement":
        if "aerobic" in v and "facultative" not in v:
            return "Aerobic"
        if "anaerobic" in v and "facultative" not in v:
            return "Anaerobic"
        if "facultative" in v:
            return "Facultative Anaerobe"
        if "microaer" in v:
            return "Microaerophilic"
        if "capno" in v:
            return "Capnophilic"
        if "intracell" in v:
            return "Intracellular"

    # Gram Stain
    if f == "Gram Stain":
        if "neg" in v:
            return "Negative"
        if "pos" in v:
            return "Positive"
        if "var" in v:
            return "Variable"

    # For Growth Temperature ranges
    if f == "Growth Temperature":
        v = v.replace("°c", "").replace(" ", "")
        if "//" in v or "-" in v:
            return v.replace("-", "//")
        try:
            float(v)
            return v
        except ValueError:
            return "Unknown"

    # If nothing matches, just capitalize nicely
    return value.strip().capitalize()


def canonical_value_set(field: str):
    """Return allowed canonical values for a field."""
    f = normalize_field(field)
    if f == "Haemolysis Type":
        return {"Alpha", "Beta", "Gamma", "None"}
    if f == "Oxygen Requirement":
        return {"Aerobic", "Anaerobic", "Microaerophilic", "Capnophilic", "Facultative Anaerobe", "Intracellular"}
    if f == "Gram Stain":
        return {"Positive", "Negative", "Variable"}
    return {"Positive", "Negative", "Variable", "Unknown"}
