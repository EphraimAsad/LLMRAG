"""
parser_rules.py — High-precision rule-based parser for microbiology text.

- Extracts BactAI-D fields from free text using regex + synonyms.
- Covers morphology, haemolysis, media, temperatures, oxygen,
  and a full panel of biochemical tests (IMViC, urease, decarboxylases,
  hydrolysis tests, ONPG, DNase, H2S, nitrate, NaCl >=6%, sugars, etc.).
- Returns a dict of {field: value}. Values are pre-normalized via schema.

Design:
- Rules are conservative: only set a field when phrasing is clear.
- Negations like "non-motile", "does not ferment lactose" are handled.
- Complex phrases like "ferments X but not Y or Z" are handled.
- Temperatures parsed to "low//high" string.
- Media names captured from a curated list.
- Oxygen requirement normalized to:
  {"Aerobic","Anaerobic","Intracellular","Microaerophilic","Capnophilic","Facultative Anaerobe"}

If a field is not confidently detected, it is omitted (caller should
canonicalize with schema.canonicalize_record to fill "Unknown" and
collect any extra keys into 'Other' if needed).
"""

from __future__ import annotations
import re
from typing import Dict, Iterable, List, Tuple
from .schema import (
    normalize_value, canonicalize_record, UNKNOWN,
)

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _lower(s: str) -> str:
    return s.lower()

def _search(pattern: str, text: str) -> re.Match | None:
    return re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)

def _findall(pattern: str, text: str) -> List[str]:
    return re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)

def _any(patterns: Iterable[str], text: str) -> bool:
    return any(_search(p, text) for p in patterns)

def _set_if_absent(out: Dict[str, str], field: str, val: str):
    if field not in out:
        out[field] = val

# ──────────────────────────────────────────────────────────────────────────────
# Canonical term maps (inputs → schema terms)
# ──────────────────────────────────────────────────────────────────────────────

GRAM_POS_PAT = r"\bgram[\-\s]?positive\b"
GRAM_NEG_PAT = r"\bgram[\-\s]?negative\b"

# Shapes: map wide variety → closest schema choice
SHAPE_MAP = {
    r"\bcocci\b|\bcoccus\b|\bdiplococci\b|\btetrads?\b": "Cocci",
    r"\brods?\b|\brod[-\s]?shaped\b": "Rods",
    r"\bbacilli?\b": "Bacilli",
    r"\bshort[-\s]?rods?\b": "Short Rods",
    r"\bspiral(s)?\b|\bspirillum\b|\bcurved rods?\b|comma[-\s]?shaped": "Spiral",
    r"\bcoccobacilli?\b": "Rods",           # collapse to closest allowed
    r"\bfilamentous\b|\bbranching\b": "Rods",
    r"\bpleomorphic\b": None,               # ambiguous → skip (leave Unknown)
}

# Oxygen requirement (exact set specified by the user)
OXYGEN_MAP = [
    (r"\b(obligate|strict)\s+anaerob", "Anaerobic"),
    (r"\b(obligate|strict)\s+aerob", "Aerobic"),
    (r"\banaerob(ic|e)?\b", "Anaerobic"),
    (r"\baerob(ic|e)?\b", "Aerobic"),
    (r"\bfacultative(ly)?\s+anaerob", "Facultative Anaerobe"),
    (r"\bmicroaerophil(ic|e)\b", "Microaerophilic"),
    (r"\bcapnophil(ic|e)\b|\bCO2[-\s]?enriched\b", "Capnophilic"),
    (r"\bintracellular\b|inside\s+cells", "Intracellular"),
]

# Media names to detect (curated; extend as needed)
MEDIA_TERMS = [
    "Blood Agar", "Chocolate Agar", "MacConkey Agar", "Nutrient Agar",
    "MRS Agar", "TCBS Agar", "PPLO Agar", "Bordet-Gengou Agar",
    "Regan-Lowe Agar", "Tryptic Soy Agar", "Sucrose Agar",
]

# Haemolysis
HAEM_BETA = r"\b(beta|β)[-\s]?(haemoly|hemoly)t(ic|is)\b"
HAEM_ALPHA = r"\b(alpha|α)[-\s]?(haemoly|hemoly)t(ic|is)\b"
HAEM_GAMMA = r"\b(gamma|γ)[-\s]?(haemoly|hemoly)t(ic|is)\b|\bnon[-\s]?haemoly\w+|\bnon[-\s]?hemoly\w+"

# NaCl tolerance (>=6%) — also accept "6.5%" / "7%" and "halophilic"
NACL_POS_PATTERNS = [
    r"\bgrows? (in|at)\s*(?:>=?\s*)?6(\.|,)?5?\s*%?\s*nacl\b",
    r"\bgrows? (in|at)\s*(?:>=?\s*)?7\s*%?\s*nacl\b",
    r"\b6(\.|,)?5?\s*%?\s*nacl\b.*\bgrows?\b",
    r"\bhalophil\w+\b",
]
NACL_NEG_PATTERNS = [
    r"\bno growth\b.*\b6(\.|,)?5?\s*%?\s*nacl\b",
]

# Biochemical “Positive/Negative/Variable” tests
# (regex captures `positive|negative|variable` in relaxed forms)
PNV_FIELDS = {
    "Catalase": [r"\bcatalase\s+(positive|negative|variable)\b"],
    "Oxidase": [r"\boxidase\s+(positive|negative|variable)\b"],
    "Coagulase": [r"\bcoagulase\s+(positive|negative|variable)\b"],
    "Urease": [r"\burease\s+(positive|negative|variable)\b"],
    "Indole": [r"\bindole\s+(positive|negative|variable)\b"],
    "Methyl Red": [r"\bmethyl\s*red\b|\bMR\b"],
    "VP": [r"\bVP\b|\bvoges[-\s]?proskauer\b"],
    "Citrate": [r"\bcitrate\s+(positive|negative|variable)\b|\bsimmons'? citrate\s+(?:test\s+)?(positive|negative|variable)\b"],
    "Dnase": [r"\bDN?ase\s+(positive|negative|variable)\b|\bDNA\s+hydrolysis\s+(positive|negative|variable)\b"],
    "Lipase Test": [r"\blipase\s+(positive|negative|variable)\b|\btributyrin\b.*\b(positive|negative|variable)\b"],
    "ONPG": [r"\bonpg\b.*\b(positive|negative|variable)\b|\bβ[-\s]?galactosidase\b.*\b(positive|negative|variable)\b"],
    "Gelatin Hydrolysis": [r"\bgelatin (?:hydrolysis|liquefaction)\s+(positive|negative|variable)\b"],
    "Esculin Hydrolysis": [r"\besculin (?:hydrolysis|hydrolyzed)\s+(positive|negative|variable)\b"],
    "Nitrate Reduction": [r"\bnitrate (?:reduction|reduc\w*)\s+(positive|negative|variable)\b|\breduces nitrate\b|\bnitrate negative\b"],
    "H2S": [r"\bH\s*2\s*S\b.*\b(positive|negative|variable)\b|\bproduces?\s+H\s*2\s*S\b"],
    "Lysine Decarboxylase": [r"\blysin[ey]\s+decarboxylase\s+(positive|negative|variable)\b"],
    "Ornithine Decarboxylase": [r"\bornithin[ey]\s+decarboxylase\s+(positive|negative|variable)\b"],
    "Arginine dihydrolase": [r"\barginine\s+(?:dihydrolase|hydrolysis)\s+(positive|negative|variable)\b"],
}

# Some PNV fields referenced by abbreviations MR/VP appear as lone tokens; handle them here.
MR_EXPL = [
    (r"\bMR\s*(positive|negative|variable)\b", "Methyl Red"),
    (r"\bmethyl\s*red\s*(positive|negative|variable)\b", "Methyl Red"),
]
VP_EXPL = [
    (r"\bVP\s*(positive|negative|variable)\b", "VP"),
    (r"\bvoges[-\s]?proskauer\s*(positive|negative|variable)\b", "VP"),
]

# Motility / Spores
MOTILITY_POS = [r"\bmotile\b|\bswarming\b"]
MOTILITY_NEG = [r"\bnon[-\s]?motile\b|\bnonmotile\b"]
SPORE_POS = [r"\bspore[-\s]?forming\b|\bes?ndospores?\b"]
SPORE_NEG = [r"\bnon[-\s]?spore[-\s]?forming\b|\bnon[-\s]?sporulating\b"]

# Colony morphology / pigment (captured as free text tokens)
PIGMENTS = [
    ("Red", r"\bred\s+pigment\b|\bpigmented\s+red\b|\bred colonies\b"),
    ("Yellow", r"\byellow\s+(?:pigment|colonies)\b"),
    ("Pink", r"\bpink\s+(?:colonies|pigment)\b|\bcoral\b"),
    ("Violet", r"\bviolet\s+(?:pigment|colonies)\b"),
    ("Green", r"\bgreen\s+(?:colonies|pigment)\b"),
    ("White", r"\bwhite\s+(?:colonies?)\b"),
    ("Grey", r"\bgrey|gray\s+(?:colonies?)\b"),
    ("Black", r"\bblack\s+(?:colonies?)\b|\bblack pigment\b"),
    ("Chalky", r"\bchalky\b"),
    ("Rough", r"\brough\b"),
    ("Smooth", r"\bsmooth\b"),
    ("Mucoid", r"\bmucoid\b"),
    ("Opaque", r"\bopaque\b"),
    ("Pearl-like", r"\bpearl[-\s]?like\b"),
    ("Medusa Head", r"\bmedusa[-\s]?head\b"),
    ("Parasporal Crystals", r"\bparasporal\s+crystals?\b"),
    ("Fried Egg", r"\bfried[-\s]?egg\b"),
    ("Non-Haemolytic", r"\bnon[-\s]?haemoly\w+|\bnon[-\s]?hemoly\w+"),
    ("Swarming", r"\bswarming\b"),
]

# Sugar fermentation detector will parse "ferments X, Y and Z but not A or B"
SUGAR_FIELDS = {
    "Glucose Fermentation": ["glucose"],
    "Lactose Fermentation": ["lactose"],
    "Sucrose Fermentation": ["sucrose"],
    "Mannitol Fermentation": ["mannitol"],
    "Maltose Fermentation": ["maltose"],
    "Xylose Fermentation": ["xylose"],
    "Raffinose Fermentation": ["raffinose"],
    "Arabinose Fermentation": ["arabinose"],
    "Sorbitol Fermentation": ["sorbitol"],
    "Trehalose Fermentation": ["trehalose"],
    "Inositol Fermentation": ["inositol"],
    "Rhamnose Fermentation": ["rhamnose"],
    # Add others here if you include them in schema
}

# ──────────────────────────────────────────────────────────────────────────────
# Core parse
# ──────────────────────────────────────────────────────────────────────────────

def parse_text_rules(text: str) -> Dict[str, str]:
    """
    Parse free text and return {field: value} for fields confidently detected.
    Values are normalized via schema.normalize_value; unknowns are omitted.
    """
    out: Dict[str, str] = {}
    t = text.strip()

    if not t:
        return out

    # 1) Gram
    if _search(GRAM_POS_PAT, t):
        _set_if_absent(out, "Gram Stain", "Positive")
    elif _search(GRAM_NEG_PAT, t):
        _set_if_absent(out, "Gram Stain", "Negative")
    elif _search(r"\bgram[-\s]?variable\b", t):
        _set_if_absent(out, "Gram Stain", "Variable")

    # 2) Shape (first good hit wins)
    for pat, mapped in SHAPE_MAP.items():
        if _search(pat, t):
            if mapped:  # None means ambiguous; skip
                _set_if_absent(out, "Shape", mapped)
            break

    # 3) Motility
    if _any(MOTILITY_NEG, t):
        _set_if_absent(out, "Motility", "Negative")
    elif _any(MOTILITY_POS, t):
        _set_if_absent(out, "Motility", "Positive")

    # 4) Spore Formation
    if _any(SPORE_NEG, t):
        _set_if_absent(out, "Spore Formation", "Negative")
    elif _any(SPORE_POS, t):
        _set_if_absent(out, "Spore Formation", "Positive")

    # 5) Haemolysis
    if _search(HAEM_BETA, t):
        _set_if_absent(out, "Haemolysis", "Positive")
        _set_if_absent(out, "Haemolysis Type", "Beta")
    elif _search(HAEM_ALPHA, t):
        _set_if_absent(out, "Haemolysis", "Positive")
        _set_if_absent(out, "Haemolysis Type", "Alpha")
    elif _search(HAEM_GAMMA, t):
        _set_if_absent(out, "Haemolysis", "Negative")

    # 6) Oxygen requirement
    for pat, val in OXYGEN_MAP:
        if _search(pat, t):
            _set_if_absent(out, "Oxygen Requirement", val)
            break

    # 7) Media
    media_hits: List[str] = []
    for m in MEDIA_TERMS:
        # allow flexible casing & hyphens
        pattern = r"\b" + re.escape(m).replace(r"\ ", r"\s*") + r"\b"
        if _search(pattern, t):
            media_hits.append(m)
    if media_hits:
        _set_if_absent(out, "Media Grown On", "; ".join(sorted(set(media_hits))))

    # 8) NaCl tolerance (>=6%)
    if _any(NACL_POS_PATTERNS, t):
        _set_if_absent(out, "NaCl Tolerant (>=6%)", "Positive")
    elif _any(NACL_NEG_PATTERNS, t):
        _set_if_absent(out, "NaCl Tolerant (>=6%)", "Negative")

    # 9) Growth Temperature(s)
    # Matches: "at 37 °C", "30–37 °C", "30-37C", "35 to 37 C"
    temp_ranges = _findall(
        r"(?:(?:at|around|~)\s*)?(\d{2})(?:\s*[–\-to]+\s*(\d{2}))?\s*°?\s*C",
        t,
    )
    # Avoid duplicates; pick the broadest range
    rng_low, rng_high = None, None
    for a, b in temp_ranges:
        low = int(a)
        high = int(b) if b else int(a)
        if rng_low is None or low < rng_low:
            rng_low = low
        if rng_high is None or high > rng_high:
            rng_high = high
    if rng_low is not None and rng_high is not None:
        _set_if_absent(out, "Growth Temperature", f"{rng_low}//{rng_high}")

    # 10) Biochem P/N/V tests (generic handlers)
    # Handle MR/VP first for their abbreviations
    for pat, fld in MR_EXPL:
        m = _search(pat, t)
        if m:
            _set_if_absent(out, fld, m.group(1).capitalize())

    for pat, fld in VP_EXPL:
        m = _search(pat, t)
        if m:
            _set_if_absent(out, fld, m.group(1).capitalize())

    # Generic patterns
    for field, patterns in PNV_FIELDS.items():
        # explicit "X positive/negative/variable"
        for p in patterns:
            m = _search(p, t)
            if m:
                # Some patterns capture only the status; others imply positive
                if m.lastindex and m.group(m.lastindex):
                    status = m.group(m.lastindex).capitalize()
                else:
                    status = "Positive"
                _set_if_absent(out, field, status)
                break

        # additional simple phrases
        if field == "Nitrate Reduction":
            if _search(r"\breduces nitrate\b", t):
                _set_if_absent(out, field, "Positive")
            elif _search(r"\bnitrate (?:reduction )?negative\b", t):
                _set_if_absent(out, field, "Negative")

        if field == "H2S":
            if _search(r"\bproduces?\s*H\s*2\s*S\b", t):
                _set_if_absent(out, field, "Positive")
            elif _search(r"\bH\s*2\s*S\s*negative\b", t):
                _set_if_absent(out, field, "Negative")

    # 11) Enzyme/IMViC short negative/positive forms like "indole negative"
    for short, field in [
        (r"\bindole\s+(positive|negative|variable)\b", "Indole"),
        (r"\bmethyl\s*red\s+(positive|negative|variable)\b", "Methyl Red"),
        (r"\bVP\s+(positive|negative|variable)\b", "VP"),
        (r"\bcitrate\s+(positive|negative|variable)\b", "Citrate"),
        (r"\burease\s+(positive|negative|variable)\b", "Urease"),
        (r"\bDN?ase\s+(positive|negative|variable)\b", "Dnase"),
        (r"\boxidase\s+(positive|negative|variable)\b", "Oxidase"),
        (r"\bcatalase\s+(positive|negative|variable)\b", "Catalase"),
        (r"\bcoagulase\s+(positive|negative|variable)\b", "Coagulase"),
        (r"\blipase\s+(positive|negative|variable)\b", "Lipase Test"),
    ]:
        m = _search(short, t)
        if m:
            _set_if_absent(out, field, m.group(1).capitalize())

    # 12) Sugar fermentation logic
    # "Ferments X, Y and Z"  → Positive
    # "but not A or B"       → Negative for A and B
    # Also detect "non-fermenter of lactose" patterns
    # First, find dedicated negatives
    for fname, sugars in SUGAR_FIELDS.items():
        for s in sugars:
            if _search(rf"\b(non[-\s]?ferment(?:er|ing)? of|does not ferment|not ferment(?:ing)?)\s+{s}\b", t):
                _set_if_absent(out, fname, "Negative")

    # Global positive list after "ferments"
    # capture segment after "ferments" up to sentence end or 'but not'
    for m in re.finditer(r"\bferments?\s+([^.]+)", t, flags=re.IGNORECASE):
        segment = m.group(1)
        # stop at "but not ..."
        segment = re.split(r"\bbut\s+not\b", segment, flags=re.IGNORECASE)[0]
        # split list by commas and 'and'
        tokens = re.split(r"[,\s]+and\s+|,\s*|\s+and\s+", segment, flags=re.IGNORECASE)
        tokens = [tok.strip(" .;") for tok in tokens if tok.strip()]
        for fname, sugars in SUGAR_FIELDS.items():
            for s in sugars:
                if any(re.search(rf"\b{s}\b", tok, flags=re.IGNORECASE) for tok in tokens):
                    _set_if_absent(out, fname, "Positive")

    # Now "but not A or B" negatives
    for m in re.finditer(r"\bbut\s+not\s+([^.]+)", t, flags=re.IGNORECASE):
        segment = m.group(1)
        tokens = re.split(r",\s*|\s+or\s+|\s+and\s+", segment)
        tokens = [tok.strip(" .;") for tok in tokens if tok.strip()]
        for fname, sugars in SUGAR_FIELDS.items():
            for s in sugars:
                if any(re.search(rf"\b{s}\b", tok, flags=re.IGNORECASE) for tok in tokens):
                    _set_if_absent(out, fname, "Negative")

    # 13) Colony morphology / pigments (collect distinct words)
    morph_hits: List[str] = []
    for label, pat in PIGMENTS:
        if _search(pat, t):
            morph_hits.append(label)
    if morph_hits:
        _set_if_absent(out, "Colony Morphology", "; ".join(sorted(set(morph_hits))))

    # 14) Normalize all values that we set
    normed = {k: normalize_value(k, v) for k, v in out.items()}

    return normed


# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrapper: parse + canonicalize
# ──────────────────────────────────────────────────────────────────────────────

def parse_and_canonicalize(text: str) -> Dict[str, str]:
    """
    Rule-parse the text, then fill in missing fields with 'Unknown' and move
    any non-schema keys into 'Other' via schema.canonicalize_record.
    """
    partial = parse_text_rules(text)
    full = canonicalize_record(partial)
    return full
