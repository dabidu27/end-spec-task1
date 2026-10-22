import re
import numpy as np

#retail id -> catering id, for the five pairs split at 5 kg / 5 litres
PAIRS = {4977: 7623,   # flour
         7788: 3834,   # sugar
         7014: 9991,   # cooking oil
         1565: 8868,   # dried pasta
         9278: 7444}   # rice
CATERING_TO_RETAIL = {c: r for r, c in PAIRS.items()}
THRESHOLD = 5.0 #"5 kg and over" / "5 litres and over" is catering

# factor to kg (or litres; 1 l is treated as 1 kg-equivalent for the threshold)
UNITS = {"kg": 1, "kgs": 1, "g": 1e-3, "gr": 1e-3, "gm": 1e-3, "grams": 1e-3,
         "l": 1, "lt": 1, "ltr": 1, "litre": 1, "liter": 1, "litres": 1, "liters": 1,
         "ml": 1e-3, "cl": 1e-2}
_RX = re.compile(r"(?:(\d+)x)?(\d*\.?\d+)([a-z]+)")

def parse_pack_size(s):
    """Total pack size in kg/litres, or NaN if it is a count ('6pk', 'each', 'x6')."""
    s = "".join(str(s).lower().split()).replace(",", ".")
    m = _RX.fullmatch(s)
    if not m or m.group(3) not in UNITS:
        return np.nan
    return float(m.group(1) or 1) * float(m.group(2)) * UNITS[m.group(3)]

def to_family(cat_id):
    """Catering id -> retail id, so each pair is ONE class for the text model."""
    return CATERING_TO_RETAIL.get(cat_id, cat_id)

def resolve_pack_side(cat_id, size):
    """If cat_id is one of the paired categories, pick the side from the pack size."""
    if cat_id in PAIRS and not np.isnan(size):
        return PAIRS[cat_id] if size >= THRESHOLD else cat_id
    return cat_id