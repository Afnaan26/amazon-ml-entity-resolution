"""
Entity Resolution Preprocessing & Normalization Module
Provides fast, robust, and open-set text normalization and feature extraction for business names and addresses.
"""

import re
import unicodedata
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional, Set


# ============================================================
# UNICODE & WHITESPACE HELPERS
# ============================================================

def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters using NFKC format."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text)


def normalize_whitespace(text: str) -> str:
    """Collapse consecutive whitespace and strip leading/trailing spaces."""
    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# ABBREVIATION DICTIONARIES
# ============================================================

# Domain legal entity suffixes and common business terms
NAME_ABBREVIATIONS: Dict[str, str] = {
    "pvt": "private",
    "pvtltd": "private limited",
    "ltd": "limited",
    "corp": "corporation",
    "co": "company",
    "inc": "incorporated",
    "llp": "limited liability partnership",
    "llc": "limited liability company",
    "plc": "public limited company",
    "gmbh": "gmbh",
    "sarl": "sarl",
    "sa": "sa",
    "assoc": "association",
    "assn": "association",
    "mfg": "manufacturing",
    "mfr": "manufacturer",
    "ind": "industries",
    "inds": "industries",
    "intl": "international",
    "natl": "national",
    "serv": "services",
    "svcs": "services",
    "tech": "technologies",
    "techno": "technology",
    "sys": "systems",
    "soln": "solutions",
    "solns": "solutions",
    "group": "group",
    "dept": "department",
    "div": "division",
    "fdn": "foundation",
    "inst": "institute",
}

# Common address token abbreviations (US, India, France/International)
ADDRESS_ABBREVIATIONS: Dict[str, str] = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "hwy": "highway",
    "pkwy": "parkway",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "cir": "circle",
    "apt": "apartment",
    "ste": "suite",
    "bldg": "building",
    "fl": "floor",
    "rm": "room",
    "dept": "department",
    "po": "po",
    "pobox": "po box",
    "opp": "opposite",
    "nr": "near",
    "adj": "adjacent",
    "ext": "extension",
    "twp": "township",
    "vill": "village",
    "dist": "district",
    "distt": "district",
    "sec": "sector",
    "sct": "sector",
    "blk": "block",
    "no": "number",
    "num": "number",
}

# Regex to strip top-level domains from business names (e.g., myshop.com -> myshop)
TLD_REGEX = re.compile(
    r"\.(com|net|org|in|co|biz|us|fr|info|io|gov|edu|me|store|online|tech|co\.in|co\.uk|org\.in)$",
    re.IGNORECASE,
)

# Regex patterns for Postal Code / ZIP Code extraction
# India: 6 digits (e.g., 110001, 400001)
# US: 5 digits or 5+4 (e.g., 10001, 90210, 90210-1234)
# France: 5 digits (e.g., 75001, 69001)
ZIP_PIN_REGEX = re.compile(
    r"\b(\d{6}|\d{5}(?:-\d{4})?)\b"
)

# Regex to standardize dotted acronyms like S.A. -> sa, P.V.T. -> pvt, L.T.D. -> ltd
DOTTED_ACRONYM_REGEX = re.compile(r"\b([a-zA-Z])\.(?=[a-zA-Z]\.|\s|$)")


# ============================================================
# CORE PREPROCESSING FUNCTIONS
# ============================================================

def clean_dotted_acronyms(text: str) -> str:
    """Convert dotted acronyms like s.a. -> sa, p.v.t. -> pvt, l.t.d. -> ltd."""
    # First handle multi-dot acronyms like s.a. or p.v.t.
    text = re.sub(r"\b([a-zA-Z])\.([a-zA-Z])\.(?:([a-zA-Z])\.)?", r"\1\2\3", text)
    return text


def normalize_name(name: Optional[str]) -> str:
    """
    Clean and normalize business name.
    1. Unicode normalization (NFKC)
    2. Lowercase & strip TLD extensions
    3. Clean dotted acronyms (e.g. S.A. -> sa, S.A.R.L. -> sarl)
    4. Replace & with ' and '
    5. Strip non-alphanumeric punctuation
    6. Standardize business entity abbreviations
    7. Whitespace cleanup
    """
    if pd.isna(name) or name is None:
        return ""

    text = normalize_unicode(str(name))
    text = text.lower().strip()

    # Strip top-level domain if name looks like a website domain
    text = TLD_REGEX.sub("", text)

    # Replace & with ' and '
    text = re.sub(r"&", " and ", text)

    # Clean dotted acronyms (e.g. s.a. -> sa, p.v.t. -> pvt)
    text = clean_dotted_acronyms(text)

    # Keep Unicode letters and digits
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    text = normalize_whitespace(text)

    tokens = text.split()
    normalized_tokens = [NAME_ABBREVIATIONS.get(tok, tok) for tok in tokens]

    return " ".join(normalized_tokens)


def extract_postal_code(address: Optional[str]) -> Optional[str]:
    """Extract PIN / Zip / Postal code from address string if present."""
    if pd.isna(address) or address is None:
        return None
    
    match = ZIP_PIN_REGEX.search(str(address))
    if match:
        return match.group(1)
    return None


def normalize_address(address: Optional[str]) -> str:
    """
    Clean and normalize address string.
    1. Unicode normalization (NFKC)
    2. Lowercase
    3. Replace & with ' and '
    4. Strip punctuation, preserving alphanumeric tokens
    5. Standardize address abbreviations (rd -> road, st -> street, etc.)
    6. Whitespace cleanup
    """
    if pd.isna(address) or address is None:
        return ""

    text = normalize_unicode(str(address))
    text = text.lower().strip()

    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = normalize_whitespace(text)

    tokens = text.split()
    normalized_tokens = [ADDRESS_ABBREVIATIONS.get(tok, tok) for tok in tokens]

    return " ".join(normalized_tokens)


def normalize_country(country: Optional[str]) -> str:
    """
    Open-set country normalization.
    Treats country as open string set.
    Performs uppercase standardization (e.g., 'us' -> 'US', 'india' -> 'INDIA', 'france' -> 'FRANCE').
    Handles missing/unseen countries smoothly without errors.
    """
    if pd.isna(country) or country is None:
        return "UNKNOWN"

    text = normalize_unicode(str(country)).upper().strip()
    text = normalize_whitespace(text)

    if not text:
        return "UNKNOWN"
    return text


def preprocess_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess a single record dict and return clean fields + boolean flags.
    """
    name_raw = record.get("business_name")
    addr_raw = record.get("business_address")
    country_raw = record.get("country")

    name_clean = normalize_name(name_raw)
    addr_clean = normalize_address(addr_raw)
    country_clean = normalize_country(country_raw)

    postal_code = extract_postal_code(addr_raw)

    has_name = bool(name_clean)
    has_address = bool(addr_clean)
    has_country = country_clean != "UNKNOWN"
    has_postal_code = postal_code is not None

    return {
        "entity_id": record.get("entity_id"),
        "business_name": name_raw,
        "business_address": addr_raw,
        "country": country_raw,
        "name_norm": name_clean,
        "address_norm": addr_clean,
        "country_norm": country_clean,
        "postal_code": postal_code if postal_code else "",
        "has_name": has_name,
        "has_address": has_address,
        "has_country": has_country,
        "has_postal_code": has_postal_code,
        "name_len": len(name_clean),
        "address_len": len(addr_clean),
        "name_word_count": len(name_clean.split()) if name_clean else 0,
        "address_word_count": len(addr_clean.split()) if addr_clean else 0,
    }


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process an entire DataFrame containing entity records.
    Adds `name_norm`, `address_norm`, `country_norm`, `postal_code`, and boolean flag features.
    """
    df_out = df.copy()

    df_out["name_norm"] = df_out["business_name"].apply(normalize_name)
    df_out["address_norm"] = df_out["business_address"].apply(normalize_address)
    df_out["country_norm"] = df_out["country"].apply(normalize_country)
    df_out["postal_code"] = df_out["business_address"].apply(extract_postal_code).fillna("")

    df_out["has_name"] = df_out["name_norm"].str.len() > 0
    df_out["has_address"] = df_out["address_norm"].str.len() > 0
    df_out["has_country"] = df_out["country_norm"] != "UNKNOWN"
    df_out["has_postal_code"] = df_out["postal_code"].str.len() > 0

    df_out["name_len"] = df_out["name_norm"].str.len()
    df_out["address_len"] = df_out["address_norm"].str.len()
    df_out["name_word_count"] = df_out["name_norm"].apply(lambda s: len(s.split()) if s else 0)
    df_out["address_word_count"] = df_out["address_norm"].apply(lambda s: len(s.split()) if s else 0)

    return df_out
