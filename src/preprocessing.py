import re
import unicodedata
import pandas as pd


# ============================================================
# HELPERS
# ============================================================

def normalize_unicode(text):
    return unicodedata.normalize("NFKC", str(text))


def normalize_spaces(text):
    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# NAME NORMALIZATION
# ============================================================

NAME_ABBREVIATIONS = {
    "pvt": "private",
    "pvt.": "private",
    "ltd": "limited",
    "ltd.": "limited",
    "corp": "corporation",
    "corp.": "corporation",
    "co": "company",
    "co.": "company",
    "inc": "incorporated",
    "inc.": "incorporated",
}


def normalize_name(name):
    if pd.isna(name):
        return ""

    name = normalize_unicode(name)
    name = name.lower().strip()

    # Remove domain suffix
    name = re.sub(
        r"\.(com|net|org|in|co|biz|us)$",
        "",
        name
    )

    # Keep Unicode letters and numbers
    name = re.sub(
        r"[^\w\s]",
        " ",
        name,
        flags=re.UNICODE
    )

    name = normalize_spaces(name)

    tokens = name.split()

    tokens = [
        NAME_ABBREVIATIONS.get(token, token)
        for token in tokens
    ]

    return " ".join(tokens)


# ============================================================
# ADDRESS NORMALIZATION
# ============================================================

ADDRESS_ABBREVIATIONS = {
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
    "apt": "apartment",
    "ste": "suite",
}


def normalize_address(address):
    if pd.isna(address):
        return ""

    address = normalize_unicode(address)
    address = address.lower().strip()

    # Keep Unicode letters and numbers
    address = re.sub(
        r"[^\w\s]",
        " ",
        address,
        flags=re.UNICODE
    )

    address = normalize_spaces(address)

    tokens = address.split()

    tokens = [
        ADDRESS_ABBREVIATIONS.get(token, token)
        for token in tokens
    ]

    return " ".join(tokens)


# ============================================================
# COUNTRY NORMALIZATION
# ============================================================

def normalize_country(country):
    if pd.isna(country):
        return ""

    country = normalize_unicode(country)
    country = country.lower().strip()

    return normalize_spaces(country)


# ============================================================
# DATAFRAME PREPROCESSING
# ============================================================

def preprocess_dataframe(df):

    df = df.copy()

    df["name_norm"] = df["business_name"].apply(
        normalize_name
    )

    df["address_norm"] = df["business_address"].apply(
        normalize_address
    )

    df["country_norm"] = df["country"].apply(
        normalize_country
    )

    return df