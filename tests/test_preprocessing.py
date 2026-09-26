from src.preprocessing import (
    normalize_name,
    normalize_address,
    normalize_country
)


# ============================================================
# NAME TESTS
# ============================================================

def test_name_lowercase():
    assert normalize_name("ABC TECHNOLOGIES") == "abc technologies"


def test_name_punctuation():
    assert normalize_name("ABC, Technologies!") == "abc technologies"


def test_name_abbreviations():
    assert normalize_name(
        "ABC Pvt. Ltd."
    ) == "abc private limited"


def test_name_corporation():
    assert normalize_name(
        "ABC Corp."
    ) == "abc corporation"


def test_name_company():
    assert normalize_name(
        "ABC Co."
    ) == "abc company"


def test_name_whitespace():
    assert normalize_name(
        "  ABC     Technologies   "
    ) == "abc technologies"


def test_name_domain():
    assert normalize_name(
        "maurewilliamscolombier.com"
    ) == "maurewilliamscolombier"


def test_name_unicode():
    assert normalize_name(
        "Dréxkor"
    ) == "dréxkor"


def test_name_nan():
    assert normalize_name(None) == ""


# ============================================================
# ADDRESS TESTS
# ============================================================

def test_address_lowercase():
    assert normalize_address(
        "85 WAYNE AVENUE"
    ) == "85 wayne avenue"


def test_address_punctuation():
    assert normalize_address(
        "85, Wayne Avenue, Ticonderoga, NY"
    ) == "85 wayne avenue ticonderoga ny"


def test_address_road_abbreviation():
    assert normalize_address(
        "12 MG Rd"
    ) == "12 mg road"


def test_address_street_abbreviation():
    assert normalize_address(
        "25 Main St."
    ) == "25 main street"


def test_address_avenue_abbreviation():
    assert normalize_address(
        "6114 10th Ave"
    ) == "6114 10th avenue"


def test_address_numbers_preserved():
    result = normalize_address(
        "85 Wayne Avenue"
    )

    assert "85" in result


def test_address_nan():
    assert normalize_address(None) == ""


# ============================================================
# COUNTRY TESTS
# ============================================================

def test_country_lowercase():
    assert normalize_country("US") == "us"


def test_country_whitespace():
    assert normalize_country("  India  ") == "india"


def test_country_open_set():
    assert normalize_country("France") == "france"


def test_country_nan():
    assert normalize_country(None) == ""
    