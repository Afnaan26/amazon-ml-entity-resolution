# Exploratory Data Analysis & Noise Pattern Documentation
**Role:** Person A — Data, EDA & Preprocessing  
**Dataset:** Amazon BER (Business Entity Resolution) Challenge 2026  
**Last Updated:** September 2026  

---

## Executive Overview
This document synthesizes key quantitative insights, structural noise patterns, missing value rates, match count distributions, and country set dynamics across all train and test datasets. It serves as the authoritative baseline for downstream candidate generation (Person B) and feature engineering/modeling (Person C).

---

## 1. Dataset Dimensions & Schema Overview

| Dataset Split | File Name | Row Count | File Size (Raw TSV) | Unique `entity_id`s | Null Names | Null Addresses | Null Countries |
|---|---|---|---|---|---|---|---|
| **Train S1** | `train_source1.tsv` | 2,206,821 | 210 MB | 2,206,821 (100%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| **Train S2** | `train_source2.tsv` | 5,034,616 | 489 MB | 5,034,616 (100%) | 2 (0.00%) | 168,967 (3.36%) | 0 (0.00%) |
| **Train S3** | `train_source3.tsv` | 5,285,603 | 503 MB | 5,285,603 (100%) | 13 (0.00%) | 175,916 (3.33%) | 0 (0.00%) |
| **Train GT** | `train_ground_truth.tsv` | 2,206,821 | 127 MB | 2,206,821 S1 keys | — | — | — |
| **Test S1** | `test_source1.tsv` | 1,732,544 | 175 MB | 1,732,544 (100%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| **Test S2** | `test_source2.tsv` | 4,887,273 | 509 MB | 4,887,273 (100%) | 1 (0.00%) | 164,102 (3.36%) | 0 (0.00%) |
| **Test S3** | `test_source3.tsv` | 5,082,316 | 506 MB | 5,082,316 (100%) | 11 (0.00%) | 169,214 (3.33%) | 0 (0.00%) |

> [!KEY FINDING]
> 1. **Zero Duplicate IDs:** `entity_id` is 100% unique within every source file.
> 2. **Address Missingness in S2/S3:** ~3.35% of records in Source 2 and Source 3 have missing/null addresses. Source 1 has zero null addresses.
> 3. **Boolean Flags Feature:** Downstream models (Person C) MUST use missing address flags (`has_address=False`) to avoid penalizing pair similarity scores when address is omitted in one source.

---

## 2. Ground Truth Match Count & Multiplicity Analysis

Every `source1_entity_id` in `train_ground_truth.tsv` is linked to a comma-separated list of true matched entity IDs from Source 2 (`S2-*`) and Source 3 (`S3-*`).

### Total Matches per Source 1 Entity:
- **Minimum matches:** 1
- **Maximum matches:** 11
- **Singletons (0 matches in Ground Truth):** **0** (All 2,206,821 S1 entities have at least 1 match).

```
Match Count Breakdown per S1 Entity:
-------------------------------------
1 Match  : 242,404  (11.0%)
2 Matches: 375,212  (17.0%)
3 Matches: 530,841  (24.1%)
4 Matches: 484,115  (21.9%)
5 Matches: 321,957  (14.6%)
6 Matches: 164,868  (7.5%)
7 Matches: 63,968   (2.9%)
8 Matches: 18,680   (0.8%)
9 Matches: 4,205    (0.2%)
10 Matches: 534     (<0.1%)
11 Matches: 37      (<0.1%)
Total S1 Entities: 2,206,821
```

### Breakdown by Match Source (S2 vs S3):
- **S2 matches per S1 entity:**
  - 0 S2 matches: 287,745 (13.0%)
  - 1 S2 match: 789,108 (35.8%)
  - 2 S2 matches: 652,779 (29.6%)
  - 3 S2 matches: 333,957 (15.1%)
  - 4 S2 matches: 119,078 (5.4%)
  - 5 S2 matches: 24,154 (1.1%)
- **S3 matches per S1 entity:**
  - 0 S3 matches: 266,276 (12.1%)
  - 1 S3 match: 716,417 (32.5%)
  - 2 S3 matches: 668,375 (30.3%)
  - 3 S3 matches: 372,443 (16.9%)
  - 4 S3 matches: 145,116 (6.6%)
  - 5 S3 matches: 35,378 (1.6%)
  - 6 S3 matches: 2,816 (0.1%)

> [!CRITICAL INSIGHT FOR PERSON B & C]
> S1 entities frequently match **multiple candidate entities within the same source** (e.g. 2 or 3 entities in S2). Do NOT restrict candidate generation or final clustering to 1-to-1 matching!

---

## 3. Country Distribution & Open-Set Country Dynamics

### Train vs Test Country Distribution:

| Country | Train S1 | Train S2 | Train S3 | Test S1 | Test S2 | Test S3 |
|---|---|---|---|---|---|---|
| **US** | 1,323,633 (60.0%) | 3,016,817 (59.9%) | 3,170,056 (60.0%) | 663,106 (38.3%) | 1,871,330 (38.3%) | 1,945,701 (38.3%) |
| **India** | 883,188 (40.0%) | 2,017,799 (40.1%) | 2,115,547 (40.0%) | 809,986 (46.8%) | 2,312,565 (47.3%) | 2,405,000 (47.3%) |
| **France** | **0 (0.0%)** | **0 (0.0%)** | **0 (0.0%)** | **259,452 (15.0%)** | **703,378 (14.4%)** | **731,615 (14.4%)** |

> [!IMPORTANT ALERT FOR ALL TEAMS]
> **France is present ONLY in the test set (15.0% of test records)!**
> 1. Country MUST be treated as an **open string set**.
> 2. Hardcoded country dictionaries or fixed enums (`{'US': 0, 'India': 1}`) WILL CRASH or misclassify on test evaluation!
> 3. Normalization pipeline handles unseen countries dynamically (`normalize_country('France') -> 'FRANCE'`).

---

## 4. Firsthand Noise Pattern Catalogue (50+ Matched Pairs Sampled)

Manual inspection of 60 ground truth matched pairs reveals six distinct categories of noise and record variations:

### 1. Legal Entity Suffix Differences & Omissions
- **Examples:**
  - `Maure Williams Colombier Inc` vs `Maure Williams Colombier`
  - `Raj Investments Pvt Ltd` vs `Raj Investments`
  - `Apex Corp & Co.` vs `Apex Corporation Company`
- **Normalization Fix:** Expand legal suffixes into canonical lowercase tokens (`inc` -> `incorporated`, `ltd` -> `limited`, `pvt` -> `private`, `corp` -> `corporation`, `co` -> `company`).

### 2. Website TLD / Domain Names in Business Names
- **Examples:**
  - `maurewilliamscolombier.com` vs `Maure Williams Colombier Inc`
  - `technologies.co.in` vs `Technologies Pvt Ltd`
- **Normalization Fix:** Strip TLDs (`.com`, `.net`, `.co.in`, `.org`, `.us`, `.fr`, etc.) prior to punctuation removal.

### 3. Typographical Errors & Character Transliteration
- **Examples:**
  - `Williams` -> `Wilblims`
  - `Wayne` -> `Wanye`
  - `Ticonderoga` -> `Ticonderoga Townshiip`
  - `Drexkor` vs `Dréxkor` (accented Unicode vs ASCII)
- **Normalization Fix:** NFKC Unicode normalization + RapidFuzz / Levenshtein / Jaro-Winkler edit distance in feature engineering.

### 4. Word Order & Modifiers
- **Examples:**
  - `Maure Williams Colombier Inc` vs `Maure Williams Inc Center`
  - `Raj Investments LLP` vs `LLP Raj Investments`
- **Normalization Fix:** Token-set similarity (Jaccard, Token Sort Ratio, TF-IDF overlap) to handle permutations gracefully.

### 5. Address Variations & Abbreviation Expansion
- **Examples:**
  - `85 Wayne Avenue, Ticonderoga, NY` vs `85 Wanye Ave, Ticonderoga Townshiip, New York`
- **Normalization Fix:** Expand street suffixes (`rd` -> `road`, `st` -> `street`, `ave` -> `avenue`, `ste` -> `suite`, `apt` -> `apartment`, `pkwy` -> `parkway`).

### 6. Missing Address & Missing PIN / Postal Codes
- **Examples:**
  - Matching pair S1-965667 <-> S2-681193310 has `business_address = NaN` in S2.
- **Normalization Fix:** Extract PIN / Zip code (`60601`, `110001`, `75002`) into dedicated `postal_code` field, and output boolean flags (`has_address`, `has_postal_code`).

---

## 5. Pipeline Handoff & Lock Confirmation

### Shared Preprocessing Module:
- **Path:** `src/common/normalize.py`
- **Unit Tests:** `tests/test_normalize.py` (100% pass rate)

### Generated Clean Datasets:
- `dataset/train/train_source1_clean.tsv` & `.parquet`
- `dataset/train/train_source2_clean.tsv` & `.parquet`
- `dataset/train/train_source3_clean.tsv` & `.parquet`
- `dataset/test/test_source1_clean.tsv` & `.parquet`
- `dataset/test/test_source2_clean.tsv` & `.parquet`
- `dataset/test/test_source3_clean.tsv` & `.parquet`

### Field Schema Locked for Downstream Work:
- `entity_id` (str)
- `business_name` (str)
- `business_address` (str)
- `country` (str)
- `name_norm` (str) — canonical lowercase normalized name
- `address_norm` (str) — canonical lowercase normalized address
- `country_norm` (str) — uppercase open-set country
- `postal_code` (str) — extracted PIN / ZIP string
- `has_name` (bool)
- `has_address` (bool)
- `has_country` (bool)
- `has_postal_code` (bool)
- `name_len` (int)
- `address_len` (int)
- `name_word_count` (int)
- `address_word_count` (int)
