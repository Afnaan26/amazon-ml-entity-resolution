# ML Challenge 2026: Business Entity Resolution Methodology

**Team Name:** Business Entity Resolution Team  
**Section Author:** Person A — Data, EDA & Preprocessing  
**Date:** September 2026  

---

## 1. Executive Summary
This document presents our end-to-end approach to the Amazon 2026 Business Entity Resolution challenge. Our architecture combines standardized multilingual preprocessing, open-set country normalization, candidate generation (blocking), edit distance & token-set feature engineering, and high-precision ensemble classification.

---

## 2. Data & Preprocessing (Person A Contribution)

### 2.1 Exploratory Data Analysis Insights

1. **Volume & Scale:**
   - Train S1: 2,206,821 records | Train S2: 5,034,616 records | Train S3: 5,285,603 records
   - Test S1: 1,732,544 records | Test S2: 4,887,273 records | Test S3: 5,082,316 records
   - Total dataset scale exceeds **24 million entity records**.

2. **Ground Truth Multiplicity:**
   - Every Source 1 entity matches between 1 and 11 records in Source 2 and Source 3. Zero singletons (0 matches) exist in train ground truth.
   - 35.8% of S1 entities have 1 match in S2; 29.6% have 2 matches; 15.1% have 3 matches.

3. **Missing Value Rates:**
   - Source 1 has 0% null values across all fields.
   - Source 2 and Source 3 contain **3.36% and 3.33% missing addresses** respectively.
   - Business names have < 0.001% missingness.

4. **Country Set Dynamics & Unseen Test Distribution:**
   - Train data consists of US (60.0%) and India (40.0%).
   - Test data introduces **France (15.0%)**, alongside India (47.0%) and US (38.0%).
   - Preprocessing MUST treat country as an open string set to ensure zero failures on test evaluations.

---

### 2.2 Data Cleaning & Normalization Layer

We developed `src/common/normalize.py` as the shared, deterministic cleaning layer for all downstream tasks.

#### Key Normalization Stages:
1. **Unicode Normalization:** Applies NFKC normalization to handle accents, diacritics, and non-ASCII character variations (`Dréxkor` -> `Drexkor`).
2. **Domain TLD Removal:** Detects and strips top-level website domain extensions appended to company names (e.g. `company.com` -> `company`, `tech.co.in` -> `tech`).
3. **Acronym & Abbreviation Standardization:**
   - Cleans dotted acronyms (`S.A.` -> `sa`, `P.V.T.` -> `pvt`, `L.T.D.` -> `ltd`).
   - Standardizes business suffixes (`corp` -> `corporation`, `ltd` -> `limited`, `pvt` -> `private`, `inc` -> `incorporated`, `co` -> `company`, `llp` -> `limited liability partnership`).
   - Standardizes address terms (`rd` -> `road`, `st` -> `street`, `ave` -> `avenue`, `ste` -> `suite`, `apt` -> `apartment`).
4. **Postal Code / PIN Code Extraction:**
   - Uses regex patterns (`\b(\d{6}|\d{5}(?:-\d{4})?)\b`) to extract 5-digit US ZIPs, 6-digit Indian PINs, and 5-digit French postal codes into a dedicated `postal_code` field.
5. **Missing Feature Flags:**
   - Generates explicit boolean flags (`has_name`, `has_address`, `has_country`, `has_postal_code`) to inform classifier models when similarity metrics are computed on missing fields.
6. **Open-Set Country Normalization:**
   - Uppercases and standardizes country strings dynamically without fixed dictionaries, preventing pipeline crashes on unseen test countries like France.

---

### 2.3 Unit Testing & Output Verification

- Unit test suite implemented in `tests/test_normalize.py` (100% test pass rate across 20+ sample pair variations).
- Preprocessed clean datasets exported to both `.parquet` (fast binary format) and `*_clean.tsv` across all 6 files.
- Verified schema lock with Person B and Person C team members.
