import sys
import os
import unittest

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.common.normalize import (
    normalize_name,
    normalize_address,
    normalize_country,
    extract_postal_code,
    preprocess_record,
)


class TestNormalization(unittest.TestCase):

    def test_name_abbreviation_expansion(self):
        self.assertEqual(
            normalize_name("Maure Williams Colombier Inc"),
            "maure williams colombier incorporated"
        )
        self.assertEqual(
            normalize_name("Raj Investments Pvt Ltd"),
            "raj investments private limited"
        )
        self.assertEqual(
            normalize_name("Apex Corp & Co."),
            "apex corporation and company"
        )

    def test_tld_removal(self):
        self.assertEqual(
            normalize_name("maurewilliamscolombier.com"),
            "maurewilliamscolombier"
        )
        self.assertEqual(
            normalize_name("technologies.co.in"),
            "technologies"
        )

    def test_address_normalization(self):
        self.assertEqual(
            normalize_address("85 Wayne Avenue, Ticonderoga, NY"),
            "85 wayne avenue ticonderoga ny"
        )
        self.assertEqual(
            normalize_address("85 Wanye Ave, Ticonderoga Townshiip, New York"),
            "85 wanye avenue ticonderoga townshiip new york"
        )

    def test_postal_code_extraction(self):
        # US zip
        self.assertEqual(extract_postal_code("85 Wayne Ave, NY 10001"), "10001")
        # India PIN
        self.assertEqual(extract_postal_code("Plot 4, Sector 18, Gurgaon 122015"), "122015")
        # France postal code
        self.assertEqual(extract_postal_code("12 Rue de la Paix, 75002 Paris"), "75002")
        # Missing postal code
        self.assertIsNone(extract_postal_code("Wayne Avenue, Ticonderoga"))

    def test_open_set_country_handling(self):
        self.assertEqual(normalize_country("US"), "US")
        self.assertEqual(normalize_country("India"), "INDIA")
        self.assertEqual(normalize_country("France"), "FRANCE")
        self.assertEqual(normalize_country("  united states  "), "UNITED STATES")
        self.assertEqual(normalize_country(None), "UNKNOWN")

    def test_sample_noise_pairs(self):
        # Pair 1: Abbreviation & missing address
        n1 = normalize_name("Maure Williams Colombier Inc")
        n2 = normalize_name("Maure Williams Colombier")
        self.assertTrue("colombier" in n1 and "colombier" in n2)

        # Pair 2: Ampersand replacement
        n3 = normalize_name("Johnson & Johnson")
        self.assertEqual(n3, "johnson and johnson")

        # Pair 3: French company
        n4 = normalize_name("Societe Generale S.A.")
        self.assertEqual(n4, "societe generale sa")

    def test_preprocess_record(self):
        rec = {
            "entity_id": "S1-12345",
            "business_name": "Acme Services Pvt. Ltd.",
            "business_address": "123 Main St, Suite 400, Chicago, IL 60601",
            "country": "US"
        }
        processed = preprocess_record(rec)
        self.assertEqual(processed["name_norm"], "acme services private limited")
        self.assertEqual(processed["address_norm"], "123 main street suite 400 chicago il 60601")
        self.assertEqual(processed["country_norm"], "US")
        self.assertEqual(processed["postal_code"], "60601")
        self.assertTrue(processed["has_postal_code"])
        self.assertTrue(processed["has_name"])
        self.assertTrue(processed["has_address"])
        self.assertTrue(processed["has_country"])


    def test_edge_cases_null_nan_empty_whitespace(self):
        import numpy as np
        # 1. None
        self.assertEqual(normalize_name(None), "")
        self.assertEqual(normalize_address(None), "")
        self.assertEqual(normalize_country(None), "UNKNOWN")
        self.assertIsNone(extract_postal_code(None))

        # 2. NaN
        self.assertEqual(normalize_name(float("nan")), "")
        self.assertEqual(normalize_address(np.nan), "")
        self.assertEqual(normalize_country(np.nan), "UNKNOWN")
        self.assertIsNone(extract_postal_code(np.nan))

        # 3. Empty string
        self.assertEqual(normalize_name(""), "")
        self.assertEqual(normalize_address(""), "")
        self.assertEqual(normalize_country(""), "UNKNOWN")
        self.assertIsNone(extract_postal_code(""))

        # 4. Whitespace-only string
        self.assertEqual(normalize_name("   \t\n  "), "")
        self.assertEqual(normalize_address("   \t  "), "")
        self.assertEqual(normalize_country("   "), "UNKNOWN")

        # 5. Punctuation-only string
        self.assertEqual(normalize_name("...---!!!,,,;;;:::"), "")
        self.assertEqual(normalize_address("...---!!!,,,;;;:::"), "")

    def test_edge_cases_abbreviations_and_canonicalization(self):
        # 7. Dotted abbreviations
        self.assertEqual(normalize_name("Acme P.V.T. L.T.D."), "acme private limited")
        self.assertEqual(normalize_name("Societe S.A."), "societe sa")

        # 8 & 9. Pvt Ltd and Private Limited canonicalization
        self.assertEqual(normalize_name("Apex Pvt Ltd"), "apex private limited")
        self.assertEqual(normalize_name("Apex Private Limited"), "apex private limited")
        self.assertEqual(normalize_name("Apex Pvt Ltd"), normalize_name("Apex Private Limited"))

        # 10. Road / Rd canonicalization
        self.assertEqual(normalize_address("123 Main Rd"), "123 main road")
        self.assertEqual(normalize_address("123 Main Road"), "123 main road")
        self.assertEqual(normalize_address("123 Main Rd"), normalize_address("123 Main Road"))

    def test_edge_cases_postal_codes(self):
        # 11. Missing postal code
        self.assertIsNone(extract_postal_code("Main Street, Springfield"))

        # 12. US ZIP and ZIP+4
        self.assertEqual(extract_postal_code("New York, NY 10001"), "10001")
        self.assertEqual(extract_postal_code("Beverly Hills, CA 90210-1234"), "90210-1234")

        # 13. Indian PIN (6 digits)
        self.assertEqual(extract_postal_code("Plot 4, Sector 18, Gurgaon 122015"), "122015")
        self.assertEqual(extract_postal_code("Connaught Place, New Delhi 110001"), "110001")

        # 14. France postal code (5 digits)
        self.assertEqual(extract_postal_code("10 Rue de la Paix, 75001 Paris"), "75001")

    def test_edge_cases_open_set_and_synthetic_dataframe(self):
        import pandas as pd
        import numpy as np
        from src.common.normalize import preprocess_dataframe

        # 14 & 15. France & unseen countries
        self.assertEqual(normalize_country("France"), "FRANCE")
        self.assertEqual(normalize_country("Germany"), "GERMANY")
        self.assertEqual(normalize_country("Japan"), "JAPAN")
        self.assertEqual(normalize_country("Brazil"), "BRAZIL")
        self.assertEqual(normalize_country("UNKNOWN"), "UNKNOWN")
        self.assertEqual(normalize_country(""), "UNKNOWN")
        self.assertEqual(normalize_country(None), "UNKNOWN")

        # Automated smoke test on tiny synthetic dataset with France and unseen country
        synthetic_df = pd.DataFrame({
            "entity_id": ["E1", "E2", "E3", "E4"],
            "business_name": ["Boutique Parisienne", "Berlin Auto GmbH", "Tokyo Tech", None],
            "business_address": [
                "10 Rue de la Paix, 75001 Paris",
                "Alexanderplatz 1, 10178 Berlin",
                "1-1 Chiyoda, Tokyo",
                None,
            ],
            "country": ["France", "Germany", "Japan", None],
        })
        processed = preprocess_dataframe(synthetic_df)

        self.assertEqual(len(processed), 4)
        self.assertEqual(processed["country_norm"].tolist(), ["FRANCE", "GERMANY", "JAPAN", "UNKNOWN"])
        self.assertEqual(processed["has_country"].tolist(), [True, True, True, False])
        self.assertEqual(processed["postal_code"].tolist(), ["75001", "10178", "", ""])
        self.assertEqual(processed["has_postal_code"].tolist(), [True, True, False, False])
        self.assertEqual(processed["has_name"].tolist(), [True, True, True, False])
        self.assertEqual(processed["has_address"].tolist(), [True, True, True, False])

    def test_edge_cases_long_and_numeric_addresses_and_unicode(self):
        # 6. Accented Unicode
        self.assertEqual(normalize_name("Dréxkor Bóral"), "dréxkor bóral")

        # 16. Very long address
        long_addr = "Suite 500, " * 50 + "100 Industrial Parkway"
        norm_long = normalize_address(long_addr)
        self.assertTrue("parkway" in norm_long)
        self.assertTrue(len(norm_long) > 200)

        # 17. Numeric address
        self.assertEqual(normalize_address("123 456 789"), "123 456 789")
        self.assertEqual(normalize_name("123 456"), "123 456")


if __name__ == "__main__":
    unittest.main()

