# Test CASING Table Parsing
import sys
import os
sys.path.append(os.path.dirname(__file__))

from main import parse_extracted_text

def test_casing_table():
    """Test parsing of CASING table with schema mapping"""
    print("=== Testing CASING Table Parsing ===")

    # CASING table with typical headers and proper spacing (simulating PDF layout)
    casing_text = """Hole Size     Depth MD KB     Casing Diameter     Casing Depth MD KB     Type     Test Fit LOT     Test Date     Test Result PPG     Test Depth MD KB
12 1/4"        0                9 5/8"              507.5                Surface  LOT            08.04.2014     11.4               507.5
7 7/8"         507.5            4 1/2"              1250.3               Int      LOT            15.05.2014     12.2               1250.3"""

    result = parse_extracted_text(casing_text)
    print(f"Parsed {len(result)} rows with CASING schema mapping:")
    for i, row in enumerate(result):
        print(f"  Row {i+1}: {row}")

    print()

if __name__ == "__main__":
    test_casing_table()