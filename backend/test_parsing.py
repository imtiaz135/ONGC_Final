# Test Script for Advanced Text Parsing
# This demonstrates the new parse_extracted_text function

import sys
import os
sys.path.append(os.path.dirname(__file__))

from main import parse_extracted_text

def test_tabular_parsing():
    """Test parsing of tabular data (like casing tables)"""
    print("=== Testing Tabular Data Parsing ===")

    # Sample casing table text (typical WCR format)
    tabular_text = """
Hole Size    Depth m    Casing Size    Weight    Grade    Connection
12 1/4"      507.5       9 5/8"        36 ppf    J-55     BTC
7 7/8"       1250.3      4 1/2"        12.6      N-80     LTC
5 7/8"       2000.0      2 7/8"        6.5       P-110    Premium
"""

    result = parse_extracted_text(tabular_text)
    print(f"Parsed {len(result)} rows:")
    for i, row in enumerate(result):
        print(f"  Row {i+1}: {row}")

    print()

def test_key_value_parsing():
    """Test parsing of key-value data"""
    print("=== Testing Key-Value Data Parsing ===")

    kv_text = """
Well Name: XYZ-001
Location: Block 15, Field A
Operator: ABC Oil Company
Spud Date: 2023-01-15
Total Depth: 2500 meters
"""

    result = parse_extracted_text(kv_text)
    print(f"Parsed {len(result)} records:")
    for record in result:
        print(f"  {record}")

    print()

def test_unstructured_parsing():
    """Test parsing of unstructured text"""
    print("=== Testing Unstructured Text Parsing ===")

    unstructured_text = """
This is a paragraph describing the well completion operations.
The casing program was designed to isolate different formations.
Multiple casing strings were run to ensure well integrity.
"""

    result = parse_extracted_text(unstructured_text)
    print(f"Parsed {len(result)} records:")
    for record in result:
        print(f"  {record}")

    print()

def test_mixed_data():
    """Test parsing of mixed tabular and non-tabular data"""
    print("=== Testing Mixed Data Parsing ===")

    mixed_text = """
Well Information:
Name: Test Well #1
Date: 2024-01-01

Casing Records:
Size      Depth    Type
9 5/8"    500      Surface
7"        1500     Intermediate
"""

    result = parse_extracted_text(mixed_text)
    print(f"Parsed {len(result)} records:")
    for record in result:
        print(f"  {record}")

if __name__ == "__main__":
    test_tabular_parsing()
    test_key_value_parsing()
    test_unstructured_parsing()
    test_mixed_data()