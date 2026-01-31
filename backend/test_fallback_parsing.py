#!/usr/bin/env python3
"""
Test script to verify extraction works without Ollama.
This ensures the fallback to layout-based parsing is robust.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from main import parse_extracted_text, parse_casing_table_specialized

def test_casing_table_fallback():
    """Test that CASING table extraction works without Ollama"""
    print("=" * 70)
    print("TEST: CASING Table Extraction (Without Ollama)")
    print("=" * 70)

    # Simulated OCR/PDF text extraction result
    casing_text = """
Hole Size     Depth MD KB     Casing Diameter     Casing Depth MD KB     Type     Test Fit LOT     Test Date     Test Result PPG     Test Depth MD KB
12 1/4"        0                9 5/8"              507.5                Surface  LOT            08.04.2014     11.4               507.5
7 7/8"         507.5            4 1/2"              1250.3               Int      LOT            15.05.2014     12.2               1250.3
5 7/8"         2000.0           2 7/8"              3500.0               Prod     LOT            20.06.2014     13.1               3500.0
""".strip()

    result = parse_extracted_text(casing_text)
    
    print(f"\nInput: {len(casing_text)} characters of text")
    print(f"Output: {len(result)} rows extracted")
    
    if result:
        print(f"\n[OK] Data successfully extracted without Ollama!")
        for i, row in enumerate(result, 1):
            print(f"\n  Row {i}:")
            for key, value in row.items():
                print(f"    {key}: {value}")
        
        # Validate schema
        required_fields = ['hole_size', 'depth_md_kb', 'casing_diameter']
        for i, row in enumerate(result, 1):
            for field in required_fields:
                if field not in row:
                    print(f"⚠ Warning: Row {i} missing field '{field}'")
                    return False
        
        print("\n✓ All required CASING schema fields present!")
        return True
    else:
        print("\n✗ Failed to extract data")
        return False

def test_wellhead_data():
    """Test extraction of well header/metadata"""
    print("\n" + "=" * 70)
    print("TEST: Well Header Data (Key-Value Extraction)")
    print("=" * 70)

    wellhead_text = """
Well Name: XYZ-001
Operator: ABC Oil Company
Location: Block 15, Offshore Field A
Spud Date: 2023-01-15
Total Depth: 2500 m
Water Depth: 85 m
""".strip()

    result = parse_extracted_text(wellhead_text)
    
    print(f"\nInput: {len(wellhead_text)} characters")
    print(f"Output: {len(result)} records extracted")
    
    if result and len(result) > 0:
        print("\n✓ Key-value data extracted without Ollama!")
        for key, value in result[0].items():
            print(f"    {key}: {value}")
        return True
    else:
        print("\n✗ Failed to extract key-value data")
        return False

def test_simple_table():
    """Test extraction of simple generic table"""
    print("\n" + "=" * 70)
    print("TEST: Simple Table (Generic Structure)")
    print("=" * 70)

    simple_table = """
Tool Type       Depth m       Manufacturer       Serial Number
Drill Collar    500           Smith Services     ABC-1001
Heavy Weight    750           Jones Drilling     XYZ-2002
Stabilizer      1200          Jones Drilling     XYZ-2003
""".strip()

    result = parse_extracted_text(simple_table)
    
    print(f"\nInput: {len(simple_table)} characters")
    print(f"Output: {len(result)} rows extracted")
    
    if result:
        print("\n✓ Generic table extracted without Ollama!")
        for i, row in enumerate(result, 1):
            print(f"\n  Row {i}: {row}")
        return True
    else:
        print("\n✗ Failed to extract table")
        return False

def test_unstructured_text():
    """Test that unstructured text returns sensible output"""
    print("\n" + "=" * 70)
    print("TEST: Unstructured Text Handling")
    print("=" * 70)

    unstructured = """
The well was completed with a successful casing shoe cement job.
Multiple formation tops were identified through the drilling process.
The formation testing confirmed the presence of hydrocarbons.
""".strip()

    result = parse_extracted_text(unstructured)
    
    print(f"\nInput: {len(unstructured)} characters of unstructured text")
    print(f"Output: {len(result)} records")
    
    if result:
        print("\n✓ Unstructured text handled gracefully!")
        if 'text' in result[0]:
            text_snippet = result[0]['text'][:100] + "..."
            print(f"    Content: {text_snippet}")
        return True
    else:
        print("\n✗ Failed to handle unstructured text")
        return False

def main():
    """Run all tests"""
    print("\n")
    print("=" * 70)
    print("PDF DATA EXTRACTION - FALLBACK PARSING TESTS (NO OLLAMA)".center(70))
    print("=" * 70)
    print()

    tests = [
        ("CASING Table", test_casing_table_fallback),
        ("Well Header", test_wellhead_data),
        ("Simple Table", test_simple_table),
        ("Unstructured", test_unstructured_text)
    ]

    results = []
    for name, test_func in tests:
        try:
            results.append((name, test_func()))
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Extraction system is working without Ollama.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    exit(main())