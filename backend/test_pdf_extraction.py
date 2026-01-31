#!/usr/bin/env python3
"""
Test extraction on actual PDF files in uploads directory.
"""

import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

import pdfplumber
from main import parse_extracted_text, parse_text_manually

def test_native_extraction(pdf_path):
    """Test pdfplumber native table extraction"""
    print("\n[1] NATIVE TABLE EXTRACTION:")
    
    try:
            page = pdf.pages[0]
            tables = page.extract_tables()
            
            if tables:
                print(f"    ✓ Found {len(tables)} table(s)")
                for i, table in enumerate(tables):
                    if table:
                        print(f"      Table {i}: {len(table)} rows x {len(table[0])} cols")
                        for j, row in enumerate(table[:2]):
                            cols_preview = ", ".join(str(cell)[:20] for cell in row[:3])
                            print(f"        Row {j}: {cols_preview}...")
                return True
            else:
                print(f"    ✗ No native tables")
                return False
                
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False

def test_text_extraction(pdf_path):
    """Test text extraction from PDF"""
    print("\n[2] TEXT EXTRACTION:")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[0]
            text = page.extract_text()
            
            if text:
                lines = text.split('\n')
                print(f"    ✓ Extracted {len(text)} chars ({len(lines)} lines)")
                print(f"      Sample:")
                for line in lines[:3]:
                    if line.strip():
                        preview = line[:55] + "..." if len(line) > 55 else line
                        print(f"        {preview}")
                return text
            else:
                print(f"    ✗ No text extracted")
                return None
                
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None

def test_ocr_extraction(pdf_path):
    """Test OCR extraction"""
    print("\n[3] ADVANCED PARSING (Layout-based):")
    
    # This would require OCR - for now skip
    return None

def test_parsing(text, label="TEST"):
    """Test advanced parsing (layout-based)"""
    print("\n[4] PARSING RESULTS:")
    
    if not text:
        print("    ✗ No text to parse")
        return None
    
    try:
        result = parse_extracted_text(text)
        
        if result:
            print(f"    ✓ {len(result)} records parsed")
            
            if isinstance(result, list) and len(result) > 0:
                first = result[0]
                if isinstance(first, dict):
                    fields = list(first.keys())
                    print(f"      Fields: {', '.join(fields[:5])}")
                    if len(fields) > 5:
                        print(f"              +{len(fields)-5} more")
            return result
        else:
            print(f"    ✗ No records")
            return None
            
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None

def main():
    """Run extraction tests on all PDFs"""
    print("\n" + "="*70)
    print("PDF EXTRACTION TEST SUITE")
    print("="*70)
    
    uploads_dir = Path(__file__).parent / "uploads"
    pdf_files = sorted(uploads_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("\n✗ No PDF files found in uploads directory")
        return 1
    
    print(f"\nFound {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"  • {pdf.name}")
    
    # Test each PDF
    for pdf_path in pdf_files:
        pdf_name = Path(pdf_path).name
        print("\n" + "-"*70)
        print(f"TESTING: {pdf_name}")
        print("-"*70)
        
        # 1. Native extraction
        test_native_extraction(str(pdf_path))
        
        # 2. Text extraction
        text = test_text_extraction(str(pdf_path))
        
        # 3. Advanced parsing
        if text:
            test_parsing(text, f"{pdf_name}")
    
    print("\n" + "="*70)
    print("TESTS COMPLETE")
    print("="*70 + "\n")
    return 0

if __name__ == "__main__":
    exit(main())
