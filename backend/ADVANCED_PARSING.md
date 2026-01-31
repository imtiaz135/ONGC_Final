# Advanced Text Parsing for PDF Data Extraction

## Overview

The `parse_extracted_text()` function provides intelligent parsing of text extracted from PDF regions. It automatically detects whether the text represents a structured table or unstructured content, then parses accordingly.

## Function Signature

```python
def parse_extracted_text(text: str) -> List[Dict[str, str]]:
    """
    Advanced text parsing for PDF extraction regions.

    Detects whether text is tabular or non-tabular and structures accordingly.

    For tabular data: Preserves table structure with headers and rows.
    For non-tabular: Converts to key-value pairs or inferred columns.

    Returns: List of dictionaries representing rows of structured data.
    """
```

## Detection Logic

### Tabular Detection Heuristics
- **Column Consistency**: >70% of rows must have the same number of columns
- **Multiple Columns**: At least 2 columns per row
- **Sufficient Rows**: At least 2 rows with multiple columns
- **Spacing Patterns**: Uses 2+ spaces as column delimiters

### Header Detection
- First row analyzed for text vs numeric content
- >50% text cells → treated as headers
- Headers normalized: lowercase, spaces → underscores, special chars removed

## Parsing Modes

### 1. Tabular Mode (Tables, Casing Records, BHA Tables)
**Input Example:**
```
Hole Size    Depth m    Casing Size    Weight    Grade
12 1/4"      507.5       9 5/8"        36 ppf    J-55
7 7/8"       1250.3      4 1/2"        12.6      N-80
```

**Output:**
```python
[
    {
        'hole_size': '12 1/4"',
        'depth_m': '507.5',
        'casing_size': '9 5/8"',
        'weight': '36 ppf',
        'grade': 'J-55'
    },
    {
        'hole_size': '7 7/8"',
        'depth_m': '1250.3',
        'casing_size': '4 1/2"',
        'weight': '12.6',
        'grade': 'N-80'
    }
]
```

### 2. Key-Value Mode (Well Headers, Metadata)
**Input Example:**
```
Well Name: XYZ-001
Location: Block 15, Field A
Operator: ABC Oil Company
Total Depth: 2500 meters
```

**Output:**
```python
[
    {
        'well_name': 'XYZ-001',
        'location': 'Block 15, Field A',
        'operator': 'ABC Oil Company',
        'total_depth': '2500 meters'
    }
]
```

### 3. Unstructured Mode (Paragraphs, Notes)
**Input Example:**
```
This is a paragraph describing well completion operations.
Multiple casing strings were run to ensure integrity.
```

**Output:**
```python
[
    {
        'text': 'This is a paragraph describing well completion operations. Multiple casing strings were run to ensure integrity.'
    }
]
```

## Integration with Extraction Pipeline

The function is integrated into the PDF extraction system as follows:

1. **Primary**: pdfplumber table extraction (preserves native PDF tables)
2. **Fallback 1**: OCR extraction (for scanned PDFs)
3. **Fallback 2**: `parse_extracted_text()` (intelligent text parsing)
4. **Fallback 3**: Raw text return (with warning)

## Usage in WCR Context

### Casing Tables
- Detects column structure automatically
- Preserves hole sizes, depths, casing specs
- Handles imperial measurements (12 1/4", 36 ppf)

### Well Headers
- Extracts metadata as key-value pairs
- Normalizes field names for database compatibility

### BHA Records
- Parses drill string components
- Maintains tool specifications and measurements

### Logs Records
- Handles formation data tables
- Preserves depth and logging measurements

## Quality Assurance

### Validation Checks
- **Column Consistency**: Rejects tables with <70% consistent column counts
- **Minimum Structure**: Requires at least 2 columns and 2 rows for tables
- **Header Validation**: Ensures headers are text-based, not numeric
- **Cell Filtering**: Skips empty or malformed cells

### Debug Output
```
DEBUG: Detected tabular structure - reconstructing table
DEBUG: Parsed table with 6 columns and 3 rows
```

## Error Handling

### Graceful Degradation
- If tabular detection fails → tries key-value parsing
- If key-value fails → returns as text block
- Always returns structured data, never fails completely

### Warning System
- `_warning` field added for fallback extractions
- Clear messages about parsing method used
- Raw text preserved when structure cannot be determined

## Performance Characteristics

- **Fast**: Regex-based parsing, no ML models
- **Memory Efficient**: Processes text in-memory
- **Defensive**: Handles malformed input gracefully
- **Extensible**: Easy to add new detection heuristics

## Testing

Run the test script to see examples:
```bash
python test_parsing.py
```

This demonstrates parsing of:
- Casing tables (tabular)
- Well metadata (key-value)
- Descriptive text (unstructured)
- Mixed content

## Future Enhancements

- **Auto Header Detection**: Search backwards from selection for missing headers
- **Data Type Inference**: Detect numeric vs text columns
- **Multi-Table Support**: Handle multiple tables in one region
- **OCR Confidence Scoring**: Weight parsing by OCR accuracy
- **Template Matching**: Learn from successful extractions