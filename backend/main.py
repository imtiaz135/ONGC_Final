import os
import shutil
import json
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pdfplumber
import pandas as pd
from sqlalchemy import text
from pydantic import BaseModel
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from PIL import Image
import pytesseract
from pytesseract import Output
import re
import requests
from dotenv import load_dotenv

load_dotenv()

from database import init_db, get_table_schema, engine
from sqlalchemy import inspect
def seed_mock_wellhead_data():
    """
    Seeds mock WELL SUMMARY data for existence check.
    Does NOT modify any extraction or endpoint logic.
    Safe to run multiple times (idempotent).
    """
    try:
        with engine.begin() as conn:
            # Check if mock record already exists
            result = conn.execute(
                text("""
                    SELECT COUNT(*) 
                    FROM wcr_wellhead 
                    WHERE WELL_NAME = :well_name
                """),
                {"well_name": "OzAlpha-1"}
            ).scalar()

            if result and result > 0:
                print("[MOCK DB] OzAlpha-1 already exists — skipping insert")
                return

            # Insert mock WELL SUMMARY record
            conn.execute(
                text("""
                    INSERT INTO wcr_wellhead (
                        UWI,
                        WELL_NAME,
                        FIELD,
                        RELEASE_NAME,
                        LOCATION_TYPE,
                        SURFACE_LONG,
                        SURFACE_LAT,
                        CATEGORY,
                        WELL_PROFILE,
                        DRILLED_DEPTH,
                        K_B,
                        G_L,
                        RIG,
                        SPUD_DATE,
                        DRILLING_COMPLETED_DATE,
                        RIG_RELEASED_DATE,
                        FORMATION_AT_TD,
                        OBJECTIVE,
                        STATUS,
                        PAGE_NUMBERS
                    ) VALUES (
                        :uwi,
                        :well_name,
                        :field,
                        :release_name,
                        :location_type,
                        :surface_long,
                        :surface_lat,
                        :category,
                        :well_profile,
                        :drilled_depth,
                        :k_b,
                        :g_l,
                        :rig,
                        :spud_date,
                        :drilling_completed_date,
                        :rig_released_date,
                        :formation_at_td,
                        :objective,
                        :status,
                        :page_numbers
                    )
                """),
                {
                    "uwi": "MOCK-OZALPHA-1",
                    "well_name": "OzAlpha-1",
                    "field": "Southern Georgina Basin",
                    "release_name": "EP 104",
                    "location_type": "Northern Territory",
                    "surface_long": 137.611636,
                    "surface_lat": -22.420414,
                    "category": "Exploration",
                    "well_profile": "Vertical",
                    "drilled_depth": 1250.3,
                    "k_b": 193.2,
                    "g_l": 189,
                    "rig": "EDA Rig #2",
                    "spud_date": "01/04/2014",
                    "drilling_completed_date": "19/04/2014",
                    "rig_released_date": "22/04/2014",
                    "formation_at_td": "Arthur Creek Formation, Thorntonia Limestone",
                    "objective": "Arthur Creek Formation Hot Shale and Thorntonia Limestone",
                    "status": "Completed",
                    "page_numbers": "Well Summary"
                }
            )

            print("[MOCK DB] Seeded WELL SUMMARY for OzAlpha-1")

    except Exception as e:
        print(f"[MOCK DB ERROR] {e}")


def seed_mock_casing_data():
    """
    Seeds mock CASING data so /check-existence can detect MATCHED.
    Safe to run multiple times (no duplicates).
    """
    try:
        with engine.begin() as conn:
            # Check if casing data already exists for this UWI
            result = conn.execute(
                text("""
                    SELECT COUNT(*)
                    FROM wcr_casing
                    WHERE UWI = :uwi
                """),
                {"uwi": "MOCK-OZALPHA-1"}
            ).scalar()

            if result and result > 0:
                print("[MOCK DB] CASING already exists — skipping")
                return

            # -------- Row 1: Conductor --------
            conn.execute(
                text("""
                    INSERT INTO wcr_casing (
                        UWI, CASING_TYPE, CASING_TOP, CASING_BOTTOM,
                        OUTER_DIAMETER, REMARKS, PAGE_NUMBERS
                    ) VALUES (
                        :uwi, :casing_type, :top, :bottom,
                        :od, :remarks, :page
                    )
                """),
                {
                    "uwi": "MOCK-OZALPHA-1",
                    "casing_type": "Conductor",
                    "top": 7.7,
                    "bottom": 6.7,
                    "od": 14,
                    "remarks": "Hole size 17 1/2\"",
                    "page": "Casing"
                }
            )

            # -------- Row 2: Surface casing --------
            conn.execute(
                text("""
                    INSERT INTO wcr_casing (
                        UWI, CASING_TYPE, CASING_TOP, CASING_BOTTOM,
                        OUTER_DIAMETER, WEIGHT, STEEL_GRADE,
                        REMARKS, PAGE_NUMBERS
                    ) VALUES (
                        :uwi, :casing_type, :top, :bottom,
                        :od, :weight, :grade,
                        :remarks, :page
                    )
                """),
                {
                    "uwi": "MOCK-OZALPHA-1",
                    "casing_type": "Surface casing",
                    "top": 507.5,
                    "bottom": 505,
                    "od": 9.625,
                    "weight": "36 ppf",
                    "grade": "J-55",
                    "remarks": "BTC; LOT test 11.4 ppg at 511 m on 08.04.2014",
                    "page": "Casing"
                }
            )

            # -------- Row 3: Production casing --------
            conn.execute(
                text("""
                    INSERT INTO wcr_casing (
                        UWI, CASING_TYPE, CASING_TOP, CASING_BOTTOM,
                        OUTER_DIAMETER, WEIGHT, STEEL_GRADE,
                        REMARKS, PAGE_NUMBERS
                    ) VALUES (
                        :uwi, :casing_type, :top, :bottom,
                        :od, :weight, :grade,
                        :remarks, :page
                    )
                """),
                {
                    "uwi": "MOCK-OZALPHA-1",
                    "casing_type": "Production casing",
                    "top": 1250.3,
                    "bottom": 1240,
                    "od": 4.5,
                    "weight": "13.5 ppf",
                    "grade": "L-80",
                    "remarks": "Tenaris Blue; Hole size 7 7/8\"",
                    "page": "Casing"
                }
            )

            print("[MOCK DB] Seeded CASING data")

    except Exception as e:
        print(f"[MOCK DB ERROR - CASING] {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("DEBUG: Server starting up...")
    try:
        init_db()
        print("[OK] Database initialized")

        # ✅ Seed mock WELLHEAD data (safe, idempotent)
        seed_mock_wellhead_data()
        
        # ✅ Seed mock CASING data (safe, idempotent)
        seed_mock_casing_data()

    except Exception as e:
        print(f"[ERROR] Database init failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Check Ollama availability
    check_ollama_availability()
    
    yield  # Server is running
    
    # Shutdown
    print("DEBUG: Server shutting down...")


app = FastAPI(title="Well Completion Extractor", lifespan=lifespan)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Backend is running successfully. Use /docs for API documentation."}

# --- MODELS ---
class RegionSelection(BaseModel):
    page_number: int
    # Percentage based (Legacy/Default)
    x_pct: Optional[float] = 0.0
    y_pct: Optional[float] = 0.0
    w_pct: Optional[float] = 0.0
    h_pct: Optional[float] = 0.0
    # Pixel based (Snip Tool)
    x: Optional[float] = 0.0
    y: Optional[float] = 0.0
    width: Optional[float] = 0.0
    height: Optional[float] = 0.0
    view_width: Optional[float] = 0.0
    view_height: Optional[float] = 0.0
    label: str    # e.g., "CASING"
    use_ai: bool = False

# --- CONFIGURATION ---
LABEL_TO_TABLE = {
    "WELL_HEADER": "wcr_wellhead",
    "WCR_WELLHEAD": "wcr_wellhead",
    "CASING": "wcr_casing",
    "WCR_CASING": "wcr_casing",
    "LOGS": "wcr_logsrecord",
    "WCR_LOGSRECORD": "wcr_logsrecord",
    "DIRSRVY": "wcr_dirsrvy",
    "WCR_DIRSRVY": "wcr_dirsrvy",
    "SWC": "wcr_swc",
    "WCR_SWC": "wcr_swc",
    "HCSHOWS": "wcr_hcshows",
    "WCR_HCSHOWS": "wcr_hcshows"
}

# --- LOGIC ---

# Configure pytesseract (handle common Windows/Linux/macOS installation paths)
TESSERACT_AVAILABLE = False

def setup_pytesseract():
    """Set up pytesseract path for different OS environments."""
    global TESSERACT_AVAILABLE
    if os.name == 'nt':  # Windows
        tesseract_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\Public\Tesseract-OCR\tesseract.exe"
        ]
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                TESSERACT_AVAILABLE = True
                print(f"[OK] Tesseract found at: {path}")
                return
        print("[WARNING] Tesseract not found in common Windows paths")
    else:
        # Linux/macOS - check if tesseract is in PATH
        try:
            pytesseract.get_tesseract_version()
            TESSERACT_AVAILABLE = True
            print("[OK] Tesseract found in PATH")
        except pytesseract.TesseractNotFoundError:
            TESSERACT_AVAILABLE = False
            print("[WARNING] Tesseract not found in PATH")

setup_pytesseract()

# --- OLLAMA LLM CONFIGURATION ---
OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2-vision"  # Try this first
OLLAMA_FALLBACK_MODELS = ["llama2", "mistral", "neural-chat"]  # Fallback options
OLLAMA_AVAILABLE = False
OLLAMA_USES_VISION = False  # Track if model supports vision

def check_ollama_availability():
    """Check if Ollama API is available on startup and find available model"""
    global OLLAMA_AVAILABLE, OLLAMA_MODEL, OLLAMA_USES_VISION
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            print(f"[OK] Ollama API available with models: {', '.join(model_names[:3])}")
            
            # Try to find the best available model
            available_models = []
            
            # Check for vision models first
            for model in model_names:
                if "llama3.2-vision" in model:
                    OLLAMA_MODEL = model.split(":")[0]
                    OLLAMA_USES_VISION = True
                    OLLAMA_AVAILABLE = True
                    print(f"[OK] Using vision model: {OLLAMA_MODEL}")
                    return True
            
            # Check for standard models
            if not OLLAMA_AVAILABLE:
                for fallback in OLLAMA_FALLBACK_MODELS:
                    for model in model_names:
                        if fallback in model:
                            OLLAMA_MODEL = model.split(":")[0]
                            OLLAMA_AVAILABLE = True
                            print(f"[OK] Using model: {OLLAMA_MODEL}")
                            return True
            
            # If we found any model
            if model_names:
                OLLAMA_MODEL = model_names[0].split(":")[0]
                OLLAMA_AVAILABLE = True
                print(f"[OK] Using available model: {OLLAMA_MODEL}")
                return True
                
            print("[WARNING] No models found in Ollama")
            OLLAMA_AVAILABLE = False
            return False
            
    except requests.exceptions.ConnectionError:
        OLLAMA_AVAILABLE = False
        print("[WARNING] Ollama API not available at http://localhost:11434")
        print("[WARNING] To use Ollama, run: ollama serve (in another terminal)")
        return False
    except Exception as e:
        OLLAMA_AVAILABLE = False
        print(f"[WARNING] Error checking Ollama: {e}")
        return False

def parse_with_ollama(text: str, label: str) -> List[Dict]:
    """
    Uses Ollama local API to parse unstructured text into structured JSON.
    Gracefully falls back if Ollama has issues (which happens frequently with 500 errors).
    """
    if not OLLAMA_AVAILABLE:
        print("DEBUG: Ollama not available, skipping AI parsing")
        return []
    
    # For now, disable Ollama due to frequent 500 errors
    # The layout-based parsing works better anyway
    print("DEBUG: Ollama parsing disabled (frequent 500 errors). Using layout-based parsing instead.")
    return []


def parse_extracted_text(text: str) -> List[Dict[str, str]]:
    """
    Advanced text parsing for PDF extraction regions.

    Detects whether text is tabular or non-tabular and structures accordingly.

    For tabular data: Preserves table structure with headers and rows.
    For non-tabular: Converts to key-value pairs or inferred columns.

    Returns: List of dictionaries representing rows of structured data.
    """
    if not text or not text.strip():
        return []

    lines = [line for line in text.split('\n') if line.strip()]
    if not lines:
        return []

    # Step 1: Detect if text is tabular using layout-based heuristics
    is_tabular = detect_tabular_layout(lines)

    if is_tabular:
        print("DEBUG: Detected tabular structure - reconstructing table layout")
        return reconstruct_table_layout(lines)
    else:
        print("DEBUG: Detected non-tabular structure - inferring structure")
        return parse_as_non_tabular(lines)

def detect_tabular_layout(lines: List[str]) -> bool:
    """
    Advanced heuristics to detect tabular structure using layout analysis:
    - Consistent column positions across rows
    - Vertical alignment patterns
    - Presence of header-like rows
    - Minimum rows and columns
    """
    if len(lines) < 3:  # Need at least header + 2 data rows
        return False

    # Analyze spacing patterns to find column boundaries
    column_positions = analyze_column_positions(lines)

    if len(column_positions) < 2:  # Need at least 2 columns
        return False

    # Check if most lines can be parsed with these column positions
    parseable_lines = 0
    for line in lines:
        if can_parse_with_columns(line, column_positions):
            parseable_lines += 1

    # At least 70% of lines should be parseable as table rows
    layout_consistency = parseable_lines / len(lines)

    print(f"DEBUG: Layout analysis - {len(column_positions)} columns detected, {layout_consistency:.1%} lines parseable")

    return layout_consistency >= 0.7 and len(column_positions) >= 2

def analyze_column_positions(lines: List[str]) -> List[int]:
    """
    Analyze text lines to identify consistent column start positions.
    Uses space patterns to find column boundaries.
    """
    if not lines:
        return []

    # Use the first line (headers) as reference for column detection
    header_line = lines[0]

    # Find positions where content starts after spaces
    column_positions = [0]  # First column always starts at 0

    i = 0
    while i < len(header_line):
        if header_line[i] == ' ':
            # Count consecutive spaces
            space_start = i
            while i < len(header_line) and header_line[i] == ' ':
                i += 1
            space_count = i - space_start

            # If we have 2+ spaces, this might be a column separator
            if space_count >= 2:
                # Check if the next non-space character exists
                next_content_start = i
                while next_content_start < len(header_line) and header_line[next_content_start] == ' ':
                    next_content_start += 1

                if next_content_start < len(header_line):
                    column_positions.append(next_content_start)
        else:
            i += 1

    # Validate columns work for most data lines
    valid_lines = 0
    for line in lines[1:]:  # Skip header
        if len(line) >= column_positions[-1]:
            valid_lines += 1

    if valid_lines >= len(lines) * 0.5:  # At least 50% of lines work
        return column_positions

    # Fallback: simpler approach - split by consistent space patterns
    return [0]  # Just return first column if detection fails

def can_parse_with_columns(line: str, column_positions: List[int]) -> bool:
    """
    Check if a line can be parsed using the given column positions.
    """
    if len(line) < max(column_positions):
        return False

    # Check if there are non-space characters near each column position
    valid_columns = 0
    for col_pos in column_positions:
        if col_pos < len(line):
            # Look for non-space within 5 chars of column position
            start = max(0, col_pos - 2)
            end = min(len(line), col_pos + 3)
            if any(c != ' ' for c in line[start:end]):
                valid_columns += 1

    # Need at least 50% of columns to have content
    return valid_columns >= len(column_positions) * 0.5

def reconstruct_table_layout(lines: List[str]) -> List[Dict[str, str]]:
    """
    Reconstruct table using layout analysis and column position detection.
    SKIPS HEADER ROW and extracts only data rows with proper mapping.
    """
    column_positions = analyze_column_positions(lines)

    if len(column_positions) < 2:
        return []

    # Parse each line into cells using column positions
    parsed_rows = []
    for line in lines:
        cells = extract_cells_from_line(line, column_positions)
        if cells and len(cells) >= 2:  # Only include rows with multiple cells
            parsed_rows.append(cells)

    if len(parsed_rows) < 2:
        return []

    # 🎯 STEP 1: Identify and skip HEADER ROW
    headers = None
    data_start_idx = 1  # Skip first row (assumed to be headers)

    if len(parsed_rows) > 1:
        first_row = parsed_rows[0]
        # Validate that first row is actually headers (more text than numbers)
        text_cells = sum(1 for cell in first_row if cell and not any(c.isdigit() for c in cell))
        is_text_row = text_cells / len(first_row) > 0.6  # >60% text cells
        
        if is_text_row:
            # Extract headers and normalize them for schema mapping
            headers = [cell.lower().replace(' ', '_').replace('.', '').replace('-', '_').strip('_')
                      for cell in first_row if cell]
        else:
            # First row is data, not headers - create generic headers
            print("DEBUG: WARNING - First row appears to be data, not headers!")
            headers = [f'col_{i}' for i in range(len(column_positions))]
            data_start_idx = 0  # Include all rows as data

    # Ensure we have the right number of headers
    if not headers:
        headers = [f'col_{i}' for i in range(len(column_positions))]
    
    while len(headers) < len(column_positions):
        headers.append(f'col_{len(headers)}')
    headers = headers[:len(column_positions)]

    # 📊 STEP 2: EXTRACT ONLY DATA ROWS (skip header)
    structured_data = []
    data_rows_skipped = 0
    
    for row_idx, row in enumerate(parsed_rows[data_start_idx:], start=data_start_idx):
        if len(row) != len(headers):
            continue
        
        row_dict = {}
        non_empty_count = 0
        
        for i, cell in enumerate(row):
            if cell and cell.strip():
                row_dict[headers[i]] = cell.strip()
                non_empty_count += 1
        
        # Only include rows with meaningful data (not empty rows)
        if non_empty_count >= len(headers) * 0.3:  # At least 30% filled
            structured_data.append(row_dict)
        else:
            data_rows_skipped += 1

    # Special handling for CASING tables - apply specialized parser
    if headers and len(headers) >= 3:
        header_text = ' '.join(str(h).lower() for h in headers)
        if any(keyword in header_text for keyword in ['hole', 'casing', 'depth', 'size', 'diameter']):
            print("DEBUG: Detected CASING table - using specialized parsing")
            # Use the original lines but parser will skip first row
            structured_data = parse_casing_table_specialized(lines)

    print(f"DEBUG: Reconstructed {len(structured_data)} data rows (header and {data_rows_skipped} empty rows skipped)")
    print(f"DEBUG: Headers: {headers}")

    return structured_data

def map_to_casing_schema(detected_headers: List[str], data_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Map detected table headers to the expected CASING table schema.
    Schema: ["hole_size", "depth_md_kb", "casing_diameter", "casing_depth_md_kb", "type", "test_fit_lot", "test_date", "test_result_ppg", "test_depth_md_kb"]
    """
    expected_schema = [
        "hole_size", "depth_md_kb", "casing_diameter", "casing_depth_md_kb",
        "type", "test_fit_lot", "test_date", "test_result_ppg", "test_depth_md_kb"
    ]

    # Create mapping from detected headers to schema
    header_mapping = {}

    # Normalize detected headers for matching
    normalized_detected = [h.lower().replace('_', ' ').replace('-', ' ') for h in detected_headers]

    for i, detected in enumerate(normalized_detected):
        best_match = None
        best_score = 0

        for schema_field in expected_schema:
            if schema_field in header_mapping.values():
                continue  # Already mapped

            # Calculate match score
            schema_words = set(schema_field.lower().replace('_', ' ').split())
            detected_words = set(detected.split())

            if schema_words & detected_words:  # Common words
                score = len(schema_words & detected_words) / len(schema_words | detected_words)
                if score > best_score:
                    best_score = score
                    best_match = schema_field

        if best_match and best_score > 0.3:  # Minimum match threshold
            header_mapping[detected_headers[i]] = best_match

    # Apply mapping to data rows
    mapped_data = []
    for row in data_rows:
        mapped_row = {}
        for orig_key, value in row.items():
            schema_key = header_mapping.get(orig_key, orig_key)
            mapped_row[schema_key] = value
        mapped_data.append(mapped_row)

    print(f"DEBUG: Mapped CASING table headers: {header_mapping}")
    return mapped_data

def extract_cells_from_line(line: str, column_positions: List[int]) -> List[str]:
    """
    Extract cell content from a line using column positions.
    Uses a simpler, more reliable approach.
    """
    cells = []
    line_len = len(line)

    for i, col_start in enumerate(column_positions):
        if col_start >= line_len:
            cells.append("")
            continue

        # Find where this cell ends
        if i < len(column_positions) - 1:
            col_end = column_positions[i + 1]
        else:
            col_end = line_len

        # Extract content from start to end, strip whitespace
        cell_content = line[col_start:col_end].strip()
        cells.append(cell_content)

    return cells

def parse_casing_table_specialized(lines: List[str]) -> List[Dict[str, str]]:
    """
    Specialized parser for CASING tables using direct pattern recognition.
    Handles the specific schema and common formatting issues.
    """
    if len(lines) < 2:
        return []

    parsed_rows = []

    # Process data lines (skip header)
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        # Use regex to split on multiple spaces while preserving quoted strings
        # This handles cases like "12 1/4"" properly
        parts = re.split(r'\s{2,}', line)

        # Clean parts
        parts = [part.strip() for part in parts if part.strip()]

        if len(parts) < 4:  # Need at least basic fields
            continue

        # Create row with expected schema
        row_data = {
            "hole_size": parts[0] if len(parts) > 0 else "",
            "depth_md_kb": parts[1] if len(parts) > 1 else "",
            "casing_diameter": parts[2] if len(parts) > 2 else "",
            "casing_depth_md_kb": parts[3] if len(parts) > 3 else "",
            "type": parts[4] if len(parts) > 4 else "",
            "test_fit_lot": parts[5] if len(parts) > 5 else "",
            "test_date": parts[6] if len(parts) > 6 else "",
            "test_result_ppg": parts[7] if len(parts) > 7 else "",
            "test_depth_md_kb": parts[8] if len(parts) > 8 else ""
        }

        # Clean up empty values
        row_data = {k: v for k, v in row_data.items() if v}

        if row_data:
            parsed_rows.append(row_data)

    print(f"DEBUG: Specialized CASING parser extracted {len(parsed_rows)} rows")
    return parsed_rows

def parse_as_non_tabular(lines: List[str]) -> List[Dict[str, str]]:
    """
    Parse non-tabular text into structured format.
    - First tries key-value pairs
    - Falls back to single text field
    """
    # Try key-value extraction
    kv_pairs = {}
    for line in lines:
        if ':' in line and len(line) < 150:
            parts = line.split(':', 1)
            key = parts[0].strip()
            value = parts[1].strip()
            if key and value and len(key) < 50:
                # Clean key
                clean_key = re.sub(r'[^a-zA-Z0-9\s]', '', key).lower().replace(' ', '_')
                kv_pairs[clean_key] = value
    
    if kv_pairs:
        print(f"DEBUG: Extracted {len(kv_pairs)} key-value pairs")
        return [kv_pairs]
    
    # Fallback: treat as single text block
    combined_text = ' '.join(lines)
    print("DEBUG: No structure found - returning as text block")
    return [{'text': combined_text}]

def parse_text_manually(text: str) -> List[Dict]:
    """
    Manually parse text into key-value pairs or table rows based on spacing.
    Used when AI extraction is disabled.
    
    CRITICAL: Do NOT extract random text as table rows. Only use structured formats.
    """
    data = []
    lines = [line for line in text.split('\n') if line.strip()]
    
    if not lines:
        return []
    
    # Pattern 1: Key-Value pairs (e.g., "Field: Value")
    kv_data = {}
    for line in lines:
        if ':' in line and len(line) < 150:
            parts = line.split(':', 1)
            k = parts[0].strip()
            v = parts[1].strip()
            # Strict validation: key < 50 chars, value > 0, looks like actual key-value
            if k and v and len(k) < 50 and len(k) > 2 and not k[0].isdigit():
                # Check it's not just random text with a colon
                if re.match(r'^[A-Za-z0-9\s\.\-_()]+$', k):
                    kv_data[k.lower().replace(" ", "_").replace(".", "").replace("-", "_")] = v
    
    if kv_data:
        data.append(kv_data)
        return data  # Return early if we found k-v pairs
    
    # Pattern 2: Table rows (ONLY if structured with consistent column count)
    table_rows = []
    
    # Find lines with multiple whitespace-separated fields
    candidate_rows = []
    for line in lines:
        if len(line.strip()) < 10:  # Skip very short lines
            continue
        # Split by multiple spaces or tabs
        cells = [cell.strip() for cell in re.split(r'\s{2,}|\t', line) if cell.strip()]
        # Only accept lines with 2-10 columns (reasonable table size)
        if 2 <= len(cells) <= 10:
            candidate_rows.append((line, cells))
    
    if len(candidate_rows) < 2:
        # Not enough data for a table
        return []
    
    # Check column consistency: all rows should have same number of columns
    col_counts = [len(cells) for _, cells in candidate_rows]
    most_common_col_count = max(set(col_counts), key=col_counts.count)
    consistency_ratio = col_counts.count(most_common_col_count) / len(col_counts)
    
    print(f"DEBUG: Table consistency check - {consistency_ratio:.1%} rows match {most_common_col_count} columns")
    
    # Only accept tables with >80% column consistency
    if consistency_ratio < 0.8:
        print(f"DEBUG: Rejecting table - poor column consistency ({consistency_ratio:.1%})")
        return []
    
    # Filter to rows with correct column count
    valid_rows = [cells for _, cells in candidate_rows if len(cells) == most_common_col_count]
    
    if len(valid_rows) < 2:
        print(f"DEBUG: Rejecting table - insufficient consistent rows")
        return []
    
    # First row as potential header (must have text-like values, not numbers)
    first_row = valid_rows[0]
    is_header = sum(1 for cell in first_row if not any(c.isdigit() for c in cell)) > len(first_row) * 0.6
    
    start_idx = 1 if is_header else 0
    
    for i, cells in enumerate(valid_rows[start_idx:]):
        row_dict = {f"col_{j}": cell for j, cell in enumerate(cells)}
        data.append(row_dict)
    
    return data if data else []

def get_canonical_bbox(
    page_w: float, page_h: float,
    view_w: float, view_h: float,
    sel_x: float, sel_y: float, sel_w: float, sel_h: float
) -> tuple:
    """
    Canonical function to convert UI pixel coordinates to Backend (PDF/Image) coordinates.
    
    CRITICAL LOGIC EXPLANATION:
    1. Coordinate Systems:
       - PDF Native: Origin is Bottom-Left.
       - pdfplumber/Image: Origin is Top-Left (abstracted).
       - Browser/UI: Origin is Top-Left.
       - We map UI (Top-Left) -> pdfplumber (Top-Left). 
       - Y-axis inversion is NOT performed manually because pdfplumber's .crop() 
         expects Top-Left coordinates (x0, top, x1, bottom).
    
    2. Trust Width Scaling:
       - Browsers often report incorrect view_height due to scrollbars, UI chrome, or CSS.
       - view_width is typically constrained by the container and is reliable.
       - If the calculated X and Y scales differ by > 2%, we assume the Y scale is 
         distorted and force it to match the X scale to preserve the selection's aspect ratio.
    """
    # 1. Validate View Dimensions
    if view_w <= 0 or view_h <= 0:
        # Fallback for legacy calls without view dims (assume 1:1 or percentage)
        return (0, 0, 0, 0)

    # 2. Calculate Scales
    scale_x = page_w / view_w
    scale_y = page_h / view_h

    # 3. Trust Width Logic (Override Height Scale if mismatch > 2%)
    if abs(scale_x - scale_y) / scale_x > 0.02:
        print(f"DEBUG: Scale mismatch (X: {scale_x:.4f}, Y: {scale_y:.4f}). Trusting Width.")
        scale_y = scale_x

    # 4. Transform Coordinates (Top-Left -> Top-Left)
    x0 = sel_x * scale_x
    top = sel_y * scale_y
    x1 = (sel_x + sel_w) * scale_x
    bottom = (sel_y + sel_h) * scale_y

    # 5. Clamp to Page Dimensions (Ensure valid bbox)
    x0 = max(0.0, min(x0, page_w))
    top = max(0.0, min(top, page_h))
    x1 = max(x0, min(x1, page_w))
    bottom = max(top, min(bottom, page_h))

    return (x0, top, x1, bottom)


def extract_table_from_ocr_image(pil_image: Image.Image) -> List[Dict[str, str]]:
    """
    High-precision OCR table extractor using layout (bounding box) information.
    Works for scanned tables/images by:
    - Running pytesseract.image_to_data to get word-level boxes
    - Clustering words into columns based on their X positions
    - Using the first non-empty row as headers and building row dictionaries

    Returns a list of row dicts, or [] if no reliable table structure is found.
    """
    try:
        # Use the same config as text OCR but request structured output
        custom_config = r'--oem 3 --psm 6'
        ocr_df = pytesseract.image_to_data(
            pil_image,
            config=custom_config,
            output_type=Output.DATAFRAME
        )
    except Exception as e:
        print(f"DEBUG: OCR layout extraction failed: {e}")
        return []

    # Filter out invalid/low confidence entries
    if ocr_df is None or ocr_df.empty:
        return []

    # Keep only words with reasonable confidence and non-empty text
    try:
        ocr_df = ocr_df[ocr_df["conf"] != -1]
        ocr_df = ocr_df[ocr_df["text"].astype(str).str.strip() != ""]
        ocr_df = ocr_df[ocr_df["conf"].astype(float) >= 40]  # confidence threshold
    except Exception as e:
        print(f"DEBUG: Cleaning OCR dataframe failed: {e}")
        return []

    if ocr_df.empty:
        return []

    # Build token list with geometry
    tokens = []
    for _, row in ocr_df.iterrows():
        txt = str(row["text"]).strip()
        if not txt:
            continue
        token = {
            "text": txt,
            "left": float(row["left"]),
            "top": float(row["top"]),
            "width": float(row["width"]),
            "height": float(row["height"]),
            "block": int(row.get("block_num", 0)),
            "line": int(row.get("line_num", 0)),
        }
        tokens.append(token)

    if not tokens:
        return []

    # Sort by vertical then horizontal position
    tokens.sort(key=lambda t: (t["top"], t["left"]))

    # 1) Detect column centers across all tokens
    column_centers: List[float] = []
    column_tolerance = max(pil_image.width * 0.03, 25)  # adaptive tolerance

    for t in tokens:
        center_x = t["left"] + t["width"] / 2.0
        assigned_idx = None
        for idx, col_center in enumerate(column_centers):
            if abs(center_x - col_center) <= column_tolerance:
                assigned_idx = idx
                # Update running average for stability
                column_centers[idx] = (col_center * 0.7) + (center_x * 0.3)
                break

        if assigned_idx is None:
            column_centers.append(center_x)
            assigned_idx = len(column_centers) - 1

        t["col_idx"] = assigned_idx

    num_cols = len(column_centers)
    if num_cols < 2 or num_cols > 15:
        # Probably not a real table
        print(f"DEBUG: OCR layout detected {num_cols} columns - rejecting as table")
        return []

    # 2) Group tokens into logical rows using (block, line)
    rows_map: Dict[tuple, List[Dict[str, Any]]] = {}
    row_order: List[tuple] = []
    for t in tokens:
        key = (t["block"], t["line"])
        if key not in rows_map:
            rows_map[key] = []
            row_order.append(key)
        rows_map[key].append(t)

    # Build dense rows with one cell per column
    dense_rows: List[List[str]] = []
    for key in row_order:
        row_tokens = sorted(rows_map[key], key=lambda x: x["left"])
        cells = ["" for _ in range(num_cols)]
        for tok in row_tokens:
            idx = tok["col_idx"]
            if cells[idx]:
                cells[idx] += " " + tok["text"]
            else:
                cells[idx] = tok["text"]
        # Skip rows that are effectively empty
        if any(c.strip() for c in cells):
            dense_rows.append([c.strip() for c in cells])

    if len(dense_rows) < 2:
        # Need at least header + one data row
        return []

    # 3) Use first row as header, rest as data
    header_row = dense_rows[0]
    # Normalize header names
    headers: List[str] = []
    for idx, cell in enumerate(header_row):
        clean = (
            cell.lower()
            .replace(" ", "_")
            .replace(".", "")
            .replace("-", "_")
            .strip("_")
        )
        if not clean:
            clean = f"col_{idx}"
        headers.append(clean)

    # Ensure uniqueness of header names
    seen = set()
    for i, h in enumerate(headers):
        original = h
        suffix = 1
        while h in seen:
            h = f"{original}_{suffix}"
            suffix += 1
        headers[i] = h
        seen.add(h)

    # Basic sanity check: if almost all headers are single characters, this is likely noise
    single_char_headers = sum(1 for h in headers if len(h) <= 1)
    if single_char_headers > len(headers) * 0.6:
        print("DEBUG: Rejecting OCR layout table - headers look like noise")
        return []

    data_rows: List[Dict[str, str]] = []
    for row_cells in dense_rows[1:]:
        row_dict: Dict[str, str] = {}
        non_empty = 0
        for idx, cell in enumerate(row_cells):
            text = cell.strip()
            if text:
                row_dict[headers[idx]] = text
                non_empty += 1
        # Require at least 2 non-empty cells to keep the row
        if non_empty >= 2:
            data_rows.append(row_dict)

    if not data_rows:
        return []

    print(
        f"DEBUG: OCR layout reconstruction produced {len(data_rows)} data rows and {len(headers)} columns"
    )
    return data_rows

def extract_with_ocr(pdf_path: str, sel: RegionSelection) -> List[Dict]:
    """Extract data from PDF using OCR with pytesseract - converts pages to images and applies OCR."""
    data = []
    try:
        print(f"DEBUG: Starting OCR extraction from {pdf_path}")
        
        # Check if tesseract is available
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            print("WARNING: Tesseract not found - OCR extraction disabled, will fall back to pdfplumber")
            return []
        
        # USE PDFPLUMBER FOR CROPPING (Better coordinate handling)
        with pdfplumber.open(pdf_path) as pdf:
            if sel.page_number < 1 or sel.page_number > len(pdf.pages):
                print(f"DEBUG: Page {sel.page_number} out of range")
                return []
                
            page = pdf.pages[sel.page_number - 1]
            width, height = float(page.width), float(page.height)
            
            # Use Canonical Coordinate Transformation
            if sel.view_width and sel.view_width > 0:
                bbox = get_canonical_bbox(width, height, float(sel.view_width), float(sel.view_height),
                                          sel.x, sel.y, sel.width, sel.height)
            else:
                # Legacy Percentage Fallback
                bbox = (sel.x_pct * width, sel.y_pct * height, 
                        (sel.x_pct + sel.w_pct) * width, (sel.y_pct + sel.h_pct) * height)

            print(f"DEBUG: PDF Dimensions: {width}x{height}")
            print(f"DEBUG: Cropping bbox: {bbox}")
            
            if bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
                return []
            
            # Crop the image to the selected region
            try:
                cropped_page = page.crop(bbox)
                # Convert crop to image (high res for OCR)
                # Resolution 600 provides better detail for small text/tables
                img = cropped_page.to_image(resolution=600).original  # PIL Image

                # DEBUG: Save crop to verify alignment
                img.save("DEBUG_CROP.png")

                # --- Advanced Image Preprocessing for OCR ---
                # 1. Convert to grayscale
                img = img.convert("L")
                # 2. Upscale 2x using LANCZOS
                new_size = (img.width * 2, img.height * 2)
                img = img.resize(
                    new_size, getattr(Image, "Resampling", Image).LANCZOS
                )
            except Exception as e:
                print(f"DEBUG: Error cropping/converting page: {e}")
                return []

            # First, try high-precision table reconstruction using OCR layout data
            table_from_layout = extract_table_from_ocr_image(img)
            if table_from_layout:
                print(
                    f"DEBUG: OCR layout-based parser extracted {len(table_from_layout)} rows from PDF image region"
                )
                data.extend(table_from_layout)
                return data

            # Fallback: plain text OCR + heuristic parsing
            print(
                "DEBUG: Falling back to text-based OCR parsing for cropped PDF region"
            )
            print("DEBUG: Applying pytesseract OCR to cropped region")
            # Config: OEM 3 (Default), PSM 6 (Block of text), Preserve spacing
            custom_config = r"--oem 3 --psm 6 -c preserve_interword_spaces=1"
            extracted_text = pytesseract.image_to_string(img, config=custom_config)

            if not extracted_text or not extracted_text.strip():
                print("DEBUG: No text extracted from OCR")
                return []

            print(f"DEBUG: OCR extracted text ({len(extracted_text)} chars)")

            if sel.use_ai:
                # Use Ollama for intelligent parsing
                ollama_data = parse_with_ollama(extracted_text, sel.label)
                if ollama_data:
                    print(
                        f"DEBUG: Ollama extracted {len(ollama_data)} records from OCR text"
                    )
                    data.extend(ollama_data)
                else:
                    # Fallback if Ollama fails
                    print(
                        "DEBUG: Ollama extraction failed, falling back to manual parsing"
                    )
                    manual_data = parse_extracted_text(extracted_text)
                    if manual_data:
                        # Mark as fallback
                        if "_warning" not in manual_data[0]:
                            manual_data[0][
                                "_warning"
                            ] = "OCR + fallback text parsing"
                        data.extend(manual_data)
                    else:
                        print(
                            "DEBUG: Text parsing failed - returning raw OCR text"
                        )
                        data.append(
                            {
                                "raw_text": extracted_text[:2000],
                                "_warning": "OCR extraction succeeded but could not parse structure",
                            }
                        )
            else:
                # Use Manual parsing
                manual_data = parse_extracted_text(extracted_text)
                if manual_data:
                    data.extend(manual_data)
                else:
                    print("DEBUG: OCR parsing failed - returning raw text")
                    data.append(
                        {
                            "raw_text": extracted_text[:2000],
                            "_warning": "OCR succeeded but no structured data found",
                        }
                    )

            if not data:
                # Return empty so we can fallback to pdfplumber (which is better at tables)
                pass
    except Exception as e:
        print(f"OCR extraction error: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    return data

def extract_from_image(image_path: str, sel: RegionSelection, use_raw_headers: bool = False) -> List[Dict]:
    """Extract data from an image using pytesseract OCR."""
    data = []
    try:
        print(f"DEBUG: Starting image OCR extraction from {image_path}")
        
        # Check if tesseract is available
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            print("WARNING: Tesseract not found - OCR extraction disabled for images")
            return []
        
        image = Image.open(image_path)
        width, height = image.size
        
        # Use Canonical Coordinate Transformation
        if sel.view_width and sel.view_width > 0:
            bbox = get_canonical_bbox(float(width), float(height), float(sel.view_width), float(sel.view_height),
                                      sel.x, sel.y, sel.width, sel.height)
        else:
            bbox = (sel.x_pct * width, sel.y_pct * height, 
                    (sel.x_pct + sel.w_pct) * width, (sel.y_pct + sel.h_pct) * height)
        
        x0, y0, x1, y1 = bbox
        
        print(f"DEBUG: Image dimensions: {width}x{height}")
        print(f"DEBUG: Crop box: ({x0}, {y0}, {x1}, {y1})")
        
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            return []
        
        # Crop the image to the selected region
        cropped_image = image.crop((x0, y0, x1, y1))
        
        # DEBUG: Save crop
        cropped_image.save("DEBUG_CROP.png")

        # --- Advanced Image Preprocessing for OCR ---
        # 1. Convert to grayscale
        cropped_image = cropped_image.convert('L')
        # 2. Upscale 2x using LANCZOS
        new_size = (cropped_image.width * 2, cropped_image.height * 2)
        resample_method = getattr(Image, 'Resampling', Image).LANCZOS
        cropped_image = cropped_image.resize(new_size, resample_method)

        # First, try high-precision table reconstruction using OCR layout data
        table_from_layout = extract_table_from_ocr_image(cropped_image)
        if table_from_layout:
            print(
                f"DEBUG: OCR layout-based parser extracted {len(table_from_layout)} rows from IMAGE region"
            )
            data.extend(table_from_layout)
            return data

        # Run pytesseract OCR on the cropped image as fallback
        print("DEBUG: Falling back to text-based OCR parsing for image region")
        print("DEBUG: Applying pytesseract OCR to image region")
        custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        extracted_text = pytesseract.image_to_string(cropped_image, config=custom_config)
        
        if not extracted_text or not extracted_text.strip():
            print("DEBUG: No text extracted from OCR")
            return []
        
        print(f"DEBUG: OCR extracted text ({len(extracted_text)} chars)")
        
        if sel.use_ai:
            # Use Ollama for intelligent parsing
            ollama_data = parse_with_ollama(extracted_text, sel.label)
            if ollama_data:
                print(f"DEBUG: Ollama extracted {len(ollama_data)} records from Image")
                data.extend(ollama_data)
            else:
                # Fallback if Ollama fails
                print("DEBUG: Ollama extraction failed, falling back to manual parsing")
                manual_data = parse_extracted_text(extracted_text)
                if manual_data:
                    manual_data[0]["_warning"] = "AI extraction failed, used manual fallback"
                    data.extend(manual_data)
                else:
                    data.append({"extracted_text": extracted_text[:1000], "_source": "ocr_raw", "_warning": "AI extraction failed"})
        else:
            # Use Manual parsing
            manual_data = parse_extracted_text(extracted_text)
            if manual_data:
                data.extend(manual_data)
            else:
                data.append({"extracted_text": extracted_text[:1000], "_source": "ocr_raw"})
            
    except Exception as e:
        print(f"Image extraction error: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    return data

def extract_key_value_pairs(cropped_page) -> List[Dict[str, str]]:
    """
    Extract key-value pairs from a PDF page region.
    Handles layouts where keys are in grey cells and values are in white cells.
    Uses pdfplumber's table extraction with improved cell pairing logic.
    """
    try:
        # First, try to extract as a table (pdfplumber handles cell boundaries well)
        tables = cropped_page.extract_tables(table_settings={
            "vertical_strategy": "lines_strict",
            "horizontal_strategy": "lines_strict",
            "snap_tolerance": 3,
            "min_words_vertical": 1,
            "min_words_horizontal": 1,
        })
        
        kv_pairs = {}
        
        # If we got tables, process them as key-value pairs
        if tables and len(tables) > 0:
            for table in tables:
                if not table or len(table) < 2:
                    continue
                
                # Process table rows - look for key-value patterns
                for row in table:
                    if not row or len(row) < 2:
                        continue
                    
                    # Clean cells
                    cells = [str(cell).strip() if cell else "" for cell in row]
                    cells = [c for c in cells if c]  # Remove empty cells
                    
                    if len(cells) < 2:
                        continue
                    
                    # Try to identify key-value pairs in the row
                    # Keys are typically: ALL CAPS, shorter, common field names
                    # Values are typically: longer, mixed case, contain numbers/dates
                    for i in range(len(cells) - 1):
                        potential_key = cells[i]
                        potential_value = cells[i + 1]
                        
                        key_upper = potential_key.upper()
                        is_key = (
                            len(potential_key) < 50 and
                            (key_upper == potential_key or potential_key.count(' ') < 4) and
                            any(keyword in key_upper for keyword in [
                                'WELL', 'NAME', 'BASIN', 'LICENCE', 'LOCATION', 'OPERATOR',
                                'OBJECTIVE', 'STRUCTURE', 'TYPE', 'DATE', 'RIG', 'NORTHING',
                                'EASTING', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'ZONE', 'GRID',
                                'SEISMIC', 'REF', 'TD', 'DAYS', 'RELEASE', 'SPUD', 'DRILLER',
                                'OFFSET', 'CLASS', 'SURVEY', 'OPERATION'
                            ])
                        )
                        
                        if is_key and potential_value:
                            # Normalize key name
                            clean_key = potential_key.lower().replace(' ', '_').replace('/', '_').replace('-', '_')
                            clean_key = re.sub(r'[^a-z0-9_]', '', clean_key)
                            # Use the value, or append if key already exists (multi-line values)
                            if clean_key in kv_pairs:
                                kv_pairs[clean_key] += ' ' + potential_value
                            else:
                                kv_pairs[clean_key] = potential_value
                            break  # Found a pair, move to next row
        
        # Fallback: text-based extraction if table extraction didn't work well
        if not kv_pairs or len(kv_pairs) < 3:
            text = cropped_page.extract_text(layout=True)
            if text and text.strip():
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                current_key = None
                current_value_parts = []
                
                for line in lines:
                    line_upper = line.upper()
                    
                    # Detect potential keys
                    is_potential_key = (
                        len(line) < 50 and
                        (line_upper == line or line_upper.count(' ') < 3) and
                        any(keyword in line_upper for keyword in [
                            'WELL', 'NAME', 'BASIN', 'LICENCE', 'LOCATION', 'OPERATOR',
                            'OBJECTIVE', 'STRUCTURE', 'TYPE', 'DATE', 'RIG', 'NORTHING',
                            'EASTING', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'ZONE', 'GRID',
                            'SEISMIC', 'REF', 'TD', 'DAYS', 'RELEASE', 'SPUD', 'DRILLER'
                        ])
                    )
                    
                    if is_potential_key:
                        if current_key and current_value_parts:
                            value = ' '.join(current_value_parts).strip()
                            if value:
                                clean_key = current_key.lower().replace(' ', '_').replace('/', '_').replace('-', '_')
                                clean_key = re.sub(r'[^a-z0-9_]', '', clean_key)
                                kv_pairs[clean_key] = value
                        current_key = line.strip()
                        current_value_parts = []
                    else:
                        if current_key:
                            current_value_parts.append(line.strip())
                
                # Don't forget the last pair
                if current_key and current_value_parts:
                    value = ' '.join(current_value_parts).strip()
                    if value:
                        clean_key = current_key.lower().replace(' ', '_').replace('/', '_').replace('-', '_')
                        clean_key = re.sub(r'[^a-z0-9_]', '', clean_key)
                        kv_pairs[clean_key] = value
        
        if kv_pairs:
            print(f"DEBUG: Extracted {len(kv_pairs)} key-value pairs: {list(kv_pairs.keys())[:5]}...")
            return [kv_pairs]
        
        return []
    except Exception as e:
        print(f"DEBUG: Key-value extraction error: {e}")
        import traceback
        traceback.print_exc()
        return []

def extract_from_region(pdf_path: str, sel: RegionSelection, use_raw_headers: bool = False) -> List[Dict]:
    """Extract tables from ONLY the cropped region you selected."""
    data = []
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[sel.page_number - 1]
        width, height = page.width, page.height
        
        print(f"DEBUG: Page dimensions: {width}x{height}")
        
        # Use Canonical Coordinate Transformation
        if sel.view_width and sel.view_width > 0:
            bbox = get_canonical_bbox(float(width), float(height), float(sel.view_width), float(sel.view_height),
                                      sel.x, sel.y, sel.width, sel.height)
        else:
            bbox = (sel.x_pct * width, sel.y_pct * height, 
                    (sel.x_pct + sel.w_pct) * width, (sel.y_pct + sel.h_pct) * height)

        x0, top, x1, bottom = bbox
        print(f"DEBUG: Cropped bbox (clamped): ({x0}, {top}, {x1}, {bottom})")
        
        if x1 - x0 <= 1 or bottom - top <= 1:
            print("DEBUG: Selection too small")
            return []
        
        # Crop page and extract tables ONLY from that region
        cropped = page.crop(bbox)
        
        # 👇 For WELL_HEADER labels, try key-value extraction first (better for structured forms)
        if sel.label in ("WELL_HEADER", "WCR_WELLHEAD"):
            print("DEBUG: WELL_HEADER detected - trying key-value pair extraction first")
            kv_data = extract_key_value_pairs(cropped)
            if kv_data and len(kv_data) > 0 and len(kv_data[0]) > 3:  # At least 3 key-value pairs
                print(f"DEBUG: Key-value extraction successful - found {len(kv_data[0])} pairs")
                return kv_data
            else:
                print("DEBUG: Key-value extraction found insufficient data, falling back to table extraction")
        
        tables = []
        
        # 🎯 STRATEGY: Standard pdfplumber extraction (Most Reliable)
        # Use pdfplumber's built-in table detection which handles headers correctly
        try:
            # Try more aggressive table settings first for structured layouts
            tables = cropped.extract_tables(table_settings={
                "vertical_strategy": "lines_strict",  # Use explicit lines for better cell detection
                "horizontal_strategy": "lines_strict",
                "snap_tolerance": 3,
                "snap_x_tolerance": 3,
                "snap_y_tolerance": 3,
                "join_tolerance": 3,
                "min_words_vertical": 1,
                "min_words_horizontal": 1,
                "intersection_x_tolerance": 5,
                "intersection_y_tolerance": 5,
                "explicit_vertical_lines": [],
                "explicit_horizontal_lines": []
            })
            
            # If that didn't work, try text-based strategy
            if not tables:
                tables = cropped.extract_tables(table_settings={
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "snap_tolerance": 5,
                    "min_words_vertical": 2,
                    "intersection_x_tolerance": 10
                })
            
            print(f"DEBUG: Found {len(tables) if tables else 0} tables in cropped region")
        except Exception as e:
            print(f"DEBUG: Error extracting tables: {e}")
            tables = []
        
        # Helper function to validate if headers are legitimate table headers
        # DEPRECATED: We're now accepting tables as-is from pdfplumber
        # This function is commented out and not used anymore

        # Extract from first table found
        if tables:
            for table_idx, table in enumerate(tables):
                if not table or len(table) < 2:
                    print(f"DEBUG: Skipping table {table_idx} - too few rows ({len(table) if table else 0})")
                    continue
                
                print(f"DEBUG: Processing table {table_idx}: {len(table)} rows x {len(table[0]) if table[0] else 0} cols")
                
                # Get number of columns from first row
                num_cols = len(table[0]) if table[0] else 0
                
                # Validate table structure: all rows should have similar column count
                col_counts = [len(row) for row in table]
                most_common_cols = max(set(col_counts), key=col_counts.count)
                col_consistency = col_counts.count(most_common_cols) / len(col_counts)
                
                print(f"DEBUG: Column consistency: {col_consistency:.1%} (expected {most_common_cols} cols)")
                
                # Reject tables with poor column consistency (likely corrupted extraction)
                if col_consistency < 0.7:
                    print(f"DEBUG: ⚠️ Rejecting table {table_idx} - poor column consistency ({col_consistency:.1%})")
                    print(f"DEBUG: Column counts per row: {col_counts[:5]}...")
                    continue
                
                # 🔍 STEP 1: Detect header rows (supports multi-line headers)
                header_rows = []
                data_start_index = 0

                for row_index, row in enumerate(table):
                    row_text_cells = [str(cell).strip() for cell in row if cell]
                    if not row_text_cells:
                        # Completely empty row; treat as separator and continue scanning
                        continue

                    numeric_cells = sum(
                        1
                        for cell in row_text_cells
                        if any(c.isdigit() for c in str(cell))
                    )
                    numeric_ratio = (
                        numeric_cells / len(row_text_cells) if row_text_cells else 0.0
                    )

                    # Heuristic:
                    #   - Header rows are mostly text (numeric_ratio <= 0.4)
                    #   - First data row will have more numbers
                    if numeric_ratio <= 0.4:
                        header_rows.append(row)
                    else:
                        data_start_index = row_index
                        break

                if not header_rows:
                    # Fallback: assume first row is header
                    header_rows = [table[0]]
                    data_start_index = 1 if len(table) > 1 else 0

                print(
                    f"DEBUG: Using {len(header_rows)} header row(s), data starts at row index {data_start_index}"
                )

                # 🔑 EXTRACT HEADERS by vertically concatenating header rows
                headers = []
                header_mapping = {}  # Maps column index to schema field name
                
                for col_idx in range(num_cols):
                    parts = []
                    for hr in header_rows:
                        if col_idx < len(hr):
                            cell = hr[col_idx]
                            if cell:
                                part = str(cell).strip()
                                if part:
                                    parts.append(part)

                    header_text = " ".join(parts).strip().lower()
                    if not header_text:
                        header_text = f"col_{col_idx}"

                    headers.append(header_text)
                
                print(f"DEBUG: Headers found: {headers}")
                
                # Auto-detect table type and create schema mapping
                headers_str = " ".join(headers).lower()
                
                # CASING Table Detection
                if any(kw in headers_str for kw in ["hole", "casing", "depth", "diameter", "size"]):
                    print(f"DEBUG: Detected CASING table - applying schema mapping")
                    expected_fields = ["hole_size", "depth_md_kb", "casing_diameter", "casing_depth_md_kb", 
                                     "type", "test_fit_lot", "test_date", "test_result_ppg", "test_depth_md_kb"]
                    
                    # Create mapping from detected headers to schema
                    for col_idx, header in enumerate(headers):
                        for schema_field in expected_fields:
                            # Simple keyword matching
                            if any(kw in header for kw in schema_field.split('_')):
                                header_mapping[col_idx] = schema_field
                                break
                        if col_idx not in header_mapping:
                            header_mapping[col_idx] = headers[col_idx]
                
                # 📊 SKIP HEADER ROW(S) AND EXTRACT ONLY DATA ROWS
                data_rows = table[data_start_index:] if len(table) > data_start_index else []
                print(f"DEBUG: Found {len(data_rows)} data rows (skipped {len(header_rows)} header row(s))")
                
                if not data_rows:
                    print(f"DEBUG: No data rows to extract")
                    continue
                
                # Process each data row with precise column mapping
                extracted_count = 0
                min_valid_cells = max(1, most_common_cols // 2)  # At least 50% of columns should have data
                
                for row_num, row in enumerate(data_rows):
                    row_dict = {}
                    non_empty_cell_count = 0
                    
                    # Extract data for ALL columns
                    for col_idx in range(num_cols):
                        cell = row[col_idx] if col_idx < len(row) else None
                        
                        # Determine the column name (use schema mapping if available)
                        col_name = header_mapping.get(col_idx, headers[col_idx] if col_idx < len(headers) else f"col_{col_idx}")
                        
                        cell_text = str(cell).strip() if cell else ""
                        # Only add non-empty cells
                        if cell_text:
                            row_dict[col_name] = cell_text
                            non_empty_cell_count += 1
                    
                    # Skip rows with too few cells (corrupted rows)
                    if non_empty_cell_count < min_valid_cells:
                        print(f"DEBUG: Skipping row {row_num} - insufficient data ({non_empty_cell_count}/{num_cols} cells)")
                        continue
                    
                    # Add row if it has sufficient data
                    if row_dict:
                        data.append(row_dict)
                        extracted_count += 1
                
                print(f"DEBUG: Extracted {extracted_count} valid data rows (headers skipped)")
                
                # If we successfully extracted rows from this table, return early
                if data:
                    return data
        
        # Fallback to text extraction only if table extraction completely failed
        print("DEBUG: No valid tables found in cropped region, trying text extraction")
        # layout=True preserves visual spacing better, crucial for regex splitting
        text = cropped.extract_text(layout=True)
        
        if text and text.strip():
            print(f"DEBUG: Extracted {len(text)} chars of text from region")
            
            if sel.use_ai:
                # Use Ollama for intelligent parsing of the text region
                ollama_data = parse_with_ollama(text, sel.label)
                if ollama_data:
                    print(f"DEBUG: Ollama extracted {len(ollama_data)} records from PDF text")
                    data.extend(ollama_data)
                else:
                    # Fallback if Ollama fails
                    print("DEBUG: Ollama extraction failed, falling back to manual parsing")
                    manual_data = parse_extracted_text(text)
                    if manual_data:
                        # Mark as fallback
                        if "_warning" not in manual_data[0]:
                            manual_data[0]["_warning"] = "Fallback text extraction (no table found)"
                        data.extend(manual_data)
                    else:
                        # Last resort: return raw text
                        print("DEBUG: Text parsing also failed - returning raw text")
                        data.append({"raw_text": text[:2000], "_warning": "Could not extract table structure - raw text returned"})
            else:
                # Use Manual parsing
                manual_data = parse_extracted_text(text)
                if manual_data:
                    data.extend(manual_data)
                else:
                    # No structure found, return as warning
                    print("DEBUG: No structured data found - returning as warning")
                    data.append({"raw_text": text[:2000], "_warning": "No table or structured data found"})
        else:
            print("DEBUG: No text found in cropped region")
    
    return data

# System columns to ignore during validation/display so we don't flag them as missing
IGNORED_COLUMNS = {
    "ID", "MODEL", "INSERT_DATE", "MATCH_PERCENT", 
    "VECTOR_IDS", "PAGE_NUMBERS", "MATCH_ID"
}

def map_columns_with_ollama(unmapped_keys: List[str], schema_columns: List[str], table_name: str) -> Dict[str, str]:
    """
    Column mapping is disabled. Using heuristic matching instead which is more reliable.
    The three-phase heuristic approach (exact, fuzzy, semantic) handles most cases well.
    """
    # Heuristics are sufficient for column mapping
    return {}

def validate_data(data: List[Dict], table_name: str):
    try:
        schema = get_table_schema(table_name)
    except ValueError:
        return {"error": f"Table {table_name} not defined in SQL."}

    validated_rows = []
    
    # Helper to normalize keys for robust matching (remove spaces, underscores, dots)
    def normalize_key(k):
        # Remove all non-alphanumeric characters for stricter matching (e.g. "Field:" -> "FIELD")
        return "".join(c for c in str(k).upper() if c.isalnum())

    # Create a map of the schema: NORMALIZED -> Original
    schema_map = {normalize_key(k): k for k in schema.keys()}
    # Also create a list for fuzzy substring matching
    sql_cols_normalized = [(normalize_key(k), k) for k in schema.keys()]
    
    # Filter schema for display/validation (exclude system columns)
    display_columns = [k for k in schema.keys() if k.upper() not in IGNORED_COLUMNS]
    
    # --- PHASE 1: Identify all unique keys in the data ---
    all_keys = set()
    for row in data:
        for k in row.keys():
            if not k.startswith("_"):
                all_keys.add(k)
    
    # --- PHASE 2: Build Column Mapping (Heuristic + LLM) ---
    key_mapping = {}
    unmapped_keys = []
    
    for key in all_keys:
        norm_key = normalize_key(key)
        real_col_name = None
        
        # 1. Exact Normalized Match
        if norm_key in schema_map:
            real_col_name = schema_map[norm_key]
        
        # 2. Fuzzy Match (Substring)
        if not real_col_name:
            for sql_norm, sql_orig in sql_cols_normalized:
                if sql_norm in norm_key and len(sql_norm) > 2: 
                    real_col_name = sql_orig
                    break
        
        # 3. Smart semantic matching (Hardcoded patterns)
        if not real_col_name:
            key_lower = key.lower()
            if "type" in key_lower and ("casing" in schema.keys() or any("CASING_TYPE" in c for c in schema.keys())):
                real_col_name = "CASING_TYPE"
            elif ("depth" in key_lower or "bottom" in key_lower) and "CASING_BOTTOM" in schema.keys():
                real_col_name = "CASING_BOTTOM"
            elif ("top" in key_lower) and "CASING_TOP" in schema.keys():
                real_col_name = "CASING_TOP"
            elif ("diameter" in key_lower or "od" in key_lower) and "OUTER_DIAMETER" in schema.keys():
                real_col_name = "OUTER_DIAMETER"
            elif ("length" in key_lower or "grade" in key_lower) and "STEEL_GRADE" in schema.keys():
                real_col_name = "STEEL_GRADE"
            elif ("material" in key_lower or "grade" in key_lower) and "MATERIAL_TYPE" in schema.keys():
                real_col_name = "MATERIAL_TYPE"
        
        if real_col_name:
            key_mapping[key] = real_col_name
        else:
            unmapped_keys.append(key)
            
    # --- PHASE 3: LLM Fallback for Unmapped Keys ---
    if unmapped_keys:
        llm_mapping = map_columns_with_ollama(unmapped_keys, list(schema.keys()), table_name)
        for k, v in llm_mapping.items():
            if v in schema:
                key_mapping[k] = v
                print(f"DEBUG: LLM mapped '{k}' -> '{v}'")

    # --- PHASE 4: Apply Mapping to Rows ---
    for row in data:
        row_status = "VALID"
        errors = []
        clean_row = {}
        
        # Skip rows with only warnings or errors
        if "_warning" in row or "_error" in row:
            # Keep these special rows as-is
            clean_row = row.copy()
            clean_row["_status"] = "WARNING" if "_warning" in row else "ERROR"
            validated_rows.append(clean_row)
            continue
        
        for key, val in row.items():
            if key.startswith("_"): continue # Skip internal flags
            
            real_col_name = key_mapping.get(key)
            
            if real_col_name:
                clean_row[real_col_name] = val
            else:
                errors.append(f"Unknown column: {key}")
                row_status = "INVALID"
        
        # 2. Only mark as INVALID if we found actual data but have unknown columns
        # If we have data that maps to known columns, keep it as VALID
        # Missing columns should just be null
        if not clean_row and not errors:
            # Empty row
            row_status = "WARNING"
            errors.append("No data extracted")
        
        # Fill in missing columns with None
        for col in display_columns:
            if col not in clean_row:
                clean_row[col] = None
        
        clean_row["_status"] = row_status
        clean_row["_errors"] = "; ".join(errors) if errors else ""
        validated_rows.append(clean_row)
        
    return {"schema": display_columns, "data": validated_rows}


# --- HELPER FUNCTIONS FOR CASING MATCHING ---

def normalize_numeric_value(val):
    """
    Convert string with units (e.g., "14\"", "507.5") to float.
    Handles cases like: "14\"", "36 ppf", "507.5 m", "J-55"
    Returns float if numeric, None if not parseable, or original string.
    """
    if val is None:
        return None
    
    val_str = str(val).strip()
    
    # Remove common units and symbols
    units_to_remove = ['"', "'", "ppf", "ppg", "m", "mm", "in", "ft", "PSI", "psi", "kg", "lb"]
    cleaned = val_str
    for unit in units_to_remove:
        cleaned = cleaned.replace(unit, "").strip()

    # Handle fractional formats like '9 5/8' or '4 1/2'
    try:
        if '/' in cleaned:
            # Examples: '9 5/8' or '5/8'
            parts = cleaned.split()
            if len(parts) == 2:
                whole = float(parts[0])
                num, den = parts[1].split('/')
                return whole + float(num) / float(den)
            elif len(parts) == 1 and '/' in parts[0]:
                num, den = parts[0].split('/')
                return float(num) / float(den)

        # Try to parse as float
        return float(cleaned)
    except Exception:
        # Not a numeric value (e.g., "J-55", "BTC")
        return val_str.upper()


def build_casing_signature(row):
    """
    Build a normalized signature for CASING matching.
    Match signature uses: UWI, CASING_TYPE, CASING_TOP, CASING_BOTTOM, OUTER_DIAMETER
    Returns dict with normalized values.
    """
    signature = {
        "UWI": str(row.get("UWI", "")).strip().upper() if row.get("UWI") else None,
        "CASING_TYPE": str(row.get("CASING_TYPE", "")).strip().upper() if row.get("CASING_TYPE") else None,
        "CASING_TOP": normalize_numeric_value(row.get("CASING_TOP")),
        "CASING_BOTTOM": normalize_numeric_value(row.get("CASING_BOTTOM")),
        "OUTER_DIAMETER": normalize_numeric_value(row.get("OUTER_DIAMETER")),
    }
    return signature


def casing_rows_match(extracted_sig, db_sig, tolerance=0.5):
    """
    Compare two CASING signatures with tolerance for numeric differences.
    Returns True if:
    - UWI matches exactly
    - At least 4 out of 5 fields match (with numeric tolerance)
    """
    if not extracted_sig.get("UWI") or not db_sig.get("UWI"):
        return False
    
    if extracted_sig["UWI"] != db_sig["UWI"]:
        return False
    
    # Compare each field
    match_count = 0
    fields = ["CASING_TYPE", "CASING_TOP", "CASING_BOTTOM", "OUTER_DIAMETER"]
    
    for field in fields:
        extracted_val = extracted_sig.get(field)
        db_val = db_sig.get(field)
        
        # If either is None, skip this field
        if extracted_val is None or db_val is None:
            continue
        
        # Compare
        if isinstance(extracted_val, float) and isinstance(db_val, float):
            # Numeric comparison with tolerance
            if abs(extracted_val - db_val) <= tolerance:
                match_count += 1
        elif isinstance(extracted_val, str) and isinstance(db_val, str):
            # String comparison
            if extracted_val == db_val:
                match_count += 1
        elif str(extracted_val) == str(db_val):
            match_count += 1
    
    # Match if at least 4 out of 5 fields match (we compare 4 fields after UWI)
    return match_count >= 4
# --- ENDPOINTS ---

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename}

@app.post("/extract")
async def extract(
    filename: str = Form(...),
    selection: str = Form(...)
):
    sel_dict = json.loads(selection)
    sel_obj = RegionSelection(**sel_dict)
    
    # Security: Ensure filename is just the name, not a path
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    table_name = LABEL_TO_TABLE.get(sel_obj.label)
    if not table_name:
        raise HTTPException(status_code=400, detail="Label not mapped to SQL table")
    
    # Determine if file is an image or PDF
    is_image = safe_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.webp', '.bmp', '.gif'))
    
    # Extract based on file type
    if is_image:
        raw_data = extract_from_image(file_path, sel_obj, use_raw_headers=True)
    else:
        # Try native PDF extraction first (Best for tables in digital PDFs)
        print(f"DEBUG: Attempting native pdfplumber extraction")
        raw_data = extract_from_region(file_path, sel_obj, use_raw_headers=True)

        # Evaluate Native Result
        native_quality_low = False
        if not raw_data or len(raw_data) == 0:
            native_quality_low = True
        elif "_warning" in raw_data[0] or "_error" in raw_data[0]:
            # Fix 2: Only OCR if PDF has NO TEXT LAYER
            # If we have substantial text, assume digital PDF and accept the text result (don't OCR)
            raw_text = raw_data[0].get("raw_text", "")
            if len(raw_text.strip()) > 50:
                native_quality_low = False
            else:
                native_quality_low = True
        elif len(raw_data[0].keys()) < 2:
            # If we only found 1 column, native extraction probably failed to split columns
            native_quality_low = True
        
        # Fallback to OCR if native extraction yields nothing
        if not raw_data or len(raw_data) == 0:
            print(f"DEBUG: Native extraction returned no data, trying OCR")
            raw_data = extract_with_ocr(file_path, sel_obj)
        # Fallback to OCR if native extraction was poor
        if native_quality_low:
            print(f"DEBUG: Native extraction quality low, trying OCR")
            ocr_data = extract_with_ocr(file_path, sel_obj)
            # Use OCR data if it found something
            if ocr_data and len(ocr_data) > 0:
                raw_data = ocr_data
    
    if not raw_data:
        msg = "No data found in selection"
        if not TESSERACT_AVAILABLE:
            msg += " (OCR unavailable: Tesseract not found on server)"
        return {"message": msg, "raw_data": [], "sql_data": [], "schema": []}

    # Validate
    result = validate_data(raw_data, table_name)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return {
        "raw_data": raw_data,
        "sql_data": result["data"],
        "schema": result["schema"]
    }

@app.post("/check-existence")
async def check_existence(
    data: str = Form(...),
    table_name: str = Form(...)
):
    try:
        rows = json.loads(data)
        if not rows:
            return {"exists": [], "new": []}
            
        # Convert to DataFrame for easier handling
        input_df = pd.DataFrame(rows)
        # Drop internal columns
        input_df = input_df.drop(columns=[c for c in input_df.columns if c.startswith("_")], errors='ignore')
        
        if input_df.empty:
             return {"exists": [], "new": rows}

        with engine.connect() as conn:
            # Check if table exists
            inspector = inspect(engine)
            if not inspector.has_table(table_name):
                return {"exists": [], "new": rows}

            # Read existing data
            # Optimization: If UWI is present, filter by UWI to reduce data load
            query = f"SELECT * FROM {table_name}"
            if "UWI" in input_df.columns:
                unique_uwis = [str(u) for u in input_df["UWI"].unique() if u]
                if unique_uwis:
                    uwis_str = "', '".join(unique_uwis)
                    query += f" WHERE \"UWI\" IN ('{uwis_str}')"
            
            existing_df = pd.read_sql(query, conn)
            
        if existing_df.empty:
            return {"exists": [], "new": rows}

        # Identify common columns for comparison
        common_cols = list(set(existing_df.columns) & set(input_df.columns))
        if not common_cols:
            # Try to map columns using validate_data (which now uses LLM)
            print("DEBUG: No common columns found. Attempting to map columns with LLM...")
            validation_result = validate_data(rows, table_name)
            
            if "error" not in validation_result and "data" in validation_result:
                mapped_rows = validation_result["data"]
                # Re-create DataFrame with mapped data
                input_df = pd.DataFrame(mapped_rows)
                input_df = input_df.drop(columns=[c for c in input_df.columns if c.startswith("_")], errors='ignore')
                # Re-check common columns
                common_cols = list(set(existing_df.columns) & set(input_df.columns))
        
        if not common_cols:
            return {"exists": [], "new": rows}

        # Special handling for CASING table: normalize meaningful fields
        if table_name.upper() == "WCR_CASING":
            exists_rows = []
            new_rows = []

            existing_dicts = existing_df.to_dict('records')
            input_dicts = input_df.to_dict('records')

            db_sigs = [build_casing_signature(r) for r in existing_dicts]

            for i, input_row in enumerate(input_dicts):
                input_sig = build_casing_signature(input_row)
                found_match = False
                for db_sig in db_sigs:
                    if casing_rows_match(input_sig, db_sig):
                        found_match = True
                        break

                if found_match:
                    exists_rows.append(rows[i])
                else:
                    new_rows.append(rows[i])

            return {"exists": exists_rows, "new": new_rows}

        # Default: Create signatures for comparison (concat all common values)
        existing_sigs = existing_df[common_cols].astype(str).agg('-'.join, axis=1)
        input_sigs = input_df[common_cols].astype(str).agg('-'.join, axis=1)

        exists_mask = input_sigs.isin(existing_sigs)

        exists_rows = []
        new_rows = []

        for i, is_exist in enumerate(exists_mask):
            if is_exist:
                exists_rows.append(rows[i])
            else:
                new_rows.append(rows[i])

        return {"exists": exists_rows, "new": new_rows}

    except Exception as e:
        print(f"Check Error: {e}")
        # Fallback: assume all new if check fails
        return {"exists": [], "new": json.loads(data)}

@app.post("/save")
async def save_to_db(
    data: str = Form(...), 
    table_name: str = Form(...)
):
    try:
        rows = json.loads(data)
        if not rows:
             return {"message": "No data to save"}
             
        # Convert to DataFrame
        df = pd.DataFrame(rows)
        
        # Remove internal fields (_status, _errors)
        cols_to_drop = [c for c in df.columns if c.startswith("_")]
        df = df.drop(columns=cols_to_drop)
        
        # Remove ID if present (let DB handle auto-increment)
        if "ID" in df.columns:
            df = df.drop(columns=["ID"])
            
        # FIX: Replace string "None" and empty strings with None to ensure NULL in DB
        # This fixes the "invalid input syntax for type double precision: 'None'" error
        df = df.replace({"None": None, "none": None, "": None})
        
        # Drop rows where all columns are None (completely empty rows)
        df = df.dropna(how='all')

        # Ensure UWI is present if it's a required field (prevents NotNullViolation)
        if "UWI" in df.columns:
            df = df.dropna(subset=["UWI"])
            
        # Filter columns to match schema (prevent "column not found" errors)
        try:
            schema = get_table_schema(table_name)
            valid_columns = set(schema.keys())
            # Keep only columns that exist in the DB schema
            existing_cols = [c for c in df.columns if c in valid_columns]
            df = df[existing_cols]
        except Exception as e:
            print(f"Schema validation warning: {e}")
            
        if df.empty:
            return {"message": "No valid data to save (rows empty or missing UWI)"}

        # Insert into DB (append mode)
        df.to_sql(table_name, engine, if_exists='append', index=False, method='multi')
        
        return {"message": f"Successfully saved {len(df)} rows to {table_name}"}
    except Exception as e:
        error_msg = str(e)
        print(f"Database Save Error: {error_msg}")
        
        # Provide user-friendly error messages
        if "UNIQUE constraint" in error_msg or "duplicate key" in error_msg:
            raise HTTPException(status_code=409, detail="Save Failed: Duplicate record exists (UWI/ID already in database).")
        elif "NOT NULL constraint" in error_msg or "null value" in error_msg:
            raise HTTPException(status_code=400, detail="Save Failed: Missing required fields (e.g., UWI is required).")
        elif "no such table" in error_msg:
            raise HTTPException(status_code=404, detail=f"Save Failed: Table {table_name} does not exist.")
        else:
            raise HTTPException(status_code=500, detail=f"Database Error: {error_msg}")

@app.post("/export")
async def export_csv(data: str = Form(...), table_name: str = Form(...)):
    """Exports ONLY valid rows to CSV matching SQL schema"""
    rows = json.loads(data)
    
    # Filter valid rows
    valid_rows = [r for r in rows if r.get('_status') == 'VALID']
    
    if not valid_rows:
        raise HTTPException(status_code=400, detail="No valid rows to export")
        
    df = pd.DataFrame(valid_rows)
    
    # Drop internal columns
    cols_to_drop = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=cols_to_drop)
    
    # Ensure column order matches SQL schema
    schema = get_table_schema(table_name)
    # Add missing columns as empty
    for col in schema.keys():
        if col not in df.columns:
            df[col] = None
            
    # Reorder
    df = df[list(schema.keys())]
    
    output_path = os.path.join(UPLOAD_DIR, f"{table_name}_export.csv")
    df.to_csv(output_path, index=False)
    
    return FileResponse(output_path, filename=f"{table_name}.csv")

@app.post("/generate-template")
async def generate_template(table_name: str = Form(...)):
    """Generates a perfect PDF template for the given SQL table"""
    try:
        schema = get_table_schema(table_name)
        # Filter system columns
        cols = [k for k in schema.keys() if k.upper() not in IGNORED_COLUMNS]
        
        output_path = os.path.join(UPLOAD_DIR, f"{table_name}_template.pdf")
        doc = SimpleDocTemplate(output_path, pagesize=landscape(letter))
        elements = []
        
        styles = getSampleStyleSheet()
        elements.append(Paragraph(f"Sample Report for {table_name}", styles['Title']))
        elements.append(Spacer(1, 20))
        
        # Create Dummy Data
        data = [cols] # Header
        dummy_row = [f"Test {c}" for c in cols] # Row 1
        data.append(dummy_row)
        
        t = Table(data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(t)
        
        doc.build(elements)
        return FileResponse(output_path, filename=f"{table_name}_template.pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export-pdf")
async def export_pdf(data: str = Form(...), table_name: str = Form(...)):
    """Exports extraction results to a PDF report"""
    rows = json.loads(data)
    if not rows:
        raise HTTPException(status_code=400, detail="No data to export")
        
    output_path = os.path.join(UPLOAD_DIR, f"{table_name}_report.pdf")
    doc = SimpleDocTemplate(output_path, pagesize=landscape(letter))
    elements = []
    styles = getSampleStyleSheet()
    
    elements.append(Paragraph(f"Extraction Report: {table_name}", styles['Title']))
    elements.append(Spacer(1, 12))
    
    # Prepare Table Data
    # Get all keys from first row (schema)
    headers = [k for k in rows[0].keys() if not k.startswith("_")]
    table_data = [headers]
    
    for row in rows:
        table_data.append([str(row.get(k, "")) for k in headers])
        
    t = Table(table_data, repeatRows=1)
    t.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(t)
    doc.build(elements)
    return FileResponse(output_path, filename=f"{table_name}_report.pdf")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9000)
