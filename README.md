Well Completion Extractor and Database Validator
Overview

This application digitizes and validates ONGC Well Completion Reports. It extracts structured data from PDF reports or images, compares the extracted data with an existing database to identify duplicate and new records, and ensures data quality before saving.

The system uses OCR, rule-based parsing, and LLM-assisted post-processing to handle both structured and unstructured technical documents.

Key Features
Data Extraction

Manual region selection on PDF pages to extract tables or key-value data.

AI-assisted extraction for complex or unstructured layouts.

Support for both digital PDFs and scanned images.

Automatic fallback between rule-based parsing, OCR, and LLM processing to ensure extraction reliability.

Database Validation

Comparison of extracted records with the database using primary keys such as UWI.

Detection of missing or incomplete values in extracted data.

Full PDF scanning to identify all records and classify them as existing or new based on database comparison.

Data Management

Saving validated data to the database.

Exporting extracted data as CSV or PDF.

Intelligent column-to-schema mapping when PDF headers differ from database column names.

Use of LLM (Ollama)

This application uses a local Large Language Model through Ollama, specifically the model llama3.2-vision:latest.

The LLM is not used for raw OCR. OCR is performed first, and the LLM is used as a post-processing layer.

The LLM is used for:

Cleaning noisy OCR output

Correcting common OCR errors

Converting unstructured text into structured data

Extracting tables and key-value pairs

Mapping extracted fields to database schema

Enforcing strict output formats to avoid hallucination

An optional cloud-based LLM (Google Gemini) can be used as a fallback for highly complex layouts.

Processing Pipeline

PDF or Image
OCR (Tesseract, pdfplumber)
Text normalization
LLM post-processing using Ollama
Structured JSON output
Database validation and storage

User Workflows
Workflow 1: Incoming Report Validation

Upload a PDF report.

Select data regions manually.

Extract data.

Check extracted records against the database.

Save new records or skip duplicates.

Workflow 2: Data Quality Check

Extract data from a PDF.

Validate extracted fields for missing values.

Correct issues before saving.

Workflow 3: Bulk PDF Validation

Upload a complete PDF report.

Scan the entire PDF without manual selection.

View summary of existing and new records.

Technology Stack
Frontend

React with TypeScript

Tailwind CSS

react-pdf

Backend

FastAPI (Python)

SQLAlchemy ORM

PostgreSQL for production

SQLite for development

Tesseract OCR

pdfplumber

Pillow

Ollama LLM (llama3.2-vision:latest)

Google Gemini API (optional)

Setup Instructions
Prerequisites

Python 3.8 or higher

Node.js 16 or higher

Tesseract OCR

Ollama installed locally

Install Ollama Model
ollama pull llama3.2-vision:latest

Backend Setup
cd backend
pip install -r requirements.txt
python main.py


Backend runs on http://127.0.0.1:9000

Frontend Setup
cd frontend
npm install
npm run dev

API Endpoints
Method	Endpoint	Description
POST	/extract	Extract data from selected region
POST	/check-existence	Compare extracted data with database
POST	/save	Save validated data
POST	/upload	Upload PDF
POST	/export	Export extracted data
Supported Database Tables

WCR_WELLHEAD

WCR_CASING

WCR_LOGSRECORD

WCR_DIRSRVY

WCR_SWC

WCR_HCSHOWS
