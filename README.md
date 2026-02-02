Well Completion Extractor and Database Validator
Overview

This application digitizes and validates ONGC Well Completion Reports. It extracts unstructured data from PDF reports or images and converts it into structured tables, compares extracted data with an existing database to identify duplicate and new records, and ensures data quality before saving.

The system combines OCR, rule-based parsing, and LLM-assisted post-processing to handle both structured and unstructured technical documents.

Key Features
1. Data Extraction

Manual region selection on PDF pages to extract tables or key-value data

AI-assisted extraction for complex or unstructured layouts

Support for both digital PDFs and scanned images

Automatic fallback between rule-based parsing, OCR, and LLM processing

2. Database Validation

Comparison of extracted records with the database using primary keys such as UWI

Detection of missing or incomplete values in extracted data

Full PDF scanning to identify all records and classify them as existing or new

3. Data Management

Save validated data to the database

Export extracted data as CSV or PDF

Intelligent column-to-schema mapping when PDF headers differ from database column names

Use of LLM (Ollama)

This application uses a local Large Language Model through Ollama, specifically llama3.2-vision:latest.

The LLM is not used for raw OCR. OCR is performed first, and the LLM acts as a post-processing and validation layer.

LLM Responsibilities

Cleaning noisy OCR output

Correcting common OCR errors

Converting unstructured text into structured data

Extracting tables and key-value pairs

Mapping extracted fields to the database schema

Enforcing strict output formats to prevent hallucination

An optional cloud-based LLM (Google Gemini) can be used as a fallback for highly complex document layouts.

Processing Pipeline

PDF or Image
OCR using Tesseract and pdfplumber
Text normalization
LLM post-processing using Ollama
Structured JSON output
Database validation and storage

User Workflows
Workflow 1: Incoming Report Validation

Upload a PDF report

Select data regions manually

Extract data

Check extracted records against the database

Save new records or skip duplicates

Workflow 2: Data Quality Check

Extract data from a PDF

Validate extracted fields for missing values

Correct issues before saving

Workflow 3: Bulk PDF Validation

Upload a complete PDF report

Scan the entire PDF without manual extraction

View a summary of existing and new records

Technology Stack
Frontend

Framework: React with TypeScript

Styling: Tailwind CSS

PDF Handling: react-pdf

Backend

Framework: FastAPI (Python)

Database: PostgreSQL (Production), SQLite (Development)

ORM: SQLAlchemy

AI / OCR: Tesseract OCR, pdfplumber, Pillow, Ollama LLM (llama3.2-vision:latest)

Setup Instructions
Prerequisites

Python 3.8 or higher

Node.js 16 or higher

Tesseract OCR

Ollama installed locally

Install Ollama Model
ollama pull llama3.2-vision:latest

Backend Setup

Navigate to the backend directory:

cd backend


Install dependencies:

pip install -r requirements.txt


Run the server:

python main.py


The backend runs on http://127.0.0.1:9000
.

Frontend Setup

Navigate to the frontend directory:

cd frontend


Install dependencies:

npm install


Start the development server:

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
