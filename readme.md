# PDF Processing with OCR and Table Extraction

## Overview
This script processes PDF files to extract structured and unstructured data using multiple techniques. It utilizes PyPDF2 for text extraction, EasyOCR for optical character recognition (OCR), and Camelot for table parsing. The extracted data is analyzed using regex-based patterns to identify invoices, purchase orders, suppliers, and product/service details. The output is saved in JSON format.

## Features
- **Text Extraction**: Uses PyPDF2 to extract digital text from PDFs.
- **OCR Processing**: Uses EasyOCR to extract text from scanned or image-based PDFs.
- **Table Extraction**: Uses Camelot to extract structured tables from PDFs.
- **Regex-Based Parsing**: Extracts invoice numbers, PO numbers, supplier information, product details, and service details.
- **Data Cleaning & Validation**: Fixes text formatting issues and validates extracted data.
- **Consistency Checks**: Compares extracted values to detect discrepancies in price calculations.
- **JSON Output**: Saves the extracted data in structured JSON format.

## Dependencies
Ensure you have the following dependencies installed:
```sh
pip install opencv-python numpy PyPDF2 easyocr pdf2image tqdm camelot-py
```

## Usage
### 1. Add PDF files
Place your PDF files inside the `data/` directory and update the `pdf_files` list in the script:

### 2. Run the script
Execute the script with:
```sh
python main.py
```
The extracted data will be saved in the `outputs/` directory as JSON files.

### 3. Debug Mode
The script includes a debug mode that prints additional information about the extraction process. Set `DEBUG_MODE = True` in the script to enable it.

## Functionality Breakdown
### 1. Text Extraction
- `extract_text_with_pypdf2(pdf_path)`: Extracts digital text from a PDF using PyPDF2.
- `ocr_with_easyocr(pdf_path)`: Extracts text using OCR when PyPDF2 fails.
- `pdf_to_text_fallback(pdf_path)`: Decides whether to use PyPDF2 or OCR based on the extracted text length.

### 2. Table Extraction
- `try_camelot_tables(pdf_path)`: Uses Camelot to detect tables.
- `parse_camelot_tables(tables)`: Parses detected tables into a structured format.

### 3. Data Processing
- `extract_invoice_details(text)`: Extracts invoice numbers using regex.
- `extract_po_numbers(text)`: Extracts purchase order numbers using regex.
- `extract_supplier_info(text)`: Extracts supplier details using regex.
- `extract_line_by_line_items(text)`: Extracts product and service details using regex.
- `generate_consistency_report_for_products(products)`: Validates price calculations.
- `generate_po_consistency_report(products, po_numbers)`: Checks if extracted PO numbers match the invoice data.

### 4. JSON Output
- `create_json_output(...)`: Converts extracted data into JSON format.
- `process_pdf(pdf_file)`: Handles the full PDF processing pipeline, including table extraction, text extraction, OCR fallback, and data validation.

## Output Format
The output JSON file contains:
```json
{
    "invoice_number": "INV-12345",
    "supplier_info": {"name": "ABC Ltd.", "address": "123 Street, City"},
    "po_numbers": ["PO-56789"],
    "products": [
        {
            "product_code": "PRD-001",
            "description": "Widget A",
            "quantity": 10,
            "unit_price": 5.5,
            "total_price": 55.0
        }
    ],
    "services": [],
    "consistency_report": {
        "product_inconsistencies": {...},
        "service_inconsistencies": {...},
        "po_inconsistencies": {...}
    }
}
```

## Notes
- If Camelot fails to extract tables, the script falls back to OCR-based text processing.
- Regex-based parsing may require adjustments depending on the document format.

## Future Improvements
- Improve OCR accuracy by preprocessing images before text recognition.
- Enhance regex patterns to handle more invoice formats.
- Implement machine learning-based entity recognition for more robust text extraction.



