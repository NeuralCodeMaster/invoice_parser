import io
import re
import json
import os
import cv2
import numpy as np
import PyPDF2
import easyocr
import pdf2image
from tqdm import tqdm
import camelot  # pip install camelot-py

DEBUG_MODE = True
MIN_CHAR_THRESHOLD = 100
MAX_LINE_MERGE = 6  # Merge up to 6 lines

# If you do not have GPU support, you can set `reader = easyocr.Reader(["en"], gpu=False)`
reader = easyocr.Reader(["en"], gpu=True)

pdf_files = [
    "data/20250221125114588.pdf",
    "data/20250221092842541.pdf",
    "data/Invaoice_2.pdf"
]

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# 1) PyPDF2 vs. OCR Fallback
# -----------------------------------------------------------------------------

def extract_text_with_pypdf2(pdf_path):
    text_output = []
    try:
        with open(pdf_path, "rb") as f:
            reader_pdf = PyPDF2.PdfReader(f)
            for page in reader_pdf.pages:
                txt = page.extract_text() or ""
                text_output.append(txt)
    except Exception as e:
        if DEBUG_MODE:
            print(f"[DEBUG] PyPDF2 error: {e}")
        return ""
    return "\n".join(text_output)

def preprocess_image(np_image):
    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def ocr_with_easyocr(pdf_path):
    images = pdf2image.convert_from_path(pdf_path)
    extracted_text = ""
    for img in tqdm(images, desc=f"Performing OCR on {pdf_path}"):
        np_img = np.array(img)
        np_img = preprocess_image(np_img)
        blocks = reader.readtext(np_img, detail=0, paragraph=True)
        extracted_text += " ".join(blocks) + "\n"
    return extracted_text

def pdf_to_text_fallback(pdf_path, min_char=MIN_CHAR_THRESHOLD):
    txt = extract_text_with_pypdf2(pdf_path)
    if len(txt) >= min_char:
        if DEBUG_MODE:
            print(f"[DEBUG] '{pdf_path}' => Enough text extracted => skipping OCR.")
        return txt
    else:
        if DEBUG_MODE:
            print(f"[DEBUG] '{pdf_path}' => Likely scanned => performing OCR.")
        return ocr_with_easyocr(pdf_path)

# -----------------------------------------------------------------------------
# 2) Attempting to parse tables with Camelot (digital/structured tables)
# -----------------------------------------------------------------------------

def try_camelot_tables(pdf_path):
    """
    If a table is found, it returns a Camelot TableList (tables).
    For image-based pages, it usually won't find tables and returns None.
    """
    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        if tables and tables.n > 0:
            return tables
    except Exception as e:
        if DEBUG_MODE:
            print(f"[DEBUG] Camelot error: {e}")
    return None

def parse_camelot_tables(tables):
    """
    Parses table rows into a list of dictionaries as a simple example.
    You can adapt this further depending on your table structure.
    """
    all_rows = []
    for i, tb in enumerate(tables):
        df = tb.df
        if df.shape[0] < 2:
            continue
        # The first row is treated as headers
        df.columns = df.iloc[0]
        df = df.drop(df.index[0])
        for idx, row in df.iterrows():
            all_rows.append(row.to_dict())
    return all_rows

# -----------------------------------------------------------------------------
# 3) cleanup_line & fix_float
# -----------------------------------------------------------------------------

def cleanup_line(line):
    # Example fixes: In some PDFs, "Pric e:" should become "Price:"
    line = re.sub(r"P\s*r\s*i\s*c\s*e\s*:", "Price:", line, flags=re.IGNORECASE)
    # Similarly, "Q ty:" => "Qty:"
    line = re.sub(r"Q\s*ty\s*:", "Qty:", line, flags=re.IGNORECASE)
    return line

def fix_float(value: str) -> float:
    """
    Attempts to parse a string into a float, handling various formatting issues.
    """
    raw = value.strip()
    # Remove unwanted characters
    raw = raw.replace('S','').replace('_','')
    raw = raw.replace('.,','.')
    raw = raw.replace(',','.')
    parts = raw.split('.')
    if len(parts) > 2:
        # If there are multiple dots, merge them
        last = parts[-1]
        middle = "".join(parts[1:-1])
        raw = parts[0] + middle + "." + last
    raw = re.sub(r'\.\.+','.', raw)
    raw = raw.replace(' ','')
    if not raw:
        raw = "0"
    try:
        return float(raw)
    except:
        if DEBUG_MODE:
            print(f"[DEBUG] fix_float parse fail: {value} => {raw}")
        return 0.0

# -----------------------------------------------------------------------------
# 4) Simple Regex (Invoice, PO, Supplier)
# -----------------------------------------------------------------------------

def extract_invoice_details(text):
    pat = r'Invoice Number[:\s]*([\w\d-]+)'
    m = re.search(pat, text, re.IGNORECASE)
    return m.group(1) if m else None

def extract_po_numbers(text):
    pat = r'(PO[-\s]?|Purchase Order[:\s#]?)\s*(\d+)'
    ms = re.findall(pat, text, re.IGNORECASE)
    return [x[1] for x in ms]

def extract_supplier_info(text):
    """
    Example pattern for supplier detection:
    ([A-Z\s]+LTD|INC|GMBH)\n(.+?)\n
    Adjust as needed based on the actual formatting in the PDF.
    """
    pat = r'([A-Z\s]+LTD|INC|GMBH)\n(.+?)\n'
    m = re.search(pat, text, re.DOTALL)
    if m:
        return {
            "name": m.group(1).strip(),
            "address": m.group(2).strip()
        }
    return None

# -----------------------------------------------------------------------------
# 5) Regex Patterns
# -----------------------------------------------------------------------------

PRODUCT_CODE_PATTERN_A = re.compile(
    r'Product Code[:;\s]*(PRD-[\w\d\-\[\]]+)\s+'
    r'Quantity[:;\s]*(\d+)(?:\s+units)?\s+'
    r'Unit\s*Price[:;\s]*\$?([\d,\.]+)\s+'
    r'Amount[:;\s]*\$?([\d,\.]+)',
    re.IGNORECASE
)

PRODUCT_CODE_PATTERN_B = re.compile(r'''
(?ix)
( PRD-[\w\d-]+ )             # (1) Product code
\s+ ( [A-Za-z0-9 /_-]+ )     # (2) Description (e.g. "Power Supply")
\s+ Qty[:;\s]* (\d+)         # (3) Qty
\s+ Price[:;\s]* \$?([\d.,]+)# (4) Unit price
\s+ Total[:;\s]* \$?([\d.,]+)# (5) Total
(?: \s+ PO:\s+ PO\s*-\s*(\d+) )?  # (6) Optional PO
''', re.VERBOSE)

SERVICE_PATTERN = re.compile(
    r'(?i)(.*?)\s+Hours[:;\s]*(\d+)\s*x\s*Rate[:;\s]*\$?([\d,\.]+)\s*/hr\s+Amount[:;\s]*\$?([\d,\.]+)',
    re.IGNORECASE | re.DOTALL
)

# -----------------------------------------------------------------------------
# 6) Line Merging + Regex
# -----------------------------------------------------------------------------

def extract_line_by_line_items(extracted_text: str):
    products = []
    services = []
    lines = extracted_text.splitlines()
    # Clean up each line for possible spacing or format issues
    lines = [cleanup_line(ln) for ln in lines]

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        matched = try_match_line(line, products, services, i, merges=0)
        if matched:
            i += 1
            continue

        merged_ok = False
        # Attempt to merge current line with following lines
        for merge_count in range(1, MAX_LINE_MERGE):
            if i + merge_count < len(lines):
                merged = line
                for k in range(1, merge_count + 1):
                    merged += " " + lines[i + k].strip()

                m_ok = try_match_line(merged, products, services, i, merges=merge_count)
                if m_ok:
                    i += (merge_count + 1)
                    merged_ok = True
                    break
            else:
                break

        if not merged_ok:
            i += 1

    return products, services

def try_match_line(line: str, products: list, services: list, line_idx: int, merges: int):
    mA = PRODUCT_CODE_PATTERN_A.search(line)
    if mA:
        if DEBUG_MODE:
            prefix = f"[MERGEx{merges}]" if merges > 0 else ""
            print(f"[DEBUG] {prefix} Pattern A matched line #{line_idx}: '{line}'")
            print(f"         Groups => {mA.groups()}")
        parse_pattern_a(mA, products)
        return True

    mB = PRODUCT_CODE_PATTERN_B.search(line)
    if mB:
        if DEBUG_MODE:
            prefix = f"[MERGEx{merges}]" if merges > 0 else ""
            print(f"[DEBUG] {prefix} Pattern B matched line #{line_idx}: '{line}'")
            print(f"         Groups => {mB.groups()}")
        parse_pattern_b(mB, products)
        return True

    mS = SERVICE_PATTERN.search(line)
    if mS:
        if DEBUG_MODE:
            prefix = f"[MERGEx{merges}]" if merges > 0 else ""
            print(f"[DEBUG] {prefix} Pattern Service matched line #{line_idx}: '{line}'")
            print(f"         Groups => {mS.groups()}")
        parse_pattern_service(mS, services)
        return True

    if DEBUG_MODE and merges == 0:
        print(f"[DEBUG] No pattern matched line #{line_idx}: '{line}'")
    return False

def parse_pattern_a(m, products_list):
    try:
        code    = m.group(1)
        qty_str = m.group(2)
        up_str  = m.group(3)
        amt_str = m.group(4)

        q = int(qty_str)
        u = fix_float(up_str)
        t = fix_float(amt_str)
        products_list.append({
            "product_code": code,
            "quantity": q,
            "unit_price": u,
            "total_price": t
        })
    except ValueError as e:
        if DEBUG_MODE:
            print(f"[DEBUG] Pattern A parse error: {e}")

def parse_pattern_b(m, products_list):
    try:
        code        = m.group(1).strip()  
        description = m.group(2).strip()  
        qty_str     = m.group(3).strip()
        price_str   = m.group(4).strip()
        total_str   = m.group(5).strip()
        po_num      = m.group(6)  

        q = int(qty_str)
        u = fix_float(price_str)
        t = fix_float(total_str)

        rec = {
            "product_code": code,
            "description": description,
            "quantity": q,
            "unit_price": u,
            "total_price": t
        }
        if po_num:
            rec["po_number"] = po_num

        products_list.append(rec)
    except ValueError as e:
        if DEBUG_MODE:
            print(f"[DEBUG] Pattern B parse error: {e}")

def parse_pattern_service(m, services_list):
    try:
        svc_name = m.group(1).strip(':,.- \n')
        hrs_str  = m.group(2).strip()
        rate_str = m.group(3).strip()
        amt_str  = m.group(4).strip()

        h = int(hrs_str)
        r = fix_float(rate_str)
        a = fix_float(amt_str)
        services_list.append({
            "service_name": svc_name,
            "hours": h,
            "rate_per_hour": r,
            "amount": a
        })
    except ValueError as e:
        if DEBUG_MODE:
            print(f"[DEBUG] Pattern Service parse error: {e}")

# -----------------------------------------------------------------------------
# 7) Basic Consistency Reports (Price + PO Matching)
# -----------------------------------------------------------------------------

def generate_consistency_report_for_products(products):
    mismatches = []
    for p in products:
        exp = p["quantity"] * p["unit_price"]
        act = p["total_price"]
        if abs(exp - act) > 0.01:
            mismatches.append({
                "product_code": p.get("product_code"),
                "expected_total": round(exp, 2),
                "actual_total": act
            })
    return {
        "price_mismatches": mismatches,
        "total_inconsistencies": len(mismatches)
    }

def generate_consistency_report_for_services(services):
    mismatches = []
    for s in services:
        exp = s["hours"] * s["rate_per_hour"]
        act = s["amount"]
        if abs(exp - act) > 0.01:
            mismatches.append({
                "service_name": s.get("service_name"),
                "expected_total": round(exp, 2),
                "actual_total": act
            })
    return {
        "price_mismatches": mismatches,
        "total_inconsistencies": len(mismatches)
    }

def generate_po_consistency_report(products, po_numbers):
    """
    Checks if PO numbers found within product lines match the ones
    extracted from the text. Also checks if there are PO numbers in
    the text that are unused in the product lines.
    """
    product_po_list = [
        p["po_number"] for p in products
        if "po_number" in p and p["po_number"]
    ]

    missing_in_extracted = [
        po for po in product_po_list if po not in po_numbers
    ]

    unused_in_products = [
        po for po in po_numbers if po not in product_po_list
    ]

    return {
        "missing_in_extracted": missing_in_extracted,
        "unused_in_products": unused_in_products,
        "total_inconsistencies": len(missing_in_extracted) + len(unused_in_products)
    }

# -----------------------------------------------------------------------------
# 8) JSON Output
# -----------------------------------------------------------------------------

def create_json_output(invoice_number, supplier_info, po_numbers, products,
                       services, product_report, service_report, po_report):
    data = {
        "invoice_number": invoice_number,
        "supplier_info": supplier_info,
        "po_numbers": po_numbers,
        "products": products,
        "services": services,
        "consistency_report": {
            "product_inconsistencies": product_report,
            "service_inconsistencies": service_report,
            "po_inconsistencies": po_report
        }
    }
    return json.dumps(data, indent=4)

# -----------------------------------------------------------------------------
# 9) process_pdf => Camelot => fallback
# -----------------------------------------------------------------------------

def process_pdf(pdf_file):
    # A) Try parsing tables with Camelot
    try:
        tables = camelot.read_pdf(pdf_file, pages='all', flavor='lattice')
        if tables and tables.n > 0:
            if DEBUG_MODE:
                print(f"[DEBUG] Found {tables.n} table(s) with Camelot.")
            # Parse table
            table_rows = parse_camelot_tables(tables)
            # Minimal parse
            products = []
            for row in table_rows:
                code  = row.get("Product Code", "")
                qty   = row.get("Qty", "0")
                price = row.get("Price", "0")
                total = row.get("Total", "0")
                try:
                    q = int(qty)
                    p = fix_float(price)
                    t = fix_float(total)
                except:
                    q = 0
                    p = 0.0
                    t = 0.0
                products.append({
                    "product_code": code,
                    "quantity": q,
                    "unit_price": p,
                    "total_price": t
                })
            # Extract other fields from text
            txt = extract_text_with_pypdf2(pdf_file)
            invoice_num = extract_invoice_details(txt)
            supp = extract_supplier_info(txt)
            pnums = extract_po_numbers(txt)
            prod_report = generate_consistency_report_for_products(products)
            serv_report = {"price_mismatches": [], "total_inconsistencies": 0}
            po_report = generate_po_consistency_report(products, pnums)

            return invoice_num, supp, pnums, products, [], prod_report, serv_report, po_report
        else:
            if DEBUG_MODE:
                print("[DEBUG] Camelot => no table => fallback.")
    except Exception as e:
        if DEBUG_MODE:
            print(f"[DEBUG] Camelot => error => fallback: {e}")

    # B) Fallback => OCR + Regex
    extracted_text = pdf_to_text_fallback(pdf_file, MIN_CHAR_THRESHOLD)
    invoice_num = extract_invoice_details(extracted_text)
    supp = extract_supplier_info(extracted_text)
    pnums = extract_po_numbers(extracted_text)
    products, services = extract_line_by_line_items(extracted_text)

    prod_report = generate_consistency_report_for_products(products)
    serv_report = generate_consistency_report_for_services(services)
    po_report = generate_po_consistency_report(products, pnums)

    return invoice_num, supp, pnums, products, services, prod_report, serv_report, po_report

# -----------------------------------------------------------------------------
# 10) MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file}")

        (
            invoice_number,
            supplier_info,
            po_numbers,
            products,
            services,
            product_report,
            service_report,
            po_report
        ) = process_pdf(pdf_file)

        js = create_json_output(
            invoice_number,
            supplier_info,
            po_numbers,
            products,
            services,
            product_report,
            service_report,
            po_report
        )

        out_path = os.path.join(output_dir, os.path.basename(pdf_file).replace(".pdf", ".json"))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(js)

        print(f"Done processing: {pdf_file}, output saved -> {out_path}\n")
