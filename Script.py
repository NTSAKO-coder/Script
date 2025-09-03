import csv
import time
import json
import os
import re
from typing import List, Dict, Union

import vertexai
from vertexai.generative_models import GenerativeModel, Part

PROJECT_ID = "lic-dev-dmt-dat-sci"
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.5-flash"

PDF_FOLDER_PATH = r"C:\\Users\\Ntsakom\\OneDrive - Lombard Insurance\\Desktop\\GIT Reports"
OUTPUT_FILE_PATH = r"C:\\Users\\Ntsakom\\Downloads\\Testing2.csv"


os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)

CSV_COLUMNS = [
    "ClaimNumber",
    "Insured",
    "PolicyType",
    "Incident",
    "Load",
    "Loading",
    "DateOfLoss",
    "Location",
    "ClaimCalculationTotal",
    "CargoOwner",
    "InsuranceCover",
    "AdequacyOfSumInsured",
    "Conveyances",
    "Driver",
    "CircumstancesOfClaim",
    "PoliceDetails"
]

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel(MODEL_NAME)

EXTRACT_POLICYHOLDER_METADATA = """
You are a specialist in extracting information from insurance claim assessor reports.
Your task is to read the report and extract the following fields. If a field is not present, return what's described in the field definition. If it's not present on the file at all and no specific instruction is given, return "null". Do not invent information. Output ONLY a single valid JSON object or a JSON array of objects with exactly the keys listed. Ensure all extracted string values are trimmed of leading/trailing whitespace.

IMPORTANT: When encountering highlighted sections, do not let the highlighting interrupt your extraction. Treat highlighted text as regular text and continue extracting all relevant information from the document, regardless of visual formatting such as highlights, bolding, or italics. Ensure you extract all data present in the document, even if it appears after a highlighted section.

**Output Format:**
- If only one incident is found, return a single JSON object.
- If multiple incidents are found, return a JSON array of objects, where each object represents a single incident.
- Every field listed below MUST be present as a key in EACH object.

**Field definitions:**
- **ClaimNumber:** The primary claim reference number. If multiple numbers are present for a single incident, focus on the first one.
- **Insured:** Name of the insured.
- **PolicyType:** Type of insurance policy (e.g., "MOBILITY"). Only take the policyType and ignore qualifiers like "GIT:".
- **Incident:** Brief description of the incident type (e.g., "Hijacking", "Collision").
- **Load:** Here you need to extract what the load is (e.g., "Tomatoes").
- **Loading:** Weight of the load in kilograms. Must always include "kg" at the end (e.g., "5000 kg", "53 582 kg.", " 51 340 kg."). Sometimes it can be an estimation; take the estimated kg. If not found, provide the summary given for its absence (e.g., "Weight not specified").
- **DateOfLoss:** The date of the loss incident. Normalize to YYYY-MM-DD (e.g., "2023-10-26"). May appear as "bi" sometimes. If not a date, extract what's there (e.g., "Date to be confirmed").
- **Location:** Extract the location (e.g., "N3 Highway near Harrismith", "Mthatha, Eastern Cape").If Unknown, extract what's there (e.g., "Location not specified", "UNKNOWN – Loss Discover at Johannesburg) or "null".
- **ClaimCalculationTotal:** Extract the value of the claim. Must be in Rand (R) format (e.g., "R 123,456.78"). May appear as "Calculation Final Value" sometimes. If not found, extract what's there (e.g., "Still to be calculated").
- **CargoOwner:** Owner of the cargo.
- **InsuranceCover:** Extract the value of the insurance cover. Must be in Rand (R) format (e.g., "R 500,000.00"). If not stated, extract what's there (e.g., "Not applicable").
- **AdequacyOfSumInsured:** The value of the load at the time of incident. This should be in Rand (R) format (e.g., "R 1,500,000.00"). If not stated, extract what's there (e.g., "Unknown").
- **Conveyances:** List all 3 if there's three, if two name 2 and if one name 1.
- **Driver:** Extract all details that appear under the Driver section. Include the driver’s full name, licence type, validity date, code, and vehicle restrictions exactly as written.
- **CircumstancesOfClaim:** Read all the paragraphs under Circumstances of the claim and provide a concise summary of its content.
- **PoliceDetails:** First check if the Police Details field is there then extracts what's there, if it's not there check on the content under Circumstances of the claim for relevant information and if there is extract it.If there's no PoliceDetails nor any relevant information under Circumstances of the claim.

**Rules:**
- Where there are two incidents (e.g., "Incident 1 and Incident 2", "1st Incident" and 2nd Incident" etc.), extract both and separate them with a "comma".Do this for all fields that can have multiple values.Ensure they share the same row in the CSV, instead of creating separate rows.
- Do not any raws empty.
- For every field, extract only information belonging to that field; if multiple values exist (e.g., Incident 1, Incident 2), keep them together under the same field as comma-separated.Which means total files should be equal to total raws.
- If the report describes multiple incidents, extract all incidents. Fields like "Incident", "DateOfLoss", "Location", "Conveyances", "Driver", "Loading", and "CircumstancesOfClaim" may have multiple values (one for each incident).
- All extracted string values must be trimmed of leading/trailing whitespace.
- All monetary amounts must be written in Rand (R) format (e.g., "R 1,234.56").
- Normalize dates to YYYY-MM-DD.
- Each field is independent - ClaimNumber, Insured, PolicyType, etc., must each extract only from their own context, not spill into others.
"""


def process_pdf_to_json_str(pdf_path: str, max_retries: int = 2) -> str:
    """
    Sends a single PDF to the model and returns the raw JSON string the model outputs.
    Includes a retry mechanism for transient errors or empty responses.
    """
    attempt = 0
    while attempt <= max_retries:
        try:
            print(f"    - Sending {os.path.basename(pdf_path)} to model (Attempt {attempt + 1}/{max_retries + 1})...")
            with open(pdf_path, "rb") as f:
                document = Part.from_data(data=f.read(), mime_type="application/pdf")

            response = model.generate_content(
                [EXTRACT_POLICYHOLDER_METADATA, document],
                generation_config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json"
                }
            )

            if response.text and response.text.strip():
                return response.text
            else:
                print(f"    - WARNING: Model returned empty response for {os.path.basename(pdf_path)}. Retrying...")

        except Exception as e:
            print(f"    - ERROR during model generation for {os.path.basename(pdf_path)}: {e}. Retrying if attempts remain.")

        attempt += 1
        if attempt <= max_retries:
            time.sleep(attempt * 2) 

    raise ValueError(f"Failed to get valid response after {max_retries + 1} attempts for {os.path.basename(pdf_path)}")


def safe_parse_model_json(text: str, filename: str) -> List[Dict]:
    """
    Attempts to parse JSON from the model output. Robustly handles code fences,
    extra text, and ensures a list of dicts is returned.
    """
    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.endswith("```"):
        text = text[:-len("```")].strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            if all(isinstance(elem, dict) for elem in data):
                return data
            else:
                raise ValueError("Parsed JSON list contains non-dictionary elements.")
        elif isinstance(data, dict):
            print(f"    - WARNING: Model returned a single object instead of a list for {filename}. Wrapping it in a list.")
            return [data]
        else:
            raise ValueError(f"Expected a JSON object or list of objects, but got: {type(data).__name__}.")

    except json.JSONDecodeError as e:
        print(f"    - CRITICAL ERROR: Could not parse direct JSON for {filename}. Attempting to find embedded JSON.")
       
        match_list = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
        match_obj = re.search(r'\{.*\}', text, re.DOTALL)

        candidate = None
        if match_list:
            candidate = match_list.group(0)
        elif match_obj:
            candidate = match_obj.group(0)

        if candidate:
            try:
                data = json.loads(candidate)
                if isinstance(data, list):
                    if all(isinstance(elem, dict) for elem in data):
                        print("    - Successfully parsed embedded JSON list.")
                        return data
                    else:
                        raise ValueError(f"Embedded JSON list contains non-dictionary elements: {type(data).__name__}")
                elif isinstance(data, dict):
                    print("    - Successfully parsed embedded JSON object, wrapping in list.")
                    return [data]
                else:
                    raise ValueError(f"Embedded JSON was not a dictionary or list of dictionaries: {type(data).__name__}")
            except json.JSONDecodeError as e2:
                raise ValueError(f"Failed to parse embedded JSON: {e2} from candidate (first 100 chars): {candidate[:100]}...") from e2
        else:
            raise ValueError(f"Could not find any JSON structure in text (first 200 chars): {text[:200]}...") from e

    except ValueError as e: 
        raise ValueError(f"Data parsing error for {filename}: {e}") from e


def normalize_row(d: Dict[str, Union[str, None]], source_file: str) -> List[str]:
    """
    Builds a CSV row in the exact column order, cleaning and defaulting values.
    Ensures that empty strings, None, or values indicating absence are always represented as "null".
    Handles specific formatting for ClaimNumber based on regex.
    """
    row = []
    for col in CSV_COLUMNS:
        if col == "SourceFile":
            val = source_file
        else:
            val = d.get(col)

        # Normalize value
        if val is None or (isinstance(val, str) and (val.strip() == "" or val.strip().lower() == "null")):
            val_str = "null"
        elif isinstance(val, (dict, list)):
            val_str = json.dumps(val, ensure_ascii=False)
        else:
            val_str = str(val).strip()

        if col == "ClaimNumber" and val_str != "null":
          
            git_mobilt_match = re.search(r'GIT:\s*MOBILT-(\d{4}-\d{7})\s*VERSION', val_str)
            if git_mobilt_match:
                val_str = git_mobilt_match.group(1)
            else:
        
                claim_match = re.search(r'[A-Za-z]{1,3}\d{2}-\d{1,5}-\d{1,5}', val_str)
                if claim_match:
                    val_str = claim_match.group(0)
                else:
                  
                    if not val_str or val_str.lower() == "none" or val_str.isspace():
                        val_str = "null"
        
        if not val_str or val_str.lower() == "none" or val_str.isspace():
            val_str = "null"

        row.append(val_str)
    return row


def main():
    """
    Main processing loop to extract data from all PDFs in a folder.
    """
    print(f"--- Starting Data Extraction ---")
    print(f"Scanning PDF folder: '{PDF_FOLDER_PATH}'")
    print(f"Output CSV file: '{OUTPUT_FILE_PATH}'")
    print(f"Using Vertex AI Model: '{MODEL_NAME}' at location '{LOCATION}' in project '{PROJECT_ID}'\n")

    with open(OUTPUT_FILE_PATH, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(CSV_COLUMNS)

        all_files = sorted([f for f in os.listdir(PDF_FOLDER_PATH) if f.lower().endswith(".pdf")])
        total_files = len(all_files)
        processed_files_count = 0
        extracted_rows_count = 0
        error_files_count = 0

        for i, filename in enumerate(all_files):
            pdf_path = os.path.join(PDF_FOLDER_PATH, filename)
            
            print(f"\nProcessing file {i + 1}/{total_files}: '{filename}'")
            
            try:
                start_time = time.time()
                
                raw_json_text = process_pdf_to_json_str(pdf_path, max_retries=2)
                data_list = safe_parse_model_json(raw_json_text, filename)

                file_incidents = 0
                for incident_data in data_list:
                    row = normalize_row(incident_data, source_file=filename)
                    writer.writerow(row) 
                    extracted_rows_count += 1
                    file_incidents += 1
                
                processed_files_count += 1
                elapsed = time.time() - start_time
                print(f"    -> SUCCESS: Extracted {file_incidents} incident(s) in {elapsed:.2f}s for '{filename}'")

            except ValueError as ve: 
                print(f"    -> FAILED (Data Error): {ve} for '{filename}'. Will write an ERROR row.")
                error_files_count += 1
                
                error_data = {k: "ERROR" for k in CSV_COLUMNS if k != "SourceFile"}
                row = normalize_row(error_data, source_file=filename)
                
                if "SourceFile" in CSV_COLUMNS:
                    row[CSV_COLUMNS.index("SourceFile")] = filename + " (ERROR)"
                writer.writerow(row) 
            except Exception as e: 
                print(f"    -> FAILED (Unexpected Error): An unrecoverable error occurred for '{filename}': {e}. Will write an ERROR row.")
                error_files_count += 1
                
                error_data = {k: "ERROR" for k in CSV_COLUMNS if k != "SourceFile"}
                row = normalize_row(error_data, source_file=filename)
                if "SourceFile" in CSV_COLUMNS:
                    row[CSV_COLUMNS.index("SourceFile")] = filename + " (ERROR)"
                writer.writerow(row) 
            
            print("----------------------------------------")

    print(f"\n--- Extraction Summary ---")
    print(f"Total PDFs attempted: {total_files}")
    print(f"Successfully processed PDFs: {processed_files_count} files")
    print(f"    - Total extracted rows (incidents): {extracted_rows_count} rows")
    print(f"PDFs with errors: {error_files_count} files")
    print(f"Extraction complete! Data saved to: '{OUTPUT_FILE_PATH}'")


if __name__ == "__main__":
    main()

