# Run gcloud auth application-default login first
import csv
import time
import json
import os
import re 
from vertexai.generative_models import Part
import vertexai
from vertexai.generative_models import GenerativeModel

PROJECT_ID = "lic-dev-dmt-dat-sci"
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.5-flash" 

PDF_FOLDER_PATH = r"C:\\Users\\Ntsakom\\Downloads\\Test"
OUTPUT_FILE_PATH = r"C:\Users\Ntsakom\Downloads\DataA.csv"

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

EXTRACT_POLICYHOLDER_METADATA = """You are a specialist in extracting information from insurance claim assessor reports.
Your task is to read the report and extract the following fields. If a field is not present, return What's in the field. if not present on the file at all, return "null".
Do not invent information. Output ONLY a single valid JSON object with exactly the keys listed.
Ensure all extracted string values are trimmed of leading/trailing whitespace.

IMPORTANT: When encountering highlighted sections, do not let the highlighting interrupt your extraction. Treat highlighted text as regular text and continue extracting all relevant information from the document, regardless of visual formatting such as highlights, bolding, or italics. Ensure you extract all data present in the document, even if it appears after a highlighted section.

Field definitions:
- ClaimNumber:If multiple numbers are present, focus on the first one.
- Insured: Name of the insured party.If multiple numbers are present, focus on the first one.
- PolicyType: Type of insurance policy.Only take the pilocyType and ignore things like e.g "GIT:"
- Incident: Brief description of the incident type.
- Load: Description of the goods (e.g., "Tomatoes", "Electronics").
- Loading: Weight of the load in kilograms. Must always include "kg" at the end (e.g., "5000 kg").Or give the summary given of why it's not there if not there(can be an explanation of why it's not there).
- DateOfLoss: The date of the loss incident. Normalize to YYYY-MM-DD (e.g., "2023-10-26"). May appear as "Date of Loss" or "bi".If not a date extract what's there(can be an explanation of why it's not there).
- Location: The geographic location where the incident occurred.
- ClaimCalculationTotal: The final calculated monetary value of the claim. Must be in Rand (R) format (e.g., "R 123,456.78"). May appear as "Calculation Final Value" or "Claim Total".If not there extract what's there(can be an explanation of why it's not there or it can be e.g "NONE").
- CargoOwner: Owner of the cargo.
- InsuranceCover: The monetary value of the insurance cover. Must be in Rand (R) format.If not stated extract what's there (can be an explanation of why it's not there).
- AdequacyOfSumInsured: The value of the load at the time of incident. This should be in Rand (R) format, If not stated extract what's there (can be an explanation of why it's not there).
- Conveyances: Always take the first one.
- Driver: Name of the driver. Sometimes the structure may be different, just look for a '"Driver" and it will correspond with the name and surname of the drive.
- CircumstancesOfClaim: Read the full paragraph describing the incident. Remove any repeated sentences or phrases. Summarize the paragraph concisely into a clear, single-sentence description.
- PoliceDetails: Police station and/or case/reference number if available. If not mentioned, extracts what's there (can be an explanation of why it's not there).

Rules:
- Do not skip any fields, do not leave a blank row.
- If there's two Incidents, List both which means some fiels will have multiple values.
- If a field is not present, return "null" for that field.
- All extracted string values must be trimmed of leading/trailing whitespace.
- If the report describes multiple incident, extract both incidents.Fiels like "Incident", "Date of loss", "Location", "Conveyances", "Driver" and "Circumstances of Claim" may have multiple values. 
- All monetary amounts must be written in Rand (R) format (e.g., "R 1,234.56").
- Normalize dates to YYYY-MM-DD.
- Summarize circumstances concisely.
- If you find files with highlights, treat that like a normal PDF and extract the same fields.Do not leave out any fields.
- Each field is independent - ClaimNumber, Insured, PolicyType, etc., must each extract only from their own context, not spill into others.
- Always return results strictly in a single valid JSON object.
"""

def process_pdf_to_json_str(pdf_path: str, max_retries: int = 2) -> str:
    """
    Sends a single PDF to the model and returns the raw JSON string the model outputs.
    Includes a retry mechanism for transient errors or empty responses.
    """
    attempt = 0
    while attempt <= max_retries:
        try:
            with open(pdf_path, "rb") as f:
                document = Part.from_data(data=f.read(), mime_type="application/pdf")

            response = model.generate_content(
                [EXTRACT_POLICYHOLDER_METADATA, document],
                generation_config={
                    "temperature": 0.0, 
                    "response_mime_type": "application/json" 
                }
            )
            
            if response.text.strip():
                return response.text
            else:
                print(f"Warning: Model returned empty response for {os.path.basename(pdf_path)}. Retrying...")
        except Exception as e:
            print(f"Error during model generation for {os.path.basename(pdf_path)} (Attempt {attempt+1}/{max_retries+1}): {e}")
        
        attempt += 1
        if attempt <= max_retries:
            time.sleep(1) 
    
    raise ValueError(f"Failed to get valid response after {max_retries+1} attempts for {os.path.basename(pdf_path)}")

def safe_parse_model_json(text: str) -> list[dict]:
    """
    Attempts to parse JSON from the model output.
    Robustly handles code fences, extra text, and ensures a list of dicts is returned.
    """
    text = text.strip()

    
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
        if text.endswith("```"):
            text = text[:-len("```")].strip()
    elif text.startswith("```"): 
        text = text[len("```"):].strip()
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
            
            return [data]
        else:
            raise ValueError(f"Expected a JSON object or list of objects, but got: {type(data).__name__} from text: {text[:100]}...")
    except json.JSONDecodeError as e:
        
        print(f"JSONDecodeError: {e}. Attempting to find embedded JSON.")
        
        match_list = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL) 
        match_obj = re.search(r'\{.*\}', text, re.DOTALL) 

        if match_list:
            candidate = match_list.group(0)
        elif match_obj:
            candidate = match_obj.group(0)
        else:
            raise ValueError(f"Could not find any JSON structure in text: {text[:200]}...") from e

        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                if all(isinstance(elem, dict) for elem in data):
                    print("Successfully parsed embedded JSON list.")
                    return data
            elif isinstance(data, dict):
                print("Successfully parsed embedded JSON object, wrapping in list.")
                return [data]
            else:
                raise ValueError(f"Embedded JSON was not a dictionary or list of dictionaries: {type(data).__name__}")
        except json.JSONDecodeError as e2:
            raise ValueError(f"Failed to parse embedded JSON: {e2} from candidate: {candidate[:100]}...") from e2
        except ValueError as e3:
             raise ValueError(f"Failed after parsing embedded JSON: {e3}") from e3
    
    
    raise ValueError(f"Could not parse valid JSON from model output: {text[:200]}...")

def normalize_row(d: dict, source_file: str) -> list:
    """
    Build a CSV row in the exact column order, defaulting missing keys to "null".
    Also coerces non-serializable types to strings and strips whitespace.
    """
    row = []
    for col in CSV_COLUMNS:
        if col == "SourceFile":
            row.append(source_file)
            continue

        val = d.get(col)

        if val is None: 
            val = "null"
        elif isinstance(val, (dict, list)):
            
            val = json.dumps(val, ensure_ascii=False)
        else:
            val = str(val)

        
        val = val.strip()

        if col == "ClaimNumber":
           
            claim_match = re.search(r'[A-Za-z]{1,3}\d{2}-\d{5}-\d{4}', val) 
            if claim_match:
                val = claim_match.group(0)
            elif val == "" or val.lower() == "null" or val.lower() == "none" or val.isspace():
                val = "null"
        
        if not val or val.lower() in ("none", "null") or val.isspace(): 
            val = "null"

        row.append(val)
    return row

def main():
    print(f"Starting data extraction from '{PDF_FOLDER_PATH}' to '{OUTPUT_FILE_PATH}'")

    with open(OUTPUT_FILE_PATH, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(CSV_COLUMNS) 

        processed_files_count = 0
        extracted_rows_count = 0
        error_files_count = 0

        for filename in os.listdir(PDF_FOLDER_PATH):
            if not filename.lower().endswith(".pdf"):
                print(f"Skipping non-PDF file: {filename}")
                continue

            pdf_path = os.path.join(PDF_FOLDER_PATH, filename)
            print(f"\nProcessing: {filename}")
            try:
                start = time.time()
                
                raw_json_text = process_pdf_to_json_str(pdf_path, max_retries=2)
                
               
                data_list = safe_parse_model_json(raw_json_text)

                
                for incident_data in data_list:
                    row = normalize_row(incident_data, source_file=filename)
                    writer.writerow(row)
                    extracted_rows_count += 1
                
                processed_files_count += 1
                elapsed = time.time() - start
                print(f"    Done: {filename} ({len(data_list)} incident(s) extracted) ({elapsed:.2f}s)")

            except Exception as e:
                print(f"    Error on {filename}: {e}")
                error_files_count += 1
                
                empty_data = {k: "null" for k in CSV_COLUMNS if k != "SourceFile"}
                row = normalize_row(empty_data, source_file=filename)
                writer.writerow(row) 

    print(f"\n--- Extraction Summary ---")
    print(f"Total PDFs processed (attempted): {processed_files_count + error_files_count}")
    print(f"Successfully processed PDFs: {processed_files_count} files")
    print(f"       Total extracted rows: {extracted_rows_count} rows")
    print(f"Errors encountered PDFs: {error_files_count} files")
    print(f"Extraction complete! Data saved to: {OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    main()
