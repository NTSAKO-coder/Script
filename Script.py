# Run gcloud auth application-default login first
import csv
import time
import json 
import os
from vertexai.generative_models import Part
import vertexai
from vertexai.generative_models import GenerativeModel

 
PROJECT_ID = "lic-dev-dmt-dat-sci"
 
LOCATION = "us-central1"
 
MODEL_NAME = "gemini-2.5-flash"

PDF_FOLDER_PATH = r"C:\\Users\\Ntsakom\\Downloads\\Test"  
OUTPUT_FILE_PATH = r"C:\\Users\\Ntsakom\\Downloads\\DataA.csv"

os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)

CSV_COLUMNS = [
    "ClaimNumber",
    "Insured",
    "PolicyType",
    "Incident",
    "Load",                    # goods (e.g., Tomatoes)
    "Loading",                 # weight in kg
    "DateOfLoss",
    "Location",
    "ClaimCalculationTotal",
    "CargoOwner",
    "InsuranceCover",
    "AdequacyOfSumInsured",
    "SubjectMatter",
    "Conveyances",
    "Driver",
    "CircumstancesOfClaim",
    "PoliceDetails",
    "SourceFile"
]

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel(MODEL_NAME)

EXTRACT_POLICYHOLDER_METADATA = """You are a specialist in extracting information from insurance claim assessor reports.
Your task is to read the report and extract the following fields. If a field is not present, return "null".
Do not invent information. Output ONLY a single valid JSON object with exactly the keys listed.

Field definitions:
- ClaimNumber
- Insured
- PolicyType
- Incident
- Load (e.g., Tomatoes, Electronics)
- Loading (weight in kg so it must always end with a kg)
- DateOfLoss (may appear as "Date of Loss" or "bi")
- Location
- ClaimCalculationTotal (may appear as "Calculation Final Value" or "Claim Total")
- CargoOwner
- InsuranceCover (monetary value of insurance cover)
- AdequacyOfSumInsured (match the value of the load it's in rands, otherwise "null")
- SubjectMatter
- Conveyances (match the last conveyance on the last and output that)
- Driver (name of the driver, otherwise "null")
- CircumstancesOfClaim (Read the full paragraph describing the incident.Remove any repeated sentences or phrases. Summarize the paragraph into a clear 3–4 sentence description. If nothing is found, output null and explain why.CircumstancesOfClaim: Read the full paragraph describing the incident. Remove any repeated sentences or phrases. Summarize the paragraph into a clear 3–4 sentence description. If nothing is found, output null and explain why.)
- PoliceDetails station and/or case/reference number if available. If not mentioned, "null".

Rules:
- If exact values are missing, return "null".
- Do not skip fields, always include all fields.
- All monetary amounts must be written in Rand (R)
- Normalize dates to YYYY-MM-DD when possible.
- Summarize circumstances if described.
- Always return results strictly in valid JSON.
"""

def process_pdf_to_json_str(pdf_path: str) -> str:
    """Sends a single PDF to the model and returns the raw JSON string the model outputs."""
    with open(pdf_path, "rb") as f:
        document = Part.from_data(data=f.read(), mime_type="application/pdf")

    response = model.generate_content(
        [EXTRACT_POLICYHOLDER_METADATA, document],
        generation_config={"temperature": 0.1}
    )
   
    return response.text

def safe_parse_model_json(text: str) -> dict:
    """
    Attempts to parse JSON from the model output.
    If the model wraps JSON in code fences or adds text, try to extract the JSON object.
    """
    text = text.strip()
  
    if text.startswith("```"):
       
        text = text.split("```", 1)[1]
        
        text = text.split("\n", 1)[1] if "\n" in text else text
       
        if "```" in text:
            text = text.rsplit("```", 1)[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            return json.loads(candidate)
      
        raise

def normalize_row(d: dict, source_file: str) -> list:
    """
    Build a CSV row in the exact column order, defaulting missing keys to "null".
    Also coerce non-serializable types to strings.
    """
    row = []
    for col in CSV_COLUMNS:
        if col == "SourceFile":
            row.append(source_file)
            continue
        val = d.get(col, "null")
       
        if val is None:
            val = "null"
        elif isinstance(val, (dict, list)):
            val = json.dumps(val, ensure_ascii=False)
        else:
            val = str(val)
        row.append(val)
    return row

def main():
   
    with open(OUTPUT_FILE_PATH, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(CSV_COLUMNS)

       
        for filename in os.listdir(PDF_FOLDER_PATH):
            if not filename.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(PDF_FOLDER_PATH, filename)
            print(f"\nProcessing: {filename}")
            try:
                start = time.time()
                raw_json_text = process_pdf_to_json_str(pdf_path)

              
                data = safe_parse_model_json(raw_json_text)

                row = normalize_row(data, source_file=filename)
                writer.writerow(row)

                elapsed = time.time() - start
                print(f" Done: {filename}  ({elapsed:.2f}s)")
               
            except Exception as e:
                
                print(f" Error on {filename}: {e}")
                empty_data = {k: "null" for k in CSV_COLUMNS if k != "SourceFile"}
                row = normalize_row(empty_data, source_file=filename)
                writer.writerow(row)

    print(f"\n Extraction complete! Data saved to: {OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    main()