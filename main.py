import re
import uuid
import os
import json

from fastapi import FastAPI, UploadFile, File, HTTPException
from ocr.llm_cleaner import analyze_image, enrich_analysis

app = FastAPI()

UPLOAD_DIR = "uploads"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

# helper function

def validate_file_type(filename: str):
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )


def unique_file_path(filename: str) -> str:
    ext = os.path.splitext(filename)[-1].lower()
    return os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")


def parse_llm_response(raw: str) -> dict:
    """
    Robustly extract JSON from model output.
    Handles thinking blocks, code fences, and plain JSON.
    """

    text = raw

    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    text = re.sub(r"<unused\d+>\s*thought[\s\S]*?<unused\d+>", "", text)
    text = re.sub(r"<unused\d+>", "", text)
    text = text.strip()

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return {"raw_response": raw}


def merge_enrichment(base: dict, enriched: dict) -> dict:
    """
    Merge enrichment into base analysis.
    Only fills null/empty fields — never overwrites document data.
    """

    def is_empty(val):
        return val is None or val == [] or val == ""

    for section in ["prescription_advisory", "recovery"]:
        if section in enriched:
            base_sec = base.get(section, {}) or {}
            enrich_sec = enriched.get(section, {}) or {}
            for key, val in enrich_sec.items():
                if is_empty(base_sec.get(key)) and not is_empty(val):
                    base_sec[key] = val
            base[section] = base_sec

    return base



@app.get("/")
def home():
    return {"message": "Health OCR server is running — gemma4:26b"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Full pipeline:
    1. Validate file type
    2. Save with unique filename
    3. Send image directly to gemma4:26b (vision model)
    4. Parse JSON response
    5. Enrich null fields with clinical knowledge (Pass 2)
    6. Return structured JSON
    """

    # Step 1: Validate
    validate_file_type(file.filename)

    # Step 2: Save
    file_path = unique_file_path(file.filename)
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        raw_response = analyze_image(file_path)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable. Ensure Ollama is running with gemma4:26b. Error: {str(e)}"
        )


    print(raw_response,"check")

    # Step 4: Parse
    analysis = parse_llm_response(raw_response)

    if "raw_response" in analysis:
        return {
            "success": False,
            "filename": file.filename,
            "analysis": analysis,
            "note": "Model response could not be parsed as JSON."
        }

    diseases = analysis.get("diseases", [])
    medicines = analysis.get("medicines", [])

    disease_names = [
        d.get("name") for d in diseases
        if d.get("name") and d.get("name") != "null"
    ]
    medicine_names = [
        m.get("name") for m in medicines
        if m.get("name") and m.get("name") != "null"
    ]

    if disease_names or medicine_names:
        try:
            raw_enrichment = enrich_analysis(disease_names, medicines)
            enrichment = parse_llm_response(raw_enrichment)
            if "raw_response" not in enrichment:
                analysis = merge_enrichment(analysis, enrichment)
        except Exception:
            pass

    # Step 6: Respond
    return {
        "success": True,
        "filename": file.filename,
        "analysis": analysis
    }
