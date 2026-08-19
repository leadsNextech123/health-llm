import re
import uuid
import os
import json

from fastapi import FastAPI, UploadFile, File, HTTPException
from ocr.engine import extract_ocr_data
from ocr.llm_cleaner import analyze_ocr_blocks, enrich_analysis

app = FastAPI()

UPLOAD_DIR = "uploads"

# Supported image formats for Tesseract
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def validate_file_type(filename: str):
    """Raise 400 if the uploaded file is not a supported image."""
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )


def unique_file_path(filename: str) -> str:
    """Unique filename to prevent collisions between uploads."""
    ext = os.path.splitext(filename)[-1].lower()
    return os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")


def parse_llm_response(raw: str) -> dict:
    """
    Robustly extract JSON from LLM output.

    Handles:
    - <unusedXX>thought...</unusedXX>  thinking blocks
    - ```json ... ```                  code fences
    - Plain JSON with no wrapper
    - Truncated JSON
    """

    text = raw

    # Strip thinking blocks: <unused94>thought ... <unused95>
    text = re.sub(r"<unused\d+>\s*thought[\s\S]*?<unused\d+>", "", text)
    text = re.sub(r"<unused\d+>", "", text)
    text = text.strip()

    # Strip ```json ... ``` or ``` ... ``` fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find outermost { ... } block
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
    Merge Pass 2 enrichment into Pass 1 result.
    Only fills fields that are null/empty — never overwrites document data.
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


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/")
def home():
    return {"message": "Health OCR server is running"}


@app.post("/ocr")
async def ocr_only(file: UploadFile = File(...)):
    """
    OCR only — returns raw Tesseract blocks. No LLM.
    """

    validate_file_type(file.filename)

    file_path = unique_file_path(file.filename)
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        ocr_blocks = extract_ocr_data(file_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"OCR failed: {str(e)}")

    return {
        "success": True,
        "filename": file.filename,
        "block_count": len(ocr_blocks),
        "blocks": ocr_blocks
    }


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Full pipeline:
    1. Validate file type
    2. Save with unique filename
    3. Tesseract OCR → raw word blocks with positions
    4. LLM Pass 1 — extract medical info from OCR blocks
    5. LLM Pass 2 — enrich null fields with clinical knowledge
    6. Return structured JSON
    """

    # Step 1: Validate
    validate_file_type(file.filename)

    # Step 2: Save
    file_path = unique_file_path(file.filename)
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    # Step 3: Tesseract OCR
    try:
        ocr_blocks = extract_ocr_data(file_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"OCR failed: {str(e)}")

    if not ocr_blocks:
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted. Check image quality."
        )

    # Step 4: LLM Pass 1 — extract from OCR blocks (with vision fallback)
    try:
        raw_response = analyze_ocr_blocks(ocr_blocks, image_path=file_path)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"LLM unavailable. Ensure Ollama is running. Error: {str(e)}"
        )

    analysis = parse_llm_response(raw_response)

    if "raw_response" in analysis:
        return {
            "success": False,
            "filename": file.filename,
            "block_count": len(ocr_blocks),
            "analysis": analysis,
            "note": "LLM response could not be parsed as JSON."
        }

    # Step 5: LLM Pass 2 — enrich nulls with clinical knowledge
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
            # Enrichment is best-effort — never fail the whole request
            pass

    # Step 6: Respond
    return {
        "success": True,
        "filename": file.filename,
        "block_count": len(ocr_blocks),
        "analysis": analysis
    }
