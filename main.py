import re
import uuid
import os
import json

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ocr.llm_cleaner import analyze_image

load_dotenv()

app = FastAPI()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
CORS_ORIGIN = os.getenv("CORS_ORIGIN", "http://localhost:8080")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGIN],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_DIR, exist_ok=True)


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


@app.get("/")

def home():
    return {"message": "Health LLM Application server is running — gemma4:26b"}


@app.post("/analyze")

# Step 1
async def analyze(file: UploadFile = File(...)):

    # Step 2
    validate_file_type(file.filename)
    
    # Step 3
    file_path = unique_file_path(file.filename)
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    # Send to gemma4:26b
    # Step 4
    try:
        raw_response = analyze_image(file_path)
        print(raw_response, 'raw_response')
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable. Ensure Ollama is running with gemma4:26b. Error: {str(e)}"
        )

    # Step 5 
    analysis = parse_llm_response(raw_response)

    if "raw_response" in analysis:
        return {
            "success": False,
            "filename": file.filename,
            "analysis": analysis,
            "note": "Model response could not be parsed as JSON."
        }

    return {
        "success": True,
        "filename": file.filename,
        "analysis": analysis
    }
