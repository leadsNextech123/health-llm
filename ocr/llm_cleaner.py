import base64
import io
import requests


OLLAMA_URL = "http://192.168.0.169:11434/api/chat"
MODEL = "gemma4:26b"

# Max image dimension before resizing — larger images = slower model
MAX_IMAGE_SIZE = 1024


# ------------------------------------------------------------------
# Single prompt — extracts + enriches in one call
# No Pass 2 needed
# ------------------------------------------------------------------

VISION_PROMPT = """You are a senior medical officer in India reviewing a photographed or scanned medical prescription.

Look carefully at the image. Read all visible text — handwritten, printed, stamps, headers.

ABBREVIATIONS:
- Tab.=Tablet, Syp.=Syrup, Cap.=Capsule, Inj.=Injection
- 1+0+1=twice daily, 1+1+1=three times daily, 0+0+1=night only, 1+0+0=morning only
- BD=twice daily, TDS=three times daily, OD=once daily, SOS=as needed, HS=bedtime
- HTN=Hypertension, DM=Diabetes Mellitus, R/A=Review after, EEG=test not medicine

RULES:
1. Return ONLY valid JSON. No markdown. No ```json. No explanation. No reasoning.
2. Extract every medicine visible — use medical knowledge to identify partial names.
3. diseases — extract from diagnosis field OR infer from medicines prescribed.
4. icd_hint — ICD-10 code for each disease.
5. prescription_advisory and recovery — extract from image if present, otherwise use standard clinical knowledge for the identified conditions and medicines. Do NOT leave these null.
6. summary — one sentence describing the case.
7. Use null only when truly unreadable.

Return exactly this JSON:

{
  "diseases": [
    {"name": null, "icd_hint": null, "severity": null, "notes": null}
  ],
  "medicines": [
    {"name": null, "dosage": null, "frequency": null, "duration": null, "route": "oral", "notes": null}
  ],
  "prescription_advisory": {
    "instructions": null,
    "warnings": null,
    "precautions": null,
    "follow_up": null
  },
  "recovery": {
    "expected_duration": null,
    "lifestyle_advice": null,
    "diet_advice": null,
    "activity_restrictions": null
  },
  "summary": null
}
"""


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _resize_and_encode(image_path: str) -> str:
    """
    Resize image to MAX_IMAGE_SIZE on the longest side
    before base64 encoding.

    Smaller image = faster vision model processing.
    Most prescriptions are readable at 1024px.
    """
    try:
        import cv2
        import numpy as np

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("cv2 could not read image")

        h, w = img.shape[:2]
        longest = max(h, w)

        if longest > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / longest
            img = cv2.resize(
                img,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA
            )

        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    except Exception:
        # cv2 not available — encode original file as-is
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def analyze_image(image_path: str) -> str:
    """
    Send image directly to gemma4:26b.

    Single call — extracts diseases, medicines, advisory,
    and recovery all at once. No second enrichment call needed.

    Args:
        image_path: path to the uploaded medical image

    Returns:
        Raw string from model (JSON inside)
    """

    image_b64 = _resize_and_encode(image_path)

    print(f"\n>>> Sending to {MODEL} ...")

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a medical assistant. Return only valid JSON. No markdown. No thinking. No explanation."
                },
                {
                    "role": "user",
                    "content": VISION_PROMPT,
                    "images": [image_b64]
                }
            ],
            "options": {
                "temperature": 0,
                "num_predict": 800
            }
        },
        timeout=600
    )

    response.raise_for_status()
    raw = response.json()["message"]["content"]

    print("\n" + "=" * 60)
    print("MODEL RESPONSE:")
    print("=" * 60)
    print(raw)
    print("=" * 60 + "\n")

    return raw
