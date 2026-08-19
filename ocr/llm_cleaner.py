import json
import base64
import requests
from ocr.layout import reconstruct_layout
from ocr.ocr_cleaner import clean_ocr_text


OLLAMA_URL = "http://192.168.0.169:11434/api/chat"
TEXT_MODEL  = "qwen3:14b"
VISION_MODEL = "gemma4:26b"

# If OCR average confidence is below this, use vision model directly
OCR_CONFIDENCE_THRESHOLD = 0.55


# ------------------------------------------------------------------
# PROMPT 1: Medical Analysis from raw OCR blocks
#
# The LLM receives raw word-level OCR blocks with:
# - text: the recognized word
# - confidence: 0.0 to 1.0
# - bounding_box: [x1, y1, x2, y2] pixel position on page
#
# The LLM uses bounding box positions to infer layout:
# - Similar Y values = same line
# - Low X = left side, High X = right side
# - Low Y = top of page (header), High Y = bottom
# ------------------------------------------------------------------

ANALYSIS_PROMPT = """You are a senior medical officer in India reviewing a scanned or photographed prescription.

The document was OCR-extracted and may have noise, garbled characters, and recognition errors due to handwriting or camera angle.
Use your medical knowledge to interpret partial or garbled text in context.

LAYOUT GUIDE:
- [HEADER] = clinic name, doctor name, qualifications
- [LEFT_COLUMN] = consultation hours, patient visit info
- [RIGHT_COLUMN] = address, date, diagnosis notes
- [CENTER_CONTENT] = medicines, dosage, clinical notes

OCR ERROR CORRECTION — apply these automatically:
- "Lacosam", "LACOSAM", "Lacosamide" → Lacosamide (anti-epileptic)
- "Lobazam", "Lobazam MD", "Clobazam" → Clobazam (anti-epileptic)
- "Sizodon", "SiZODoN" → Risperidone
- "Qutipin", "Quetipin" → Quetiapine
- "Ativan", "ATivan" → Lorazepam
- "Rivotil", "Riv0til" → Clonazepam
- "Diclofinae", "Diclofenac" → Diclofenac
- "Paracetmol" → Paracetamol
- "Amoxcillin" → Amoxicillin
- "Eltracal-D", "Ultracal" → Calcium + Vitamin D3
- "Cartilix", "Cartligen" → Glucosamine + Chondroitin
- "Dp Sciatry", "Sciatic" → Sciatica
- "HTN" → Hypertension
- "DM" → Diabetes Mellitus
- "Syp." = Syrup (route: oral)
- "Tab." = Tablet (route: oral)
- "Cap." = Capsule (route: oral)
- "Inj." = Injection (route: injection)
- "R/A" = Review/Advice after
- "EEG" = Electroencephalogram (test, not medicine)
- "1+0+1" → twice daily (morning + night)
- "1+1+1" → three times daily
- "0+0+1" → once at night
- "1+0+0" → once in morning
- "2+0+2" → twice daily (2 tablets)
- "BD" → twice daily
- "TDS" → three times daily
- "OD" → once daily
- "HS" → at bedtime
- "SOS" → as needed
- "x 4 months", "X4 mon" → duration: 4 months

RULES:
1. Return ONLY valid JSON. No markdown, no ```json, no explanation.
2. Do NOT output <think>, thought, or reasoning.
3. Extract every medicine visible — even if partially garbled, use context to identify it.
4. For diseases — extract from diagnosis notes AND infer from medicine names (e.g. Lacosamide → Epilepsy).
5. For prescription_advisory and recovery — extract from document if mentioned. If not, provide standard clinical guidance for the identified conditions.
6. icd_hint — provide ICD-10 code for each disease.
7. severity — infer from document context if possible (e.g. "overall much better" → improving/mild).
8. summary — write one sentence describing the case.
9. Use null only when truly unextractable.

Return exactly this structure:

{
  "diseases": [
    {
      "name": null,
      "icd_hint": null,
      "severity": null,
      "notes": null
    }
  ],
  "medicines": [
    {
      "name": null,
      "dosage": null,
      "frequency": null,
      "duration": null,
      "route": "oral",
      "notes": null
    }
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

MEDICAL DOCUMENT:
"""


# ------------------------------------------------------------------
# PROMPT 2: Clinical Enrichment — Pass 2
#
# Fills prescription_advisory and recovery fields that
# were null after Pass 1, using clinical knowledge.
# ------------------------------------------------------------------

ENRICHMENT_PROMPT_TEMPLATE = """You are a clinical knowledge assistant.

A patient has been diagnosed with: {diseases}
Prescribed medicines: {medicines}

IMPORTANT RULES:
1. Return ONLY valid JSON. No markdown. No ```json. No explanation.
2. Do NOT output <think>, thought, plan, or any internal reasoning.
3. Use standard clinical knowledge for this condition and these medicines.
4. All fields are required — do not use null.
5. Keep responses concise and practical.

Return exactly this structure:

{{
  "prescription_advisory": {{
    "instructions": null,
    "warnings": null,
    "precautions": null,
    "follow_up": null
  }},
  "recovery": {{
    "expected_duration": null,
    "lifestyle_advice": null,
    "diet_advice": null,
    "activity_restrictions": null
  }}
}}
"""


# ------------------------------------------------------------------
# Shared Ollama request helper
# ------------------------------------------------------------------

def _call_ollama(prompt: str, timeout: int = 600, num_predict: int = 2048) -> str:
    """Send a text prompt to the text LLM and return the raw response."""

    print("\n" + "=" * 60)
    print("PROMPT SENT TO LLM:")
    print("=" * 60)
    print(prompt)
    print("=" * 60 + "\n")

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": TEXT_MODEL,
            "think": False,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a medical assistant. Return only valid JSON. No markdown. No thinking. No explanation."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "options": {
                "temperature": 0,
                "num_predict": num_predict
            }
        },
        timeout=timeout
    )

    response.raise_for_status()
    raw = response.json()["message"]["content"]

    print("\n" + "=" * 60)
    print("RAW RESPONSE FROM LLM:")
    print("=" * 60)
    print(raw)
    print("=" * 60 + "\n")

    return raw


def _call_vision(image_path: str, prompt: str, timeout: int = 600, num_predict: int = 2048) -> str:
    """Send image + prompt to the vision LLM (gemma4:26b)."""

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    print("\n" + "=" * 60)
    print(f"VISION MODEL CALL: {VISION_MODEL}")
    print("=" * 60)
    print(prompt[:500])
    print("=" * 60 + "\n")

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": VISION_MODEL,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a medical assistant. Return only valid JSON. No markdown. No thinking. No explanation."
                },
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64]
                }
            ],
            "options": {
                "temperature": 0,
                "num_predict": num_predict
            }
        },
        timeout=timeout
    )

    response.raise_for_status()
    raw = response.json()["message"]["content"]

    print("\n" + "=" * 60)
    print("RAW VISION RESPONSE:")
    print("=" * 60)
    print(raw)
    print("=" * 60 + "\n")

    return raw


# ------------------------------------------------------------------
# Function 1: Analyze raw OCR blocks (Pass 1)
# ------------------------------------------------------------------

def analyze_ocr_blocks(ocr_blocks: list, image_path: str = None) -> str:
    """
    Pass 1: Extract medical info from the document.

    Strategy:
    - Clean OCR blocks → layout reconstruction → text LLM (qwen3:14b)
    - If OCR confidence is too low (handwriting/poor scan),
      fall back to vision LLM (gemma4:26b) with the original image.

    Args:
        ocr_blocks: list of { text, confidence, bounding_box } dicts
        image_path: original image path (used for vision fallback)

    Returns:
        Raw string from model (JSON inside)
    """

    # Check average OCR confidence
    if ocr_blocks:
        avg_conf = sum(b["confidence"] for b in ocr_blocks) / len(ocr_blocks)
    else:
        avg_conf = 0.0

    print(f"Average OCR confidence: {avg_conf:.2f} (threshold: {OCR_CONFIDENCE_THRESHOLD})")

    # --- Vision fallback for low-confidence OCR (handwriting / phone photos) ---
    if avg_conf < OCR_CONFIDENCE_THRESHOLD and image_path:
        print("Low OCR confidence → using vision model directly")
        return _call_vision(image_path, ANALYSIS_PROMPT, timeout=600, num_predict=2048)

    # --- Standard text path ---
    # Step 1: Clean OCR blocks
    cleaned_blocks = []
    for block in ocr_blocks:
        cleaned_blocks.append({
            "text": clean_ocr_text(block["text"]),
            "confidence": block["confidence"],
            "bounding_box": block["bounding_box"]
        })

    # Step 2: Reconstruct layout
    layout_result = reconstruct_layout(cleaned_blocks)
    document_text = layout_result["text"]

    print("\n" + "=" * 60)
    print("RECONSTRUCTED DOCUMENT TEXT:")
    print("=" * 60)
    print(document_text)
    print("=" * 60 + "\n")

    # Step 3: Send to text LLM
    prompt = ANALYSIS_PROMPT + "\n\n" + document_text
    return _call_ollama(prompt, timeout=600, num_predict=2048)


# ------------------------------------------------------------------
# Function 2: Clinical Enrichment (Pass 2)
# ------------------------------------------------------------------

def enrich_analysis(disease_names: list, medicines: list) -> str:
    """
    Pass 2: Fill null prescription_advisory and recovery fields
    using clinical knowledge of the identified diseases and medicines.

    Runs only when Pass 1 left those fields null because the
    document didn't mention them explicitly.

    Args:
        disease_names: list of disease name strings from Pass 1
        medicines:     list of medicine dicts from Pass 1

    Returns:
        Raw string from model (JSON inside)
    """

    diseases_str = ", ".join(disease_names) if disease_names else "unknown condition"

    medicine_names = [
        m.get("name") for m in medicines
        if m.get("name") and m.get("name") != "null"
    ]
    medicines_str = ", ".join(medicine_names) if medicine_names else "not specified"

    prompt = ENRICHMENT_PROMPT_TEMPLATE.format(
        diseases=diseases_str,
        medicines=medicines_str
    )

    return _call_ollama(prompt, timeout=300, num_predict=1024)
