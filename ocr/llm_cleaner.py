import base64
import requests


OLLAMA_URL = "http://192.168.0.169:11434/api/chat"
MODEL = "gemma4:26b"


VISION_ANALYSIS_PROMPT = """You are a senior medical officer in India reviewing a photographed or scanned medical prescription.

Look carefully at the image. Read all visible text — handwritten cursive, printed text, stamps, and headers.

READING GUIDE:
- Read the doctor name and clinic/hospital name from the header
- Read patient name, age, sex, date from the top fields
- Read all medicines — they usually start with Tab., Syp., Cap., Inj.
- Read dosage patterns like 1+0+1, 2+0+2, 50mg, 2.5ml, BD, TDS, OD
- Read diagnosis from any diagnosis field or infer from medicines prescribed
- Read any instructions, follow-up notes, warnings, or advisory text

COMMON ABBREVIATIONS:
- Tab. = Tablet, Syp. = Syrup, Cap. = Capsule, Inj. = Injection
- 1+0+1 = twice daily (morning + night)
- 1+1+1 = three times daily
- 0+0+1 = once at night
- 1+0+0 = once in morning
- 2+0+2 = twice daily 2 tablets
- BD = twice daily, TDS = three times daily, OD = once daily
- SOS = as needed, HS = at bedtime
- R/A = Review after, EEG = test not medicine
- HTN = Hypertension, DM = Diabetes Mellitus

RULES:
1. Return ONLY valid JSON. No markdown. No ```json. No explanation.
2. Do NOT output <think>, thought, or any reasoning.
3. Read every medicine visible — use medical knowledge to identify partial names.
4. For diseases — extract from diagnosis field AND infer from medicines if not written.
5. For prescription_advisory and recovery — extract from image if present. If not, provide standard clinical guidance for the identified conditions.
6. icd_hint — provide ICD-10 code for each disease.
7. severity — infer from clinical notes if possible.
8. summary — one sentence describing the case.
9. Use null only when truly unreadable.

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
"""


ENRICHMENT_PROMPT_TEMPLATE = """You are a clinical knowledge assistant.

A patient has been diagnosed with: {diseases}
Prescribed medicines: {medicines}

RULES:
1. Return ONLY valid JSON. No markdown. No ```json. No explanation.
2. Do NOT output <think>, thought, or reasoning.
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



def _encode_image(image_path: str) -> str:
    """Base64-encode an image file for the Ollama API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _call_vision(image_path: str, prompt: str, timeout: int = 600, num_predict: int = 2048) -> str:
    """Send image + prompt to gemma4:26b and return raw response."""

    image_b64 = _encode_image(image_path)

    print("\n" + "=" * 60)
    print(f"VISION CALL → {MODEL}")
    print("=" * 60)
    print(prompt[:300], "...")
    print("=" * 60 + "\n")

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
    print("RAW RESPONSE FROM MODEL:")
    print("=" * 60)
    print(raw)
    print("=" * 60 + "\n")

    return raw


def _call_text(prompt: str, timeout: int = 300, num_predict: int = 1024) -> str:
    """Send a text-only prompt to gemma4:26b for enrichment."""

    print("\n" + "=" * 60)
    print(f"TEXT CALL → {MODEL}")
    print("=" * 60)
    print(prompt[:300], "...")
    print("=" * 60 + "\n")

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
    print("RAW ENRICHMENT RESPONSE:")
    print("=" * 60)
    print(raw)
    print("=" * 60 + "\n")

    return raw


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def analyze_image(image_path: str) -> str:
    """
    Send image directly to gemma4:26b for medical analysis.

    No Tesseract. No layout reconstruction. No preprocessing.
    The vision model reads the image and returns structured JSON.

    Args:
        image_path: path to the uploaded medical image

    Returns:
        Raw string from model (JSON inside)
    """
    return _call_vision(image_path, VISION_ANALYSIS_PROMPT, timeout=600, num_predict=2048)


def enrich_analysis(disease_names: list, medicines: list) -> str:
    """
    Fill null prescription_advisory and recovery fields
    using clinical knowledge of identified diseases and medicines.

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

    return _call_text(prompt, timeout=300, num_predict=1024)
