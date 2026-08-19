from ocr.engine import extract_ocr_data
from ocr.layout import reconstruct_layout


image_path = "uploads/test1_medical_report.webp"

ocr_results = extract_ocr_data(image_path)

lines = reconstruct_layout(ocr_results)

for line in lines:

    texts = [
        item["text"]
        for item in line
    ]

    print(" | ".join(texts))

    