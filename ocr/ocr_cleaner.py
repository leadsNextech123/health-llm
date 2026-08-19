import re


def clean_ocr_text(text: str) -> str:
    """
    Conservative OCR text cleaning.

    IMPORTANT:
    - Does NOT guess medical terms.
    - Does NOT correct medicines.
    - Does NOT correct diseases.
    - Does NOT alter names, dates, dosages, or numbers.
    """

    if not text:
        return ""

    # --------------------------------------------------
    # 1. Basic whitespace cleanup
    # --------------------------------------------------

    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)

    # Remove spaces around common separators
    text = re.sub(r"\s*\|\s*", " | ", text)

    # --------------------------------------------------
    # 2. Fix obvious formatting issues
    # --------------------------------------------------

    # Example:
    # "M.B.B.S., M.D., MsC."
    # We can normalize academic abbreviation formatting.
    text = re.sub(
        r"\bMsC\.",
        "MSc.",
        text,
        flags=re.IGNORECASE
    )

    # Normalize common degree formatting
    text = re.sub(r"\bMBBS\b", "M.B.B.S.", text, flags=re.IGNORECASE)
    text = re.sub(r"\bMD\b", "M.D.", text)

    # --------------------------------------------------
    # 3. Normalize AM / PM formatting
    # --------------------------------------------------

    text = re.sub(
        r"\b(\d{1,2})\s*[Aa]\s*[Mm]\b",
        r"\1 AM",
        text
    )

    text = re.sub(
        r"\b(\d{1,2})\s*[Pp]\s*[Mm]\b",
        r"\1 PM",
        text
    )

    # --------------------------------------------------
    # 4. Remove obvious OCR garbage characters
    # --------------------------------------------------

    # Keep medical symbols such as:
    # +, -, /, %, (), :, ., #, α
    #
    # We intentionally DO NOT remove unknown characters
    # because they may contain medical information.

    # --------------------------------------------------
    # 5. Clean line endings
    # --------------------------------------------------

    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Remove excessive spaces again
        line = re.sub(r"\s{2,}", " ", line)

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)