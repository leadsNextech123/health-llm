from typing import List, Dict, Any


def get_box_info(item: Dict[str, Any]):
    """
    Extract geometric information from PaddleOCR bounding box.

    bounding_box:
        [x1, y1, x2, y2]
    """

    x1, y1, x2, y2 = item["bounding_box"]

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "center_x": (x1 + x2) / 2,
        "center_y": (y1 + y2) / 2,
        "height": y2 - y1,
        "width": x2 - x1,
    }


def group_into_lines(
    results: List[Dict[str, Any]],
    y_tolerance: float = 15
):
    """
    Group OCR boxes that belong to the same horizontal line.
    """

    items = []

    for item in results:

        geometry = get_box_info(item)

        items.append({
            **item,
            **geometry
        })

    # Start from top of document
    items.sort(key=lambda x: x["center_y"])

    lines = []

    for item in items:

        placed = False

        for line in lines:

            # Average Y position of current line
            line_y = sum(
                x["center_y"] for x in line
            ) / len(line)

            if abs(item["center_y"] - line_y) <= y_tolerance:

                line.append(item)
                placed = True
                break

        if not placed:
            lines.append([item])

    # Sort each line left → right
    for line in lines:
        line.sort(key=lambda x: x["x1"])

    # Sort lines top → bottom
    lines.sort(
        key=lambda line: min(
            x["center_y"] for x in line
        )
    )

    return lines


def detect_columns(
    lines,
    column_gap: float = 100
):
    """
    Detect multiple columns in the document.

    Example:

    LEFT COLUMN             RIGHT COLUMN

    Consultation            Mudichur Road
    Monday                  Krishna Nagar
    7 PM                    West Tambaram
    """

    sections = []

    for line in lines:

        # Sort left → right
        line.sort(key=lambda x: x["x1"])

        current_section = []

        previous_x2 = None

        for item in line:

            if previous_x2 is not None:

                gap = item["x1"] - previous_x2

                if gap > column_gap:

                    if current_section:
                        sections.append(current_section)

                    current_section = []

            current_section.append(item)

            previous_x2 = item["x2"]

        if current_section:
            sections.append(current_section)

    return sections


def reconstruct_layout(ocr_results: List[Dict]) -> Dict:
    """
    Reconstruct OCR text into a more natural reading order.
    Tuned for Tesseract word-level output (larger Y tolerance).
    """

    if not ocr_results:
        return {"sections": [], "text": ""}

    items = []
    for item in ocr_results:
        x1, y1, x2, y2 = item["bounding_box"]
        items.append({
            "text": item["text"].strip(),
            "confidence": item["confidence"],
            "bounding_box": item["bounding_box"],
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
            "width": x2 - x1,
            "height": y2 - y1
        })

    items = [i for i in items if i["text"]]
    if not items:
        return {"sections": [], "text": ""}

    page_width = max(i["x2"] for i in items)
    page_height = max(i["y2"] for i in items)

    # ----------------------------------------------------------
    # Group words into lines
    # Use larger Y tolerance (20px) since Tesseract gives words
    # ----------------------------------------------------------
    items_sorted = sorted(items, key=lambda x: x["cy"])
    lines = []

    for item in items_sorted:
        placed = False
        for line in lines:
            line_y = sum(x["cy"] for x in line) / len(line)
            if abs(item["cy"] - line_y) <= 20:
                line.append(item)
                placed = True
                break
        if not placed:
            lines.append([item])

    for line in lines:
        line.sort(key=lambda x: x["x1"])
    lines.sort(key=lambda line: min(x["cy"] for x in line))

    # ----------------------------------------------------------
    # Detect header: top 15% of page
    # ----------------------------------------------------------
    header_threshold = page_height * 0.15
    header_lines = [l for l in lines if min(x["cy"] for x in l) < header_threshold]
    body_lines   = [l for l in lines if min(x["cy"] for x in l) >= header_threshold]

    # ----------------------------------------------------------
    # Detect columns in body
    # ----------------------------------------------------------
    left_lines   = []
    right_lines  = []
    center_lines = []

    for line in body_lines:
        line_cx = sum(x["cx"] for x in line) / len(line)
        if line_cx < page_width * 0.38:
            left_lines.append(line)
        elif line_cx > page_width * 0.62:
            right_lines.append(line)
        else:
            center_lines.append(line)

    has_columns = len(left_lines) >= 2 and len(right_lines) >= 2

    def lines_to_text(lines_list):
        return "\n".join(" ".join(w["text"] for w in line) for line in lines_list)

    sections = []

    if header_lines:
        sections.append({"section": "header",       "text": lines_to_text(header_lines)})

    if has_columns:
        if left_lines:
            sections.append({"section": "left_column",  "text": lines_to_text(left_lines)})
        if right_lines:
            sections.append({"section": "right_column", "text": lines_to_text(right_lines)})
        if center_lines:
            sections.append({"section": "center_content", "text": lines_to_text(center_lines)})
    else:
        all_body = sorted(
            left_lines + center_lines + right_lines,
            key=lambda line: min(x["cy"] for x in line)
        )
        if all_body:
            sections.append({"section": "document", "text": lines_to_text(all_body)})

    final_text = "\n\n".join(
        f"[{s['section'].upper()}]\n{s['text']}" for s in sections
    )

    return {"sections": sections, "text": final_text}