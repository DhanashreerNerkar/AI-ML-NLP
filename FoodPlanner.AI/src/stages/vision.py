"""Stage 1 — refrigerator vision ingestion."""

from __future__ import annotations

import base64
from pathlib import Path

from src.llm import GLOBAL_RULES, complete_json
from src.schemas import DetectedItem, DetectedItems

SYSTEM = f"""
ROLE: Food-item extraction module.
{GLOBAL_RULES}
TASK: List only food items identifiable with reasonable confidence.
CONSTRAINTS: food items only; ignore brands, containers, shelving, non-food; use lowercase generic names.
OUTPUT (JSON only):
{{"detected_items":[{{"item":"<name>","confidence":"high|medium|low"}}],"needs_user_confirmation":true}}
""".strip()


def detect_from_image_bytes(image_bytes: bytes, media_type: str = "image/jpeg") -> DetectedItems:
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    data = complete_json(
        system=SYSTEM,
        user="Extract food items from this fridge/pantry photo.",
        image_base64=b64,
        image_media_type=media_type,
    )
    return DetectedItems.model_validate(data)


def detect_from_image_path(path: str | Path) -> DetectedItems:
    path = Path(path)
    suffix = path.suffix.lower()
    media = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/jpeg")
    return detect_from_image_bytes(path.read_bytes(), media_type=media)


def detect_from_typed_inventory(raw: str | list[str]) -> DetectedItems:
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
    else:
        parts = list(raw)
    items = [
        DetectedItem(item=p.lower(), confidence="high")
        for p in parts
        if p and p.strip()
    ]
    return DetectedItems(detected_items=items, needs_user_confirmation=True)
