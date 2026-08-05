"""Claude client helpers for structured JSON stages."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

load_dotenv()

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None


def get_model() -> str:
    return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")


def get_client() -> Any:
    if anthropic is None:
        raise RuntimeError("anthropic package not installed")
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key or key == "your_anthropic_api_key_here":
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return anthropic.Anthropic(api_key=key)


def extract_json(text: str) -> Union[dict, list]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def complete_json(
    *,
    system: str,
    user: str,
    temperature: Optional[float] = None,
    max_tokens: int = 2048,
    image_base64: Optional[str] = None,
    image_media_type: str = "image/jpeg",
) -> Union[dict, list]:
    """Call Claude and parse JSON.

    Newer Anthropic models reject `temperature`, so it is omitted unless
    explicitly provided via the temperature argument.
    """
    client = get_client()
    content: List[dict] = []
    if image_base64:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_media_type,
                    "data": image_base64,
                },
            }
        )
    content.append({"type": "text", "text": user})

    kwargs: Dict[str, Any] = {
        "model": get_model(),
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": content}],
    }
    # Only send temperature when callers opt in — many current models reject it.
    if temperature is not None:
        kwargs["temperature"] = temperature

    msg = client.messages.create(**kwargs)
    text = "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")
    return extract_json(text)


GLOBAL_RULES = """
GLOBAL RULES (apply always):
1. SAFETY IS A HARD CONSTRAINT. Never output a plan, recipe, or substitute containing an item on dietary_restrictions or dislikes.
2. GROUND, DON'T INVENT. Use only provided references and recipe candidates.
3. Return valid JSON matching the requested schema. No markdown fences unless asked.
4. If you cannot produce a grounded, compliant answer, return the refusal object specified.
5. Do not infer an item's exact age or expiry from an image; score ingredient TYPE only when scoring.
""".strip()
