"""Shared paths and KB/fixture loaders."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIXTURES = DATA / "fixtures"


def _load_json(path: Path) -> dict | list:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_shelf_life_kb() -> dict:
    return _load_json(DATA / "shelf_life_kb.json")


@lru_cache(maxsize=1)
def load_substitution_kb() -> dict:
    return _load_json(DATA / "substitution_kb.json")


@lru_cache(maxsize=1)
def load_derivation_kb() -> dict:
    return _load_json(DATA / "derivation_kb.json")


@lru_cache(maxsize=1)
def load_pantry_staples() -> set[str]:
    data = _load_json(DATA / "pantry_staples.json")
    return {s.lower().strip() for s in data.get("staples", [])}


@lru_cache(maxsize=1)
def load_fixture_recipes() -> list[dict]:
    return _load_json(FIXTURES / "recipes.json")


def normalize_name(name: str) -> str:
    text = " ".join(name.lower().strip().replace("_", " ").split())
    # Light singularization so lemon/lemons and tomato/tomatoes align
    if text.endswith("oes") and len(text) > 4:
        return text[:-2]
    if text.endswith("ies") and len(text) > 4:
        return text[:-3] + "y"
    if text.endswith("s") and not text.endswith("ss") and len(text) > 3:
        return text[:-1]
    return text
