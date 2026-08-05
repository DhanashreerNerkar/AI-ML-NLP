"""Spoonacular recipe search + diet filtering."""

from __future__ import annotations

import os

import httpx

from src.diet_filter import filter_recipe_candidates
from src.kb import load_fixture_recipes
from src.schemas import RecipeCandidate, UserProfile

SPOONACULAR_SEARCH = "https://api.spoonacular.com/recipes/findByIngredients"
SPOONACULAR_INFO = "https://api.spoonacular.com/recipes/{id}/information"


def _api_key() -> str | None:
    return os.getenv("SPOONACULAR_API_KEY") or None


def _clean_ingredient_name(raw: str) -> str:
    text = raw.strip().lower()
    # Spoonacular sometimes embeds instructions; keep short heads
    if "." in text and len(text) > 40:
        text = text.split(".")[0]
    return " ".join(text.split())[:80]


def _candidate_from_fixture(row: dict) -> RecipeCandidate:
    return RecipeCandidate(
        id=row.get("id"),
        title=row["title"],
        ingredients=[_clean_ingredient_name(i) for i in row.get("ingredients", [])],
        ready_in_minutes=row.get("ready_in_minutes"),
        steps=row.get("steps", []),
        source=row.get("source", "fixture"),
    )


def search_by_ingredients(
    ingredients: list[str],
    profile: UserProfile,
    number: int = 8,
    use_fixtures_on_failure: bool = True,
) -> list[RecipeCandidate]:
    key = _api_key()
    candidates: list[RecipeCandidate] = []

    if key and key != "your_spoonacular_api_key_here":
        try:
            candidates = _search_live(ingredients, key, number=number)
        except Exception:
            candidates = []

    if not candidates and use_fixtures_on_failure:
        candidates = [_candidate_from_fixture(r) for r in load_fixture_recipes()]

    # Enforce hard constraints before candidates are supplied to the planner.
    # The LLM receives only recipe options that have passed this deterministic
    # check; stage_plan applies a second check to its final output.
    return filter_recipe_candidates(candidates, profile)


def _search_live(ingredients: list[str], api_key: str, number: int = 8) -> list[RecipeCandidate]:
    params = {
        "apiKey": api_key,
        "ingredients": ",".join(ingredients),
        "number": number,
        "ranking": 1,
        "ignorePantry": True,
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(SPOONACULAR_SEARCH, params=params)
        resp.raise_for_status()
        rows = resp.json()

        out: list[RecipeCandidate] = []
        for row in rows:
            ing = []
            for bucket in ("usedIngredients", "missedIngredients", "unusedIngredients"):
                for item in row.get(bucket, []) or []:
                    name = item.get("name") or item.get("original") or ""
                    if name:
                        ing.append(_clean_ingredient_name(name))
            # Deduplicate while preserving order
            seen = set()
            cleaned = []
            for i in ing:
                if i not in seen:
                    seen.add(i)
                    cleaned.append(i)

            steps: list[str] = []
            rid = row.get("id")
            if rid:
                try:
                    info = client.get(
                        SPOONACULAR_INFO.format(id=rid),
                        params={"apiKey": api_key},
                    )
                    if info.status_code == 200:
                        data = info.json()
                        for block in data.get("analyzedInstructions", []) or []:
                            for step in block.get("steps", []) or []:
                                s = (step.get("step") or "").strip()
                                if s:
                                    steps.append(s)
                        # Prefer extended ingredients if present
                        ext = []
                        for item in data.get("extendedIngredients", []) or []:
                            name = item.get("name") or item.get("original") or ""
                            if name:
                                ext.append(_clean_ingredient_name(name))
                        if ext:
                            cleaned = list(dict.fromkeys(ext))
                except Exception:
                    pass

            out.append(
                RecipeCandidate(
                    id=rid,
                    title=row.get("title") or "Untitled recipe",
                    ingredients=cleaned,
                    ready_in_minutes=row.get("readyInMinutes"),
                    steps=steps[:8],
                    source="spoonacular",
                )
            )
        return out
