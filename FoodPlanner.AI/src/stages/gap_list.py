"""Stage 6 — deterministic gap-only shopping list."""

from __future__ import annotations

from typing import List, Optional, Set, Union

from src.kb import load_derivation_kb, load_pantry_staples, normalize_name
from src.schemas import GapList, MealPlan, ShoppingListItem

CATEGORY_MAP = {
    "lemon": "produce",
    "tomato": "produce",
    "spinach": "produce",
    "onion": "produce",
    "garlic": "produce",
    "mushroom": "produce",
    "milk": "dairy_or_alternatives",
    "yogurt": "dairy_or_alternatives",
    "curd": "dairy_or_alternatives",
    "paneer": "dairy_or_alternatives",
    "cheese": "dairy_or_alternatives",
    "coconut milk": "dairy_or_alternatives",
    "chicken": "protein",
    "tofu": "protein",
    "egg": "protein",
    "rice": "grains_and_bakery",
    "bread": "grains_and_bakery",
    "pasta": "grains_and_bakery",
    "chickpea": "canned_and_packaged",
    "lentil": "canned_and_packaged",
}


def _normalize_set(items: Union[List[str], Set[str]]) -> Set[str]:
    return {normalize_name(i) for i in items if i and normalize_name(i)}


def _category_for(name: str) -> str:
    n = normalize_name(name)
    for key, cat in CATEGORY_MAP.items():
        if key in n:
            return cat
    return "other"


def _homemade_available(name: str) -> bool:
    kb = load_derivation_kb()
    n = normalize_name(name)
    for entry in kb.get("entries", []):
        names = [entry.get("derivative", ""), *entry.get("aliases", [])]
        if any(normalize_name(x) == n for x in names if x):
            return True
    return False


def compute_gap_list(
    plan: Union[MealPlan, dict],
    inventory: List[str],
    pantry_staples: Optional[Set[str]] = None,
    purchased_or_made: Optional[List[str]] = None,
) -> GapList:
    if isinstance(plan, dict):
        plan = MealPlan.model_validate(plan)

    staples = pantry_staples if pantry_staples is not None else load_pantry_staples()
    staples_n = _normalize_set(staples)
    inv = _normalize_set(inventory)
    made = _normalize_set(purchased_or_made or [])

    required: Set[str] = set()
    meal_map = {}
    for meal in plan.plan:
        label = f"Day {meal.day or meal.night} {meal.meal_type}"
        for group in (
            meal.ingredients_from_inventory,
            meal.extra_pantry_items,
            meal.missing_ingredients,
        ):
            for raw in group:
                n = normalize_name(raw)
                if not n or len(n.split()) > 4:
                    continue
                required.add(n)
                meal_map.setdefault(n, []).append(label)

    gaps = sorted(required - inv - staples_n - made)
    already = sorted((required & inv) | (required & staples_n) | (required & made))

    items = []
    for g in gaps:
        items.append(
            ShoppingListItem(
                normalized_name=g,
                display_name=g.title(),
                category=_category_for(g),
                required_for_meals=sorted(set(meal_map.get(g, []))),
                homemade_option_available=_homemade_available(g),
            )
        )

    markdown = _format_gaps_markdown(items, already)
    return GapList(gaps=gaps, markdown=markdown, items=items, already_in_kitchen=already)


def _format_gaps_markdown(items: List[ShoppingListItem], already: List[str]) -> str:
    if not items:
        body = "### Shopping list\n\nNothing to buy — your inventory covers this plan."
    else:
        groups = {}
        for it in items:
            groups.setdefault(it.category, []).append(it)
        lines = ["### Shopping list (gaps only)", ""]
        titles = {
            "produce": "Produce",
            "dairy_or_alternatives": "Dairy or alternatives",
            "protein": "Protein",
            "grains_and_bakery": "Grains and bakery",
            "canned_and_packaged": "Canned and packaged",
            "other": "Other",
        }
        for cat, title in titles.items():
            rows = groups.get(cat) or []
            if not rows:
                continue
            lines.append(f"**{title}**")
            for r in rows:
                meals = ", ".join(r.required_for_meals) if r.required_for_meals else "plan"
                alt = " _(homemade option available)_" if r.homemade_option_available else ""
                lines.append(f"- [ ] {r.display_name} — used in: {meals}{alt}")
            lines.append("")
        body = "\n".join(lines)

    if already:
        body += "\n\n**Already in your kitchen and not added:**\n" + ", ".join(already)
    return body
