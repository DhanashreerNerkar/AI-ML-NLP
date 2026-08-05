"""Deterministic diet / banned-ingredient filtering."""

from __future__ import annotations

from typing import Set

from src.kb import normalize_name
from src.schemas import RecipeCandidate, UserProfile

MEAT_KEYWORDS = {
    "chicken", "beef", "pork", "lamb", "mutton", "turkey", "duck", "bacon",
    "ham", "sausage", "meat", "steak", "veal", "prosciutto", "pepperoni",
    "salami", "anchovy", "anchovies", "poultry",
}

SEAFOOD_KEYWORDS = {
    "fish", "salmon", "tuna", "cod", "tilapia", "shrimp", "prawn", "crab",
    "lobster", "clam", "mussel", "oyster", "seafood", "fish sauce",
    "oyster sauce", "anchovy", "shellfish",
}

HIDDEN_NONVEG = {
    "gelatin", "gelatine", "fish sauce", "oyster sauce", "lard", "tallow",
    "rennet", "worcestershire", "chicken broth", "beef stock", "chicken stock",
    "bone broth",
}

DAIRY_KEYWORDS = {
    "milk", "butter", "cream", "cheese", "paneer", "yogurt", "yoghurt", "curd",
    "ghee", "whey", "casein", "buttermilk", "ricotta", "parmesan", "mozzarella",
}

EGG_KEYWORDS = {"egg", "eggs", "mayonnaise", "mayo"}

# These labels commonly hide animal-derived ingredients.  They may be safe
# only when a recipe explicitly says so; treat ambiguity as unsafe for users
# with a vegetarian or vegan profile instead of guessing.
AMBIGUOUS_VEG_LABELS = {"kimchi", "bibimbap"}

ALLERGEN_MAP = {
    "peanuts": {"peanut", "peanuts"},
    "tree nuts": {"almond", "cashew", "walnut", "pecan", "pistachio", "hazelnut", "nut"},
    "dairy": DAIRY_KEYWORDS,
    "eggs": EGG_KEYWORDS,
    "gluten": {"wheat", "gluten", "flour", "bread", "pasta", "barley", "rye"},
    "soy": {"soy", "soya", "tofu", "edamame", "soy sauce"},
    "sesame": {"sesame", "tahini"},
    "shellfish": {"shrimp", "prawn", "crab", "lobster", "shellfish", "clam", "mussel"},
    "fish": SEAFOOD_KEYWORDS - {"shrimp", "prawn", "crab", "lobster", "shellfish", "clam", "mussel", "oyster"},
}

JAIN_BANNED = {
    "onion", "garlic", "potato", "carrot", "beet", "radish", "ginger", "root",
}


def _tokens(text: str) -> str:
    return normalize_name(text)


def contains_any(text: str, keywords: set) -> bool:
    t = _tokens(text)
    return any(k in t for k in keywords)


def banned_ingredients_for_profile(profile: UserProfile) -> Set[str]:
    profile.sync_aliases()
    banned: Set[str] = set()
    banned.update(normalize_name(x) for x in profile.dietary_restrictions)
    banned.update(normalize_name(x) for x in profile.hard_exclusions)
    banned.update(normalize_name(x) for x in profile.allergies)
    banned.update(normalize_name(x) for x in profile.dislikes)
    # Ambient rules are human-readable phrases ("no onion"); concrete banned
    # ingredients are already expanded into hard_exclusions by sync_aliases.
    for x in profile.ambient_rules:
        n = normalize_name(x)
        if n.startswith("no "):
            continue
        banned.add(n)

    for allergy in profile.allergies:
        key = normalize_name(allergy)
        for label, words in ALLERGEN_MAP.items():
            if key == normalize_name(label) or key in words:
                banned.update(words)

    diet = profile.diet_type
    cultural_l = [normalize_name(c) for c in (profile.cultural_rules or profile.cultural_constraints)]
    cultural = set(cultural_l)

    if diet in {"vegetarian", "vegan", "eggetarian"}:
        banned.update(MEAT_KEYWORDS)
        banned.update(SEAFOOD_KEYWORDS)
        banned.update(HIDDEN_NONVEG)
    if diet == "vegetarian":
        # Egg-tolerant users select the explicit ``eggetarian`` profile.  A
        # regular vegetarian profile must not receive egg-based recipes.
        banned.update(EGG_KEYWORDS)
    if diet == "eggetarian":
        banned.update(MEAT_KEYWORDS)
        banned.update(SEAFOOD_KEYWORDS)
        banned.update(HIDDEN_NONVEG)
        # eggs allowed
        banned -= EGG_KEYWORDS
    if diet == "vegan":
        banned.update(DAIRY_KEYWORDS)
        banned.update(EGG_KEYWORDS)
        banned.add("honey")
    if diet == "pescatarian":
        banned.update(MEAT_KEYWORDS - {"anchovy", "anchovies"})
        banned.update({"pork", "bacon", "ham", "lard", "gelatin", "gelatine", "chicken", "beef", "turkey"})

    if "jain" in cultural or any("jain" in c for c in cultural_l):
        banned.update(JAIN_BANNED)
        banned.update(MEAT_KEYWORDS)
        banned.update(SEAFOOD_KEYWORDS)
        banned.update(EGG_KEYWORDS)
        banned.update(HIDDEN_NONVEG)
    if any("halal" in c for c in cultural_l):
        banned.update({"pork", "bacon", "ham", "lard", "alcohol", "wine", "beer"})
    if any("kosher" in c for c in cultural_l):
        banned.update({"pork", "bacon", "ham", "shellfish", "shrimp", "crab", "lobster"})
    # Hindu dietary preference ⇒ no beef. Eggs stay allowed for non-veg /
    # eggetarian / flexitarian unless the user also picked vegetarian/vegan/jain
    # or listed egg as an allergy.
    if any("hindu" in c for c in cultural_l) or any("no beef" in c for c in cultural_l):
        banned.add("beef")
        banned.update({"beef stock", "beef broth"})
    if any("no pork" in c for c in cultural_l):
        banned.update({"pork", "bacon", "ham"})

    # Never keep diet-type labels themselves in the banned ingredient set —
    # substring checks would turn "eggetarian" into an egg ban.
    banned -= {
        "vegetarian", "non-vegetarian", "non vegetarian", "vegan",
        "eggetarian", "pescatarian", "flexitarian", "hindu",
        "hindu dietary preference", "halal", "kosher", "jain",
    }

    return {b for b in banned if b and b != "none"}


def ambient_rules_from_profile(profile: UserProfile) -> list:
    profile.sync_aliases()
    return list(profile.ambient_rules)


def ingredient_violates(ingredient: str, banned: set) -> bool:
    """Return True if ``ingredient`` matches a banned term.

    Uses word-boundary style matching so short stems like ``egg`` do not
    false-positive inside labels such as ``eggetarian``.
    """
    text = _tokens(ingredient)
    if not text:
        return False
    padded = f" {text} "
    for raw in banned:
        if not raw or raw == "none":
            continue
        b = _tokens(raw)
        if not b or b == "none":
            continue
        if text == b:
            return True
        if f" {b} " in padded:
            return True
        # Multi-word ban phrases (e.g. "fish sauce") may appear inside a longer blob.
        if " " in b and b in text:
            return True
    return False


def text_violates(text: str, banned: set) -> bool:
    return ingredient_violates(text, banned)


def _ambiguous_diet_label(text: str, profile: UserProfile) -> bool:
    if profile.diet_type not in {"vegetarian", "vegan", "eggetarian"}:
        return False
    normalized = _tokens(text)
    requires_explicit_safe_label = any(label in normalized for label in AMBIGUOUS_VEG_LABELS)
    is_explicitly_safe = "vegan" in normalized or "egg free" in normalized or "egg-free" in normalized
    return requires_explicit_safe_label and not is_explicitly_safe


def filter_recipe_candidates(candidates, profile: UserProfile):
    banned = banned_ingredients_for_profile(profile)
    kept = []
    for c in candidates:
        blob = " ".join([c.title, *c.ingredients])
        if text_violates(blob, banned) or _ambiguous_diet_label(blob, profile):
            continue
        if profile.diet_type in {"vegetarian", "vegan", "eggetarian"} and contains_any(c.title, MEAT_KEYWORDS):
            continue
        if profile.diet_type == "vegan" and contains_any(blob, DAIRY_KEYWORDS | EGG_KEYWORDS):
            continue
        kept.append(c)
    return kept


def scan_plan_for_violations(plan_dict: dict, profile: UserProfile) -> list:
    banned = banned_ingredients_for_profile(profile)
    violations = []
    for meal in plan_dict.get("plan", []):
        fields = [
            meal.get("recipe", ""),
            *meal.get("ingredients_from_inventory", []),
            *meal.get("extra_pantry_items", []),
            *meal.get("uses_critical", []),
            *meal.get("steps", []),
        ]
        for field in fields:
            if isinstance(field, str) and (
                text_violates(field, banned) or _ambiguous_diet_label(field, profile)
            ):
                violations.append(field)
    return violations


def inventory_conflicts(item_name: str, profile: UserProfile) -> bool:
    banned = banned_ingredients_for_profile(profile)
    # Dislikes are soft — still surface as conflicts for confirmation but allergies/diet are hard
    hard = banned_ingredients_for_profile(profile)
    # remove soft dislikes for "hard conflict" check
    soft = {normalize_name(d) for d in profile.dislikes}
    hard = {b for b in hard if b not in soft}
    return ingredient_violates(item_name, hard)
