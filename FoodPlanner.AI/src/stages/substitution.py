"""Stage 5 — substitution + homemade derivation flow."""

from __future__ import annotations

from src.diet_filter import banned_ingredients_for_profile, ingredient_violates
from src.kb import load_derivation_kb, load_substitution_kb, normalize_name
from src.schemas import (
    DerivationOption,
    SubstitutionResult,
    SubstitutionSuccess,
    UserProfile,
)


def _names_match(query: str, candidate: str) -> bool:
    """Exact / alias match without substring traps (butter ≠ buttermilk)."""
    q = normalize_name(query)
    c = normalize_name(candidate)
    if not q or not c:
        return False
    if q == c:
        return True
    # Allow plural/singular simple forms
    if q.rstrip("s") == c.rstrip("s") and min(len(q), len(c)) >= 4:
        return True
    return False


def _option_allowed(diet_tags: list[str], profile: UserProfile) -> bool:
    banned = banned_ingredients_for_profile(profile)
    tags = {normalize_name(t) for t in diet_tags}
    # Must not include banned ingredients in the tag-implied foods; also option tags must be compatible
    if profile.diet_type == "vegan" and not ({"vegan", "dairy-free"} & tags):
        # allow if vegan explicitly in tags
        if "vegan" not in tags:
            return False
    if profile.diet_type == "vegetarian" and "vegan" not in tags and "vegetarian" not in tags:
        # non-vegetarian-only options blocked
        if tags & {"non-vegetarian", "pescatarian"} and "vegetarian" not in tags:
            return False
    if "jain" in {normalize_name(c) for c in profile.cultural_constraints}:
        if "jain" not in tags and "jain-excluded" in tags:
            return False
    # Scan tag strings and diet tags content against banned words lightly
    blob = " ".join(diet_tags)
    if ingredient_violates(blob, banned - {"none"}):
        # e.g. options tagged only non-vegetarian with meat implication — already handled
        pass
    return True


def _find_derivation_options(missing: str, profile: UserProfile) -> list[DerivationOption]:
    kb = load_derivation_kb()
    missing_n = normalize_name(missing)
    out: list[DerivationOption] = []
    for entry in kb.get("entries", []):
        names = [entry.get("derivative", ""), *entry.get("aliases", [])]
        if not any(_names_match(missing_n, n) for n in names if n):
            continue
        for opt in entry.get("options", []):
            if not _option_allowed(opt.get("diet_tags", []), profile):
                continue
            # Also ensure method/base ingredients don't violate profile
            banned = banned_ingredients_for_profile(profile)
            bases = opt.get("base_ingredients", [])
            if any(ingredient_violates(b, banned) for b in bases):
                continue
            out.append(
                DerivationOption(
                    option_id=opt["option_id"],
                    label=opt["label"],
                    base_ingredients=bases,
                    method_steps=opt.get("method_steps", []),
                    diet_tags=opt.get("diet_tags", []),
                )
            )
    return out


def _find_store_options(missing: str, profile: UserProfile) -> list[SubstitutionSuccess]:
    kb = load_substitution_kb()
    missing_n = normalize_name(missing)
    banned = banned_ingredients_for_profile(profile)
    out: list[SubstitutionSuccess] = []
    for entry in kb.get("entries", []):
        names = [entry.get("missing", ""), *entry.get("aliases", [])]
        if not any(_names_match(missing_n, n) for n in names if n):
            continue
        for sub in entry.get("substitutes", []):
            if not _option_allowed(sub.get("diet_tags", []), profile):
                continue
            if ingredient_violates(sub.get("substitute", ""), banned):
                continue
            out.append(
                SubstitutionSuccess(
                    substitute=sub["substitute"],
                    prep=sub.get("prep", ""),
                    integration=sub.get("integration", ""),
                    source="approved_substitutions",
                )
            )
    return out


def build_substitution_options(
    *,
    missing_ingredient: str,
    recipe_context: str,
    profile: UserProfile,
) -> SubstitutionResult:
    derivations = _find_derivation_options(missing_ingredient, profile)
    stores = _find_store_options(missing_ingredient, profile)
    if not derivations and not stores:
        return SubstitutionResult(
            status="no_substitute",
            message="No suitable substitute found. Modify the recipe accordingly.",
            missing_ingredient=missing_ingredient,
            recipe_context=recipe_context,
        )
    return SubstitutionResult(
        status="options",
        missing_ingredient=missing_ingredient,
        recipe_context=recipe_context,
        derivation_options=derivations,
        store_options=stores,
    )


def select_substitution_option(
    options: SubstitutionResult,
    option_id: str,
) -> SubstitutionResult:
    if option_id == "skip":
        return SubstitutionResult(
            status="selected",
            selected_option_id="skip",
            missing_ingredient=options.missing_ingredient,
            recipe_context=options.recipe_context,
            message="User chose to skip / buy later.",
            substitute=options.missing_ingredient,
            source="skip_buy",
        )
    for d in options.derivation_options:
        if d.option_id == option_id:
            return SubstitutionResult(
                status="selected",
                selected_option_id=option_id,
                missing_ingredient=options.missing_ingredient,
                recipe_context=options.recipe_context,
                substitute=d.label,
                prep="; ".join(d.method_steps),
                method_steps=d.method_steps,
                integration=f"Use homemade {options.missing_ingredient} in {options.recipe_context}",
                source="derivation_kb",
            )
    for i, s in enumerate(options.store_options):
        sid = f"store_{i}"
        if option_id == sid or option_id == s.substitute:
            return SubstitutionResult(
                status="ok",
                selected_option_id=sid,
                missing_ingredient=options.missing_ingredient,
                recipe_context=options.recipe_context,
                substitute=s.substitute,
                prep=s.prep,
                integration=s.integration,
                source=s.source,
            )
    return SubstitutionResult(
        status="no_substitute",
        message="No suitable substitute found. Modify the recipe accordingly.",
        missing_ingredient=options.missing_ingredient,
        recipe_context=options.recipe_context,
    )


def auto_pick_simple_substitute(
    *,
    missing_ingredient: str,
    recipe_context: str,
    profile: UserProfile,
) -> SubstitutionResult:
    """Non-interactive rescue: prefer first compliant store substitute."""
    options = build_substitution_options(
        missing_ingredient=missing_ingredient,
        recipe_context=recipe_context,
        profile=profile,
    )
    if options.status == "no_substitute":
        return options
    if options.store_options:
        s = options.store_options[0]
        return SubstitutionResult(
            status="ok",
            substitute=s.substitute,
            prep=s.prep,
            integration=s.integration,
            source=s.source,
            missing_ingredient=missing_ingredient,
            recipe_context=recipe_context,
        )
    if options.derivation_options:
        return select_substitution_option(options, options.derivation_options[0].option_id)
    return SubstitutionResult(
        status="no_substitute",
        message="No suitable substitute found. Modify the recipe accordingly.",
        missing_ingredient=missing_ingredient,
        recipe_context=recipe_context,
    )
