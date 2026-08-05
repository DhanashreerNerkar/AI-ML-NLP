"""Unit tests for diet filtering."""

from src.diet_filter import filter_recipe_candidates, banned_ingredients_for_profile
from src.schemas import RecipeCandidate, UserProfile


def test_vegetarian_strips_meat_and_hidden():
    profile = UserProfile(diet_type="vegetarian", diet_style="strict-vegetarian")
    candidates = [
        RecipeCandidate(title="Chicken Kale Bake", ingredients=["chicken", "kale"]),
        RecipeCandidate(title="Spinach Rice", ingredients=["spinach", "rice", "gelatin"]),
        RecipeCandidate(title="Lemon Dal", ingredients=["lentils", "lemon", "cumin"]),
        RecipeCandidate(title="Veg Stir Fry", ingredients=["tofu", "spinach", "fish sauce"]),
    ]
    kept = filter_recipe_candidates(candidates, profile)
    titles = {c.title for c in kept}
    assert titles == {"Lemon Dal"}


def test_vegan_strips_dairy():
    profile = UserProfile(diet_type="vegan", diet_style="none")
    candidates = [
        RecipeCandidate(title="Paneer Curry", ingredients=["paneer", "spinach"]),
        RecipeCandidate(title="Tofu Curry", ingredients=["tofu", "spinach", "oil"]),
    ]
    kept = filter_recipe_candidates(candidates, profile)
    assert [c.title for c in kept] == ["Tofu Curry"]


def test_vegetarian_strips_egg_but_eggetarian_allows_it():
    candidates = [
        RecipeCandidate(title="Egg Fried Rice", ingredients=["rice", "eggs"]),
        RecipeCandidate(title="Tofu Fried Rice", ingredients=["rice", "tofu"]),
    ]

    vegetarian = UserProfile(diet_type="vegetarian", diet_style="strict-vegetarian")
    assert [c.title for c in filter_recipe_candidates(candidates, vegetarian)] == ["Tofu Fried Rice"]

    eggetarian = UserProfile(diet_type="eggetarian")
    assert [c.title for c in filter_recipe_candidates(candidates, eggetarian)] == [
        "Egg Fried Rice",
        "Tofu Fried Rice",
    ]


def test_jain_ambient_bans_onion_garlic():
    profile = UserProfile(
        diet_type="vegetarian",
        cultural_constraints=["jain"],
        ambient_rules=["no onion", "no garlic"],
    )
    banned = banned_ingredients_for_profile(profile)
    assert "onion" in banned
    assert "garlic" in banned


def test_hard_allergy_is_filtered_before_planning():
    profile = UserProfile(diet_type="vegetarian", allergies=["peanuts"])
    candidates = [
        RecipeCandidate(title="Peanut Noodles", ingredients=["noodles", "peanuts"]),
        RecipeCandidate(title="Tofu Noodles", ingredients=["noodles", "tofu"]),
    ]
    assert [c.title for c in filter_recipe_candidates(candidates, profile)] == ["Tofu Noodles"]


def test_vegetarian_requires_explicitly_vegan_kimchi_and_bibimbap():
    profile = UserProfile(diet_type="vegetarian", allergies=["egg"])
    candidates = [
        RecipeCandidate(title="Vegetable Bibimbap", ingredients=["cabbage kimchi", "rice"]),
        RecipeCandidate(title="Vegan Bibimbap", ingredients=["vegan kimchi", "rice"]),
    ]
    assert [c.title for c in filter_recipe_candidates(candidates, profile)] == ["Vegan Bibimbap"]


def test_non_veg_hindu_allows_eggs_bans_beef():
    """Non-veg + Hindu must not treat eggs as a restriction; beef is the cultural ban."""
    profile = UserProfile(
        diet_type="non-vegetarian",
        cultural_rules=["Hindu dietary preference"],
        allergies=[],
        # Stale leftovers from a previous "Eggs" allergy answer — must not stick.
        hard_exclusions=["egg"],
        dietary_restrictions=["egg"],
    ).sync_aliases()

    assert "egg" not in {x.lower() for x in profile.hard_exclusions}
    assert "egg" not in {x.lower() for x in profile.dietary_restrictions}
    assert "beef" in {x.lower() for x in profile.hard_exclusions}

    banned = banned_ingredients_for_profile(profile)
    assert "egg" not in banned and "eggs" not in banned
    assert "beef" in banned

    candidates = [
        RecipeCandidate(title="Egg Bhurji", ingredients=["eggs", "onion"]),
        RecipeCandidate(title="Beef Curry", ingredients=["beef", "onion"]),
        RecipeCandidate(title="Chicken Curry", ingredients=["chicken", "onion"]),
    ]
    kept = {c.title for c in filter_recipe_candidates(candidates, profile)}
    assert kept == {"Egg Bhurji", "Chicken Curry"}


def test_clearing_allergies_drops_stale_hard_exclusions():
    profile = UserProfile(
        diet_type="non-vegetarian",
        allergies=["egg"],
        hard_exclusions=["egg"],
        dietary_restrictions=["egg"],
    ).sync_aliases()
    assert "egg" in {x.lower() for x in profile.dietary_restrictions}

    profile.allergies = []
    profile.sync_aliases()
    assert profile.hard_exclusions == []
    assert profile.dietary_restrictions == []
    assert "egg" not in banned_ingredients_for_profile(profile)


def test_egg_does_not_match_eggetarian_label():
    from src.diet_filter import ingredient_violates

    assert ingredient_violates("egg", {"eggetarian"}) is False
    assert ingredient_violates("eggs", {"eggetarian"}) is False
    assert ingredient_violates("egg", {"egg", "eggs"}) is True
    assert ingredient_violates("chicken egg scramble", {"egg"}) is True
