"""Hard-constraint checks around recipe retrieval and LLM planning."""

from src.diet_filter import scan_plan_for_violations
from src.pipeline import stage_plan
from src.schemas import RecipeCandidate, UserProfile


def test_stage_plan_blocks_an_unsafe_llm_response(monkeypatch):
    from src.stages.planning import plan_meals as real_plan_meals

    profile = UserProfile(diet_type="vegetarian", allergies=["peanuts"])
    unsafe = {
        "plan": [
            {
                "night": 1,
                "meal_type": "dinner",
                "recipe": "Peanut Chicken Bowl",
                "time_min": 20,
                "servings": 2,
                "uses_critical": [],
                "ingredients_from_inventory": ["peanuts", "chicken"],
                "extra_pantry_items": [],
                "steps": ["Cook chicken with peanuts"],
                "missing_ingredients": [],
            }
        ],
        "flagged_for_other_use": [],
        "clarification": None,
    }

    def fake_plan_meals(**kwargs):
        if kwargs["use_llm"]:
            from src.schemas import MealPlan

            return MealPlan.model_validate(unsafe)
        return real_plan_meals(**kwargs)

    monkeypatch.setattr("src.pipeline.plan_meals", fake_plan_meals)
    state = {
        "profile": profile.model_dump(),
        "confirmed_items": ["tofu", "rice"],
        "critical_priority": [],
        "recipe_candidates": [
            RecipeCandidate(title="Tofu Rice Bowl", ingredients=["tofu", "rice"]).model_dump()
        ],
        "use_llm": True,
    }

    result = stage_plan(state)
    assert result["validation_ok"] is False
    assert result["plan"]["plan"][0]["recipe"] == "Tofu Rice Bowl"


def test_plan_scan_flags_ambiguous_bibimbap_and_kimchi():
    profile = UserProfile(diet_type="vegetarian", allergies=["egg"])
    plan = {
        "plan": [
            {
                "recipe": "Vegetable Bibimbap",
                "ingredients_from_inventory": [],
                "extra_pantry_items": ["cabbage kimchi"],
                "uses_critical": [],
                "steps": [],
            }
        ]
    }
    assert scan_plan_for_violations(plan, profile)
