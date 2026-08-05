"""Unit tests for validators."""

from src.schemas import GapList, MealPlan, PlanMeal, UserProfile
from src.validators import validate_gap_list, validate_meal_plan


def test_plan_validator_catches_meat_for_vegetarian():
    profile = UserProfile(diet_type="vegetarian", diet_style="strict-vegetarian")
    plan = {
        "plan": [
            {
                "night": 1,
                "recipe": "Chicken Rice",
                "time_min": 20,
                "servings": 2,
                "ingredients_from_inventory": ["chicken", "rice"],
                "extra_pantry_items": [],
                "steps": ["cook chicken"],
            }
        ]
    }
    report = validate_meal_plan(plan, profile)
    assert report.ok is False
    assert any("banned" in e for e in report["errors"])


def test_gap_validator_math():
    profile = UserProfile()
    plan = MealPlan(
        plan=[
            PlanMeal(
                night=1,
                recipe="Dal",
                time_min=20,
                servings=2,
                ingredients_from_inventory=["lentils"],
                missing_ingredients=["tomato"],
            )
        ]
    )
    bad = GapList(gaps=["paneer"], markdown="")
    report = validate_gap_list(bad, plan, inventory=["lentils"])
    assert report.ok is False
