"""Regression tests for requested meal-slot coverage."""

from src.schemas import RecipeCandidate, UserProfile
from src.stages.planning import plan_meals_deterministic


def test_one_day_lunch_and_dinner_creates_two_meal_entries():
    profile = UserProfile(nights=1, meal_types=["lunch", "dinner"], servings=1)
    candidate = RecipeCandidate(
        title="Tofu Rice Bowl",
        ingredients=["tofu", "rice"],
        ready_in_minutes=20,
        steps=["Cook and serve"],
    )

    plan = plan_meals_deterministic(
        inventory=["tofu", "rice"],
        critical_priority=[],
        recipe_candidates=[candidate],
        profile=profile,
    )

    assert len(plan.plan) == 2
    assert [(meal.night, meal.meal_type) for meal in plan.plan] == [
        (1, "lunch"),
        (1, "dinner"),
    ]
    assert all(meal.servings == 1 for meal in plan.plan)
