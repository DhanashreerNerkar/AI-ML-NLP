"""Unit tests for gap-list set math."""

from src.schemas import MealPlan, PlanMeal
from src.stages.gap_list import compute_gap_list


def test_gap_list_excludes_inventory_and_staples():
    plan = MealPlan(
        plan=[
            PlanMeal(
                night=1,
                recipe="Spinach Rice",
                time_min=20,
                servings=2,
                ingredients_from_inventory=["spinach", "rice"],
                extra_pantry_items=["salt", "cumin", "paneer"],
                missing_ingredients=[],
            )
        ]
    )
    gaps = compute_gap_list(plan, inventory=["spinach", "rice"])
    assert "paneer" in gaps.gaps
    assert "salt" not in gaps.gaps
    assert "cumin" not in gaps.gaps
    assert "spinach" not in gaps.gaps


def test_skip_buy_paneer_appears():
    plan = MealPlan(
        plan=[
            PlanMeal(
                night=1,
                recipe="Palak Paneer",
                time_min=25,
                servings=2,
                ingredients_from_inventory=["spinach"],
                extra_pantry_items=[],
                missing_ingredients=["paneer"],
            )
        ]
    )
    gaps = compute_gap_list(plan, inventory=["spinach"])
    assert gaps.gaps == ["paneer"]


def test_rejects_prose_extras():
    plan = MealPlan(
        plan=[
            PlanMeal(
                night=1,
                recipe="Bowl",
                time_min=15,
                servings=2,
                ingredients_from_inventory=["rice"],
                extra_pantry_items=["use a vegetarian swap instead of chicken tonight"],
            )
        ]
    )
    gaps = compute_gap_list(plan, inventory=["rice"])
    assert gaps.gaps == []
