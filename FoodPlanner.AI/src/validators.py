"""Pre-response validators: schema, banned ingredients, gap math."""

from __future__ import annotations

from pydantic import ValidationError

from src.diet_filter import scan_plan_for_violations
from src.kb import load_pantry_staples, normalize_name
from src.schemas import GapList, MealPlan, SubstitutionResult, UserProfile
from src.stages.gap_list import compute_gap_list


class ValidationReport(dict):
    @property
    def ok(self) -> bool:
        return self.get("ok", False)


def validate_meal_plan(plan_data: dict | MealPlan, profile: UserProfile) -> ValidationReport:
    try:
        plan = plan_data if isinstance(plan_data, MealPlan) else MealPlan.model_validate(plan_data)
    except ValidationError as e:
        return ValidationReport(ok=False, errors=[f"schema: {e}"], plan=None)

    violations = scan_plan_for_violations(plan.model_dump(), profile)
    if violations:
        return ValidationReport(
            ok=False,
            errors=[f"banned ingredient: {v}" for v in violations],
            plan=plan,
        )
    return ValidationReport(ok=True, errors=[], plan=plan)


def validate_substitution(data: dict | SubstitutionResult) -> ValidationReport:
    try:
        result = (
            data if isinstance(data, SubstitutionResult) else SubstitutionResult.model_validate(data)
        )
    except ValidationError as e:
        return ValidationReport(ok=False, errors=[f"schema: {e}"], substitution=None)
    return ValidationReport(ok=True, errors=[], substitution=result)


def validate_gap_list(
    gap_data: dict | GapList,
    plan: MealPlan,
    inventory: list[str],
    purchased_or_made: list[str] | None = None,
) -> ValidationReport:
    try:
        gaps = gap_data if isinstance(gap_data, GapList) else GapList.model_validate(gap_data)
    except ValidationError as e:
        return ValidationReport(ok=False, errors=[f"schema: {e}"], gap_list=None)

    expected = compute_gap_list(plan, inventory, purchased_or_made=purchased_or_made)
    got = {normalize_name(g) for g in gaps.gaps}
    want = {normalize_name(g) for g in expected.gaps}
    if got != want:
        return ValidationReport(
            ok=False,
            errors=[f"gap math mismatch: got={sorted(got)} expected={sorted(want)}"],
            gap_list=gaps,
            expected=expected,
        )
    return ValidationReport(ok=True, errors=[], gap_list=gaps)


def run_all_validators(
    *,
    profile: UserProfile,
    plan: MealPlan | dict | None,
    gap_list: GapList | dict | None,
    inventory: list[str],
    substitution: SubstitutionResult | dict | None = None,
    purchased_or_made: list[str] | None = None,
) -> ValidationReport:
    errors: list[str] = []
    plan_obj = None
    gap_obj = None
    sub_obj = None

    if plan is not None:
        pr = validate_meal_plan(plan, profile)
        errors.extend(pr.get("errors", []))
        plan_obj = pr.get("plan")

    if substitution is not None:
        sr = validate_substitution(substitution)
        errors.extend(sr.get("errors", []))
        sub_obj = sr.get("substitution")

    if gap_list is not None and plan_obj is not None:
        gr = validate_gap_list(gap_list, plan_obj, inventory, purchased_or_made=purchased_or_made)
        errors.extend(gr.get("errors", []))
        gap_obj = gr.get("gap_list")

    return ValidationReport(
        ok=len(errors) == 0,
        errors=errors,
        plan=plan_obj,
        gap_list=gap_obj,
        substitution=sub_obj,
    )
