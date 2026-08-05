"""Pipeline node functions and intent routing."""

from __future__ import annotations

import os
from typing import Any, Callable, Literal, Optional, TypedDict

from src.diet_filter import filter_recipe_candidates, scan_plan_for_violations
from src.schemas import (
    GapList,
    MealPlan,
    RecipeCandidate,
    SubstitutionResult,
    UserProfile,
)
from src.stages.critical import critical_priority_filter
from src.stages.freshness import score_freshness
from src.stages.gap_list import compute_gap_list
from src.stages.planning import plan_meals
from src.stages.substitution import build_substitution_options, select_substitution_option
from src.tools.spoonacular import search_by_ingredients
from src.validators import validate_gap_list


class PipelineState(TypedDict, total=False):
    intent: Literal["plan", "substitute"]
    profile: dict
    confirmed_items: list[str]
    ranked: list[dict]
    critical_priority: list[str]
    critical_explanation: str
    recipe_candidates: list[dict]
    plan: dict
    substitution: dict
    gap_list: dict
    missing_ingredient: str
    recipe_context: str
    selected_option_id: str
    purchased_or_made: list[str]
    use_llm: bool
    errors: list[str]
    validation_ok: bool
    conversation_summary: str
    user_request: str


def route_intent(state: PipelineState) -> Literal["plan", "substitute"]:
    intent = state.get("intent") or "plan"
    if intent == "substitute":
        return "substitute"
    return "plan"


def _profile(state: PipelineState) -> UserProfile:
    return UserProfile.model_validate(state["profile"])


def _use_llm(state: PipelineState) -> bool:
    if "use_llm" in state:
        return bool(state["use_llm"])
    key = os.getenv("ANTHROPIC_API_KEY", "")
    return bool(key and key != "your_anthropic_api_key_here")


def stage_freshness(state: PipelineState) -> dict[str, Any]:
    ranked = score_freshness(state.get("confirmed_items", []), use_llm=_use_llm(state))
    return {"ranked": [r.model_dump(by_alias=True) for r in ranked.ranked]}


def stage_critical(state: PipelineState) -> dict[str, Any]:
    result = critical_priority_filter(state.get("ranked", []))
    return {
        "critical_priority": result.critical_priority,
        "critical_explanation": result.explanation,
    }


def stage_retrieve(state: PipelineState) -> dict[str, Any]:
    profile = _profile(state)
    candidates = search_by_ingredients(state.get("confirmed_items", []), profile)
    candidates = filter_recipe_candidates(candidates, profile)
    return {"recipe_candidates": [c.model_dump() for c in candidates]}


def stage_plan(state: PipelineState) -> dict[str, Any]:
    profile = _profile(state)
    candidates = [RecipeCandidate.model_validate(c) for c in state.get("recipe_candidates", [])]
    candidates = filter_recipe_candidates(candidates, profile)
    plan = plan_meals(
        inventory=state.get("confirmed_items", []),
        critical_priority=state.get("critical_priority", []),
        recipe_candidates=candidates,
        profile=profile,
        use_llm=_use_llm(state),
        conversation_summary=state.get("conversation_summary", ""),
        user_request=state.get("user_request", ""),
    )
    violations = scan_plan_for_violations(plan.model_dump(), profile)
    if violations:
        # Never expose an unsafe LLM result. Rebuild from the already-filtered
        # candidate pool without an LLM so the fallback is deterministic.
        plan = plan_meals(
            inventory=state.get("confirmed_items", []),
            critical_priority=state.get("critical_priority", []),
            recipe_candidates=candidates,
            profile=profile,
            use_llm=False,
        )
    return {
        "plan": plan.model_dump(),
        "errors": list(state.get("errors") or []) + (
            ["Unsafe generated plan blocked and replaced with a safe fallback."] if violations else []
        ),
        "validation_ok": not violations,
    }


def stage_gap(state: PipelineState) -> dict[str, Any]:
    plan = MealPlan.model_validate(state["plan"])
    gaps = compute_gap_list(
        plan,
        inventory=state.get("confirmed_items", []),
        purchased_or_made=state.get("purchased_or_made") or [],
    )
    # Pure gap-math check (deterministic set arithmetic, not a diet judgment) —
    # kept as-is. Diet compliance is not re-validated here; see stage_plan.
    report = validate_gap_list(
        gap_data=gaps,
        plan=plan,
        inventory=state.get("confirmed_items", []),
        purchased_or_made=state.get("purchased_or_made") or [],
    )
    errors = list(state.get("errors") or [])
    if not report.ok:
        errors.extend(report.get("errors", []))
        gaps = report.get("expected") or gaps
    return {
        "gap_list": gaps.model_dump() if isinstance(gaps, GapList) else gaps,
        "errors": errors,
        "validation_ok": report.ok and state.get("validation_ok", True),
    }


def stage_substitute(state: PipelineState) -> dict[str, Any]:
    profile = _profile(state)
    missing = state.get("missing_ingredient") or ""
    recipe_context = state.get("recipe_context") or ""
    selected = state.get("selected_option_id")

    if selected:
        # Rebuild options then apply selection
        options = build_substitution_options(
            missing_ingredient=missing,
            recipe_context=recipe_context,
            profile=profile,
        )
        result = select_substitution_option(options, selected)
    else:
        result = build_substitution_options(
            missing_ingredient=missing,
            recipe_context=recipe_context,
            profile=profile,
        )

    purchased = list(state.get("purchased_or_made") or [])
    if result.status in {"ok", "selected"} and result.source == "derivation_kb":
        if result.missing_ingredient and result.missing_ingredient not in purchased:
            purchased.append(result.missing_ingredient)
    if result.status == "selected" and result.selected_option_id == "skip":
        if result.missing_ingredient and result.missing_ingredient not in purchased:
            # skip/buy later means it remains a gap — do not add to purchased
            pass

    out: dict[str, Any] = {"substitution": result.model_dump(), "purchased_or_made": purchased}

    # If we already have a plan and user selected skip/buy, refresh gaps to include missing item
    if state.get("plan") and result.selected_option_id == "skip" and result.missing_ingredient:
        plan = MealPlan.model_validate(state["plan"])
        # Ensure missing ingredient is represented on the plan
        if plan.plan:
            if result.missing_ingredient not in plan.plan[0].missing_ingredients:
                plan.plan[0].missing_ingredients.append(result.missing_ingredient)
        gaps = compute_gap_list(plan, inventory=state.get("confirmed_items", []))
        out["plan"] = plan.model_dump()
        out["gap_list"] = gaps.model_dump()
    elif state.get("plan") and result.source == "derivation_kb" and result.missing_ingredient:
        plan = MealPlan.model_validate(state["plan"])
        gaps = compute_gap_list(
            plan,
            inventory=state.get("confirmed_items", []),
            purchased_or_made=purchased,
        )
        out["gap_list"] = gaps.model_dump()

    return out


# Human-readable labels for real LangGraph nodes (shown in the UI progress panel).
PLAN_STEP_LABELS = {
    "stage_freshness": "Scoring freshness from the shelf-life knowledge base",
    "stage_critical": "Picking use-first / at-risk ingredients",
    "stage_retrieve": "Searching diet-safe recipes",
    "stage_plan": "Building your multi-day meal plan",
    "stage_gap": "Computing the gap-only shopping list",
}

ProgressCallback = Callable[[str, str, str], None]


def _step_detail(node: str, update: dict, *, use_llm: bool) -> str:
    """Turn the actual stage output into a short glimpse for the UI."""
    if node == "stage_freshness":
        ranked = update.get("ranked") or []
        top = [r.get("item") for r in ranked[:4] if isinstance(r, dict) and r.get("item")]
        return f"Ranked {len(ranked)} items" + (f" · top: {', '.join(top)}" if top else "")
    if node == "stage_critical":
        crit = update.get("critical_priority") or []
        return f"{len(crit)} use-first: {', '.join(crit[:5])}" if crit else "No high-urgency items"
    if node == "stage_retrieve":
        cands = update.get("recipe_candidates") or []
        sources = sorted({(c.get("source") or "unknown") for c in cands if isinstance(c, dict)})
        src = ", ".join(sources) if sources else "none"
        return f"{len(cands)} verified candidates · source: {src}"
    if node == "stage_plan":
        plan = update.get("plan") or {}
        meals = plan.get("plan") or []
        titles = [m.get("recipe") for m in meals if isinstance(m, dict) and m.get("recipe")]
        engine = "Claude planner" if use_llm else "offline planner"
        if update.get("validation_ok") is False:
            engine += " · unsafe draft blocked, safe fallback used"
        preview = ", ".join(titles[:3])
        return f"{engine} · {len(meals)} meal(s)" + (f" · {preview}" if preview else "")
    if node == "stage_gap":
        gaps = (update.get("gap_list") or {}).get("gaps") or []
        return f"{len(gaps)} missing item(s) for the shopping list" if gaps else "No shopping gaps yet"
    return ""


def run_plan_pipeline(
    *,
    profile: UserProfile,
    confirmed_items: list[str],
    use_llm: bool | None = None,
    conversation_summary: str = "",
    user_request: str = "",
    on_step: Optional[ProgressCallback] = None,
) -> PipelineState:
    """Run the plan graph. If on_step is provided, stream each real node as it finishes."""
    from src.graph import build_plan_graph

    graph = build_plan_graph()
    init: PipelineState = {
        "intent": "plan",
        "profile": profile.model_dump(),
        "confirmed_items": confirmed_items,
        "errors": [],
        "purchased_or_made": [],
        "conversation_summary": conversation_summary,
        "user_request": user_request,
    }
    if use_llm is not None:
        init["use_llm"] = use_llm

    effective_llm = _use_llm(init)

    if on_step is None:
        return graph.invoke(init)

    # Stream node completions so the UI can show genuine backend progress.
    final: PipelineState = dict(init)
    for event in graph.stream(init, stream_mode="updates"):
        for node, update in event.items():
            if not isinstance(update, dict):
                continue
            final.update(update)
            label = PLAN_STEP_LABELS.get(node, node.replace("_", " ").title())
            detail = _step_detail(node, update, use_llm=effective_llm)
            on_step(node, label, detail)
    return final


def run_substitute_pipeline(
    *,
    profile: UserProfile,
    missing_ingredient: str,
    recipe_context: str,
    selected_option_id: str | None = None,
    confirmed_items: list[str] | None = None,
    plan: dict | None = None,
) -> PipelineState:
    from src.graph import build_substitute_graph

    graph = build_substitute_graph()
    init: PipelineState = {
        "intent": "substitute",
        "profile": profile.model_dump(),
        "missing_ingredient": missing_ingredient,
        "recipe_context": recipe_context,
        "confirmed_items": confirmed_items or [],
        "purchased_or_made": [],
        "errors": [],
    }
    if selected_option_id:
        init["selected_option_id"] = selected_option_id
    if plan:
        init["plan"] = plan
    return graph.invoke(init)
