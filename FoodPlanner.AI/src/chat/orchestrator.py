"""Conversational orchestrator for FoodPlanner.AI."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.chat.intents import classify_intent
from src.chat.nlu import llm_route, update_conversation_summary
from src import waste_tracker
from src.diet_filter import banned_ingredients_for_profile, ingredient_violates, inventory_conflicts
from src.llm import complete_json
from src.kb import normalize_name
from src.memory import profile_summary, save_profile, save_session
from src.pipeline import run_plan_pipeline, run_substitute_pipeline
from src.schemas import (
    ChatMessage,
    InventoryItem,
    MealPlan,
    PlanMeal,
    PlanPreferences,
    RecipeCandidate,
    SessionState,
    SubstitutionResult,
    UserProfile,
)
from src.stages.freshness import format_freshness_summary, score_freshness
from src.stages.gap_list import compute_gap_list
from src.stages.substitution import build_substitution_options, select_substitution_option
from src.stages.vision import detect_from_image_bytes, detect_from_typed_inventory
from src.stages.critical import critical_priority_filter


STAGE_LABELS = [
    ("profile", "Profile"),
    ("preferences", "Preferences"),
    ("inventory", "Inventory"),
    ("confirmation", "Confirmation"),
    ("freshness", "Freshness"),
    ("meal_plan", "Meal Plan"),
    ("adjustments", "Adjustments"),
    ("shopping_list", "Shopping List"),
]


def _assistant(session: SessionState, content: str, kind: str = "text", meta: Optional[dict] = None, quick: Optional[List[str]] = None) -> SessionState:
    session.messages.append(ChatMessage(role="assistant", content=content, kind=kind, meta=meta or {}))
    if quick is not None:
        session.quick_replies = quick
    return session


def _user(session: SessionState, content: str) -> SessionState:
    session.messages.append(ChatMessage(role="user", content=content))
    return session


def new_session(profile: Optional[UserProfile] = None, use_llm: bool = True) -> SessionState:
    session = SessionState(use_llm=use_llm)
    if profile and profile.profile_confirmed:
        session.stage = "preferences"
        session.pref_step = "duration"
        # Long-term memory: surface waste-tracker patterns from past inventories.
        hint = waste_tracker.history_hint(profile.user_id)
        hint_block = f"{hint}\n\n" if hint else ""
        session = _assistant(
            session,
            "Hi! I’m your FoodPlanner.AI assistant.\n\n"
            "I’ll help you use what you already have, prioritize ingredients that may spoil soon, "
            "and create meals that match your saved dietary profile.\n\n"
            + hint_block +
            "To get started, I’ll ask a few quick questions. You can change any answer later.\n\n"
            "How many days or meals would you like me to plan?",
            quick=["1 meal", "3 days", "5 days", "7 days", "Custom"],
        )
        session.awaiting = "pref_duration"
    else:
        session.stage = "profile"
        session.profile_step = "diet_type"
        session = _assistant(
            session,
            "Welcome to FoodPlanner.AI\n\n"
            "Before we look inside your fridge, let’s make sure every recommendation fits your dietary needs.\n\n"
            "What type of diet should every recommendation follow?",
            quick=[
                "Vegetarian",
                "Vegan",
                "Non-vegetarian",
                "Pescatarian",
                "Eggetarian",
                "Flexitarian",
                "Other",
            ],
        )
        session.awaiting = "diet_type"
    return session


def start_new_plan(session: SessionState, profile: UserProfile) -> SessionState:
    """Reset short-term session memory; keep profile."""
    use_llm = session.use_llm
    session = SessionState(use_llm=use_llm, stage="preferences", pref_step="duration")
    return new_session(profile, use_llm=use_llm)


def clear_inventory_for_reupload(session: SessionState) -> SessionState:
    """Clear inventory and reopen the photo / typed-inventory step.

    Used by the sidebar "Clear current inventory" action so the user can upload
    a second fridge photo without having to start an entirely new plan.
    """
    session.inventory = []
    session.pending_conflicts = []
    session.ranked = []
    session.critical_priority = []
    session.recipe_candidates = []
    session.plan = None
    session.substitution = None
    session.gap_list = None
    session.purchased_or_made = []
    session.last_image_hash = None
    session.pending_outcomes = []
    session.rejected_recipes = []
    session.stage = "inventory"
    session = _assistant(
        session,
        "Inventory cleared. Upload a new fridge or pantry photo below, or type the ingredients you have now.",
        kind="inventory_prompt",
        quick=["Type ingredients instead", "Add another photo"],
    )
    session.awaiting = "inventory_input"
    return session


def _parse_list(text: str) -> List[str]:
    parts = re.split(r"[,;\n]| and ", text.lower())
    return [normalize_name(p) for p in parts if normalize_name(p) and normalize_name(p) not in {"none", "no", "n/a"}]


def _active_inventory_names(session: SessionState) -> List[str]:
    return [
        i.normalized_name
        for i in session.inventory
        if i.confirmed and not i.exclude_from_plan and not i.do_not_use
    ]


def _find_item(session: SessionState, name: str) -> Optional[InventoryItem]:
    n = normalize_name(name)
    for item in session.inventory:
        if normalize_name(item.normalized_name) == n or normalize_name(item.display_name) == n:
            return item
        if n in normalize_name(item.normalized_name):
            return item
    return None


# ---- Profile conversation ----

def _handle_profile(session: SessionState, profile: UserProfile, text: str) -> Tuple[SessionState, UserProfile]:
    step = session.profile_step
    t = text.strip().lower()

    if step == "diet_type":
        mapping = {
            "vegetarian": "vegetarian",
            "vegan": "vegan",
            "non-vegetarian": "non-vegetarian",
            "non vegetarian": "non-vegetarian",
            "pescatarian": "pescatarian",
            "eggetarian": "eggetarian",
            "flexitarian": "flexitarian",
            "other": "other",
        }
        diet = mapping.get(t, t.replace(" ", "-"))
        if diet not in mapping.values():
            diet = "other"
        profile.diet_type = diet  # type: ignore[assignment]
        session.profile_step = "cultural"
        session = _assistant(
            session,
            "Got it. Do you follow any cultural or religious food rules?",
            quick=[
                "None",
                "Jain",
                "Halal",
                "Kosher",
                "Hindu dietary preference",
                "No beef",
                "No pork",
                "No onion or garlic",
                "No root vegetables",
            ],
        )
        session.awaiting = "cultural"
        return session, profile

    if step == "cultural":
        if t in {"none", "no"}:
            profile.cultural_rules = []
            session.profile_step = "allergies"
            session = _assistant(
                session,
                "Do you have any food allergies or ingredients that must never be included?\n\n"
                "_FoodPlanner.AI can help filter ingredients, but it cannot verify cross-contamination, "
                "manufacturing environments, or medical suitability. Always check product labels when an allergy is severe._",
                quick=["None", "Peanuts", "Tree nuts", "Dairy", "Eggs", "Gluten", "Soy", "Sesame", "Shellfish", "Fish"],
            )
            session.awaiting = "allergies"
            return session, profile
        if "jain" in t:
            profile.cultural_rules = list(set(profile.cultural_rules + ["jain"]))
            session.profile_step = "jain_detail"
            session = _assistant(
                session,
                "Which Jain food rules should I apply?",
                quick=[
                    "No onion and garlic",
                    "No root vegetables",
                    "No onion, garlic, or root vegetables",
                    "Custom Jain restrictions",
                ],
            )
            session.awaiting = "jain_detail"
            return session, profile
        # allow multi via comma
        rules = _parse_list(text) if "," in text else [text.strip()]
        profile.cultural_rules = list({*profile.cultural_rules, *[r for r in rules if r != "none"]})
        if any("halal" in r or "kosher" in r for r in profile.cultural_rules):
            note = (
                "\n\n_Note: I can filter for ingredient compatibility with your selected preference, "
                "but certification and kitchen preparation cannot be verified by FoodPlanner.AI._"
            )
        else:
            note = ""
        session.profile_step = "allergies"
        session = _assistant(
            session,
            f"Saved. I’ll treat those as hard restrictions.{note}\n\n"
            "Do you have any food allergies or ingredients that must never be included?",
            quick=["None", "Peanuts", "Tree nuts", "Dairy", "Eggs", "Gluten", "Soy", "Sesame", "Shellfish", "Fish"],
        )
        session.awaiting = "allergies"
        return session, profile

    if step == "jain_detail":
        if "custom" in t:
            profile.jain_rules = ["custom — confirm with user notes"]
        elif "onion" in t and "root" in t:
            profile.jain_rules = ["no onion and garlic", "no root vegetables"]
        elif "root" in t:
            profile.jain_rules = ["no root vegetables"]
        else:
            profile.jain_rules = ["no onion and garlic"]
        profile.cultural_rules = list(set(profile.cultural_rules + ["jain"]))
        session.profile_step = "allergies"
        session = _assistant(
            session,
            "I’ll treat those Jain rules as hard restrictions. Do you have any food allergies?",
            quick=["None", "Peanuts", "Tree nuts", "Dairy", "Eggs", "Gluten", "Soy"],
        )
        session.awaiting = "allergies"
        return session, profile

    if step == "allergies":
        # Allergies are the source of truth; sync_aliases rebuilds hard_exclusions /
        # dietary_restrictions so a previously saved "Eggs" ban cannot stick after "None".
        profile.allergies = [] if t in {"none", "no"} else _parse_list(text)
        profile.hard_exclusions = []
        profile.dietary_restrictions = []
        profile.sync_aliases()
        session.profile_step = "servings"
        session = _assistant(
            session,
            "How many people are you cooking for?",
            quick=["1", "2", "3", "4"],
        )
        session.awaiting = "servings"
        return session, profile

    if step == "servings":
        m = re.search(r"\d+", t)
        profile.servings = int(m.group(0)) if m else 2
        session.profile_step = "time"
        session = _assistant(
            session,
            "How much time do you normally want to spend cooking?",
            quick=["Under 15 minutes", "15–30 minutes", "30–45 minutes", "Up to 60 minutes", "No strict limit"],
        )
        session.awaiting = "time"
        return session, profile

    if step == "time":
        if "15" in t and "under" in t:
            profile.cooking_time_preference = "under_15"
        elif "60" in t or "up to" in t:
            profile.cooking_time_preference = "up_to_60"
        elif "45" in t:
            profile.cooking_time_preference = "30_45"
        elif "no strict" in t or "no limit" in t:
            profile.cooking_time_preference = "no_limit"
        else:
            profile.cooking_time_preference = "15_30"
        session.profile_step = "confirm"
        profile.sync_aliases()
        session = _assistant(
            session,
            "Here is what I’ll follow:\n\n" + profile_summary(profile) + "\n\nDoes everything look correct?",
            quick=["Save and continue", "Edit profile"],
        )
        session.awaiting = "profile_confirm"
        return session, profile

    if step == "confirm":
        if "edit" in t:
            session.profile_step = "diet_type"
            session = _assistant(
                session,
                "No problem — let’s rebuild your profile. What diet should every recommendation follow?",
                quick=["Vegetarian", "Vegan", "Non-vegetarian", "Pescatarian", "Eggetarian", "Flexitarian"],
            )
            session.awaiting = "diet_type"
            return session, profile
        profile.profile_confirmed = True
        profile.sync_aliases()
        save_profile(profile)
        session.stage = "preferences"
        session.pref_step = "duration"
        hint = waste_tracker.history_hint(profile.user_id)
        hint_block = f"{hint}\n\n" if hint else ""
        session = _assistant(
            session,
            "Great — profile saved.\n\n"
            "Hi! I’m your FoodPlanner.AI assistant. I’ll help you use what you already have and match your dietary profile.\n\n"
            + hint_block +
            "How many days or meals would you like me to plan?",
            quick=["1 meal", "3 days", "5 days", "7 days", "Custom"],
        )
        session.awaiting = "pref_duration"
        return session, profile

    session = _assistant(session, "Please choose one of the options above, or type your answer.")
    return session, profile


# ---- Preferences ----

def _handle_preferences(session: SessionState, profile: UserProfile, text: str) -> SessionState:
    step = session.pref_step
    t = text.strip().lower()
    prefs = session.preferences

    if step == "duration":
        if "1" in t and "meal" in t:
            prefs.days = 1
        elif "7" in t:
            prefs.days = 7
        elif "5" in t:
            prefs.days = 5
        elif "custom" in t:
            session = _assistant(session, "How many days should I plan?", quick=["2", "3", "4", "6"])
            session.awaiting = "pref_duration"
            return session
        else:
            m = re.search(r"\d+", t)
            prefs.days = int(m.group(0)) if m else 3
        profile.nights = prefs.days
        session.pref_step = "meals"
        session = _assistant(
            session,
            "Which meals should I include?",
            quick=["Dinner", "Lunch", "Breakfast", "Dinner + Lunch", "All meals"],
        )
        session.awaiting = "pref_meals"
        return session

    if step == "meals":
        if "all" in t:
            prefs.meal_types = ["breakfast", "lunch", "dinner"]
        elif "lunch" in t and "dinner" in t:
            prefs.meal_types = ["lunch", "dinner"]
        elif "breakfast" in t:
            prefs.meal_types = ["breakfast"]
        elif "lunch" in t:
            prefs.meal_types = ["lunch"]
        else:
            prefs.meal_types = ["dinner"]
        profile.meal_types = prefs.meal_types
        # Assume common pantry basics by default; users can adjust in chat anytime.
        prefs.assume_staples = True
        if not profile.assumed_staples:
            profile.assumed_staples = ["salt", "pepper", "oil", "water"]
        session.stage = "inventory"
        session = _assistant(
            session,
            "Great. Now show me what you already have.\n\n"
            "Upload or take a clear photo of your fridge or pantry, or type your ingredients. "
            "I'll assume you have basics like salt, pepper, oil, and water — tell me if not.\n\n"
            "You'll be able to correct everything I detect before I plan anything.",
            kind="inventory_prompt",
            quick=["Type ingredients instead", "Skip photo and enter manually"],
        )
        session.awaiting = "inventory_input"
        return session

    return session


def ingest_typed_inventory(session: SessionState, text: str) -> SessionState:
    detected = detect_from_typed_inventory(text)
    for d in detected.detected_items:
        name = normalize_name(d.item)
        if _find_item(session, name):
            continue
        session.inventory.append(
            InventoryItem(
                normalized_name=name,
                display_name=d.item,
                source="typed",
                confidence=1.0,
                confirmed=False,
            )
        )
    session.stage = "confirmation"
    session = _assistant(
        session,
        "I found these items. Please check the list before I create your plan—"
        "I may miss items hidden behind containers or misread packaging.",
        kind="inventory_review",
        meta={"inventory": [i.model_dump() for i in session.inventory]},
        quick=["Confirm inventory", "Add ingredient manually", "Start over"],
    )
    session.awaiting = "confirm_inventory"
    # Detect conflicts
    return _annotate_conflicts(session)


def ingest_photo(session: SessionState, image_bytes: bytes, media_type: str = "image/jpeg") -> SessionState:
    # Hash the image for duplicate-upload detection (never used to identify content).
    session.last_image_hash = waste_tracker.image_hash(image_bytes)
    try:
        detected = detect_from_image_bytes(image_bytes, media_type=media_type)
    except Exception as e:
        session = _assistant(
            session,
            "I couldn’t identify the ingredients clearly enough from that photo. "
            f"Please try another angle, upload an additional photo, or type the ingredients manually.\n\n"
            f"_(Technical note hidden from users in production; detail: {type(e).__name__})_",
            quick=["Type ingredients instead", "Try another photo"],
        )
        # cleaner message without technical detail for UX:
        session.messages[-1].content = (
            "I couldn’t identify the ingredients clearly enough from that photo. "
            "Please try another angle, upload an additional photo, or type the ingredients manually."
        )
        session.awaiting = "inventory_input"
        return session

    for d in detected.detected_items:
        name = normalize_name(d.name or d.item)
        conf_map = {"high": 0.9, "medium": 0.7, "low": 0.45}
        conf = conf_map.get(d.confidence, 0.7) if isinstance(d.confidence, str) else float(d.confidence)
        if _find_item(session, name):
            continue
        session.inventory.append(
            InventoryItem(
                normalized_name=name,
                display_name=d.name or d.item,
                source="photo",
                confidence=conf,
                needs_confirmation=conf < 0.75 or d.needs_confirmation,
                quantity_text=d.quantity_text,
                confirmed=False,
            )
        )
    session.stage = "confirmation"
    names = ", ".join(i.display_name for i in session.inventory) or "nothing clear"
    session = _assistant(
        session,
        f"I found: {names}.\n\n"
        "Please check the list before I create your plan—I may miss items hidden behind containers or misread packaging.",
        kind="inventory_review",
        meta={"inventory": [i.model_dump() for i in session.inventory]},
        quick=["Confirm inventory", "Add ingredient manually", "Add another photo"],
    )
    session.awaiting = "confirm_inventory"
    return _annotate_conflicts(session)


def _annotate_conflicts(session: SessionState, profile: Optional[UserProfile] = None) -> SessionState:
    # Conflicts are resolved when confirming; message if any hard conflicts exist and profile provided later
    return session


def _conflict_items(session: SessionState, profile: UserProfile) -> List[InventoryItem]:
    return [i for i in session.inventory if not i.exclude_from_plan and inventory_conflicts(i.normalized_name, profile)]


def confirm_inventory(session: SessionState, profile: UserProfile) -> SessionState:
    conflicts = _conflict_items(session, profile)
    if conflicts:
        # Ask ONCE about all conflicting items to avoid a question loop.
        session.pending_conflicts = [
            {"id": i.id, "name": i.normalized_name} for i in conflicts
        ]
        names = ", ".join(f"**{i.display_name}**" for i in conflicts)
        session = _assistant(
            session,
            f"I detected {names} — these conflict with your saved dietary profile. Should I:\n\n"
            "1. Keep them recorded but exclude them from your plan (e.g., they're for someone else)\n"
            "2. Remove them from the inventory (e.g., detected incorrectly)",
            quick=[
                "Exclude from plan",
                "Remove from inventory",
            ],
        )
        session.awaiting = "conflict:bulk"
        return session

    for item in session.inventory:
        if not item.exclude_from_plan:
            item.confirmed = True

    # Freshness
    session.stage = "freshness"
    ranked = score_freshness([i for i in session.inventory if i.confirmed and not i.exclude_from_plan])
    session.ranked = ranked.ranked
    crit = critical_priority_filter(ranked)
    session.critical_priority = crit.critical_priority

    # Long-term waste tracking: persist the CONFIRMED inventory as a snapshot.
    source = "image" if session.last_image_hash else "typed"
    snapshot, duplicate, disappeared = waste_tracker.record_snapshot(
        profile.user_id,
        ranked.ranked,
        source=source,
        image_reference=session.last_image_hash,
    )
    session.last_snapshot_id = snapshot.snapshot_id
    session.last_image_hash = None
    session.pending_outcomes = disappeared

    session = _assistant(
        session,
        format_freshness_summary(ranked)
        + "\n\nShould I generate your meal plan now?",
        kind="freshness",
        meta={"ranked": [r.model_dump(by_alias=True) for r in ranked.ranked]},
        quick=["Yes, generate plan", "Adjust priorities", "Edit inventory"],
    )
    session.awaiting = "generate_plan"

    if duplicate is not None:
        session = _assistant(
            session,
            "One quick check: this photo looks identical to one you uploaded recently, so I haven’t "
            "counted it as a new grocery purchase yet. Is this the same inventory as before, or new groceries?",
            quick=["Same inventory as before", "New grocery purchase"],
        )
        session.awaiting = f"dup_check:{snapshot.snapshot_id}"
    elif session.pending_outcomes:
        session = _ask_next_outcome(session)
    return session


OUTCOME_QUICK = ["Used", "Still have it", "Bought again", "Spoiled", "Thrown away", "Donated", "Not sure"]


def _ask_next_outcome(session: SessionState) -> SessionState:
    name = session.pending_outcomes[0]
    session = _assistant(
        session,
        f"Quick check for your waste tracker — what happened to the **{name}** from your previous inventory?",
        quick=OUTCOME_QUICK,
    )
    session.awaiting = f"outcome:{name}"
    return session


def generate_plan(
    session: SessionState,
    profile: UserProfile,
    on_step: Optional[Callable[[str, str, str], None]] = None,
) -> SessionState:
    """Build a meal plan. on_step(key, label, detail) reports real backend stages to the UI."""
    inventory = _active_inventory_names(session)
    if not inventory:
        session = _assistant(session, "I need a confirmed inventory before I can plan. Please add ingredients first.")
        session.stage = "inventory"
        return session

    profile.nights = session.preferences.days or profile.nights
    profile.sync_aliases()

    pipeline_steps: List[Dict[str, str]] = []

    def emit(key: str, label: str, detail: str = "") -> None:
        pipeline_steps.append({"key": key, "label": label, "detail": detail})
        if on_step:
            on_step(key, label, detail)

    emit(
        "inventory",
        "Reading your confirmed inventory",
        f"{len(inventory)} ingredients: {', '.join(inventory[:8])}"
        + ("…" if len(inventory) > 8 else ""),
    )
    emit(
        "profile",
        "Loading dietary profile (long-term memory)",
        f"{profile.diet_type}"
        + (f" · {', '.join(profile.cultural_rules)}" if profile.cultural_rules else "")
        + (f" · allergies: {', '.join(profile.allergies)}" if profile.allergies else "")
        + f" · {profile.servings} serving(s) · {profile.nights} day(s)",
    )

    # Refresh short-term memory, then build this call's prompt as
    # long-term memory (profile) + short-term memory (summary) + the user's
    # current request — all handed to the planning LLM.
    session = update_conversation_summary(session)
    user_request = (
        session.messages[-1].content
        if session.messages and session.messages[-1].role == "user"
        else ""
    )
    if session.conversation_summary:
        emit("memory", "Refreshing short-term conversation memory", session.conversation_summary[:160])
    else:
        emit("memory", "Refreshing short-term conversation memory", "No prior summary yet — using this session’s inventory and profile")

    engine = "Claude" if session.use_llm else "offline deterministic planner"
    emit("pipeline_start", f"Starting plan pipeline ({engine})", "freshness → critical → retrieve → plan → gaps")

    try:
        result = run_plan_pipeline(
            profile=profile,
            confirmed_items=inventory,
            use_llm=session.use_llm,
            conversation_summary=session.conversation_summary,
            user_request=user_request,
            on_step=emit,
        )
    except Exception:
        emit("retry", "Primary pipeline failed — retrying with offline recipes", "Claude/API unavailable or invalid output")
        session = _assistant(
            session,
            "I had trouble structuring that result. I’m retrying with the same confirmed information using the offline recipe collection.",
        )
        result = run_plan_pipeline(
            profile=profile,
            confirmed_items=inventory,
            use_llm=False,
            on_step=emit,
        )

    if not result.get("recipe_candidates") and not result.get("plan"):
        session = _assistant(
            session,
            "I couldn’t find a verified recipe that uses your current ingredients while meeting all of your restrictions. "
            "I can loosen a soft preference, suggest a simple meal from the local fallback collection, or help you add one or two ingredients.",
            quick=["Use offline recipes", "Add an ingredient", "Edit profile"],
        )
        return session

    from src.schemas import RankedItem

    raw_ranked = result.get("ranked") or []
    session.ranked = [
        r if isinstance(r, RankedItem) else RankedItem.model_validate(r) for r in raw_ranked
    ] or session.ranked
    session.critical_priority = result.get("critical_priority") or session.critical_priority
    raw_cands = result.get("recipe_candidates") or []
    session.recipe_candidates = [
        c if isinstance(c, RecipeCandidate) else RecipeCandidate.model_validate(c) for c in raw_cands
    ]

    plan = MealPlan.model_validate(result["plan"])
    # If the user previously rejected recipes (via Replace), rebuild the plan from the
    # remaining candidates so regenerating doesn't just return the same dishes.
    rejected = {normalize_name(r) for r in session.rejected_recipes}
    if rejected and session.recipe_candidates:
        from src.stages.planning import plan_meals_deterministic

        alt_cands = [c for c in session.recipe_candidates if normalize_name(c.title) not in rejected]
        if alt_cands:
            emit(
                "rejected",
                "Skipping recipes you already rejected this session",
                ", ".join(session.rejected_recipes[:6]),
            )
            plan = plan_meals_deterministic(
                inventory=inventory,
                critical_priority=session.critical_priority,
                recipe_candidates=alt_cands,
                profile=profile,
            )
        # If every remaining title was still rejected, keep the pipeline plan but tell the user below.

    # Enrich why_selected
    for meal in plan.plan:
        if not meal.why_selected and meal.uses_critical:
            meal.why_selected = f"Uses {', '.join(meal.uses_critical)} first."
        elif not meal.why_selected:
            meal.why_selected = "Fits your inventory, time limit, and dietary profile."
        meal.status = "proposed"
    session.plan = plan
    session.stage = "meal_plan"

    flagged = plan.flagged_for_other_use
    flag_note = ""
    if flagged:
        flag_note = "\n\nExcluded from cooking (diet conflict): " + ", ".join(f.item for f in flagged)

    titles = [m.recipe for m in plan.plan]
    emit("finalize", "Finalizing meal cards", ", ".join(titles) if titles else "empty plan")

    replace_day = plan.plan[0].day or plan.plan[0].night if plan.plan else 1
    if len(plan.plan) > 1:
        # Prefer offering a mid-plan day when more than one meal exists.
        replace_day = plan.plan[min(1, len(plan.plan) - 1)].day or plan.plan[min(1, len(plan.plan) - 1)].night
    session = _assistant(
        session,
        "Here’s a draft plan based on your confirmed inventory and profile." + flag_note
        + "\n\nHow does this plan look? You can accept it, replace one meal, change the number of days, "
        "request a different cuisine, or tell me what you don’t like.",
        kind="meal_cards",
        meta={"plan": plan.model_dump(), "pipeline_steps": pipeline_steps},
        quick=[
            "Accept plan",
            f"Replace Day {replace_day}",
            "Make it faster",
            "Help with missing ingredients",
        ],
    )
    session.awaiting = "plan_feedback"
    return session


def _candidate_to_meal(
    cand: RecipeCandidate,
    *,
    day: int,
    meal_type: str,
    profile: UserProfile,
    inventory: List[str],
    critical: List[str],
    why: str,
) -> PlanMeal:
    inv_n = {normalize_name(i) for i in inventory}
    staples = {"salt", "oil", "water", "pepper"}
    inv_used = [i for i in cand.ingredients if normalize_name(i) in inv_n]
    missing = [
        i
        for i in cand.ingredients
        if normalize_name(i) not in inv_n and normalize_name(i) not in staples
    ]
    uses_crit = [
        c for c in critical if any(normalize_name(c) in normalize_name(i) for i in cand.ingredients)
    ]
    return PlanMeal(
        night=day,
        day=day,
        meal_type=meal_type,
        recipe=cand.title,
        recipe_id=str(cand.id) if cand.id is not None else None,
        time_min=min(cand.ready_in_minutes or profile.time_limit_min, profile.time_limit_min),
        servings=profile.servings,
        uses_critical=uses_crit,
        ingredients_from_inventory=inv_used,
        steps=cand.steps[:6] or [f"Prepare {cand.title} using available ingredients."],
        missing_ingredients=missing[:6],
        why_selected=why,
        status="replaced",
    )


def _rank_candidates(
    candidates: List[RecipeCandidate],
    inventory: List[str],
    critical: List[str],
) -> List[RecipeCandidate]:
    inv_n = {normalize_name(i) for i in inventory}
    crit_n = {normalize_name(c) for c in critical}
    scored = []
    for cand in candidates:
        ing_n = {normalize_name(i) for i in cand.ingredients}
        scored.append((len(ing_n & crit_n), len(ing_n & inv_n), cand))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [c for _, _, c in scored]


def _recipe_key(title: str) -> str:
    """Stable key for recipe titles. Do NOT use ingredient normalize_name — that
    singularizes trailing 's' and can collide unrelated titles."""
    return " ".join((title or "").strip().lower().split())


def replace_meal(session: SessionState, profile: UserProfile, day: int) -> SessionState:
    """Swap one day for a DIFFERENT verified recipe. Never re-suggest the same title."""
    if not session.plan:
        return _assistant(session, "There’s no plan to edit yet.")

    day = int(day)
    target_idx = next(
        (i for i, m in enumerate(session.plan.plan) if int(m.day or m.night or 0) == day),
        None,
    )
    if target_idx is None:
        return _assistant(
            session,
            f"I couldn’t find Day {day} in the current plan.",
            quick=[f"Replace Day {m.day or m.night}" for m in session.plan.plan],
        )

    target = session.plan.plan[target_idx]
    current_title = target.recipe
    inventory = _active_inventory_names(session)

    # Track rejected titles so we never cycle back within this session.
    if current_title and _recipe_key(current_title) not in {_recipe_key(r) for r in session.rejected_recipes}:
        session.rejected_recipes.append(current_title)

    exclude = {_recipe_key(r) for r in session.rejected_recipes}
    exclude |= {_recipe_key(m.recipe) for m in session.plan.plan if m.recipe}
    exclude.add(_recipe_key(current_title))

    def _alts(cands: List[RecipeCandidate]) -> List[RecipeCandidate]:
        return [c for c in cands if _recipe_key(c.title) not in exclude]

    def _merge_candidates(extra: List[RecipeCandidate]) -> None:
        seen = {_recipe_key(c.title) for c in session.recipe_candidates}
        for c in extra:
            key = _recipe_key(c.title)
            if key and key not in seen:
                session.recipe_candidates.append(c)
                seen.add(key)

    # Always widen the pool with diet-safe fixtures + a fresh search so we are not
    # stuck with a single Spoonacular hit (the usual cause of "same plan again").
    from src.diet_filter import filter_recipe_candidates
    from src.kb import load_fixture_recipes
    from src.tools.spoonacular import _candidate_from_fixture, search_by_ingredients

    fixtures = filter_recipe_candidates(
        [_candidate_from_fixture(r) for r in load_fixture_recipes()],
        profile,
    )
    _merge_candidates(fixtures)
    try:
        _merge_candidates(search_by_ingredients(inventory, profile, number=12))
    except Exception:
        pass

    alts = _alts(session.recipe_candidates)
    if not alts:
        session = _assistant(
            session,
            f"I don’t have a different verified recipe for Day {day} that still fits your "
            f"inventory and dietary profile. **{current_title}** is the best match I have right now.\n\n"
            "I can keep it, help with a missing ingredient, or you can add something to your inventory "
            "so I have more options.",
            kind="meal_cards",
            meta={"plan": session.plan.model_dump()},
            quick=["Keep this meal", "Help with missing ingredients", "Add an ingredient"],
        )
        session.awaiting = "plan_feedback"
        return session

    ranked = _rank_candidates(alts, inventory, session.critical_priority)
    # Rotate through alternatives so repeated "Replace" clicks keep changing.
    pick = ranked[(len(session.rejected_recipes) - 1) % len(ranked)]
    if _recipe_key(pick.title) == _recipe_key(current_title):
        pick = next((c for c in ranked if _recipe_key(c.title) != _recipe_key(current_title)), None)
    if pick is None:
        session = _assistant(
            session,
            f"I couldn’t find a different dish than **{current_title}** that still fits your constraints.",
            kind="meal_cards",
            meta={"plan": session.plan.model_dump()},
            quick=["Keep this meal", "Help with missing ingredients"],
        )
        session.awaiting = "plan_feedback"
        return session

    replacement = _candidate_to_meal(
        pick,
        day=day,
        meal_type=target.meal_type,
        profile=profile,
        inventory=inventory,
        critical=session.critical_priority,
        why=f"Replacement for Day {day} based on your feedback.",
    )

    # Rebuild the plan list so the change cannot be lost to a stale reference.
    new_meals = list(session.plan.plan)
    new_meals[target_idx] = replacement
    session.plan = MealPlan(
        plan_id=session.plan.plan_id,
        plan=new_meals,
        flagged_for_other_use=session.plan.flagged_for_other_use,
        clarification=session.plan.clarification,
        confirmed=False,
        unresolved_gaps=session.plan.unresolved_gaps,
    )
    session.stage = "adjustments"
    session = _assistant(
        session,
        f"Understood. I kept the other days unchanged and updated Day {day} to **{replacement.recipe}**.",
        kind="meal_cards",
        meta={"plan": session.plan.model_dump()},
        quick=["Accept plan", "Replace another meal", "Help with missing ingredients"],
    )
    session.awaiting = "plan_feedback"
    return session


DETAILED_RECIPE_SYSTEM = """You write complete, kitchen-ready recipes for FoodPlanner.AI.
Return ONLY valid JSON with this schema:
{
  "title": string,
  "servings": integer,
  "total_time_min": integer,
  "ingredients": [{"item": string, "quantity": string}],
  "steps": [string],
  "tips": [string]
}
Rules:
- Scale EVERY ingredient quantity precisely to the requested number of servings,
  using specific measurements (grams/cups/tablespoons/pieces).
- The recipe must strictly respect the dietary profile provided. Never include a
  restricted or allergenic ingredient.
- Write 6-12 detailed steps with pan/heat levels, timings, and visual doneness cues,
  so a beginner can follow them.
- Base the recipe on the dish name and the listed ingredients; simple staples
  (salt, pepper, oil, water) may be assumed.
"""


def get_detailed_recipe(session: SessionState, profile: UserProfile, day: int) -> SessionState:
    """Produce a full recipe for a planned meal: measured ingredients scaled to the
    profile's servings plus detailed instructions. Cached on the meal after first use."""
    if not session.plan or not session.plan.plan:
        return _assistant(session, "There’s no meal plan yet — generate one first.", quick=["Yes, generate plan"])
    meal = next((m for m in session.plan.plan if (m.day or m.night) == day), None)
    if not meal:
        return _assistant(session, f"I couldn’t find a meal on day {day}.", quick=["Review plan"])

    detail = meal.detailed_recipe
    if not detail and session.use_llm:
        available = meal.ingredients_from_inventory + meal.extra_pantry_items
        prompt = (
            f"Dish: {meal.recipe}\n"
            f"Servings required: {profile.servings}\n"
            f"Meal type: {meal.meal_type}; target total time: about {meal.time_min} minutes\n"
            f"Ingredients from the user's kitchen: {', '.join(available) or 'unknown'}\n"
            f"Ingredients they plan to buy: {', '.join(meal.missing_ingredients) or 'none'}\n"
            f"Assumed staples: {', '.join(profile.assumed_staples) or 'salt, pepper, oil, water'}\n"
            f"Outline steps to expand: {' | '.join(meal.steps) or 'none'}\n\n"
            f"DIETARY PROFILE (STRICT):\n{profile_summary(profile)}\n\n"
            "Respond with the JSON object only."
        )
        try:
            result = complete_json(system=DETAILED_RECIPE_SYSTEM, user=prompt, max_tokens=1800)
            if isinstance(result, dict) and result.get("ingredients") and result.get("steps"):
                # Deterministic safety net: reject the detail if any ingredient
                # violates the profile, regardless of what the model claims.
                banned = banned_ingredients_for_profile(profile)
                safe = not any(
                    ingredient_violates(str(i.get("item", "")), banned)
                    for i in result["ingredients"]
                    if isinstance(i, dict)
                )
                if safe:
                    detail = result
        except Exception:
            detail = None
        if detail:
            meal.detailed_recipe = detail

    if detail:
        ing_lines = "\n".join(
            f"• {i.get('quantity', '')} {i.get('item', '')}".strip()
            for i in detail.get("ingredients", [])
            if isinstance(i, dict)
        )
        step_lines = "\n".join(f"{n}. {s}" for n, s in enumerate(detail.get("steps", []), 1))
        tip_lines = "\n".join(f"• {t}" for t in detail.get("tips", []) or [])
        body = (
            f"**{detail.get('title', meal.recipe)}** — serves {detail.get('servings', profile.servings)} · "
            f"about {detail.get('total_time_min', meal.time_min)} min\n\n"
            f"**Ingredients**\n{ing_lines}\n\n"
            f"**Steps**\n{step_lines}"
        )
        if tip_lines:
            body += f"\n\n**Tips**\n{tip_lines}"
    else:
        # Offline fallback: everything we know, with a note about measurements.
        all_ingredients = meal.ingredients_from_inventory + meal.extra_pantry_items + meal.missing_ingredients
        ing_lines = "\n".join(f"• {i} — adjust for {profile.servings} servings" for i in all_ingredients)
        step_lines = "\n".join(f"{n}. {s}" for n, s in enumerate(meal.steps, 1)) or "1. No stored steps for this recipe."
        body = (
            f"**{meal.recipe}** — serves {profile.servings} · about {meal.time_min} min\n\n"
            f"**Ingredients**\n{ing_lines}\n\n"
            f"**Steps**\n{step_lines}\n\n"
            "_Enable Claude (sidebar toggle) for exact measurements and fully detailed instructions._"
        )

    session = _assistant(
        session,
        body,
        kind="recipe_detail",
        quick=["Accept plan", f"Replace Day {day}", "Help with missing ingredients"],
    )
    session.awaiting = "plan_feedback"
    return session


def start_substitution(session: SessionState, profile: UserProfile, missing: str, recipe: str) -> SessionState:
    options = build_substitution_options(missing_ingredient=missing, recipe_context=recipe, profile=profile)
    session.substitution = options
    if options.status == "no_substitute":
        session = _assistant(
            session,
            "I couldn’t find a verified substitution that meets your saved dietary constraints. "
            "I won’t suggest an unverified replacement. I can replace the recipe or add the missing ingredient to the shopping list.",
            quick=["Replace this meal", "Add to shopping list"],
        )
        session.awaiting = "plan_feedback"
        return session

    lines = [
        f"**{missing}** isn’t in your confirmed inventory for **{recipe}**.",
        "Before I add it to your shopping list, here are grounded options:",
        "",
    ]
    quick = []
    for d in options.derivation_options:
        lines.append(f"• {d.label} (needs: {', '.join(d.base_ingredients)})")
        quick.append(d.label)
    for s in options.store_options:
        lines.append(f"• Use {s.substitute}")
        quick.append(f"Use {s.substitute}")
    lines.append("• Choose a different meal")
    lines.append("• Add to shopping list")
    quick.extend(["Choose a different meal", "Add to shopping list"])
    session = _assistant(session, "\n".join(lines), kind="substitution_options", meta=options.model_dump(), quick=quick)
    session.awaiting = f"sub_select:{missing}"
    return session


def apply_substitution_choice(session: SessionState, profile: UserProfile, text: str) -> SessionState:
    sub = session.substitution
    if not sub:
        return _assistant(session, "I don’t have an open substitution question right now.")
    t = text.strip().lower()
    if "shopping" in t or "add to" in t:
        missing = sub.missing_ingredient or ""
        if session.plan and missing:
            # ensure on first meal missing list
            if session.plan.plan and missing not in session.plan.plan[0].missing_ingredients:
                session.plan.plan[0].missing_ingredients.append(missing)
        session = _assistant(session, f"I’ll keep **{missing}** as a shopping gap. You can confirm the plan when ready.", quick=["Accept plan", "Review missing ingredients"])
        session.awaiting = "plan_feedback"
        return session
    if "different meal" in t or "replace" in t:
        return replace_meal(session, profile, day=(session.plan.plan[0].day if session.plan and session.plan.plan else 1))

    # Match derivation / store option
    option_id = None
    for d in sub.derivation_options:
        if d.label.lower() in t or d.option_id.lower() in t or t in d.label.lower():
            option_id = d.option_id
            break
    if not option_id:
        for i, s in enumerate(sub.store_options):
            if s.substitute.lower() in t or t in s.substitute.lower():
                option_id = f"store_{i}"
                break
    if not option_id and sub.derivation_options:
        # "make from milk" style
        for d in sub.derivation_options:
            if any(b in t for b in d.base_ingredients):
                option_id = d.option_id
                break

    if not option_id:
        session = _assistant(session, "Please pick one of the listed options.", quick=session.quick_replies)
        return session

    selected = select_substitution_option(sub, option_id)
    session.substitution = selected
    if selected.status in {"selected", "ok"} and selected.source == "derivation_kb":
        steps = "\n".join(f"{i}. {s}" for i, s in enumerate(selected.method_steps, 1))
        session = _assistant(
            session,
            f"**{selected.substitute}**\n\n{steps}\n\n"
            "Would you like me to use this homemade option in the plan?",
            quick=["Yes, use homemade", "Show another option", "Add to shopping list"],
        )
        session.awaiting = "sub_confirm_homemade"
        return session

    if selected.status == "ok":
        session.purchased_or_made.append(selected.substitute or "")
        # treat as resolved missing
        if selected.missing_ingredient:
            session.purchased_or_made.append(selected.missing_ingredient)
        session = _assistant(
            session,
            f"Great — I’ll use **{selected.substitute}**. Prep: {selected.prep}",
            quick=["Accept plan", "Help with another missing ingredient"],
        )
        session.awaiting = "plan_feedback"
        return session

    return session


def accept_plan_and_shop(session: SessionState, profile: UserProfile) -> SessionState:
    if not session.plan:
        return _assistant(session, "There’s no plan to confirm yet.")
    session.plan.confirmed = True
    staples = set(profile.assumed_staples) if profile.assumed_staples else None
    gaps = compute_gap_list(
        session.plan,
        inventory=_active_inventory_names(session),
        pantry_staples=staples,
        purchased_or_made=session.purchased_or_made,
    )
    session.gap_list = gaps
    session.stage = "shopping_list"
    session = _assistant(
        session,
        "Your plan is confirmed. Here’s your gap-only shopping list:\n\n" + gaps.markdown,
        kind="shopping",
        meta=gaps.model_dump(),
        quick=["Start new plan", "Edit profile", "Review plan"],
    )
    session.awaiting = None
    return session


def handle_message(
    session: SessionState,
    profile: UserProfile,
    text: str,
) -> Tuple[SessionState, UserProfile]:
    text = (text or "").strip()
    if not text:
        return session, profile

    session = _user(session, text)
    intent, meta = classify_intent(text, session.stage, session.awaiting)

    if intent == "unsafe_or_out_of_scope":
        session = _assistant(
            session,
            "I can’t ignore saved allergies or dietary restrictions, and I won’t change system safety rules "
            "from a chat message. Your hard constraints stay in place. How would you like to continue planning?",
            quick=["Continue planning", "Edit profile", "Start new plan"],
        )
        save_session(session)
        return session, profile

    if intent == "reset_current_plan":
        session = start_new_plan(session, profile)
        save_session(session)
        return session, profile

    if intent == "edit_profile":
        profile.profile_confirmed = False
        session.stage = "profile"
        session.profile_step = "diet_type"
        session = _assistant(
            session,
            "Let’s update your profile. What diet should every recommendation follow?",
            quick=["Vegetarian", "Vegan", "Non-vegetarian", "Pescatarian", "Eggetarian"],
        )
        session.awaiting = "diet_type"
        save_session(session)
        return session, profile

    # Profile flow
    if session.stage == "profile":
        session, profile = _handle_profile(session, profile, text)
        save_profile(profile) if profile.profile_confirmed else None
        save_session(session)
        return session, profile

    if session.stage == "preferences":
        session = _handle_preferences(session, profile, text)
        save_session(session)
        return session, profile

    # Conflict resolution (handles ALL pending conflicts in one turn — no loops)
    if session.awaiting and session.awaiting.startswith("conflict"):
        conflict_ids = {c["id"] for c in session.pending_conflicts}
        conflict_names = [i.display_name for i in session.inventory if i.id in conflict_ids]
        t = text.lower()
        if conflict_ids:
            if "incorrect" in t or "remove" in t or "delete" in t:
                session.inventory = [i for i in session.inventory if i.id not in conflict_ids]
                session = _assistant(
                    session,
                    "Removed from inventory: " + ", ".join(f"**{n}**" for n in conflict_names) + ".",
                )
            else:
                for i in session.inventory:
                    if i.id in conflict_ids:
                        i.exclude_from_plan = True
                        i.do_not_use = True
                        i.confirmed = True
                session = _assistant(
                    session,
                    "I'll keep " + ", ".join(f"**{n}**" for n in conflict_names)
                    + " recorded but exclude them from your plan.",
                )
        session.pending_conflicts = []
        session.awaiting = None
        return confirm_inventory(session, profile), profile

    # Duplicate-photo check ("same inventory or new groceries?")
    if session.awaiting and session.awaiting.startswith("dup_check:"):
        snap_id = session.awaiting.split(":", 1)[1]
        same = "same" in text.lower()
        waste_tracker.set_new_purchase_flag(profile.user_id, snap_id, not same)
        msg = (
            "Got it — I’ll treat it as the same inventory, not a new purchase."
            if same
            else "Noted — I’ve logged this as a new grocery purchase."
        )
        session = _assistant(session, msg, quick=["Yes, generate plan", "Edit inventory"])
        session.awaiting = "generate_plan"
        if session.pending_outcomes:
            session = _ask_next_outcome(session)
        save_session(session)
        return session, profile

    # Ingredient outcome follow-up ("what happened to the spinach?")
    if session.awaiting and session.awaiting.startswith("outcome:"):
        if intent == "generate_plan" or "generate" in text.lower():
            # User skipped the follow-up; leave outcomes unresolved and move on.
            session.pending_outcomes = []
            session.awaiting = None
            session = generate_plan(session, profile)
            save_session(session)
            return session, profile
        name = session.awaiting.split(":", 1)[1]
        outcome = waste_tracker.parse_outcome_answer(text)
        waste_tracker.record_outcome(profile.user_id, name, outcome, session.last_snapshot_id)
        session.pending_outcomes = [n for n in session.pending_outcomes if n != name]
        if outcome in waste_tracker.WASTE_OUTCOMES:
            ack = (
                f"Thanks — I’ve logged **{name}** as confirmed waste. "
                "Over time this helps me suggest better buying amounts."
            )
        elif outcome == "not_sure":
            ack = f"No problem — I’ll leave **{name}** as unresolved."
        else:
            ack = f"Thanks — noted **{name}** as {outcome.replace('_', ' ')}."
        if session.pending_outcomes:
            session = _assistant(session, ack)
            session = _ask_next_outcome(session)
        else:
            session = _assistant(
                session,
                ack + "\n\nShould I generate your meal plan now?",
                quick=["Yes, generate plan", "Edit inventory"],
            )
            session.awaiting = "generate_plan"
        save_session(session)
        return session, profile

    if session.awaiting and session.awaiting.startswith("sub_select"):
        session = apply_substitution_choice(session, profile, text)
        save_session(session)
        return session, profile

    if session.awaiting == "sub_confirm_homemade":
        if text.lower().startswith("yes"):
            missing = (session.substitution.missing_ingredient if session.substitution else None) or ""
            if missing and missing not in session.purchased_or_made:
                session.purchased_or_made.append(missing)
            # remove from missing lists
            if session.plan:
                for meal in session.plan.plan:
                    meal.missing_ingredients = [m for m in meal.missing_ingredients if normalize_name(m) != normalize_name(missing)]
            session = _assistant(
                session,
                f"Done. I updated the plan to use homemade **{missing}** and removed it from shopping gaps.",
                kind="meal_cards",
                meta={"plan": session.plan.model_dump() if session.plan else {}},
                quick=["Accept plan", "Help with another missing ingredient"],
            )
            session.awaiting = "plan_feedback"
        else:
            session = _assistant(session, "Okay — pick another option or add it to the shopping list.", quick=["Add to shopping list", "Choose a different meal"])
            session.awaiting = f"sub_select:{missing if False else (session.substitution.missing_ingredient if session.substitution else '')}"
        save_session(session)
        return session, profile

    # Inventory typed
    if session.stage == "inventory" or session.awaiting in {"inventory_input", "typed_inventory"}:
        if "type" in text.lower() or "manual" in text.lower() or "skip photo" in text.lower():
            session = _assistant(
                session,
                "Type your ingredients as a comma-separated list (for example: spinach, rice, tomatoes).",
                quick=["Add another photo"],
            )
            session.awaiting = "typed_inventory"
            save_session(session)
            return session, profile
        if intent == "answer_question" and (
            session.awaiting == "typed_inventory"
            or ("," in text and session.stage in {"inventory", "confirmation"})
        ):
            session = ingest_typed_inventory(session, text)
            # Check conflicts immediately if profile known
            conflicts = _conflict_items(session, profile)
            if conflicts:
                save_session(session)
                return confirm_inventory(session, profile), profile
            save_session(session)
            return session, profile

    if intent == "upload_inventory" and meta.get("mode") == "photo":
        session.stage = "inventory"
        session = _assistant(
            session,
            "Sure — upload or take another photo below, or type your ingredients instead.",
            kind="inventory_prompt",
            quick=["Type ingredients instead"],
        )
        session.awaiting = "inventory_input"
        save_session(session)
        return session, profile

    if intent == "edit_inventory":
        session.stage = "confirmation"
        session = _assistant(
            session,
            "Here’s your current inventory — make any changes, then confirm.",
            kind="inventory_review",
            meta={"inventory": [i.model_dump() for i in session.inventory]},
            quick=["Confirm inventory", "Add ingredient manually"],
        )
        session.awaiting = "confirm_inventory"
        save_session(session)
        return session, profile

    if intent == "update_freshness":
        raw = (meta.get("raw") or text).lower()
        m = re.search(r"use\s+(?:the\s+)?(.+?)\s+first", raw)
        handled = None
        if m:
            item = _find_item(session, m.group(1))
            if item:
                item.use_soon_user_flag = True
                handled = f"Got it — I’ll prioritize **{item.display_name}** first."
        if not handled and any(x in raw for x in ["actually fresh", "don't prioritize", "do not prioritize", "froze", "frozen"]):
            for i in session.inventory:
                if i.normalized_name in raw:
                    i.use_soon_user_flag = False
                    i.frozen = "froz" in raw
                    handled = f"Understood — I won’t rush **{i.display_name}**."
                    break
        session = _assistant(
            session,
            handled
            or "Tell me what to change — for example, “use the spinach first” or “the milk is actually fresh”.",
            quick=["Yes, generate plan", "Edit inventory"],
        )
        session.awaiting = "generate_plan"
        save_session(session)
        return session, profile

    if intent == "show_plan":
        if session.plan:
            session = _assistant(
                session,
                "Here’s your current plan.",
                kind="meal_cards",
                meta={"plan": session.plan.model_dump()},
                quick=["Accept plan", "Replace a meal", "Help with missing ingredients"],
            )
            session.awaiting = "plan_feedback"
        else:
            session = _assistant(session, "There’s no plan yet — want me to generate one?", quick=["Yes, generate plan"])
            session.awaiting = "generate_plan"
        save_session(session)
        return session, profile

    if intent == "continue_planning":
        if session.plan:
            session = _assistant(
                session,
                "Here’s where we left off — your current plan.",
                kind="meal_cards",
                meta={"plan": session.plan.model_dump()},
                quick=["Accept plan", "Replace a meal", "Help with missing ingredients"],
            )
            session.awaiting = "plan_feedback"
        elif session.stage == "freshness":
            session = _assistant(session, "Ready when you are — should I generate the plan?", quick=["Yes, generate plan", "Edit inventory"])
            session.awaiting = "generate_plan"
        elif session.inventory:
            session = _assistant(session, "Your inventory is ready to confirm.", quick=["Confirm inventory", "Add ingredient manually"])
            session.awaiting = "confirm_inventory"
        else:
            session.stage = "inventory"
            session = _assistant(
                session,
                "Let’s get your ingredients — upload a photo or type them.",
                kind="inventory_prompt",
                quick=["Type ingredients instead"],
            )
            session.awaiting = "inventory_input"
        save_session(session)
        return session, profile

    if intent == "make_faster":
        order = ["no_limit", "up_to_60", "30_45", "15_30", "under_15"]
        try:
            idx = order.index(profile.cooking_time_preference)
        except ValueError:
            idx = 2
        profile.cooking_time_preference = order[min(idx + 1, len(order) - 1)]  # type: ignore[assignment]
        profile.sync_aliases()
        session = _assistant(session, f"Okay — I’ll aim for quicker meals (about {profile.time_limit_min} minutes or less) and rebuild the plan.")
        session = generate_plan(session, profile)
        save_session(session)
        return session, profile

    if intent == "use_offline":
        session.use_llm = False
        session = generate_plan(session, profile)
        save_session(session)
        return session, profile

    if intent == "add_to_shopping_list":
        item_name = meta.get("item") or (session.substitution.missing_ingredient if session.substitution else None)
        if item_name:
            if session.plan and session.plan.plan and item_name not in session.plan.plan[0].missing_ingredients:
                session.plan.plan[0].missing_ingredients.append(item_name)
            session = _assistant(
                session,
                f"I’ll keep **{item_name}** on the shopping list. You can confirm the plan when ready.",
                quick=["Accept plan", "Replace a meal"],
            )
            session.awaiting = "plan_feedback"
        else:
            session = _assistant(session, "Tell me the item — for example, “add tofu to shopping list”.")
        save_session(session)
        return session, profile

    if intent == "correct_inventory_item":
        frm = meta.get("from", "")
        to = meta.get("to", "")
        item = _find_item(session, frm)
        if item:
            item.normalized_name = normalize_name(to)
            item.display_name = to.strip()
            session = _assistant(session, f"Got it — I replaced **{frm}** with **{to}**. Anything else you’d like to change?", quick=["Confirm inventory", "Add ingredient"])
        else:
            session = _assistant(session, f"I couldn’t find “{frm}” in the list. You can add **{to}** instead.", quick=["Confirm inventory"])
        save_session(session)
        return session, profile

    if intent == "add_inventory_item":
        if not meta.get("items_text"):
            session = _assistant(session, "What should I add? Type the ingredients, comma-separated.")
            session.awaiting = "typed_inventory"
            save_session(session)
            return session, profile
        for name in _parse_list(meta.get("items_text", text)):
            if inventory_conflicts(name, profile):
                session.inventory.append(
                    InventoryItem(normalized_name=name, display_name=name, source="typed", confirmed=False)
                )
                session = _assistant(
                    session,
                    f"I can record **{name}**, but it conflicts with your profile. Exclude it from the plan, or remove it?",
                    quick=["Exclude from plan", "Remove from inventory"],
                )
                session.pending_conflicts = [
                    {"id": session.inventory[-1].id, "name": name}
                ]
                session.awaiting = "conflict:bulk"
                save_session(session)
                return session, profile
            if not _find_item(session, name):
                session.inventory.append(
                    InventoryItem(normalized_name=name, display_name=name, source="typed", confirmed=session.stage != "confirmation")
                )
        session = _assistant(session, "Added. Anything else to change?", quick=["Confirm inventory", "Add another"])
        save_session(session)
        return session, profile

    if intent == "remove_inventory_item":
        name = meta.get("item", "")
        before = len(session.inventory)
        session.inventory = [i for i in session.inventory if normalize_name(name) not in normalize_name(i.normalized_name)]
        session = _assistant(session, "Removed." if len(session.inventory) < before else "I couldn’t find that item.", quick=["Confirm inventory"])
        save_session(session)
        return session, profile

    if intent == "confirm_inventory" or (session.awaiting == "confirm_inventory" and text.lower() in {"confirm inventory", "confirm", "yes", "looks good", "done"}):
        session = confirm_inventory(session, profile)
        save_session(session)
        return session, profile

    if intent == "generate_plan" or (session.awaiting == "generate_plan" and text.lower().startswith("yes")):
        session = generate_plan(session, profile)
        save_session(session)
        return session, profile

    if intent == "request_recipe_steps":
        day = meta.get("day")
        dish = meta.get("dish", "")
        if not day and session.plan and session.plan.plan:
            words = set(re.findall(r"[a-z]{3,}", (dish or text).lower()))
            for meal in session.plan.plan:
                if set(re.findall(r"[a-z]{3,}", meal.recipe.lower())) & words:
                    day = meal.day or meal.night
                    break
            if not day and len(session.plan.plan) == 1:
                day = session.plan.plan[0].day or session.plan.plan[0].night
        if day:
            session = get_detailed_recipe(session, profile, day=int(day))
        elif session.plan and session.plan.plan:
            session = _assistant(
                session,
                "Which meal would you like the full recipe for?",
                quick=[f"View steps {m.day or m.night}" for m in session.plan.plan],
            )
            session.awaiting = "plan_feedback"
        else:
            session = _assistant(session, "There’s no meal plan yet — generate one first and I’ll write out the full recipe.", quick=["Yes, generate plan"])
            session.awaiting = "generate_plan"
        save_session(session)
        return session, profile

    if intent == "replace_meal":
        day = meta.get("day")
        if not day and session.plan and session.plan.plan:
            # "Replace this meal" — resolve from substitution context or a 1-meal plan.
            if session.substitution and session.substitution.recipe_context:
                for meal in session.plan.plan:
                    if meal.recipe == session.substitution.recipe_context:
                        day = meal.day or meal.night
                        break
            if not day and len(session.plan.plan) == 1:
                day = session.plan.plan[0].day or session.plan.plan[0].night
            if not day:
                session = _assistant(
                    session,
                    "Which day should I replace?",
                    quick=[f"Replace Day {m.day or m.night}" for m in session.plan.plan],
                )
                session.awaiting = "plan_feedback"
                save_session(session)
                return session, profile
        session = replace_meal(session, profile, day=int(day or 1))
        save_session(session)
        return session, profile

    if intent == "change_servings":
        profile.servings = int(meta.get("servings") or profile.servings)
        save_profile(profile)
        session = _assistant(session, f"Updated servings to {profile.servings}. I can regenerate the plan with that change.", quick=["Generate plan", "Keep current plan"])
        save_session(session)
        return session, profile

    if intent == "request_substitution" or "missing" in text.lower():
        # Try to extract the ingredient: match any missing ingredient from the plan.
        missing = None
        if session.plan:
            mentioned = set(re.findall(r"[a-z]{3,}", text.lower()))
            for meal in session.plan.plan:
                for ing in meal.missing_ingredients:
                    if set(re.findall(r"[a-z]{3,}", ing.lower())) & mentioned:
                        missing = ing
                        break
                if missing:
                    break
        if not missing and session.plan:
            for meal in session.plan.plan:
                if meal.missing_ingredients:
                    missing = meal.missing_ingredients[0]
                    recipe = meal.recipe
                    break
            else:
                recipe = session.plan.plan[0].recipe if session.plan.plan else "your recipe"
        else:
            recipe = session.plan.plan[0].recipe if session.plan and session.plan.plan else "your recipe"
        if missing:
            session = start_substitution(session, profile, missing, recipe)
            save_session(session)
            return session, profile
        if session.plan:
            session = _assistant(
                session,
                "Good news — nothing is missing across your plan. You can accept it when ready.",
                quick=["Accept plan", "Replace a meal"],
            )
            session.awaiting = "plan_feedback"
            save_session(session)
            return session, profile

    if intent == "accept_plan" or text.lower() in {"accept plan", "confirm plan", "yes, confirm"}:
        # First ask confirmation then shop
        if session.stage in {"meal_plan", "adjustments"} and not (session.plan and session.plan.confirmed):
            session = _assistant(
                session,
                "Your plan is ready. Before I create the shopping list, please confirm that these meals work for you.",
                kind="meal_cards",
                meta={"plan": session.plan.model_dump() if session.plan else {}},
                quick=["Confirm plan", "Replace a meal", "Review missing ingredients"],
            )
            session.awaiting = "final_confirm"
            if text.lower() in {"confirm plan", "yes, confirm"} or "confirm" in text.lower():
                session = accept_plan_and_shop(session, profile)
            save_session(session)
            return session, profile

    if session.awaiting == "final_confirm" and "confirm" in text.lower():
        session = accept_plan_and_shop(session, profile)
        save_session(session)
        return session, profile

    if intent == "generate_shopping_list":
        session = accept_plan_and_shop(session, profile)
        save_session(session)
        return session, profile

    t_low = text.lower()

    # Offline rule: declining a pending yes/no question ("no", "not now", "don't generate").
    if session.awaiting == "generate_plan" and re.match(r"^\s*(no|nope|not now|later|don'?t|dont)\b", t_low):
        session = _assistant(
            session,
            "No problem — I won’t generate the plan yet. You can edit your inventory, tell me what to "
            "prioritize, or just say the word when you’re ready.",
            quick=["Yes, generate plan", "Edit inventory", "Adjust priorities"],
        )
        save_session(session)
        return session, profile

    # Offline rule: "I don't know how to cook X / give me another option" about a planned meal.
    if session.plan and session.plan.plan and any(
        cue in t_low for cue in ["alternative", "something else", "other option", "different meal", "don't know how", "dont know how", "can't cook", "cant cook"]
    ):
        words = set(re.findall(r"[a-z]{3,}", t_low))
        for meal in session.plan.plan:
            recipe_words = set(re.findall(r"[a-z]{3,}", meal.recipe.lower()))
            if words & recipe_words:
                session = _assistant(
                    session,
                    f"Sure — let me swap **{meal.recipe}** for something simpler.",
                )
                session = replace_meal(session, profile, day=int(meal.day or meal.night))
                save_session(session)
                return session, profile
        if not session.use_llm:
            # Couldn't tell which meal they mean — ask instead of falling through.
            session = _assistant(
                session,
                "Happy to find an alternative. Which meal should I replace?",
                quick=[f"Replace Day {m.day or m.night}" for m in session.plan.plan],
            )
            session.awaiting = "plan_feedback"
            save_session(session)
            return session, profile

    # Conversational fallback: let Claude interpret the free-form message with
    # long-term memory (saved profile) and short-term memory (summary, recent chat, plan, inventory).
    if session.use_llm:
        session = update_conversation_summary(session, min_new_messages=4)
        route = llm_route(text, session, profile)
        if route:
            session, profile = _apply_llm_route(session, profile, route)
            save_session(session)
            return session, profile

    # Last resort — offer context-appropriate next steps so the user is never stuck.
    if session.plan:
        fallback_quick = ["Accept plan", "Replace a meal", "Start new plan"]
    elif session.stage == "freshness":
        fallback_quick = ["Yes, generate plan", "Edit inventory"]
    elif session.inventory:
        fallback_quick = ["Confirm inventory", "Add ingredient manually"]
    else:
        fallback_quick = ["Type ingredients instead", "Start new plan"]
    session = _assistant(
        session,
        "I want to make sure I help with the right next step. You can confirm inventory, generate a plan, "
        "replace a meal, ask for a substitution, or start a new plan.",
        quick=fallback_quick,
    )
    save_session(session)
    return session, profile


def _contextual_quick(session: SessionState) -> List[str]:
    if session.plan:
        return ["Accept plan", "Replace a meal", "Help with missing ingredients"]
    if session.stage == "freshness":
        return ["Yes, generate plan", "Edit inventory"]
    if session.inventory:
        return ["Confirm inventory", "Add ingredient manually"]
    return ["Type ingredients instead", "Start new plan"]


def _apply_llm_route(session: SessionState, profile: UserProfile, route: dict) -> Tuple[SessionState, UserProfile]:
    """Execute the action chosen by the conversational router.

    The LLM only picks WHAT to do and writes the reply; all dietary filtering
    and plan changes run through the same deterministic code as button clicks.
    """
    action = route.get("action", "chat")
    reply = (route.get("reply") or "").strip()
    items = [str(i).strip().lower() for i in (route.get("items") or []) if str(i).strip()]
    day = route.get("day")

    # Long-term memory: persist newly revealed dislikes to the saved profile.
    new_dislikes = [str(d).strip().lower() for d in (route.get("remember_dislikes") or []) if str(d).strip()]
    remembered = [d for d in new_dislikes if d not in profile.dislikes]
    if remembered:
        profile.dislikes.extend(remembered)
        save_profile(profile)
        reply = (reply + f"\n\n_I’ll remember you’d rather avoid {', '.join(remembered)} in future plans._").strip()

    if action in {"chat", "decline"}:
        session = _assistant(session, reply or "Happy to help — what would you like to do next?", quick=_contextual_quick(session))
        return session, profile

    if reply:
        session = _assistant(session, reply)

    if action == "replace_meal" and session.plan and session.plan.plan:
        if not day:
            words = set(items)
            for meal in session.plan.plan:
                recipe_words = set(re.findall(r"[a-z]{3,}", meal.recipe.lower()))
                if words & recipe_words:
                    day = meal.day or meal.night
                    break
        if not day and len(session.plan.plan) == 1:
            day = session.plan.plan[0].day or session.plan.plan[0].night
        if day:
            session = replace_meal(session, profile, day=int(day))
        else:
            session = _assistant(session, "Which day should I replace?", quick=[f"Replace Day {m.day or m.night}" for m in session.plan.plan])
            session.awaiting = "plan_feedback"
        return session, profile

    if action == "recipe_steps" and session.plan and session.plan.plan:
        if not day:
            words = set(items)
            for meal in session.plan.plan:
                if set(re.findall(r"[a-z]{3,}", meal.recipe.lower())) & words:
                    day = meal.day or meal.night
                    break
        if not day and len(session.plan.plan) == 1:
            day = session.plan.plan[0].day or session.plan.plan[0].night
        if day:
            session = get_detailed_recipe(session, profile, day=int(day))
        else:
            session = _assistant(
                session,
                "Which meal would you like the full recipe for?",
                quick=[f"View steps {m.day or m.night}" for m in session.plan.plan],
            )
            session.awaiting = "plan_feedback"
        return session, profile

    if action == "generate_plan":
        session = generate_plan(session, profile)
    elif action == "accept_plan":
        session = accept_plan_and_shop(session, profile)
    elif action == "show_plan" and session.plan:
        session = _assistant(session, "Here’s your current plan.", kind="meal_cards", meta={"plan": session.plan.model_dump()}, quick=_contextual_quick(session))
        session.awaiting = "plan_feedback"
    elif action == "edit_inventory":
        session.stage = "confirmation"
        session = _assistant(
            session,
            "Here’s your current inventory — make any changes, then confirm.",
            kind="inventory_review",
            meta={"inventory": [i.model_dump() for i in session.inventory]},
            quick=["Confirm inventory", "Add ingredient manually"],
        )
        session.awaiting = "confirm_inventory"
    elif action == "substitution" and items:
        recipe = session.plan.plan[0].recipe if session.plan and session.plan.plan else "your recipe"
        session = start_substitution(session, profile, items[0], recipe)
    elif action == "add_items" and items:
        for name in items:
            if not _find_item(session, name) and not inventory_conflicts(name, profile):
                session.inventory.append(InventoryItem(normalized_name=normalize_name(name), display_name=name, source="typed", confirmed=session.stage != "confirmation"))
        session = _assistant(session, f"Added: {', '.join(items)}.", quick=_contextual_quick(session))
    elif action == "remove_items" and items:
        for name in items:
            session.inventory = [i for i in session.inventory if normalize_name(name) != i.normalized_name]
        session = _assistant(session, f"Removed: {', '.join(items)}.", quick=_contextual_quick(session))
    elif action == "shopping_list":
        if session.plan:
            session = accept_plan_and_shop(session, profile)
        else:
            session = _assistant(session, "I build the shopping list from a confirmed plan — let’s make one first.", quick=_contextual_quick(session))
    elif action == "new_plan":
        session = start_new_plan(session, profile)
    elif action == "edit_profile":
        profile.profile_confirmed = False
        session.stage = "profile"
        session.profile_step = "diet_type"
        session = _assistant(session, "Let’s update your profile. What diet should every recommendation follow?", quick=["Vegetarian", "Vegan", "Non-vegetarian", "Pescatarian", "Eggetarian"])
        session.awaiting = "diet_type"
    elif not reply:
        session = _assistant(session, "Happy to help — what would you like to do next?", quick=_contextual_quick(session))

    return session, profile
