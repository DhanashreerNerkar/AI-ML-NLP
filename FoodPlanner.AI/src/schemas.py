"""Pydantic contracts for FoodPlanner.AI chatbot pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field


def _id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid4().hex[:10]}"


DietType = Literal[
    "vegetarian",
    "vegan",
    "non-vegetarian",
    "pescatarian",
    "eggetarian",
    "flexitarian",
    "other",
]

CookingSkill = Literal["beginner", "comfortable", "experienced"]
SpiceLevel = Literal["not_spicy", "mild", "medium", "hot", "varies"]
PriorityLabel = Literal["use_first", "use_soon", "moderate", "longer", "shelf_stable"]

ChatStage = Literal[
    "profile",
    "preferences",
    "inventory",
    "confirmation",
    "freshness",
    "meal_plan",
    "adjustments",
    "shopping_list",
]

Intent = Literal[
    "create_profile",
    "edit_profile",
    "start_plan",
    "answer_question",
    "upload_inventory",
    "add_inventory_item",
    "remove_inventory_item",
    "correct_inventory_item",
    "confirm_inventory",
    "update_freshness",
    "generate_plan",
    "replace_meal",
    "modify_meal",
    "change_servings",
    "change_schedule",
    "request_substitution",
    "select_substitution",
    "request_recipe_steps",
    "accept_plan",
    "generate_shopping_list",
    "mark_item_used",
    "add_new_purchase",
    "reset_current_plan",
    "ask_general_question",
    "unsafe_or_out_of_scope",
    "add_to_shopping_list",
    "edit_inventory",
    "show_plan",
    "continue_planning",
    "make_faster",
    "use_offline",
]


class UserProfile(BaseModel):
    user_id: str = Field(default_factory=lambda: _id("user"))
    diet_type: DietType = "vegetarian"
    # Backward-compatible alias used by older pipeline stages
    diet_style: str = "strict-vegetarian"
    cultural_rules: List[str] = Field(default_factory=list)
    cultural_constraints: List[str] = Field(default_factory=list)  # alias for older code
    jain_rules: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    hard_exclusions: List[str] = Field(default_factory=list)
    dietary_restrictions: List[str] = Field(default_factory=list)  # alias
    dislikes: List[str] = Field(default_factory=list)
    ambient_rules: List[str] = Field(default_factory=list)
    servings: int = 2
    household_includes_children: bool = False
    cooking_time_preference: str = "15_30"
    time_limit_min: int = 30
    nights: int = 3
    cooking_skill: CookingSkill = "beginner"
    equipment: List[str] = Field(default_factory=lambda: ["stovetop"])
    preferred_cuisines: List[str] = Field(default_factory=list)
    spice_level: SpiceLevel = "mild"
    meal_types: List[str] = Field(default_factory=lambda: ["dinner"])
    assumed_staples: List[str] = Field(
        default_factory=lambda: ["salt", "pepper", "oil", "water"]
    )
    profile_confirmed: bool = False

    def sync_aliases(self) -> "UserProfile":
        """Keep legacy field names in sync for diet_filter / planning.

        ``allergies`` + cultural rules are the source of truth.
        ``hard_exclusions`` / ``dietary_restrictions`` are derived outputs and
        must be recomputed each time — never unioned with their previous values,
        or cleared allergies (e.g. Eggs → None) would stick forever.
        """
        if self.cultural_rules:
            self.cultural_constraints = list(self.cultural_rules)
        elif self.cultural_constraints:
            self.cultural_rules = list(self.cultural_constraints)

        # Recompute from allergies only — do not reuse stale hard/dietary lists.
        hard = {x for x in self.allergies if x and str(x).strip().lower() not in {"none", "no"}}
        # Map diet type into hard exclusions + ambient rules
        diet = self.diet_type
        if diet in {"vegetarian", "eggetarian"}:
            self.diet_style = "strict-vegetarian"
        elif diet == "vegan":
            self.diet_style = "strict-vegetarian"
        elif diet == "flexitarian":
            self.diet_style = "vegetarian-flexible"
        else:
            self.diet_style = "none"

        # Recompute ambient rules from cultural constraints (avoid sticky leftovers).
        rules: set[str] = set()
        cultural_l = [c.lower() for c in self.cultural_rules]
        cultural = set(cultural_l)
        if "jain" in cultural or any("jain" in c for c in cultural_l):
            for r in self.jain_rules or ["no onion and garlic", "no root vegetables"]:
                rules.add(r)
            hard.update(["onion", "garlic", "potato", "carrot", "beet", "radish"])
            rules.update(["no onion", "no garlic", "no root vegetables"])
        # Hindu preference (common): avoid beef. Never implies an egg ban —
        # eggs follow diet_type (vegetarian/vegan/jain) or an explicit allergy.
        if any("hindu" in c for c in cultural_l) or "no beef" in cultural:
            hard.add("beef")
            rules.add("no beef")
        if "no pork" in cultural or any("halal" in c for c in cultural_l):
            hard.update(["pork", "bacon", "ham"])
        if "no onion or garlic" in cultural:
            hard.update(["onion", "garlic"])
            rules.update(["no onion", "no garlic"])
        if "no root vegetables" in cultural:
            hard.update(["potato", "carrot", "beet", "radish", "onion", "garlic", "ginger"])
            rules.add("no root vegetables")

        self.ambient_rules = sorted(rules)
        self.hard_exclusions = sorted(hard)
        self.dietary_restrictions = sorted(hard)
        # Map cooking time preference to minutes
        mapping = {
            "under_15": 15,
            "15_30": 30,
            "30_45": 45,
            "up_to_60": 60,
            "no_limit": 120,
        }
        self.time_limit_min = mapping.get(self.cooking_time_preference, self.time_limit_min)
        return self


class InventoryItem(BaseModel):
    id: str = Field(default_factory=lambda: _id("ing"))
    normalized_name: str
    display_name: str = ""
    quantity: Optional[float] = None
    unit: Optional[str] = None
    quantity_text: Optional[str] = None
    source: str = "typed"  # photo / typed / leftover
    confidence: float = 1.0
    confirmed: bool = False
    opened: bool = False
    cooked: bool = False
    frozen: bool = False
    use_soon_user_flag: bool = False
    exclude_from_plan: bool = False
    do_not_use: bool = False
    needs_confirmation: bool = False
    notes: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.display_name:
            self.display_name = self.normalized_name


class DetectedItem(BaseModel):
    item: str
    confidence: Literal["high", "medium", "low"] = "medium"
    name: Optional[str] = None
    quantity_text: Optional[str] = None
    source: str = "photo"
    needs_confirmation: bool = False


class DetectedItems(BaseModel):
    detected_items: List[DetectedItem]
    needs_user_confirmation: bool = True
    warnings: List[str] = Field(default_factory=list)


class RankedItem(BaseModel):
    item: str
    score: int = Field(ge=1, le=5)
    class_name: str = Field(alias="class", default="unknown")
    reason: str = ""
    in_reference: bool = True
    priority_label: PriorityLabel = "moderate"
    uncertain: bool = False
    ingredient_id: Optional[str] = None

    model_config = {"populate_by_name": True}


class RankedItems(BaseModel):
    ranked: List[RankedItem]


class CriticalPriority(BaseModel):
    critical_priority: List[str]
    explanation: str = ""


class PlanMeal(BaseModel):
    night: int
    day: Optional[int] = None
    meal_type: str = "dinner"
    recipe: str
    recipe_id: Optional[str] = None
    time_min: int
    servings: int
    uses_critical: List[str] = Field(default_factory=list)
    ingredients_from_inventory: List[str] = Field(default_factory=list)
    extra_pantry_items: List[str] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list)
    missing_ingredients: List[str] = Field(default_factory=list)
    why_selected: str = ""
    status: str = "proposed"
    diet_validation_passed: bool = True
    # Full recipe (measured ingredients + detailed steps), generated on demand and cached.
    detailed_recipe: Optional[Dict[str, Any]] = None

    def model_post_init(self, __context: Any) -> None:
        if self.day is None:
            self.day = self.night


class FlaggedItem(BaseModel):
    item: str
    reason: str


class MealPlan(BaseModel):
    plan_id: str = Field(default_factory=lambda: _id("plan"))
    plan: List[PlanMeal]
    flagged_for_other_use: List[FlaggedItem] = Field(default_factory=list)
    clarification: Optional[str] = None
    confirmed: bool = False
    unresolved_gaps: List[str] = Field(default_factory=list)


class RecipeCandidate(BaseModel):
    id: Optional[Union[int, str]] = None
    title: str
    ingredients: List[str] = Field(default_factory=list)
    ready_in_minutes: Optional[int] = None
    steps: List[str] = Field(default_factory=list)
    source: str = "spoonacular"
    cuisine: Optional[str] = None


class SubstitutionSuccess(BaseModel):
    status: Literal["ok"] = "ok"
    substitute: str
    prep: str = ""
    integration: str = ""
    source: str = "approved_substitutions"


class DerivationOption(BaseModel):
    option_id: str
    label: str
    base_ingredients: List[str]
    method_steps: List[str]
    diet_tags: List[str] = Field(default_factory=list)
    option_type: str = "homemade_derivation"
    estimated_minutes: Optional[int] = None


class SubstitutionResult(BaseModel):
    status: Literal["ok", "no_substitute", "options", "selected"]
    substitute: Optional[str] = None
    prep: Optional[str] = None
    integration: Optional[str] = None
    source: Optional[str] = None
    message: Optional[str] = None
    missing_ingredient: Optional[str] = None
    recipe_context: Optional[str] = None
    derivation_options: List[DerivationOption] = Field(default_factory=list)
    store_options: List[SubstitutionSuccess] = Field(default_factory=list)
    selected_option_id: Optional[str] = None
    method_steps: List[str] = Field(default_factory=list)


class ShoppingListItem(BaseModel):
    normalized_name: str
    display_name: str = ""
    quantity: Optional[float] = None
    unit: Optional[str] = None
    category: str = "other"
    required_for_meals: List[str] = Field(default_factory=list)
    optional: bool = False
    homemade_option_available: bool = False


class GapList(BaseModel):
    gaps: List[str]
    markdown: str = ""
    items: List[ShoppingListItem] = Field(default_factory=list)
    already_in_kitchen: List[str] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: Literal["assistant", "user", "system"]
    content: str
    kind: str = "text"  # text | meal_cards | inventory | freshness | shopping | options
    meta: Dict[str, Any] = Field(default_factory=dict)


class PlanPreferences(BaseModel):
    days: int = 3
    meal_types: List[str] = Field(default_factory=lambda: ["dinner"])
    busy_notes: str = ""
    priorities: List[str] = Field(default_factory=list)
    assume_staples: bool = True
    custom_staples: List[str] = Field(default_factory=list)
    leftovers: List[str] = Field(default_factory=list)


# ---- Long-term waste / over-purchasing tracking ----

FreshnessStatus = Literal["fresh", "use_soon", "use_first", "high_priority", "near_expiry", "stale"]

OutcomeChoice = Literal[
    "used", "still_have", "bought_again", "spoiled", "thrown_away", "donated", "not_sure"
]


class SnapshotIngredient(BaseModel):
    ingredient_id: Optional[str] = None
    name: str
    normalized_name: str
    category: str = "unknown"
    quantity: Optional[str] = None
    unit: Optional[str] = None
    freshness_score: int = 60  # 0-100 scale (KB score 1-5 mapped x20)
    freshness_status: FreshnessStatus = "fresh"
    priority_rank: int = 0


class SnapshotSummary(BaseModel):
    total_items: int = 0
    at_risk_items: int = 0
    at_risk_percentage: float = 0.0
    top_at_risk_items: List[str] = Field(default_factory=list)


class InventorySnapshot(BaseModel):
    snapshot_id: str = Field(default_factory=lambda: _id("snap"))
    user_id: str
    created_at: str = ""
    source: str = "typed"  # "image" | "typed" | "mixed"
    image_reference: Optional[str] = None
    confirmed: bool = True
    possible_duplicate: bool = False
    duplicate_of: Optional[str] = None
    # None = unknown; False = user said same inventory; True = user said new purchase
    is_new_purchase: Optional[bool] = None
    ingredients: List[SnapshotIngredient] = Field(default_factory=list)
    summary: SnapshotSummary = Field(default_factory=SnapshotSummary)


class IngredientOutcome(BaseModel):
    outcome_id: str = Field(default_factory=lambda: _id("outcome"))
    user_id: str
    ingredient_name: str
    related_snapshot_id: Optional[str] = None
    recorded_at: str = ""
    outcome: OutcomeChoice = "not_sure"
    confirmed_by_user: bool = True


class PatternFinding(BaseModel):
    ingredient: str
    pattern: str  # repeated_at_risk | persistent_at_risk | disappeared | reappeared
    occurrences: int = 0
    confidence: float = 0.5
    status: str = "unresolved"  # disappearances are unresolved until the user confirms
    requires_user_confirmation: bool = False


class PurchaseRecommendation(BaseModel):
    ingredient: str
    recommendation_type: str = "buy_less"  # buy_less | smaller_package | wait_before_buying
    suggested_reduction_percentage: int = 10
    reason: str = ""
    confidence: float = 0.5
    supporting_snapshot_ids: List[str] = Field(default_factory=list)


class SessionState(BaseModel):
    stage: ChatStage = "profile"
    profile_step: str = "diet_type"
    pref_step: str = "duration"
    messages: List[ChatMessage] = Field(default_factory=list)
    inventory: List[InventoryItem] = Field(default_factory=list)
    pending_conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    ranked: List[RankedItem] = Field(default_factory=list)
    critical_priority: List[str] = Field(default_factory=list)
    recipe_candidates: List[RecipeCandidate] = Field(default_factory=list)
    plan: Optional[MealPlan] = None
    substitution: Optional[SubstitutionResult] = None
    gap_list: Optional[GapList] = None
    preferences: PlanPreferences = Field(default_factory=PlanPreferences)
    purchased_or_made: List[str] = Field(default_factory=list)
    awaiting: Optional[str] = None  # what the bot last asked for
    quick_replies: List[str] = Field(default_factory=list)
    use_llm: bool = True
    demo_mode: bool = False
    conversation_summary: str = ""
    summarized_through: int = 0
    # Waste-tracking state
    last_image_hash: Optional[str] = None
    last_snapshot_id: Optional[str] = None
    pending_outcomes: List[str] = Field(default_factory=list)
    # Recipes the user asked to replace — never re-suggest these in this session.
    rejected_recipes: List[str] = Field(default_factory=list)
