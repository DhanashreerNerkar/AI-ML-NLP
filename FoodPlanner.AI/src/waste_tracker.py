"""Long-term food-waste and over-purchasing analysis.

Every CONFIRMED inventory becomes a snapshot in persistent, per-user history.
Historical snapshots are compared to find recurring at-risk ingredients and to
produce conservative purchase recommendations.

Key behavioral rules implemented here:
- Only confirmed inventories are stored (raw detections are never persisted).
- An ingredient missing from a later snapshot is "unresolved", never "wasted".
- Only user-confirmed outcomes ("spoiled", "thrown_away") count as waste.
- No prices or monetary estimates anywhere.
- All storage is keyed by user_id; users' data is never mixed.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from src.kb import DATA
from src.schemas import (
    IngredientOutcome,
    InventorySnapshot,
    PatternFinding,
    PurchaseRecommendation,
    RankedItem,
    SnapshotIngredient,
    SnapshotSummary,
)

USER_DATA = DATA / "user_data"
HISTORY_PATH = USER_DATA / "inventory_history.json"
OUTCOMES_PATH = USER_DATA / "ingredient_outcomes.json"

# Freshness labels that count as "at risk" (KB scores 4-5).
AT_RISK_STATUSES = {"use_first", "use_soon", "high_priority", "near_expiry", "stale"}
WASTE_OUTCOMES = {"spoiled", "thrown_away"}

DUPLICATE_WINDOW_HOURS = 48
MIN_SNAPSHOTS_FOR_RECOMMENDATIONS = 3
MIN_AT_RISK_OCCURRENCES = 2
TOP_AT_RISK_LIMIT = 4
COMPARISON_WINDOW = 6  # how many recent snapshots patterns are computed over
MAX_OUTCOME_QUESTIONS = 3


# ---------------------------------------------------------------- storage

def _load_store(path) -> Dict[str, list]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_store(path, store: Dict[str, list]) -> None:
    USER_DATA.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2), encoding="utf-8")


def load_history(user_id: str) -> List[InventorySnapshot]:
    store = _load_store(HISTORY_PATH)
    return [InventorySnapshot.model_validate(s) for s in store.get(user_id, [])]


def load_outcomes(user_id: str) -> List[IngredientOutcome]:
    store = _load_store(OUTCOMES_PATH)
    return [IngredientOutcome.model_validate(o) for o in store.get(user_id, [])]


def _append(path, user_id: str, record: dict) -> None:
    store = _load_store(path)
    store.setdefault(user_id, []).append(record)
    _save_store(path, store)


def image_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()[:24]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- snapshots

_STATUS_MAP = {
    "use_first": "use_first",
    "use_soon": "use_soon",
    "moderate": "fresh",
    "longer": "fresh",
    "shelf_stable": "fresh",
}


def build_snapshot(
    user_id: str,
    ranked: List[RankedItem],
    source: str = "typed",
    image_reference: Optional[str] = None,
) -> InventorySnapshot:
    ingredients: List[SnapshotIngredient] = []
    for rank, r in enumerate(ranked, start=1):
        status = _STATUS_MAP.get(r.priority_label, "fresh")
        ingredients.append(
            SnapshotIngredient(
                ingredient_id=r.ingredient_id,
                name=r.item,
                normalized_name=r.item,
                category=r.class_name,
                freshness_score=int(r.score) * 20,
                freshness_status=status,  # type: ignore[arg-type]
                priority_rank=rank,
            )
        )
    at_risk = [i for i in ingredients if i.freshness_status in AT_RISK_STATUSES]
    total = len(ingredients)
    summary = SnapshotSummary(
        total_items=total,
        at_risk_items=len(at_risk),
        at_risk_percentage=round(len(at_risk) / total * 100, 2) if total else 0.0,
        top_at_risk_items=[i.normalized_name for i in at_risk[:TOP_AT_RISK_LIMIT]],
    )
    return InventorySnapshot(
        user_id=user_id,
        created_at=_now_iso(),
        source=source,
        image_reference=image_reference,
        confirmed=True,
        ingredients=ingredients,
        summary=summary,
    )


def find_duplicate(user_id: str, img_hash: Optional[str]) -> Optional[InventorySnapshot]:
    """Duplicate detection is based on the IMAGE hash, never on the ingredient
    list — two different photos can legitimately contain similar groceries."""
    if not img_hash:
        return None
    cutoff = datetime.now(timezone.utc) - timedelta(hours=DUPLICATE_WINDOW_HOURS)
    for snap in reversed(load_history(user_id)):
        if snap.image_reference != img_hash:
            continue
        try:
            created = datetime.fromisoformat(snap.created_at)
        except ValueError:
            continue
        if created >= cutoff:
            return snap
    return None


def record_snapshot(
    user_id: str,
    ranked: List[RankedItem],
    source: str = "typed",
    image_reference: Optional[str] = None,
) -> Tuple[InventorySnapshot, Optional[InventorySnapshot], List[str]]:
    """Persist a confirmed snapshot.

    Returns (snapshot, possible_duplicate_of, disappeared_at_risk_names) where
    disappeared_at_risk_names are ingredients that were at risk in the previous
    snapshot and are absent now — candidates for the outcome follow-up question.
    Their fate is UNKNOWN until the user answers; nothing is marked as waste here.
    """
    history = load_history(user_id)
    snapshot = build_snapshot(user_id, ranked, source=source, image_reference=image_reference)

    duplicate = find_duplicate(user_id, image_reference)
    if duplicate is not None:
        snapshot.possible_duplicate = True
        snapshot.duplicate_of = duplicate.snapshot_id
        # Not counted as a new purchase unless the user says otherwise.
        snapshot.is_new_purchase = None

    disappeared: List[str] = []
    if history and not snapshot.possible_duplicate:
        previous = history[-1]
        current_names = {i.normalized_name for i in snapshot.ingredients}
        already_asked = {o.ingredient_name for o in load_outcomes(user_id)}
        for name in previous.summary.top_at_risk_items:
            if name not in current_names and name not in already_asked:
                disappeared.append(name)
    disappeared = disappeared[:MAX_OUTCOME_QUESTIONS]

    _append(HISTORY_PATH, user_id, json.loads(snapshot.model_dump_json()))
    return snapshot, duplicate, disappeared


def set_new_purchase_flag(user_id: str, snapshot_id: str, is_new_purchase: bool) -> None:
    store = _load_store(HISTORY_PATH)
    for raw in store.get(user_id, []):
        if raw.get("snapshot_id") == snapshot_id:
            raw["is_new_purchase"] = is_new_purchase
            break
    _save_store(HISTORY_PATH, store)


# ---------------------------------------------------------------- outcomes

def record_outcome(
    user_id: str,
    ingredient_name: str,
    outcome: str,
    related_snapshot_id: Optional[str] = None,
) -> IngredientOutcome:
    entry = IngredientOutcome(
        user_id=user_id,
        ingredient_name=ingredient_name,
        related_snapshot_id=related_snapshot_id,
        recorded_at=_now_iso(),
        outcome=outcome,  # type: ignore[arg-type]
        confirmed_by_user=True,
    )
    _append(OUTCOMES_PATH, user_id, json.loads(entry.model_dump_json()))
    return entry


def parse_outcome_answer(text: str) -> str:
    t = text.strip().lower()
    if re.search(r"\bstill\b", t):
        return "still_have"
    if re.search(r"\bbought\b|\bagain\b", t):
        return "bought_again"
    if "spoil" in t:
        return "spoiled"
    if re.search(r"\bthrown\b|\bthrew\b|\btrash\b|\btossed?\b", t):
        return "thrown_away"
    if "donat" in t:
        return "donated"
    if re.search(r"\bused?\b|\bate\b|\beaten\b|\bcooked?\b|\bfinished\b", t):
        return "used"
    return "not_sure"


# ---------------------------------------------------------------- analysis

def _countable_snapshots(history: List[InventorySnapshot]) -> List[InventorySnapshot]:
    """Snapshots used for pattern counting: exclude ones the user confirmed as
    duplicates of the same inventory (is_new_purchase == False)."""
    return [s for s in history if not (s.possible_duplicate and s.is_new_purchase is False)]


def compare_with_history(user_id: str) -> List[PatternFinding]:
    history = _countable_snapshots(load_history(user_id))
    if len(history) < 2:
        return []
    recent = history[-COMPARISON_WINDOW:]
    findings: List[PatternFinding] = []

    at_risk_counts: Dict[str, int] = {}
    for snap in recent:
        for name in {i.normalized_name for i in snap.ingredients if i.freshness_status in AT_RISK_STATUSES}:
            at_risk_counts[name] = at_risk_counts.get(name, 0) + 1
    for name, count in sorted(at_risk_counts.items(), key=lambda kv: -kv[1]):
        if count >= MIN_AT_RISK_OCCURRENCES:
            findings.append(
                PatternFinding(
                    ingredient=name,
                    pattern="repeated_at_risk",
                    occurrences=count,
                    confidence=round(min(0.95, 0.5 + 0.1 * count), 2),
                    status="at_risk",
                    requires_user_confirmation=True,
                )
            )

    latest, previous = recent[-1], recent[-2]
    latest_risk = {i.normalized_name for i in latest.ingredients if i.freshness_status in AT_RISK_STATUSES}
    prev_risk = {i.normalized_name for i in previous.ingredients if i.freshness_status in AT_RISK_STATUSES}
    for name in sorted(latest_risk & prev_risk):
        streak = 0
        for snap in reversed(recent):
            names = {i.normalized_name for i in snap.ingredients if i.freshness_status in AT_RISK_STATUSES}
            if name in names:
                streak += 1
            else:
                break
        findings.append(
            PatternFinding(
                ingredient=name,
                pattern="persistent_at_risk",
                occurrences=streak,
                confidence=round(min(0.95, 0.55 + 0.1 * streak), 2),
                status="at_risk",
            )
        )

    latest_names = {i.normalized_name for i in latest.ingredients}
    prev_names = {i.normalized_name for i in previous.ingredients}
    for name in sorted(prev_risk - latest_names):
        # A missing ingredient may have been consumed, moved, donated, discarded,
        # or mis-detected earlier. It stays UNRESOLVED until the user confirms.
        findings.append(
            PatternFinding(
                ingredient=name,
                pattern="disappeared",
                occurrences=1,
                confidence=0.4,
                status="unresolved",
                requires_user_confirmation=True,
            )
        )

    if len(recent) >= 3:
        older_names = set()
        for snap in recent[:-2]:
            older_names |= {i.normalized_name for i in snap.ingredients}
        for name in sorted((latest_names - prev_names) & older_names):
            findings.append(
                PatternFinding(
                    ingredient=name,
                    pattern="reappeared",
                    occurrences=1,
                    confidence=0.5,
                    status="repeated_purchase",
                )
            )

    return findings


def compute_analytics(user_id: str) -> dict:
    history = load_history(user_id)
    countable = _countable_snapshots(history)
    outcomes = load_outcomes(user_id)
    non_empty = [s for s in countable if s.summary.total_items > 0]

    latest = history[-1] if history else None
    waste_events = [o for o in outcomes if o.outcome in WASTE_OUTCOMES and o.confirmed_by_user]
    waste_counts: Dict[str, int] = {}
    for o in waste_events:
        waste_counts[o.ingredient_name] = waste_counts.get(o.ingredient_name, 0) + 1

    repeated = [
        f for f in compare_with_history(user_id) if f.pattern == "repeated_at_risk"
    ]
    consecutive = {
        f.ingredient: f.occurrences
        for f in compare_with_history(user_id)
        if f.pattern == "persistent_at_risk"
    }

    over_purchasing = []
    snapshots_considered = non_empty[-COMPARISON_WINDOW:]
    if snapshots_considered:
        for f in repeated:
            score = f.occurrences / len(snapshots_considered)
            score += 0.25 * waste_counts.get(f.ingredient, 0)
            over_purchasing.append({"ingredient": f.ingredient, "score": round(min(score, 1.0), 2)})

    return {
        "snapshot_count": len(history),
        "latest_snapshot_at": latest.created_at if latest else None,
        "current_at_risk_count": latest.summary.at_risk_items if latest else 0,
        "current_at_risk_percentage": latest.summary.at_risk_percentage if latest else 0.0,
        "current_top_at_risk": latest.summary.top_at_risk_items if latest else [],
        "average_at_risk_percentage": (
            round(sum(s.summary.at_risk_percentage for s in non_empty) / len(non_empty), 2)
            if non_empty
            else 0.0
        ),
        "at_risk_percentage_series": [
            {"date": s.created_at[:10], "at_risk_percentage": s.summary.at_risk_percentage}
            for s in non_empty
        ],
        "repeated_at_risk": [
            {"ingredient": f.ingredient, "occurrences": f.occurrences, "confidence": f.confidence}
            for f in repeated
        ],
        "confirmed_waste_events": len(waste_events),
        "most_wasted": sorted(
            ({"ingredient": k, "count": v} for k, v in waste_counts.items()),
            key=lambda d: -d["count"],
        ),
        "consecutive_at_risk": consecutive,
        "unresolved": [
            f.ingredient for f in compare_with_history(user_id) if f.pattern == "disappeared"
        ],
    }


# ------------------------------------------------------- recommendations

def build_recommendations(user_id: str) -> List[PurchaseRecommendation]:
    history = _countable_snapshots(load_history(user_id))
    non_empty = [s for s in history if s.summary.total_items > 0]
    if len(non_empty) < MIN_SNAPSHOTS_FOR_RECOMMENDATIONS:
        return []

    recent = non_empty[-COMPARISON_WINDOW:]
    outcomes = load_outcomes(user_id)
    waste_counts: Dict[str, int] = {}
    for o in outcomes:
        if o.outcome in WASTE_OUTCOMES and o.confirmed_by_user:
            waste_counts[o.ingredient_name] = waste_counts.get(o.ingredient_name, 0) + 1

    occurrences: Dict[str, List[str]] = {}
    categories: Dict[str, str] = {}
    for snap in recent:
        for ing in snap.ingredients:
            if ing.freshness_status in AT_RISK_STATUSES:
                occurrences.setdefault(ing.normalized_name, []).append(snap.snapshot_id)
                categories[ing.normalized_name] = ing.category

    reappeared = {
        f.ingredient for f in compare_with_history(user_id) if f.pattern == "reappeared"
    }

    recommendations: List[PurchaseRecommendation] = []
    for name, snap_ids in sorted(occurrences.items(), key=lambda kv: -len(kv[1])):
        count = len(snap_ids)
        if count < MIN_AT_RISK_OCCURRENCES:
            continue
        waste = waste_counts.get(name, 0)
        # Conservative reduction, clamped to 10-50%.
        reduction = max(10, min(50, 10 + 5 * (count - MIN_AT_RISK_OCCURRENCES) + 15 * waste))
        confidence = round(min(0.95, 0.5 + 0.08 * count + 0.15 * waste), 2)

        if name in reappeared and waste == 0:
            rec_type = "wait_before_buying"
        elif categories.get(name, "") in {"leafy_green", "herb_fresh", "berry", "dairy"}:
            rec_type = "smaller_package"
        else:
            rec_type = "buy_less"

        reason = f"Marked at risk in {count} of the last {len(recent)} inventories"
        if waste:
            reason += f"; confirmed spoiled or thrown away {waste} time{'s' if waste > 1 else ''}"
        reason += "."

        recommendations.append(
            PurchaseRecommendation(
                ingredient=name,
                recommendation_type=rec_type,
                suggested_reduction_percentage=reduction,
                reason=reason,
                confidence=confidence,
                supporting_snapshot_ids=snap_ids,
            )
        )
    return recommendations


# ------------------------------------------------------------- chat hints

def history_hint(user_id: str) -> Optional[str]:
    """Short long-term-memory line for chat greetings. Carefully worded: talks
    about "use-first" frequency, never claims anything was wasted."""
    analytics = compute_analytics(user_id)
    if analytics["snapshot_count"] < 2:
        return None
    repeated = [r["ingredient"] for r in analytics["repeated_at_risk"][:3]]
    if not repeated:
        return None
    names = ", ".join(repeated)
    return (
        f"Based on your recent inventories, {names} "
        f"{'are' if len(repeated) > 1 else 'is'} most often in the use-first group. "
        f"Your average at-risk percentage is {analytics['average_at_risk_percentage']:.0f}%."
    )
