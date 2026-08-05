"""Tests for long-term waste tracking, outcomes, analytics, and recommendations."""

import json

from src import waste_tracker as wt
from src.chat.orchestrator import (
    confirm_inventory,
    handle_message,
    ingest_typed_inventory,
    new_session,
)
from src.schemas import RankedItem, UserProfile


def _ranked(*items):
    """items: (name, score). Score 5 -> use_first, 4 -> use_soon, else fresh."""
    out = []
    for name, score in items:
        label = "use_first" if score >= 5 else ("use_soon" if score == 4 else "moderate")
        out.append(
            RankedItem(item=name, score=score, reason="test", priority_label=label, **{"class": "leafy_green"})
        )
    return out


def _profile(**kwargs):
    p = UserProfile(diet_type="vegetarian", servings=2, profile_confirmed=True, **kwargs)
    p.sync_aliases()
    return p


def _chat_session(profile, items="spinach, rice, tomato"):
    session = new_session(profile, use_llm=False)
    session.stage = "inventory"
    session = ingest_typed_inventory(session, items)
    return session


# ---- snapshot persistence ----

def test_confirmed_inventory_snapshot_is_saved():
    profile = _profile()
    session = _chat_session(profile)
    session = confirm_inventory(session, profile)
    history = wt.load_history(profile.user_id)
    assert len(history) == 1
    snap = history[0]
    assert snap.confirmed is True
    assert snap.summary.total_items > 0
    assert snap.user_id == profile.user_id


def test_unconfirmed_detection_is_not_saved():
    profile = _profile()
    _chat_session(profile)  # detected but never confirmed
    assert wt.load_history(profile.user_id) == []


def test_at_risk_percentage_calculation():
    snap = wt.build_snapshot("u1", _ranked(("spinach", 5), ("milk", 4), ("rice", 2), ("pasta", 1)))
    assert snap.summary.total_items == 4
    assert snap.summary.at_risk_items == 2
    assert snap.summary.at_risk_percentage == 50.0
    assert "spinach" in snap.summary.top_at_risk_items


def test_zero_ingredients_handled_safely():
    snap, dup, disappeared = wt.record_snapshot("u1", [])
    assert snap.summary.at_risk_percentage == 0.0
    assert dup is None and disappeared == []
    # Empty snapshots are excluded from averages.
    wt.record_snapshot("u1", _ranked(("spinach", 5), ("rice", 2)))
    analytics = wt.compute_analytics("u1")
    assert analytics["average_at_risk_percentage"] == 50.0


# ---- patterns ----

def test_repeated_at_risk_ingredient_detected():
    for _ in range(3):
        wt.record_snapshot("u1", _ranked(("spinach", 5), ("rice", 2)))
    findings = wt.compare_with_history("u1")
    repeated = [f for f in findings if f.pattern == "repeated_at_risk"]
    assert repeated and repeated[0].ingredient == "spinach"
    assert repeated[0].occurrences == 3
    assert repeated[0].requires_user_confirmation is True


def test_missing_ingredient_is_unresolved_not_wasted():
    wt.record_snapshot("u1", _ranked(("spinach", 5), ("rice", 2)))
    _, _, disappeared = wt.record_snapshot("u1", _ranked(("rice", 2)))
    assert disappeared == ["spinach"]
    findings = wt.compare_with_history("u1")
    gone = [f for f in findings if f.pattern == "disappeared"]
    assert gone and gone[0].status == "unresolved"
    # Nothing was confirmed by the user, so there is NO waste.
    assert wt.compute_analytics("u1")["confirmed_waste_events"] == 0


def test_only_user_confirmed_spoilage_counts_as_waste():
    wt.record_snapshot("u1", _ranked(("spinach", 5), ("milk", 4)))
    wt.record_outcome("u1", "milk", "used")
    wt.record_outcome("u1", "spinach", "spoiled")
    wt.record_outcome("u1", "avocado", "not_sure")
    analytics = wt.compute_analytics("u1")
    assert analytics["confirmed_waste_events"] == 1
    assert analytics["most_wasted"] == [{"ingredient": "spinach", "count": 1}]


# ---- duplicate images ----

def test_duplicate_image_upload_is_flagged_not_new_purchase():
    ranked = _ranked(("spinach", 5))
    wt.record_snapshot("u1", ranked, source="image", image_reference="hash-abc")
    snap2, dup, _ = wt.record_snapshot("u1", ranked, source="image", image_reference="hash-abc")
    assert dup is not None
    assert snap2.possible_duplicate is True
    assert snap2.duplicate_of == dup.snapshot_id
    assert snap2.is_new_purchase is None  # unknown until the user answers
    # A different image is not a duplicate even with the same ingredients.
    snap3, dup3, _ = wt.record_snapshot("u1", ranked, source="image", image_reference="hash-xyz")
    assert dup3 is None and snap3.possible_duplicate is False


def test_user_answer_updates_purchase_flag():
    wt.record_snapshot("u1", _ranked(("spinach", 5)), source="image", image_reference="h")
    snap2, _, _ = wt.record_snapshot("u1", _ranked(("spinach", 5)), source="image", image_reference="h")
    wt.set_new_purchase_flag("u1", snap2.snapshot_id, False)
    stored = [s for s in wt.load_history("u1") if s.snapshot_id == snap2.snapshot_id][0]
    assert stored.is_new_purchase is False


# ---- recommendations ----

def test_recommendations_after_sufficient_history():
    for _ in range(4):
        wt.record_snapshot("u1", _ranked(("spinach", 5), ("rice", 2)))
    wt.record_outcome("u1", "spinach", "thrown_away")
    recs = wt.build_recommendations("u1")
    assert recs
    rec = recs[0]
    assert rec.ingredient == "spinach"
    assert 10 <= rec.suggested_reduction_percentage <= 50
    assert 0 < rec.confidence <= 0.95
    assert len(rec.supporting_snapshot_ids) >= 2
    assert "thrown away" in rec.reason or "spoiled" in rec.reason


def test_no_recommendations_with_insufficient_history():
    wt.record_snapshot("u1", _ranked(("spinach", 5)))
    wt.record_snapshot("u1", _ranked(("spinach", 5)))
    assert wt.build_recommendations("u1") == []


# ---- isolation & scope ----

def test_users_data_is_isolated():
    wt.record_snapshot("user-a", _ranked(("spinach", 5)))
    wt.record_snapshot("user-a", _ranked(("spinach", 5)))
    wt.record_snapshot("user-b", _ranked(("milk", 4)))
    wt.record_outcome("user-a", "spinach", "spoiled")
    assert len(wt.load_history("user-a")) == 2
    assert len(wt.load_history("user-b")) == 1
    assert wt.compute_analytics("user-b")["confirmed_waste_events"] == 0
    assert wt.load_outcomes("user-b") == []


def test_no_prices_or_monetary_values_anywhere():
    for _ in range(3):
        wt.record_snapshot("u1", _ranked(("spinach", 5), ("milk", 4)))
    wt.record_outcome("u1", "spinach", "spoiled")
    blobs = [
        json.dumps([s.model_dump() for s in wt.load_history("u1")]),
        json.dumps(wt.compute_analytics("u1")),
        json.dumps([r.model_dump() for r in wt.build_recommendations("u1")]),
    ]
    for blob in blobs:
        low = blob.lower()
        assert "price" not in low and "cost" not in low and "$" not in blob


# ---- chat integration ----

def test_history_hint_loaded_in_new_chat_session():
    profile = _profile()
    for _ in range(3):
        wt.record_snapshot(profile.user_id, _ranked(("spinach", 5), ("rice", 2)))
    session = new_session(profile, use_llm=False)
    greeting = session.messages[0].content
    assert "use-first group" in greeting
    assert "spinach" in greeting
    assert "at-risk percentage" in greeting


def test_outcome_followup_flow_records_confirmed_waste():
    profile = _profile()
    # First inventory: spinach is at risk.
    session = _chat_session(profile, "spinach, rice")
    session = confirm_inventory(session, profile)
    assert len(wt.load_history(profile.user_id)) == 1

    # Second inventory: spinach is gone -> follow-up question, never auto-waste.
    session2 = new_session(profile, use_llm=False)
    session2.stage = "inventory"
    session2 = ingest_typed_inventory(session2, "rice, pasta")
    session2 = confirm_inventory(session2, profile)
    assert session2.awaiting == "outcome:spinach"
    assert wt.compute_analytics(profile.user_id)["confirmed_waste_events"] == 0

    session2, profile = handle_message(session2, profile, "Thrown away")
    analytics = wt.compute_analytics(profile.user_id)
    assert analytics["confirmed_waste_events"] == 1
    assert session2.awaiting == "generate_plan"
