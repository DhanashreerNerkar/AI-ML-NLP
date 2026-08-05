"""Render the Streamlit app with chat histories that used to crash it.

Regression for StreamlitDuplicateElementKey: replacing a meal produces two
meal-card messages in history; both used to render buttons with identical keys.
"""

from streamlit.testing.v1 import AppTest

from src.chat.orchestrator import (
    confirm_inventory,
    generate_plan,
    handle_message,
    ingest_typed_inventory,
    new_session,
)
from src.schemas import UserProfile


def _session_with_replaced_meal():
    profile = UserProfile(
        diet_type="vegetarian",
        cultural_rules=["halal"],
        allergies=["egg"],
        servings=2,
        profile_confirmed=True,
    )
    profile.sync_aliases()
    session = new_session(profile, use_llm=False)
    session.stage = "inventory"
    session = ingest_typed_inventory(session, "spinach, rice, tomato, garlic")
    session = confirm_inventory(session, profile)
    session = generate_plan(session, profile)
    # Second meal_cards message — this used to duplicate widget keys.
    session, profile = handle_message(session, profile, "Replace day 1")
    assert sum(1 for m in session.messages if m.kind == "meal_cards") >= 2
    return session, profile


def test_app_renders_with_duplicate_meal_card_messages():
    session, profile = _session_with_replaced_meal()
    at = AppTest.from_file("app.py", default_timeout=30)
    at.session_state["chat"] = session
    at.session_state["profile"] = profile
    at.session_state["use_llm"] = False
    at.run()
    assert not at.exception, f"App crashed while rendering: {at.exception}"


def test_waste_tracker_panel_renders():
    session, profile = _session_with_replaced_meal()
    at = AppTest.from_file("app.py", default_timeout=30)
    at.session_state["chat"] = session
    at.session_state["profile"] = profile
    at.session_state["use_llm"] = False
    at.session_state["show_tracer"] = True
    at.run()
    assert not at.exception, f"Tracer panel crashed: {at.exception}"
