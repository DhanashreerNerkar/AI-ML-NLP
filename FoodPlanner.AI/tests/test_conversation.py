"""Free-form conversational messages must get a real answer, not the fallback menu."""

from src.chat.orchestrator import (
    confirm_inventory,
    generate_plan,
    handle_message,
    ingest_typed_inventory,
    new_session,
)
from src.schemas import UserProfile

FALLBACK_SNIPPET = "I want to make sure I help with the right next step"


def _ready_session():
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
    return session, profile


def test_declining_generate_plan_is_acknowledged():
    session, profile = _ready_session()
    assert session.awaiting == "generate_plan"
    session, profile = handle_message(session, profile, "no dont generete")
    reply = session.messages[-1].content
    assert FALLBACK_SNIPPET not in reply
    assert "won’t generate" in reply or "won't generate" in reply


def test_asking_for_alternative_replaces_matching_meal():
    session, profile = _ready_session()
    session = generate_plan(session, profile)
    target = session.plan.plan[0].recipe.split()[0].lower()
    session, profile = handle_message(
        session, profile, f"I dont know how to cook {target} can you give me some other alternative"
    )
    reply = session.messages[-1].content
    assert FALLBACK_SNIPPET not in reply
    assert "updated Day" in reply or "Which meal" in reply


def test_view_steps_returns_full_recipe_with_servings():
    session, profile = _ready_session()
    session = generate_plan(session, profile)
    session, profile = handle_message(session, profile, "View steps 1")
    reply = session.messages[-1].content
    assert FALLBACK_SNIPPET not in reply
    assert "**Ingredients**" in reply
    assert "**Steps**" in reply
    assert f"serves {profile.servings}" in reply


def test_how_do_i_make_maps_to_detailed_recipe():
    session, profile = _ready_session()
    session = generate_plan(session, profile)
    dish_word = session.plan.plan[0].recipe.split()[0]
    session, profile = handle_message(session, profile, f"how do I make {dish_word}?")
    reply = session.messages[-1].content
    assert FALLBACK_SNIPPET not in reply
    assert "**Ingredients**" in reply and "**Steps**" in reply


def test_alternative_request_with_unknown_food_asks_which_meal():
    session, profile = _ready_session()
    session = generate_plan(session, profile)
    session, profile = handle_message(
        session, profile, "I dont know how to cook xyzfood can you give me some other alternative"
    )
    reply = session.messages[-1].content
    assert FALLBACK_SNIPPET not in reply
    assert "Which meal" in reply
