"""Long-term profile + session persistence for FoodPlanner.AI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.kb import DATA, ROOT
from src.schemas import SessionState, UserProfile

USER_DATA = DATA / "user_data"
PROFILE_PATH = USER_DATA / "user_profile.json"
SESSION_PATH = USER_DATA / "session.json"


def ensure_user_data_dir() -> None:
    USER_DATA.mkdir(parents=True, exist_ok=True)


def save_profile(profile: UserProfile) -> Path:
    ensure_user_data_dir()
    profile.sync_aliases()
    PROFILE_PATH.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return PROFILE_PATH


def load_profile() -> Optional[UserProfile]:
    if not PROFILE_PATH.exists():
        # Fall back to fixture for first boot demos
        fixture = DATA / "fixtures" / "user_profile.json"
        if fixture.exists():
            raw = json.loads(fixture.read_text(encoding="utf-8"))
            # Only treat as saved if confirmed flag present and true
            if raw.get("profile_confirmed"):
                return UserProfile.model_validate(raw).sync_aliases()
        return None
    raw = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    return UserProfile.model_validate(raw).sync_aliases()


def clear_profile() -> None:
    if PROFILE_PATH.exists():
        PROFILE_PATH.unlink()


def save_session(session: SessionState) -> Path:
    ensure_user_data_dir()
    SESSION_PATH.write_text(session.model_dump_json(indent=2), encoding="utf-8")
    return SESSION_PATH


def load_session() -> Optional[SessionState]:
    if not SESSION_PATH.exists():
        return None
    raw = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
    return SessionState.model_validate(raw)


def clear_session() -> None:
    if SESSION_PATH.exists():
        SESSION_PATH.unlink()


def profile_summary(profile: UserProfile) -> str:
    p = profile.sync_aliases()
    lines = [
        f"Diet: {p.diet_type.replace('-', ' ').title()}",
        f"Food rules: {', '.join(p.cultural_rules) if p.cultural_rules else 'None'}",
    ]
    if p.jain_rules:
        lines.append(f"Jain details: {', '.join(p.jain_rules)}")
    lines.append(f"Allergies: {', '.join(p.allergies) if p.allergies else 'None reported'}")
    lines.append(f"Dislikes: {', '.join(p.dislikes) if p.dislikes else 'None'}")
    lines.append(f"Servings: {p.servings}")
    lines.append(f"Cooking time: about {p.time_limit_min} minutes")
    lines.append(f"Equipment: {', '.join(p.equipment) if p.equipment else 'Not specified'}")
    if p.preferred_cuisines:
        lines.append(f"Preferred cuisines: {', '.join(p.preferred_cuisines)}")
    lines.append(f"Skill level: {p.cooking_skill}")
    return "\n".join(f"• {line}" for line in lines)
