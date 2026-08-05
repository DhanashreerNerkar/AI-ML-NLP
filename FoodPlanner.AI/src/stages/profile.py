"""Stage 0 helpers — profile construction."""

from __future__ import annotations

from typing import List, Optional

from src.schemas import UserProfile


def build_profile(
    *,
    diet_type: str = "vegetarian",
    diet_style: str = "strict-vegetarian",
    cultural_constraints: Optional[List[str]] = None,
    dietary_restrictions: Optional[List[str]] = None,
    dislikes: Optional[List[str]] = None,
    servings: int = 2,
    nights: int = 3,
    time_limit_min: int = 30,
    allergies: Optional[List[str]] = None,
    cultural_rules: Optional[List[str]] = None,
) -> UserProfile:
    profile = UserProfile(
        diet_type=diet_type,  # type: ignore[arg-type]
        diet_style=diet_style,
        cultural_constraints=cultural_constraints or cultural_rules or [],
        cultural_rules=cultural_rules or cultural_constraints or [],
        dietary_restrictions=dietary_restrictions or [],
        allergies=allergies or [],
        dislikes=dislikes or [],
        servings=servings,
        nights=nights,
        time_limit_min=time_limit_min,
        profile_confirmed=True,
    )
    return profile.sync_aliases()


def profile_is_complete(profile: Optional[UserProfile]) -> bool:
    return profile is not None and bool(profile.profile_confirmed)
