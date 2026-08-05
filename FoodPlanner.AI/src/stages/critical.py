"""Stage 3 — deterministic critical-priority filter."""

from __future__ import annotations

from src.schemas import CriticalPriority, RankedItem, RankedItems


def critical_priority_filter(
    ranked: RankedItems | list[RankedItem] | list[dict],
    threshold: int = 4,
) -> CriticalPriority:
    items: list[RankedItem]
    if isinstance(ranked, RankedItems):
        items = ranked.ranked
    else:
        items = [
            r if isinstance(r, RankedItem) else RankedItem.model_validate(r)
            for r in ranked
        ]

    critical = [r.item for r in items if r.score >= threshold]
    if not critical:
        explanation = "No items scored at or above the critical threshold."
    else:
        explanation = (
            f"{', '.join(critical)} scored >= {threshold} and should be used first "
            "to reduce waste and food-safety risk."
        )
    return CriticalPriority(critical_priority=critical, explanation=explanation)
