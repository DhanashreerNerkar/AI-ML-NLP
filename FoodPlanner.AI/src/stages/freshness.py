"""Stage 2 — freshness / perishability scoring grounded in shelf_life_kb."""

from __future__ import annotations

from typing import List, Union

from src.kb import load_shelf_life_kb, normalize_name
from src.schemas import InventoryItem, RankedItem, RankedItems


def _priority_label(score: int) -> str:
    if score >= 5:
        return "use_first"
    if score == 4:
        return "use_soon"
    if score == 3:
        return "moderate"
    if score == 2:
        return "longer"
    return "shelf_stable"


def _lookup_item(name: str, kb: dict):
    n = normalize_name(name)
    default_score = int(kb.get("default_score", 3))
    default_class = kb.get("default_class", "unknown")
    for entry in kb.get("entries", []):
        candidates = [entry.get("item", ""), *entry.get("aliases", [])]
        for c in candidates:
            c_n = normalize_name(c)
            if n == c_n or c_n in n or n in c_n:
                return (
                    int(entry["score"]),
                    entry.get("class", default_class),
                    f"{entry.get('class', default_class)} from shelf-life reference",
                    True,
                )
    return default_score, default_class, "not in shelf-life reference; default score 3", False


def score_freshness_deterministic(
    confirmed_items: Union[List[str], List[InventoryItem]],
) -> RankedItems:
    kb = load_shelf_life_kb()
    ranked: List[RankedItem] = []
    for raw in confirmed_items:
        if isinstance(raw, InventoryItem):
            if raw.exclude_from_plan or raw.do_not_use:
                continue
            name = raw.normalized_name
            opened = raw.opened
            cooked = raw.cooked
            frozen = raw.frozen
            use_soon = raw.use_soon_user_flag
            ingredient_id = raw.id
        else:
            name = str(raw)
            opened = cooked = frozen = use_soon = False
            ingredient_id = None

        score, cls, reason, in_ref = _lookup_item(name, kb)
        uncertain = not in_ref
        if frozen:
            score = max(1, score - 2)
            reason = f"{reason}; frozen — lower planning urgency"
        if opened and score < 5:
            score = min(5, score + 1)
            reason = f"{reason}; opened"
        if cooked:
            score = min(5, max(score, 4))
            reason = f"{reason}; leftover/cooked — use soon"
        if use_soon:
            score = 5
            reason = f"{reason}; user marked use soon"

        ranked.append(
            RankedItem(
                item=normalize_name(name),
                score=score,
                **{"class": cls},
                reason=reason,
                in_reference=in_ref,
                priority_label=_priority_label(score),  # type: ignore[arg-type]
                uncertain=uncertain,
                ingredient_id=ingredient_id,
            )
        )
    ranked.sort(key=lambda r: r.score, reverse=True)
    return RankedItems(ranked=ranked)


def score_freshness(confirmed_items, use_llm: bool = False) -> RankedItems:
    # Freshness is grounded in KB; LLM path intentionally unused for honesty/safety.
    return score_freshness_deterministic(confirmed_items)


def format_freshness_summary(ranked: RankedItems) -> str:
    buckets = {
        "Use first": [],
        "Use soon": [],
        "Longer-lasting": [],
    }
    for r in ranked.ranked:
        if r.score >= 5:
            buckets["Use first"].append(f"• {r.item} — {r.reason}")
        elif r.score == 4:
            buckets["Use soon"].append(f"• {r.item} — {r.reason}")
        else:
            buckets["Longer-lasting"].append(f"• {r.item} — {r.reason}")
    parts = ["Here's what I recommend using first:\n"]
    for title, lines in buckets.items():
        if lines:
            parts.append(f"**{title}:**")
            parts.extend(lines)
            parts.append("")
    parts.append(
        "_These are planning-priority estimates, not food-safety guarantees. "
        "Check smell, texture, storage history, and package dates before eating._"
    )
    return "\n".join(parts)
