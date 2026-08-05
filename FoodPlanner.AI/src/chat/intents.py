"""Rule-based intent classification for FoodPlanner.AI chat turns."""

from __future__ import annotations

import re
from typing import Optional, Tuple

from src.schemas import ChatStage, Intent


def classify_intent(text: str, stage: ChatStage, awaiting: Optional[str] = None) -> Tuple[Intent, dict]:
    t = (text or "").strip().lower()
    meta: dict = {}

    # Safety / injection attempts
    injection_markers = [
        "ignore previous",
        "ignore all previous",
        "ignore my allergy",
        "ignore your system",
        "pretend you are a chef with no",
        "disable restrictions",
        "show me another user's",
        "reveal system prompt",
        "update your knowledge base",
        "don't mention chicken",
        "do not mention chicken",
        "jailbreak",
    ]
    if any(m in t for m in injection_markers):
        return "unsafe_or_out_of_scope", {"reason": "prompt_injection"}

    if any(x in t for x in ["new plan", "start over", "reset plan", "start a new plan"]):
        return "reset_current_plan", {}

    if any(x in t for x in ["edit profile", "change my profile", "update profile"]):
        return "edit_profile", {}

    # Quick-reply navigation phrases. These MUST be matched before the generic
    # add/remove regexes below, otherwise buttons like "Add to shopping list"
    # get captured as literal inventory items and fall into dead-end loops.
    m = re.search(r"add\s+(.+?)\s+to\s+(?:the\s+)?shopping\s*list", t)
    if m:
        return "add_to_shopping_list", {"item": m.group(1).strip()}
    if "add to shopping list" in t or "keep as shopping gap" in t:
        return "add_to_shopping_list", {}

    if stage == "shopping_list" or "shopping list" in t:
        if "shopping" in t or "buy" in t or "grocery" in t:
            return "generate_shopping_list", {}
    if any(x in t for x in ["edit inventory", "review inventory", "show inventory", "change inventory"]):
        return "edit_inventory", {}
    if any(x in t for x in ["review plan", "show plan", "view plan", "see the plan", "keep current plan", "keep the plan", "keep plan"]):
        return "show_plan", {}
    if t in {"continue planning", "continue", "keep going"}:
        return "continue_planning", {}
    if any(x in t for x in ["make it faster", "faster meals", "quicker meals", "less time", "too slow"]):
        return "make_faster", {}
    if "offline recipe" in t or "use offline" in t:
        return "use_offline", {}
    if "another photo" in t or "add a photo" in t or "upload photo" in t:
        return "upload_inventory", {"mode": "photo"}
    if "adjust priorit" in t or re.search(r"use\s+.+\s+first", t):
        return "update_freshness", {"raw": text}
    if re.search(r"replace\s+(?:this|a|another)\s+meal", t) or t == "replace meal":
        return "replace_meal", {}

    # Detailed recipe requests ("View steps 2", "full recipe", "how do I make X").
    m = re.search(r"(?:view|show|see|full|detailed)\s+(?:steps|recipe)(?:\s+(?:for\s+)?(?:day\s*)?(\d+))?", t)
    if m:
        return "request_recipe_steps", ({"day": int(m.group(1))} if m.group(1) else {})
    wants_alternative = any(x in t for x in ["alternative", "something else", "different", "replace", "other option", "don't know", "dont know"])
    m = re.search(r"how\s+(?:do\s+i|to|can\s+i)\s+(?:make|cook|prepare)\s+(.+)", t)
    if m and not wants_alternative:
        return "request_recipe_steps", {"dish": m.group(1).strip()}

    # Inventory edits
    m = re.search(r"(?:that is not|it's not|its not|not)\s+(.+?)[,.]?\s*(?:it is|it's|its|=)\s*(.+)", t)
    if m:
        return "correct_inventory_item", {"from": m.group(1).strip(), "to": m.group(2).strip()}

    m = re.search(r"(?:rename|change)\s+(.+?)\s+to\s+(.+)", t)
    if m:
        return "correct_inventory_item", {"from": m.group(1).strip(), "to": m.group(2).strip()}

    m = re.search(r"(?:you missed|also have|i (?:also )?have|add)\s+(.+)", t)
    if m and stage in {"confirmation", "inventory", "freshness", "meal_plan", "adjustments"}:
        captured = m.group(1).strip()
        # Button labels like "Add an ingredient" are requests, not item names.
        generic = {
            "an ingredient", "ingredient", "ingredients", "ingredient manually",
            "an item", "item", "items", "another", "something", "more",
        }
        if captured in generic:
            return "add_inventory_item", {"items_text": ""}
        return "add_inventory_item", {"items_text": captured}

    m = re.search(r"(?:remove|exclude|delete)\s+(.+)", t)
    if m and stage in {"confirmation", "inventory", "freshness", "meal_plan", "adjustments"}:
        return "remove_inventory_item", {"item": m.group(1).strip()}

    if stage == "confirmation" and any(x in t for x in ["confirm", "looks good", "that's correct", "thats correct", "yes", "done"]):
        return "confirm_inventory", {}

    # Plan feedback
    m = re.search(r"(?:replace|change|don't like|dont like|dislike)\s+(?:day|meal)\s*(\d+)", t)
    if not m:
        m = re.search(r"day\s*(\d+).*(?:too complicated|too long|replace|don't like|dont like)", t)
    if m:
        meta["day"] = int(m.group(1))
        return "replace_meal", meta

    if any(x in t for x in ["i don't like day", "i dont like day", "replace day", "keep day"]):
        m = re.search(r"day\s*(\d+)", t)
        if m:
            meta["day"] = int(m.group(1))
            return "replace_meal", meta

    if "servings" in t or "people" in t:
        m = re.search(r"(\d+)\s*(?:people|servings|persons)", t)
        if m:
            meta["servings"] = int(m.group(1))
            return "change_servings", meta

    if any(x in t for x in ["accept plan", "confirm plan", "looks good", "finalize", "yes, confirm"]):
        if stage in {"meal_plan", "adjustments"}:
            return "accept_plan", {}

    if any(x in t for x in ["generate plan", "create plan", "make a plan", "plan now", "yes generate", "yes, generate"]):
        return "generate_plan", {}

    if any(x in t for x in ["substitute", "missing", "make from", "homemade", "derivation"]):
        return "request_substitution", {}

    if awaiting and awaiting.startswith("sub_select"):
        return "select_substitution", {"raw": text}

    if awaiting and awaiting.startswith("conflict"):
        return "answer_question", {"raw": text}

    # Freshness corrections
    if stage == "freshness" and any(x in t for x in ["actually fresh", "expires", "froze", "do not prioritize", "don't prioritize"]):
        return "update_freshness", {"raw": text}

    if stage in {"profile", "preferences"}:
        return "answer_question", {"raw": text}

    if any(x in t for x in ["type inventory", "manual", "no photo", "enter manually"]):
        return "upload_inventory", {"mode": "typed"}

    return "answer_question", {"raw": text}
