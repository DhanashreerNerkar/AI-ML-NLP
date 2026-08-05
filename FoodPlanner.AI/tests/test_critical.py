"""Unit tests for critical-priority filter."""

from src.schemas import RankedItem, RankedItems
from src.stages.critical import critical_priority_filter


def test_critical_threshold():
    ranked = RankedItems(
        ranked=[
            RankedItem(item="spinach", score=5, **{"class": "leafy_green"}),
            RankedItem(item="chicken", score=5, **{"class": "raw_poultry"}),
            RankedItem(item="lemon", score=2, **{"class": "citrus"}),
            RankedItem(item="rice", score=1, **{"class": "grain"}),
        ]
    )
    result = critical_priority_filter(ranked)
    assert result.critical_priority == ["spinach", "chicken"]


def test_empty_critical():
    ranked = [{"item": "rice", "score": 1, "class": "grain"}]
    result = critical_priority_filter(ranked)
    assert result.critical_priority == []
