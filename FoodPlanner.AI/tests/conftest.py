"""Pytest path bootstrap."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolate_waste_storage(tmp_path, monkeypatch):
    """Keep waste-tracker history out of real user data during tests."""
    from src import waste_tracker

    monkeypatch.setattr(waste_tracker, "USER_DATA", tmp_path)
    monkeypatch.setattr(waste_tracker, "HISTORY_PATH", tmp_path / "inventory_history.json")
    monkeypatch.setattr(waste_tracker, "OUTCOMES_PATH", tmp_path / "ingredient_outcomes.json")
