"""Test model invocations must never charge the live accounting database."""
import pytest


@pytest.fixture(autouse=True)
def isolate_group_model_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("GROUP_CHAT_BUDGET_DB", str(tmp_path / "group-model-budget.db"))
