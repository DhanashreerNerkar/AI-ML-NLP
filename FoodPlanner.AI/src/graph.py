"""LangGraph orchestration for PantryPilot."""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from src.pipeline import (
    PipelineState,
    route_intent,
    stage_critical,
    stage_freshness,
    stage_gap,
    stage_plan,
    stage_retrieve,
    stage_substitute,
)


def build_plan_graph():
    g = StateGraph(PipelineState)
    g.add_node("stage_freshness", stage_freshness)
    g.add_node("stage_critical", stage_critical)
    g.add_node("stage_retrieve", stage_retrieve)
    g.add_node("stage_plan", stage_plan)
    g.add_node("stage_gap", stage_gap)

    g.add_edge(START, "stage_freshness")
    g.add_edge("stage_freshness", "stage_critical")
    g.add_edge("stage_critical", "stage_retrieve")
    g.add_edge("stage_retrieve", "stage_plan")
    g.add_edge("stage_plan", "stage_gap")
    g.add_edge("stage_gap", END)
    return g.compile()


def build_substitute_graph():
    g = StateGraph(PipelineState)
    g.add_node("stage_substitute", stage_substitute)
    g.add_edge(START, "stage_substitute")
    g.add_edge("stage_substitute", END)
    return g.compile()


@lru_cache(maxsize=1)
def build_router_graph():
    """Single graph with conditional routing by intent."""
    g = StateGraph(PipelineState)
    g.add_node("stage_freshness", stage_freshness)
    g.add_node("stage_critical", stage_critical)
    g.add_node("stage_retrieve", stage_retrieve)
    g.add_node("stage_plan", stage_plan)
    g.add_node("stage_gap", stage_gap)
    g.add_node("stage_substitute", stage_substitute)

    def _entry(state: PipelineState) -> str:
        return "stage_substitute" if route_intent(state) == "substitute" else "stage_freshness"

    g.add_conditional_edges(START, _entry, {
        "stage_freshness": "stage_freshness",
        "stage_substitute": "stage_substitute",
    })
    g.add_edge("stage_freshness", "stage_critical")
    g.add_edge("stage_critical", "stage_retrieve")
    g.add_edge("stage_retrieve", "stage_plan")
    g.add_edge("stage_plan", "stage_gap")
    g.add_edge("stage_gap", END)
    g.add_edge("stage_substitute", END)
    return g.compile()
