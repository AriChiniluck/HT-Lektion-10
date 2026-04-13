"""
test_tools.py — Tool Correctness tests for the multi-agent system.

Verifies that each agent calls the correct tools for a given input:
  1. Planner receives a research request → must call `knowledge_search`
  2. Researcher receives a plan → must call at least one search tool
     (knowledge_search, web_search, or read_url)
  3. Supervisor completes a full pipeline → must call `save_report`

Metric: ToolCorrectnessMetric (deepeval)
  threshold=0.5 — lenient because tool order / extra tool calls are acceptable.

Run with debug output:
  deepeval test run tests/test_tools.py --debug
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import deepeval
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from config import settings
from tools import debug_print
from agents.planner import get_planner_agent
from agents.research import get_research_agent
from supervisor import supervisor, reset_supervisor_limits

EVAL_MODEL = os.getenv("DEEPEVAL_MODEL", settings.eval_model)

TOOL_METRIC = ToolCorrectnessMetric(threshold=0.5, model=EVAL_MODEL)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_ai_tool_calls(messages: list) -> list[ToolCall]:
    """Parse AIMessages from an agent result and return ToolCall objects."""
    captured: list[ToolCall] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in (getattr(msg, "tool_calls", []) or []):
                name = tc.get("name", "")
                args = tc.get("args", {}) or {}
                if name:
                    debug_print(f"  [test_tools] AI tool call: {name}({list(args.keys())})")
                    captured.append(ToolCall(name=name, input_parameters=args))
    return captured


def _stream_supervisor_collect(payload, config: dict) -> tuple[list[ToolCall], list, str]:
    """Stream the supervisor and collect tool calls, interrupts, and final text."""
    tool_calls: list[ToolCall] = []
    interrupts: list = []
    final_texts: list[str] = []

    for chunk in supervisor.stream(
        payload,
        config=config,
        stream_mode=["updates"],
        version="v2",
    ):
        if chunk["type"] != "updates":
            continue

        data = chunk["data"]

        if "__interrupt__" in data:
            interrupts = list(data["__interrupt__"])
            continue

        model_payload = data.get("model") or {}
        for msg in model_payload.get("messages", []):
            for tc in (getattr(msg, "tool_calls", []) or []):
                name = tc.get("name", "")
                args = tc.get("args", {}) or {}
                if name:
                    debug_print(f"  [test_tools] supervisor tool call: {name}")
                    tool_calls.append(ToolCall(name=name, input_parameters=args))

            text = ""
            content = getattr(msg, "content", "") or ""
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                text = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                ).strip()
            if text and not getattr(msg, "tool_calls", None):
                final_texts.append(text)

    return tool_calls, interrupts, "\n".join(final_texts).strip()


# ── Test 1: Planner → knowledge_search ───────────────────────────────────────


def test_tool_correctness_planner_uses_knowledge_search() -> None:
    """Planner should call `knowledge_search` when handling a course-specific RAG query."""
    request = "Explain naive RAG, sentence-window retrieval, and parent-child chunking from the course knowledge base"

    debug_print(f"\n[test_tools] Test 1 — Planner tool correctness")
    debug_print(f"  request: {request!r}")

    agent = get_planner_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": request}]})
    messages = result.get("messages", [])
    tools_called = _extract_ai_tool_calls(messages)
    tool_names_called = [tc.name for tc in tools_called]

    debug_print(f"  tools called: {tool_names_called}")

    # Build final text for LLMTestCase
    final_output = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            content = getattr(msg, "content", "") or ""
            if isinstance(content, str) and content.strip():
                final_output = content[:500]
                break
    if not final_output:
        final_output = str(result.get("structured_response", "plan produced"))

    test_case = LLMTestCase(
        input=request,
        actual_output=final_output,
        tools_called=tools_called,
        expected_tools=[ToolCall(name="knowledge_search")],
    )

    deepeval.assert_test(test_case, [TOOL_METRIC])


# ── Test 2: Researcher → at least one search tool ────────────────────────────


def test_tool_correctness_researcher_uses_search_tools() -> None:
    """Researcher should call at least one search tool when executing a plan."""
    plan_text = (
        "Research goal: Describe main differences between naive RAG and agentic RAG.\n\n"
        "Search queries:\n"
        "- naive RAG chunk splitting\n"
        "- agentic RAG orchestration 2025 2026\n\n"
        "Sources to consult: knowledge_base, web\n\n"
        "Expected output: Comparative analysis with sources."
    )

    debug_print(f"\n[test_tools] Test 2 — Researcher tool correctness")

    agent = get_research_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": plan_text}]},
        config={"recursion_limit": 15},
    )
    messages = result.get("messages", [])
    tools_called = _extract_ai_tool_calls(messages)
    tool_names_called = [tc.name for tc in tools_called]

    debug_print(f"  tools called: {tool_names_called}")

    final_output = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            content = getattr(msg, "content", "") or ""
            if isinstance(content, str) and content.strip():
                final_output = content[:600]
                break

    search_tools = {"knowledge_search", "web_search", "read_url"}
    assert search_tools & set(tool_names_called), (
        f"Researcher should call at least one of {search_tools}, "
        f"but only called: {tool_names_called}"
    )

    test_case = LLMTestCase(
        input=plan_text,
        actual_output=final_output or "researcher output",
        tools_called=tools_called,
        expected_tools=[
            ToolCall(name="knowledge_search"),
        ],
    )

    deepeval.assert_test(test_case, [TOOL_METRIC])


# ── Test 3: Supervisor → save_report after full pipeline ─────────────────────


def test_tool_correctness_supervisor_calls_save_report() -> None:
    """After the full Planner→Researcher→Critic pipeline, Supervisor must call `save_report`."""
    thread_id = f"test-tools-{uuid4().hex[:8]}"
    reset_supervisor_limits(thread_id)  # ensure clean state for new thread
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.graph_recursion_limit,
    }
    # Use a complex, multi-part query that forces the full research pipeline
    user_input = (
        "Research all major RAG approaches from the course materials: naive RAG, "
        "advanced RAG, and agentic RAG. Compare their architectures and create a report."
    )

    debug_print(f"\n[test_tools] Test 3 — Supervisor → save_report, thread={thread_id}")
    debug_print(f"  query: {user_input[:80]!r}...")

    payload: dict | Command = {"messages": [{"role": "user", "content": user_input}]}

    all_tool_calls: list[ToolCall] = []
    max_rounds = 6  # plan + research + critique + optional revision + save + approve

    for round_num in range(max_rounds):
        tool_calls, interrupts, _ = _stream_supervisor_collect(payload, config)
        all_tool_calls.extend(tool_calls)

        names_this_round = [tc.name for tc in tool_calls]
        debug_print(f"  round {round_num}: tool calls = {names_this_round}")

        if not interrupts:
            break

        # Auto-approve save_report interrupt
        debug_print("  [test_tools] auto-approving save_report interrupt")
        payload = Command(resume={"decisions": [{"type": "approve"}]})

    reset_supervisor_limits(thread_id)

    all_names = [tc.name for tc in all_tool_calls]
    debug_print(f"  all tools called across full run: {all_names}")

    assert any(t in all_names for t in ("plan", "research")), (
        f"Supervisor did not invoke any research tools — query may be too simple or pipeline was skipped. "
        f"Called: {all_names}"
    )
    assert "save_report" in all_names, (
        f"Supervisor should call `save_report` after completing the pipeline, "
        f"but only called: {all_names}"
    )

    test_case = LLMTestCase(
        input=user_input,
        actual_output="supervisor completed pipeline",
        tools_called=all_tool_calls,
        expected_tools=[
            ToolCall(name="plan"),
            ToolCall(name="research"),
            ToolCall(name="critique"),
            ToolCall(name="save_report"),
        ],
    )

    deepeval.assert_test(test_case, [TOOL_METRIC])
