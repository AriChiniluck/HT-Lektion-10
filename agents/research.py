from functools import lru_cache
import json

from langchain.agents import create_agent
from langchain_core.tools import tool

from config import RESEARCH_SYSTEM_PROMPT, build_chat_model, settings
from tools import web_search, read_url, knowledge_search


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content) if content else ""


@lru_cache(maxsize=1)
def get_research_agent():
    return create_agent(
        model=build_chat_model(temperature=0.2, model=settings.researcher_model),
        tools=[web_search, read_url, knowledge_search],
        system_prompt=RESEARCH_SYSTEM_PROMPT,
    )


@tool
def research(plan: str) -> str:
    """Execute the research plan and return concise findings with sources."""
    plan_text = str(plan or "").strip()
    if not plan_text:
        return "Research plan is empty."

    # First calls arrive as a ResearchPlan JSON from the Planner.
    # Revision calls arrive as plain text from the middleware — pass through as-is.
    try:
        parsed = json.loads(plan_text)
        if isinstance(parsed, dict) and "goal" in parsed:
            goal = parsed.get("goal", "")
            queries = parsed.get("search_queries") or []
            sources = parsed.get("sources_to_check") or []
            output_format = parsed.get("output_format", "")
            parts = [f"Research goal: {goal}"]
            if queries:
                parts.append("Search queries to execute:\n" + "\n".join(f"- {q}" for q in queries))
            if sources:
                parts.append("Sources to consult: " + ", ".join(sources))
            if output_format:
                parts.append(f"Expected output format: {output_format}")
            plan_text = "\n\n".join(parts)
    except (json.JSONDecodeError, TypeError):
        pass  # plain text revision plan — use as-is

    research_request = (
        "Execute the following research plan and gather evidence.\n\n"
        f"{plan_text}\n\n"
        "Important: answer in the same language as the user's request, consult the local knowledge base first for course topics, and keep source metadata in the form 'Source / page / Relevance' when available."
    )
    try:
        result = get_research_agent().invoke(
            {"messages": [{"role": "user", "content": research_request}]},
            config={"recursion_limit": 17},  # ~8 tool calls max (2 nodes per call + 1 final)
        )
    except Exception as exc:
        return f"Research agent failed: {exc}"

    messages = result.get("messages", [])
    if not messages:
        return "Researcher did not return any findings."

    return _content_to_text(getattr(messages[-1], "content", ""))
