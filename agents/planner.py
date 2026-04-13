from functools import lru_cache

from langchain.agents import create_agent
from langchain_core.tools import tool

from config import PLANNER_SYSTEM_PROMPT, build_chat_model, settings
from schemas import ResearchPlan
from tools import knowledge_search


@lru_cache(maxsize=1)
def get_planner_agent():
    return create_agent(
        model=build_chat_model(temperature=0.1, model=settings.planner_model),
        tools=[knowledge_search],
        system_prompt=PLANNER_SYSTEM_PROMPT,
        response_format=ResearchPlan,
    )


@tool
def plan(request: str) -> str:
    """Create a structured research plan for the user's request."""
    planner_request = (
        f"{request}\n\n"
        "Important: keep all ResearchPlan text fields in the same language as the user's request."
    )
    result = get_planner_agent().invoke(
        {"messages": [{"role": "user", "content": planner_request}]}
    )

    structured = result.get("structured_response")
    if isinstance(structured, ResearchPlan):
        return structured.model_dump_json(indent=2)

    messages = result.get("messages", [])
    if messages:
        return str(getattr(messages[-1], "content", ""))

    return "Planner did not return a valid plan."
