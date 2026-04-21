from langchain.agents import create_agent
from langchain_core.tools import tool

from config import PLANNER_SYSTEM_PROMPT, build_chat_model, settings
from schemas import ResearchPlan
from tools import knowledge_search as _knowledge_search

# Hard cap on knowledge_search calls per plan() invocation to prevent infinite loops.
_MAX_KS_CALLS = 3


def _make_planner_agent():
    """Create a fresh planner agent with a bounded knowledge_search for this invocation."""
    call_count = [0]

    @tool
    def knowledge_search(query: str) -> str:
        """Search the local knowledge base to understand course topics before planning."""
        call_count[0] += 1
        if call_count[0] > _MAX_KS_CALLS:
            return (
                f"knowledge_search call limit ({_MAX_KS_CALLS}) reached. "
                "Stop searching and produce the ResearchPlan now based on results already gathered."
            )
        return _knowledge_search.invoke(query)

    return create_agent(
        model=build_chat_model(temperature=0.1, model=settings.planner_model),
        tools=[knowledge_search],
        system_prompt=PLANNER_SYSTEM_PROMPT,
        response_format=ResearchPlan,
    )


def get_planner_agent():
    """Public alias used by tests — returns a fresh bounded planner agent."""
    return _make_planner_agent()


@tool
def plan(request: str) -> str:
    """Create a structured research plan for the user's request."""
    planner_request = (
        f"{request}\n\n"
        "Important: keep the `goal` and `output_format` fields in the same language as the user's request. "
        "However, the `search_queries` field MUST always use the original English technical terms — "
        "do NOT translate 'RAG', 'sentence-window retrieval', 'FAISS', 'BM25', 'LangGraph', 'LLM', "
        "'embedding', 'reranker', or any other technical term. Write search queries in English only."
    )
    result = _make_planner_agent().invoke(
        {"messages": [{"role": "user", "content": planner_request}]},
        config={"recursion_limit": 12},
    )

    structured = result.get("structured_response")
    if isinstance(structured, ResearchPlan):
        return structured.model_dump_json(indent=2)

    messages = result.get("messages", [])
    if messages:
        return str(getattr(messages[-1], "content", ""))

    return "Planner did not return a valid plan."
