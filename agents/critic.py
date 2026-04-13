from functools import lru_cache

from langchain.agents import create_agent
from langchain_core.tools import tool

from config import CRITIC_SYSTEM_PROMPT, build_chat_model, settings
from schemas import CritiqueResult
from tools import knowledge_search


@lru_cache(maxsize=1)
def get_critic_agent():
    return create_agent(
        model=build_chat_model(temperature=0.0, model=settings.critic_model),
        tools=[knowledge_search],
        system_prompt=CRITIC_SYSTEM_PROMPT,
        response_format=CritiqueResult,
    )


@tool
def critique(original_request: str, findings: str, plan: str = "") -> str:
    """Verify findings against the original request and return a structured approve/revise critique."""
    # Limit findings size to avoid unnecessary token spend — critic evaluates quality, not volume.
    MAX_FINDINGS_LEN = 8000
    findings_text = str(findings or "").strip()
    if len(findings_text) > MAX_FINDINGS_LEN:
        findings_text = findings_text[:MAX_FINDINGS_LEN] + "\n\n[... findings truncated to fit evaluation context ...]"

    critique_request = "Original user request:\n" + original_request + "\n\n"
    if plan.strip():
        critique_request += "Research plan that was executed:\n" + plan.strip() + "\n\n"
    critique_request += (
        "Current research findings:\n"
        f"{findings_text}\n\n"
        "Important: return all explanation fields in the same language as the user's request/findings."
    )
    try:
        result = get_critic_agent().invoke(
            {"messages": [{"role": "user", "content": critique_request}]}
        )

        structured = result.get("structured_response")
        if isinstance(structured, CritiqueResult):
            return structured.model_dump_json(indent=2)

        messages = result.get("messages", [])
        if messages:
            return str(getattr(messages[-1], "content", ""))

    except Exception as exc:
        fallback = CritiqueResult(
            verdict="REVISE",
            is_fresh=False,
            is_complete=False,
            is_well_structured=False,
            strengths=[],
            gaps=[
                "The critic agent failed to return fully valid structured output.",
                str(exc),
            ],
            revision_requests=[
                "Re-run the verification and ensure all required CritiqueResult fields are present.",
            ],
        )
        return fallback.model_dump_json(indent=2)

    return CritiqueResult(
        verdict="REVISE",
        is_fresh=False,
        is_complete=False,
        is_well_structured=False,
        strengths=[],
        gaps=["Critic did not return a valid critique."],
        revision_requests=["Run the critique step again with clearer evidence."],
    ).model_dump_json(indent=2)
