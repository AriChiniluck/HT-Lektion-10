import json
import re
from collections import defaultdict
from functools import lru_cache
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from config import SUPERVISOR_SYSTEM_PROMPT, build_chat_model, settings
from tools import save_report
from agents import plan, research, critique


BEST_EFFORT_DISCLAIMER = (
    "> **⚠️ Best-effort draft:** this report was saved after reaching the maximum "
    "number of revise cycles and may still contain unresolved gaps noted by the Critic.\n\n"
)

def _make_counters(original_request: str | None = None) -> dict:
    return {
        "research_calls": 0,
        "revise_cycles": 0,
        "limit_reached": False,
        "awaiting_save": False,
        "force_research": False,
        "last_critique_payload": None,
        "last_findings": None,
        "original_request": original_request,
    }


_RUN_LIMITS: dict[str, dict[str, object]] = defaultdict(_make_counters)


def _get_thread_id(request: ToolCallRequest) -> str:
    config = getattr(request.runtime, "config", {}) or {}
    configurable = config.get("configurable", {}) or {}
    return str(configurable.get("thread_id", "default-thread"))


_INVALID_JSON_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_text(text: str) -> str:
    """Strip control characters that break JSON serialisation of the OpenAI API payload."""
    return _INVALID_JSON_CHARS.sub("", text)


def _tool_content_to_text(content) -> str:
    if isinstance(content, str):
        return _sanitize_text(content)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return _sanitize_text("".join(parts))
    return _sanitize_text(str(content)) if content else ""


def _prepend_best_effort_disclaimer(content: str) -> str:
    text = content or ""
    if text.startswith(BEST_EFFORT_DISCLAIMER):
        return text
    return BEST_EFFORT_DISCLAIMER + text


def _suggest_report_filename(content: str, default_stem: str = "report") -> str:
    for line in str(content or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            slug = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
            if slug:
                return f"{slug[:80]}.md"
    return f"{default_stem}.md"


def reset_supervisor_limits(thread_id: str | None = None) -> None:
    if thread_id is None:
        _RUN_LIMITS.clear()
        return
    _RUN_LIMITS.pop(thread_id, None)


def get_last_critique_payload(thread_id: str) -> dict | None:
    """Return the last critique payload for a given thread (used by main.py for debug display)."""
    return _RUN_LIMITS.get(thread_id, {}).get("last_critique_payload")


def reset_awaiting_save(thread_id: str) -> None:
    """Reset awaiting_save flag so auto-save does not trigger after a HITL rejection."""
    if thread_id in _RUN_LIMITS:
        _RUN_LIMITS[thread_id]["awaiting_save"] = False


def _build_research_followup_from_critique(
    payload: dict | None,
    last_findings: str | None = None,
    original_request: str | None = None,
) -> str:
    prefix = f"Original user request:\n{original_request}\n\n" if original_request else ""

    if not isinstance(payload, dict):
        base = (
            "Revise the previous research using the critic's feedback. "
            "Strengthen the answer, verify the sources, and improve the structure."
        )
        if last_findings:
            preview = last_findings[:1500] + ("..." if len(last_findings) > 1500 else "")
            return f"{prefix}{base}\n\nContext from previous findings (improve upon these):\n{preview}"
        return f"{prefix}{base}"

    revision_requests = [
        str(item).strip() for item in (payload.get("revision_requests") or []) if str(item).strip()
    ]
    gaps = [str(item).strip() for item in (payload.get("gaps") or []) if str(item).strip()]

    lines = [f"{prefix}Revise the previous research using the critic's feedback."]
    if revision_requests:
        lines.append("Address these revision requests:")
        lines.extend(f"- {item}" for item in revision_requests[:8])
    elif gaps:
        lines.append("Fix these gaps:")
        lines.extend(f"- {item}" for item in gaps[:8])

    if last_findings:
        preview = last_findings[:1500] + ("..." if len(last_findings) > 1500 else "")
        lines.append("\nContext from previous findings (improve upon these):")
        lines.append(preview)

    lines.append("Return updated findings with verified sources and a clearer structure.")
    return "\n".join(lines)


class RevisionLimitMiddleware(AgentMiddleware):
    """Hard-stop repeated revise loops after the configured maximum and force a save step when needed."""

    def wrap_model_call(self, request, handler):
        thread_id = _get_thread_id(request)
        counters = _RUN_LIMITS[thread_id]

        # Inject research BEFORE calling the LLM to avoid wasted token spend
        if counters.get("force_research"):
            followup_plan = _build_research_followup_from_critique(
                counters.get("last_critique_payload"),
                counters.get("last_findings"),
                counters.get("original_request"),
            )
            counters["force_research"] = False
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "research",
                        "args": {"plan": followup_plan},
                        "id": f"call_auto_research_{uuid4().hex[:8]}",
                        "type": "tool_call",
                    }
                ],
            )

        # Fix A (Critical): when revision limit reached, inject save_report BEFORE the LLM call.
        # This prevents the LLM from seeing the final REVISE signal and outputting plain text
        # instead of calling save_report. Mirrors the same pre-call injection pattern as force_research.
        if counters.get("limit_reached") and counters.get("awaiting_save"):
            last_findings = (
                str(counters.get("last_findings") or "").strip()
                or "No research findings were collected before the revision limit was reached."
            )
            content = _prepend_best_effort_disclaimer(last_findings)
            filename = _suggest_report_filename(content, default_stem="best_effort_report")
            counters["awaiting_save"] = False
            return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "save_report",
                            "args": {"filename": filename, "content": content},
                            "id": f"call_auto_save_{uuid4().hex[:8]}",
                            "type": "tool_call",
                        }
                    ],
                )

        response = handler(request)

        if not counters.get("awaiting_save"):
            return response

        # Fix B (Backup for APPROVE-flow): robust extraction of the last AIMessage regardless
        # of the wrapper type returned by handler() in langchain >= 1.2.0.
        last_message = None
        if isinstance(response, AIMessage):
            last_message = response
        elif hasattr(response, "message") and isinstance(response.message, AIMessage):
            last_message = response.message
        else:
            candidates = list(getattr(response, "result", None) or getattr(response, "messages", None) or [])
            for msg in reversed(candidates):
                if isinstance(msg, AIMessage):
                    last_message = msg
                    break

        if last_message is None:
            return response

        tool_calls = getattr(last_message, "tool_calls", []) or []
        if tool_calls:
            return response

        content = _tool_content_to_text(getattr(last_message, "content", "")).strip()
        if not content or len(content) < 120:
            return response

        lowered = content.lower()
        if any(marker in lowered for marker in ["report saved to", "reviewer requested changes", "user rejected"]):
            counters["awaiting_save"] = False
            return response

        filename = _suggest_report_filename(
            content,
            default_stem="best_effort_report" if counters.get("limit_reached") else "report",
        )
        counters["awaiting_save"] = False
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "save_report",
                    "args": {"filename": filename, "content": content},
                    "id": f"call_auto_save_{uuid4().hex[:8]}",
                    "type": "tool_call",
                }
            ],
        )

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        tool_name = request.tool_call.get("name") or getattr(request.tool, "name", "")
        tool_call_id = request.tool_call.get("id") or getattr(
            request.runtime, "tool_call_id", "tool-call"
        )
        thread_id = _get_thread_id(request)

        if tool_name == "plan":
            _RUN_LIMITS[thread_id] = _make_counters(
                original_request=str((request.tool_call.get("args") or {}).get("request", "") or ""),
            )
            return handler(request)

        if tool_name == "research":
            counters = _RUN_LIMITS[thread_id]
            if (
                int(counters["research_calls"]) > 0
                and int(counters["revise_cycles"]) >= settings.critique_max_rounds
            ):
                counters["limit_reached"] = True
                return ToolMessage(
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    status="error",
                    content=(
                        "⛔ Hard revision limit reached. "
                        f"The Critic has already requested {counters['revise_cycles']} revise cycle(s), "
                        f"which matches the configured maximum of {settings.critique_max_rounds}. "
                        "Do not call `research` again. Immediately compose a best-effort draft report from the "
                        "evidence already collected, add a first-line warning that the draft was saved after "
                        "reaching the revision limit, and call `save_report` now."
                    ),
                )
            counters["research_calls"] = int(counters["research_calls"]) + 1
            # Fix 1: store findings so revision rounds can build upon them
            result = handler(request)
            findings_text = _tool_content_to_text(getattr(result, "content", ""))
            if findings_text.strip():
                counters["last_findings"] = findings_text
            return result

        if tool_name == "save_report":
            counters = _RUN_LIMITS[thread_id]
            counters["awaiting_save"] = True
            if counters.get("limit_reached"):
                args = request.tool_call.setdefault("args", {})
                content = str(args.get("content", ""))
                args["content"] = _prepend_best_effort_disclaimer(content)
                if not args.get("filename"):
                    args["filename"] = "best_effort_report.md"
            result = handler(request)
            # Fix 2: reset awaiting_save when HITL rejects so explanation text isn't mis-saved
            result_text = _tool_content_to_text(getattr(result, "content", "")).lower()
            if any(m in result_text for m in ["user rejected", "reviewer requested changes"]):
                counters["awaiting_save"] = False
            return result

        result = handler(request)

        if tool_name == "critique":
            counters = _RUN_LIMITS[thread_id]
            text = _tool_content_to_text(getattr(result, "content", ""))
            verdict = None
            payload = None

            try:
                payload = json.loads(text)
                verdict = payload.get("verdict")
            except Exception:
                if "REVISE" in text:
                    verdict = "REVISE"
                elif "APPROVE" in text:
                    verdict = "APPROVE"

            counters["last_critique_payload"] = payload

            if verdict == "REVISE":
                counters["awaiting_save"] = False
                counters["revise_cycles"] = int(counters["revise_cycles"]) + 1
                counters["force_research"] = False
                if int(counters["revise_cycles"]) >= settings.critique_max_rounds:
                    counters["limit_reached"] = True
                    counters["awaiting_save"] = True
                else:
                    counters["limit_reached"] = False
                    counters["force_research"] = True
            elif verdict == "APPROVE":
                counters["limit_reached"] = False
                counters["awaiting_save"] = True
                counters["force_research"] = False

        return result


@lru_cache(maxsize=1)
def build_supervisor():
    return create_agent(
        model=build_chat_model(temperature=0.0, model=settings.supervisor_model),
        tools=[plan, research, critique, save_report],
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        middleware=[
            RevisionLimitMiddleware(),
            HumanInTheLoopMiddleware(
                interrupt_on={"save_report": True},
            ),
        ],
        checkpointer=InMemorySaver(),
    )


supervisor = build_supervisor()
