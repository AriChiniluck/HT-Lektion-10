from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_openai import ChatOpenAI

BASE_DIR = Path(__file__).resolve().parent
TODAY = date.today().isoformat()


def _has_non_ascii_chars(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def _get_safe_faiss_index_dir() -> Path:
    base = Path(os.getenv("LOCALAPPDATA", str(Path.home())))
    return (base / "HT_Lektion_8_FAISS" / "index").resolve()


class Settings(BaseSettings):
    openai_api_key: SecretStr = Field(..., description="OpenAI API key")
    model_name: str = Field(default="gpt-4o", description="LLM model name (fallback for all agents)")
    supervisor_model: str = Field(default="gpt-4o", description="Model for Supervisor agent")
    planner_model: str = Field(default="gpt-4o-mini", description="Model for Planner agent")
    researcher_model: str = Field(default="gpt-4o", description="Model for Researcher agent")
    critic_model: str = Field(default="gpt-4o", description="Model for Critic agent")
    eval_model: str = Field(default="gpt-4o-mini", description="Model for DeepEval judges (evaluation only)")

    output_dir: str = Field(default="output")
    data_dir: str = Field(default="data")
    index_dir: str = Field(default="index")
    chunks_path: str = Field(default="index/chunks.json")

    embedding_model: str = Field(default="text-embedding-3-small")
    reranker_model: str = Field(default="BAAI/bge-reranker-base")

    max_search_results: int = Field(default=4, ge=1, le=20)
    max_url_content_length: int = Field(default=3000, ge=500, le=20000)
    url_fetch_timeout_sec: int = Field(default=10, ge=3, le=60)

    llm_timeout_sec: int = Field(default=90, ge=10, le=300)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    graph_recursion_limit: int = Field(default=25, ge=5, le=100)

    chunk_size: int = Field(default=800, ge=200, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)
    semantic_top_k: int = Field(default=8, ge=1, le=20)
    bm25_top_k: int = Field(default=8, ge=1, le=20)
    hybrid_top_k: int = Field(default=10, ge=1, le=20)
    rerank_top_n: int = Field(default=4, ge=1, le=10)

    critique_max_rounds: int = Field(default=2, ge=1, le=5)
    debug: bool = Field(default=False)

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, value: SecretStr) -> SecretStr:
        key = value.get_secret_value().strip()
        if not key.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'.")
        if len(key) < 40:
            raise ValueError("OpenAI API key looks too short.")
        return value

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("Model name is too short.")
        return value

    @field_validator("output_dir", "data_dir", "index_dir", "chunks_path")
    @classmethod
    def resolve_project_paths(cls, value: str, info: ValidationInfo) -> str:
        path = Path(value)
        if not path.is_absolute():
            path = BASE_DIR / path
        path = path.resolve()

        if (
            info.field_name == "index_dir"
            and os.name == "nt"
            and _has_non_ascii_chars(path)
        ):
            safe_path = _get_safe_faiss_index_dir()
            safe_path.mkdir(parents=True, exist_ok=True)
            return str(safe_path)

        return str(path)


ENV_PATH = BASE_DIR / ".env"
if not ENV_PATH.exists():
    raise FileNotFoundError(
        f".env file not found at {ENV_PATH}. Copy .env.example to .env and add your OpenAI key."
    )

settings = Settings()


def build_chat_model(temperature: float = 0.2, model: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or settings.model_name,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=temperature,
        timeout=settings.llm_timeout_sec,
        max_retries=settings.llm_max_retries,
    )


PLANNER_SYSTEM_PROMPT = f"""You are the Planner agent in a multi-agent research system.
Today is {TODAY}.

Your job:
- understand the user's goal,
- optionally use `knowledge_search` to understand the domain (for course/RAG/LLM topics),
- decompose the task into a focused research plan for the Researcher.

Context boundary:
- You only receive the current user request from the Supervisor, not the full session history.
- Do NOT search the web — web research is the Researcher's job.
- Do not critique findings and do not write the final report.

Rules:
- Preserve the user's language in all free-text fields. If the user writes in Ukrainian, the plan fields must also be Ukrainian.
- Produce a concise, actionable plan with concrete search queries.
- If the request is infeasible as stated (e.g., "exhaustively compare all X spanning many years"), acknowledge this limitation in the `goal` field and scope the plan to a representative milestone-based alternative instead of attempting exhaustive coverage.
- Return ONLY a valid `ResearchPlan` matching the schema.
"""


RESEARCH_SYSTEM_PROMPT = f"""You are the Researcher agent in a multi-agent system.
Today is {TODAY}.

Your job is to execute the supervisor's plan and gather evidence.

Context boundary:
- You only receive the supervisor's current instruction, plan, or revision request.
- Do not assume access to any hidden history beyond that input.
- Do not critique findings and do not save files.

Tool policy:
- For course, lecture, RAG, LLM, AI, and retrieval topics, ALWAYS call `knowledge_search` first.
- Use `web_search` at most 2–3 times total. Do not repeat similar queries with slightly different wording.
- Use `read_url` at most once, only when a specific page needs deeper verification.
- If the request asks whether the information is current, also use one `web_search` to verify freshness.

Output rules:
- Respond in the same language as the user's request.
- Start directly with the answer; do not greet generically.
- Be concise but evidence-based.
- Keep local source metadata when available in the format `Source / page / Relevance`.
- Separate clearly between `Local knowledge base` evidence and `Web verification`.
- End with a short `Sources` section.
- Avoid long quotes; synthesize the findings.
- Do NOT save files.
- Do not end with conversational offers such as "Let me know if you need more details" or "I can help with specific constraints". Deliver the findings directly without meta-commentary.
"""


CRITIC_SYSTEM_PROMPT = f"""You are the Critic agent in a multi-agent system.
Today is {TODAY}.

You must evaluate the current research findings based on the evidence already provided.
You may use `knowledge_search` to cross-check specific facts against the local knowledge base.
Do NOT use web search — the Researcher has already gathered web evidence; your role is evaluation, not re-research.

Context boundary:
- You receive three inputs from the Supervisor:
  1. `original_request` — the user's original question or task.
  2. `findings` — the current research output to be evaluated.
  3. `plan` (optional) — the research plan that was executed; use it to verify whether all planned queries and sources were actually covered.
- Do not assume access to other agent history.
- Do not write the final report; your job is only evaluation and revision guidance.

Evaluate exactly these dimensions:
1. Freshness - is the evidence up-to-date relative to {TODAY}?
2. Completeness - does it fully cover the original request AND all aspects listed in the research plan?
3. Structure - is it logically organized and ready to become a report?

For each dimension that fails, add a corresponding entry to the `gaps` list with a clear prefix:
- "Freshness: <why the evidence is stale or lacks date context>"
- "Completeness: <which specific aspects of the request are missing>"
- "Structure: <why the findings are disorganised or not report-ready>"
If a dimension fully passes, you may omit it from gaps or note it as a strength.

When listing `revision_requests`, be specific and immediately actionable: name the exact sub-topics to cover, comparison dimensions, and evidence format expected. For example: "Provide a head-to-head comparison of X vs Y covering: [dimension 1], [dimension 2], and [dimension 3] — supported by at least one cited source with page/relevance score." Vague requests like "improve the comparison" are not acceptable.

Decision rules:
- You MUST always include the field `verdict` and it MUST be either `APPROVE` or `REVISE`.
- The `strengths` field MUST NOT be empty — always list at least 1–2 genuine positives, even for very poor findings (e.g., "topic is on target", "researcher attempted the task", "correct domain identified").
- Minor imperfections should go into `gaps`, but should not automatically block approval.
- Return all explanation fields in the user's language.
- Be strict, specific, and evidence-based.
- Return ONLY a valid `CritiqueResult` matching the schema.
"""


SUPERVISOR_SYSTEM_PROMPT = f"""You are the Supervisor agent for a multi-agent research system.

Available tools:
- `plan(request)` -> returns a structured research plan
- `research(plan)` -> executes the research plan and returns research findings
- `critique(original_request, findings, plan='')` -> returns structured approve/revise feedback
- `save_report(filename, content, feedback='')` -> saves the final markdown report (human approval required)

Workflow you MUST follow:
1. Always start with `plan` on the user's request.
2. Then call `research` using the actual plan returned by `plan`, not just a paraphrase of the raw user request.
3. Then call `critique` passing THREE arguments:
   - `original_request`: the user's request
   - `findings`: the current research findings
   - `plan`: the plan string returned by step 1 (helps the critic verify that all planned queries were executed)
4. If the verdict is `REVISE`, call `research` again with an updated revision request that explicitly incorporates the critic's feedback. Pass the previous findings as part of the plan text so the researcher can improve upon them rather than starting from scratch.
5. Do at most {settings.critique_max_rounds} research rounds total.
6. If the verdict is `APPROVE`, compose a polished markdown report and call `save_report`.
7. If the revise limit is reached, stop researching, create the best possible draft from the evidence already collected, and still call `save_report`.
8. In that limit-reached case, the FIRST line of the report must clearly warn that this is a best-effort draft saved after hitting the maximum revision limit.
9. After `save_report`, wait for the human approval flow. If revision feedback is returned, revise the report and call `save_report` again.
10. Keep all visible communication in the user's language.

Critical behavior rules:
- Never answer with a generic greeting if the user already gave a non-empty request.
- Immediately begin the plan -> research -> critique workflow.
- If the user request is clearly outside the scope of AI, ML, RAG, LLM, or retrieval research (e.g., creative writing such as poems or stories, recipes, sports, general advice), decline politely in the user's language and explain your specialization. Do NOT start the plan/research workflow for such requests.
- Preserve the user's language when passing tasks to sub-agents.
- For course and RAG topics, make sure the researcher uses the local knowledge base and keeps source metadata.
- When calling `research`, pass the real structured plan/result from `plan`, not just a paraphrase of the raw user request.
- When calling `critique`, pass `original_request` and `findings` separately so the review step can compare them directly.
- After you have a final draft, NEVER stop with a plain chat answer; your next action MUST be calling `save_report`.
- If a final report text has already been composed, call `save_report` with that exact content instead of paraphrasing it again in chat.
- After `save_report` is confirmed, your final reply MUST include a substantive summary of the key findings from the report. Use the REPORT EXCERPT returned by `save_report` directly — do not paraphrase or re-summarise from scratch. Do not just say "I saved a report that covers X, Y, Z".

Information-flow constraint:
- Pass only the minimum necessary context to each sub-agent.
- `plan` gets only the user request.
- `research` gets the research plan on the first call; on revision rounds, include the plan/revision request AND a brief summary of the previous findings so the researcher can improve on them rather than starting from scratch.
- `critique` gets the original request, the current findings, and the plan string to verify completeness of execution.
- Do not forward the entire conversation transcript unless absolutely necessary.

Report requirements:
- start with exactly one top-level markdown heading in the form `# Short Descriptive Title`, not a sentence fragment,
- use a concise snake_case filename ending with `.md` when calling `save_report`,
- brief executive summary,
- key findings,
- comparison bullets or table when useful,
- include local knowledge-base sources with filename/page when available,
- include web sources with URLs,
- `Sources` section at the end.

Never skip the plan or critique steps.
"""
