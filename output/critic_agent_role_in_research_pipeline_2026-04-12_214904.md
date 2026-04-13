# Role of the Critic Agent in a Research Pipeline

## Executive summary

In a multi‑agent research pipeline, the Critic agent is the quality‑control and governance component. It does not perform the primary research itself; instead, it evaluates the work produced by other agents (such as a Planner and Researcher/Solver), checks it against the original request and evidence, and decides whether the result is ready to deliver or needs revision. This role underpins reliability, faithfulness to sources, and adherence to user and system constraints.

## Key role and responsibilities

In a planner–researcher–critic (or planner–solver–critic) pipeline, the Critic sits after the "doing" stage. A typical flow is:

1. **Planner**: interprets the user’s request and creates a structured plan (subtasks, tools, sources).
2. **Researcher/Solver**: executes the plan, queries tools or the web, synthesizes evidence, and drafts an answer or intermediate artifact.
3. **Critic**: inspects the draft, the plan, and the original instructions and evaluates whether the output is acceptable.

The Critic’s core responsibilities usually cover several dimensions at once:

- **Alignment with the original request**: Does the answer actually address the user’s question, at the requested depth, scope, language, and format? Are all sub‑questions covered?
- **Factual consistency and grounding**: Are key claims supported by the cited sources or underlying data? Are there hallucinations, contradictions, or outdated facts relative to the evidence gathered?
- **Reasoning quality**: Do conclusions follow logically from the evidence? Are there gaps, non sequiturs, or unjustified leaps in reasoning?
- **Adherence to the plan and constraints**: Were all planned subtasks executed? Were instructions respected (e.g., required tools, forbidden data sources, length, tone, audience)?
- **Clarity and structure**: Is the answer organized, readable, and appropriately framed for the target audience, with clear sections or steps where needed?
- **Safety and compliance (when applicable)**: Does the content comply with safety policies, domain regulations, or organizational guidelines (e.g., avoiding disallowed topics, protecting privacy, or flagging high‑risk recommendations)?

Importantly, the Critic generally **does not rewrite the entire answer from scratch**. Its role is diagnostic and evaluative: to surface problems and guide improvements.

## Position in the workflow and interaction with other agents

The Critic is invoked **after** a draft answer or intermediate artifact has been produced, and often participates in one or more **revision loops**:

1. Planner defines the research strategy.
2. Researcher/Solver follows the plan and drafts a response.
3. Critic evaluates the draft against explicit criteria (coverage, correctness, reasoning, style, safety, etc.).
4. If issues are found, the Critic returns structured feedback (often with labels like "approve", "needs revision", or error categories).
5. The Researcher/Solver (sometimes with an updated plan from the Planner) revises the work in line with the Critic’s comments.
6. The Critic re‑evaluates the revised draft and either approves it for delivery or requests further changes.

In many systems, the Critic may also assign **scores or tags** (e.g., numerical quality scores, "blocking" vs. "non‑blocking" issues). An orchestrator can then use these signals to decide when to stop iterating or when to escalate (for instance, to a human reviewer).

Although often discussed in the context of text answers, the same pattern applies to other artifact types in research‑like workflows: code, experiment plans, data analyses, or reports can all be passed through a Critic that checks correctness, robustness, and compliance with requirements before they are adopted.

## Best practices for designing a Critic agent

Effective Critic agents rely heavily on **clear prompts and structured outputs**. Common best practices include:

- **Explicit criteria**: Clearly list what the Critic should check—task alignment, factual grounding, reasoning soundness, plan coverage, formatting/audience fit, and safety/compliance.
- **Structured feedback format**: Ask the Critic to produce an overall **verdict** (e.g., APPROVE / REVISE), followed by per‑criterion comments and concrete revision suggestions.
- **Separation of blocking vs. non‑blocking issues**: Instruct the Critic to distinguish between issues that must be fixed before release (e.g., factual errors, missing sub‑questions) and those that are nice‑to‑have (e.g., style improvements, extra examples).
- **Diagnostic, not generative**: Emphasize that the Critic should focus on analysis and guidance rather than fully rewriting the answer, so that the Researcher/Solver remains responsible for content creation.

These practices make the Critic’s output predictable and machine‑actionable, which in turn makes it easier to orchestrate automated revision cycles or escalate to a human reviewer when necessary.

## Example of Critic feedback

Below is a simplified example of how a Critic might respond to a draft answer about its own role:

- **Verdict**: Needs revision.
- **Alignment with request**: The answer explains what a "Critic agent" is but does not clearly describe where it appears in the pipeline. Add 1–2 sentences summarizing when the Critic is invoked relative to the Planner and Researcher/Solver.
- **Factual consistency**: No obvious contradictions, but you refer to "multi‑agent systems" in general terms. Briefly ground the description in at least one common pattern (e.g., planner–solver–critic workflows used in multi‑agent LLM systems).
- **Reasoning quality**: The explanation of "governance" is vague. Clarify that the Critic evaluates outputs against predefined criteria (task coverage, correctness, safety) rather than inventing new tasks.
- **Clarity and structure**: Consider using bullets to list the Critic’s main responsibilities so they are easier to scan.

This style of structured, actionable feedback illustrates how the Critic improves the reliability and usability of research outputs without taking over the primary research role.

## Sources

- Local knowledge base: large-language-model.pdf, pp. 7–8 (discussion of multi‑agent LLM systems, planning, and evaluation roles).
- SAGE: Multi-Agent Self-Evolution for LLM Reasoning (arXiv), describing a Planner–Solver–Critic loop where the Critic evaluates Solver solutions and guides further evolution.
- "Multi-Agent Workflows for Vertical AI Applications" (industry article, e.g., ombrulla.com), outlining planner–solver–critic and reviewer agents as quality‑control components in applied AI pipelines.