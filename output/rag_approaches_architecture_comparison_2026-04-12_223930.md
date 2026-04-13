# Architectural Comparison of Naive, Advanced, and Agentic RAG

> **⚠️ Best-effort draft:** this report was saved after reaching the maximum number of revise cycles and may still contain unresolved gaps noted by the Critic.

# Architectural Comparison of Naive, Advanced, and Agentic RAG

## Executive summary

This report synthesizes the course materials on Retrieval‑Augmented Generation (RAG) to compare three major approaches:

- **Naive RAG** – a simple, single‑step pipeline: retrieval + one LLM call.
- **Advanced RAG** – a stronger but still single‑step pipeline with optimized retrieval (hybrid search, re‑ranking, better chunking, evaluation).
- **Agentic RAG** – an agent architecture where RAG is exposed as a tool inside a multi‑step planning loop with other tools and memory.

Architecturally, they form an evolution chain:

- naive RAG is the minimal pattern for “LLM + external documents”;
- advanced RAG refines the retrieval layer while keeping the overall one‑shot structure;
- agentic RAG wraps naive/advanced RAG in an agent loop that can plan, call tools repeatedly, and reason over multiple steps.

The choice between them should be driven by task complexity, data scale/heterogeneity, infrastructure budget, and latency/quality trade‑offs. For small FAQ‑style systems naive RAG is typically sufficient; for large heterogeneous corpora advanced RAG becomes necessary; for complex analytic workflows and tool‑rich environments, agentic RAG is justified.

---

## 1. What RAG is and why it is used

RAG (Retrieval‑Augmented Generation) combines a **retrieval system** with a **language model**:

1. Documents are indexed (often via embeddings in a vector store).
2. A user’s question is converted into a query and used to retrieve relevant documents.
3. Retrieved snippets are injected into the LLM prompt as context.
4. The LLM generates an answer conditioned on the retrieved context.

This solves key limitations of “pure” LLMs:

- **Freshness & domain specificity** – use private, up‑to‑date corpora without full model fine‑tuning.
- **Reduced hallucinations** – ground the answer in concrete sources.
- **Reuse of general models** – one base LLM can be adapted to multiple domains via different RAG layers.

**Course support**

- Basic definition and high‑level RAG process (indexing with embeddings, vector DB, retrieval, and LLM answer).  
  Source: `retrieval-augmented-generation.pdf / p.0–1`  
- RAG as a way to integrate external knowledge into LLM context.  
  Source: `large-language-model.pdf / p.7`

---

## 2. Naive RAG

### 2.1. Architecture

Naive RAG implements the canonical “embeddings + vector DB + LLM” pattern in a **single forward pass**:

1. **Offline indexing**
   - Input documents are split into fixed‑size chunks (with possible overlap).
   - Each chunk is embedded with an embedding model.
   - (id, chunk text, embedding, metadata) tuples are stored in a vector database.

2. **Online query handling**
   - User question → query embedding (same embedding model).
   - k‑nearest neighbour search in the vector store (cosine or dot‑product similarity).
   - Top‑k chunks are concatenated as a flat “context” section.
   - Prompt = system instructions + user question + context.
   - Single LLM call generates the final answer.

This is the high‑level “overview of RAG process” diagram from the course.  
Source: `retrieval-augmented-generation.pdf / p.1`

### 2.2. Components

- **Chunker** – naive text splitter (by characters/tokens, sometimes with fixed overlap).
- **Embedding model** – produces dense vector for each chunk and for each query.
- **Vector store** – FAISS/Pinecone/Chroma‑like storage with similarity search.
- **Retriever** – thin wrapper over vector search (top‑k by similarity).
- **LLM** – a single generative model (chat/completion) used once per query.

### 2.3. Request processing pipeline

1. Receive user question.
2. Compute query embedding.
3. Run vector similarity search to get top‑k chunks.
4. Form a prompt with instructions, context, and question.
5. Call LLM once to generate the answer.
6. Return answer (optionally with source snippets/links).

### 2.4. Strengths

- **Simplicity** – very few moving parts, easy to implement and reason about.
- **Low latency** – one vector DB call + one LLM call.
- **Low cost** – minimal infrastructure and token usage beyond basic retrieval.
- **Good baseline** – often a big improvement over raw LLM on private corpora.

### 2.5. Weaknesses

- **Sensitive to chunking and embeddings** – bad chunks or weak embeddings degrade retrieval sharply.
- **No query rewriting** – struggles with very long, ambiguous, or multi‑topic questions.
- **No re‑ranking** – top‑k by raw similarity often includes noisy or tangential chunks.
- **Flat context** – LLM sees a flat text blob, not a structured representation of the corpus.
- **Limited scalability** – for large heterogeneous corpora, simple top‑k becomes a bottleneck.

### 2.6. Typical mistakes

- Oversized chunks causing context truncation; tiny chunks losing cross‑paragraph semantics.
- Only dense similarity without metadata filters (e.g., by document type, timestamp).
- Weak or missing system instructions, causing the LLM to hallucinate from irrelevant context.

### 2.7. Typical use cases

- FAQ‑style Q&A over small corpora (internal docs, product manuals).
- Simple “chat with documentation” where questions are fairly direct.
- MVPs to validate whether RAG on a given corpus is useful at all.

**Sources**

- Naive RAG definition and pipeline: `retrieval-augmented-generation.pdf / p.0–1`
- RAG as integration of external knowledge into chat context: `large-language-model.pdf / p.7`

---

## 3. Advanced RAG

### 3.1. Architectural idea

Advanced RAG keeps the **single LLM call per user query**, but makes the retrieval pipeline much more powerful and controllable:

- **Query pipeline** – normalization, query rewriting/expansion, sometimes query classification/routing.
- **Advanced retrieval** – hybrid search (dense + sparse) plus re‑ranking of candidates.
- **Post‑retrieval processing** – merging and pruning of passages, optional summarization.
- **Generation** – LLM answers based on a higher‑quality, more focused context.

The core change vs naive RAG: the retrieval layer becomes a **multi‑step IR pipeline** instead of “one similarity lookup”.

**Course support**

- Hybrid retrieval: combining classic (sparse) and vector search for better recall.  
  Source: `retrieval-augmented-generation.pdf / p.3`
- Re‑ranking techniques to improve retriever performance.  
  Source: `retrieval-augmented-generation.pdf / p.2`
- Retrieval optimization: similarity metrics (dot‑product, ANN) and Late Interactions.  
  Source: `retrieval-augmented-generation.pdf / p.2`
- LangChain as SDK for building RAG pipelines.  
  Source: `langchain.pdf / p.0–3`

### 3.2. Components

1. **Advanced retriever**
   - Dense index (vector store) + sparse index (e.g., BM25).
   - Hybrid merging strategy for candidate lists.
   - Re‑ranking model (cross‑encoder or LLM‑based scorer).

2. **Query processing module**
   - Query normalization.
   - Query rewriting/expansion (LLM or rule‑based).
   - Optional query classification to route to specific indexes.

3. **Chunking & indexing**
   - Format‑aware chunking (e.g., respecting function boundaries in code, DOM structure in HTML).  
     Source: `retrieval-augmented-generation.pdf / p.3`

4. **Generator (LLM)**
   - Same or similar base LLM as naive RAG, but with stronger instructions (e.g., cite sources, do not speculate beyond context).

5. **Evaluation and monitoring**
   - Metrics for retrieval quality and end‑to‑end RAG performance.
   - Use of RAG benchmarks such as BEIR, Natural Questions.  
     Source: `retrieval-augmented-generation.pdf / p.3`

6. **Framework integration**
   - LangChain “chains” to link retriever, re‑ranker, and LLM in a pipeline.  
     Source: `langchain.pdf / p.0–3`

### 3.3. Request processing pipeline

1. **User query → query preprocessing**
   - Normalize; optionally rewrite using an LLM prompt.
   - Optionally classify (e.g., “code question” vs “policy question”).

2. **Hybrid retrieval**
   - BM25 (or other sparse search) over text index.
   - Dense vector search over embedding index.
   - Merge candidate lists.

3. **Re‑ranking**
   - Run re‑ranking model over top‑N candidates.
   - Select final top‑k passages.

4. **Context post‑processing**
   - Merge neighbouring passages.
   - Remove redundant/noisy chunks.

5. **Generation**
   - LLM gets instructions + refined context + user question.
   - Produces answer (often with citations or structured format).

6. **Evaluation/logging**
   - Log retriever and generator outputs for offline analysis.

### 3.4. Strengths

- **Higher relevance and faithfulness** thanks to hybrid retrieval and re‑ranking.  
  Source: `retrieval-augmented-generation.pdf / p.2–3`
- **Better recall + precision** combination: more likely to retrieve all relevant docs and to surface the best ones.
- **Tunability** – can improve retrieval independently of the LLM (IR‑style tuning, benchmarks, ablations).
- **Good ecosystem support** – frameworks like LangChain provide ready‑made patterns for such pipelines.  
  Source: `langchain.pdf / p.0–3`

### 3.5. Weaknesses

- **Higher infrastructure complexity** – multiple indices, extra models for re‑ranking, more services to maintain.
- **Higher latency and cost** – additional retrieval steps and model calls.
- **Harder debugging** – errors can stem from the query rewriter, base retriever, re‑ranker, or LLM.
- **Still single‑step generation** – no dynamic planning or multi‑turn reasoning over tools.

### 3.6. Typical mistakes and anti‑patterns

1. **Over‑reliance on re‑ranking**
   - Treating re‑ranker as a fix for a weak base retriever.
   - In reality, re‑ranker only rescored what was retrieved; if relevant docs never reach top‑N, re‑ranker can’t help.  
   Source: `retrieval-augmented-generation.pdf / p.2`

2. **Poor chunking**
   - Ignoring file/semantic structure and splitting at fixed lengths only.
   - Leads to incoherent passages and poor retrieval.  
   Source: `retrieval-augmented-generation.pdf / p.3`

3. **No retrieval evaluation**
   - Focusing only on LLM answer quality without measuring retrievability/retrieval accuracy using benchmarks like BEIR/NQ.  
   Source: `retrieval-augmented-generation.pdf / p.3`

4. **Unconstrained query rewriting**
   - LLM rewrites may drift semantically, especially in specialized domains.
   - Mitigation: controlled prompts, logging before/after, domain vocabulary constraints.

5. **Stacking optimizations without measurement**
   - Introducing hybrid search, re‑ranking, and complex chunking all at once without ablation studies makes it hard to attribute gains and debug issues.

### 3.7. Typical use cases

- Large, heterogeneous document collections where naive RAG’s simple top‑k retrieval starts to fail.
- Compliance/knowledge‑intensive domains where false negatives and hallucinations are costly.
- Systems that need measurable, tunable retrieval quality (benchmarked RAG).

**Sources**

- Hybrid retrieval, file‑format‑based chunking, RAG evaluation: `retrieval-augmented-generation.pdf / p.3`
- Similarity optimization, Late Interactions, re‑ranking: `retrieval-augmented-generation.pdf / p.2`
- LangChain for RAG pipelines: `langchain.pdf / p.0–3`

---

## 4. Agentic RAG

### 4.1. Intuition and course connection

Agentic RAG is not a new retrieval algorithm but an **agent architecture** where:

- an LLM functions as an **agent/planner**;
- RAG (naive or advanced) is exposed as a **tool** among others;
- the system operates in a **multi‑step loop**: plan → act (call tools) → observe → refine.

Course materials describe several agentic/interactive methods:

- **DEPS** – “Describe, Explain, Plan and Select”: LLM plans actions for complex tasks.  
  Source: `large-language-model.pdf / p.7`
- **Reflexion** – agent improves via reflection over previous episodes.  
  Source: `large-language-model.pdf / p.7`
- **Generative Agents** – multi‑step agents with long‑term memory.  
  Source: `large-language-model.pdf / p.28`
- **LangChain agents & tools** – framework to build agents that select tools (including RAG retrievers) and act in loops.  
  Source: `langchain.pdf / p.0–3`

Agentic RAG is a concrete instantiation: an agent uses a RAG retriever tool (internally naive or advanced RAG) alongside other tools.

### 4.2. Architecture

Key components:

1. **Agent (LLM‑policy)**
   - Core LLM that:
     - reads current state (dialogue history, tool outputs, partial conclusions),
     - decides next action: ask user, call RAG, call another tool, or finalize answer.

2. **Planner / reasoning module**
   - Often realized via a prompt pattern enforcing a loop:
     - *Plan → Act → Observe → Reflect* (aligned with DEPS/Reflexion ideas).  
       Source: `large-language-model.pdf / p.7`

3. **Tools (functions)**
   - **RAG retriever tool**:
     - under the hood: naive or advanced RAG pipeline (sections 2–3),
     - interface: e.g., `search_knowledge_base(query) -> documents/snippets`.
   - Other tools:
     - web search,
     - code execution/“Python REPL”,
     - SQL/analytics queries,
     - domain‑specific APIs.

4. **Memory**
   - **Short‑term memory** – current conversation, recent tool outputs and decisions.
   - **Long‑term memory** – sometimes implemented as a separate retriever storing agent’s experiences, similar in spirit to Generative Agents.  
     Source: `large-language-model.pdf / p.28`

5. **Orchestrator / agent loop executor**
   - External runtime that repeatedly:
     - feeds the current state to the agent LLM,
     - executes chosen tools,
     - updates the state,
     - checks termination conditions (max steps, explicit “finish” decision).

6. **Framework support**
   - LangChain **Agents**:
     - accept tool descriptions (including RAG retrievers),
     - use an LLM to pick tools and arguments at each step,
     - implement the agent loop.  
     Source: `langchain.pdf / p.0–3`

### 4.3. Typical agentic RAG loop

1. **User task arrives**
   - Possibly complex: “Audit these policies”, “Compare several sources and propose a plan”, etc.

2. **Initial planning**
   - Agent LLM generates a brief internal plan: e.g.,
     - “Step 1: use RAG to gather docs about X. Step 2: analyze conflicts. Step 3: summarize recommendations.”

3. **Tool selection and first RAG call**
   - Agent chooses the RAG tool.
   - Forms a focused retrieval query (can be a rewritten/clarified version of the original task).
   - RAG tool returns relevant snippets.

4. **Observation and reasoning**
   - Agent reads snippets and decides:
     - whether to call RAG again with refined query,
     - whether to call other tools (e.g., run analysis code, query a database),
     - whether to ask user clarifying questions.

5. **Multi‑step iteration**
   - Loop of *Plan → Act → Observe* continues:
     - RAG → reasoning → more RAG or other tools → intermediate conclusions.

6. **Finalization**
   - Agent decides it has enough evidence.
   - Produces a final answer (report, plan, recommendations), often referencing multiple tool calls and sources.

The central architectural difference vs advanced RAG: **the retrieval pipeline is now a tool invoked multiple times within a dynamic control loop**, instead of being a fixed pre‑generation stage.

### 4.4. How agentic RAG uses naive/advanced RAG

- The **RAG tool** inside an agent can be:
  - a thin wrapper around **naive RAG** (simple vector search + LLM), or
  - a wrapper around **advanced RAG** (hybrid retrieval, re‑ranking, etc.).
- Architecturally: agentic RAG is a **superset/composition** pattern:
  - **Naive/advanced RAG** define *how* a single retrieval+generation step works.
  - **Agentic RAG** defines *when and how often* to call that step, and how to combine it with other tools and memory.

### 4.5. Strengths

- **Multi‑step reasoning and analysis**
  - Can decompose tasks, gather information iteratively, and refine hypotheses over several RAG calls.

- **Flexibility**
  - Dynamically chooses when to use RAG vs other tools vs user interaction.

- **Natural fit for real workflows**
  - Mirrors how humans investigate complex topics: search → read → compute → search again → decide.

- **Alignment with modern agent research**
  - DEPS, Reflexion, Generative Agents patterns are directly applicable.  
    Source: `large-language-model.pdf / p.7, p.28`

### 4.6. Weaknesses

- **Highest implementation complexity**
  - Requires agent runtime, tool design, error handling, monitoring.

- **Unstable/opaque behaviour if unconstrained**
  - Agents may loop, over‑query tools, or take non‑intuitive paths.

- **Higher latency and cost**
  - Multiple LLM and RAG calls per user task.

- **Harder debuggability**
  - Need full traces of agent steps to understand and tune behaviour.

### 4.7. Typical mistakes and anti‑patterns

1. **Treating RAG as the agent itself**
   - Building an increasingly complex single RAG pipeline and expecting agent‑like behaviour, instead of adding an explicit agent layer that orchestrates tool use.

2. **Underspecified tools**
   - Vague tool descriptions; agent cannot reliably choose when to use RAG vs other tools.

3. **No step limits or guards**
   - Agents performing unbounded numbers of RAG calls, causing runaway latency/cost.

4. **Missing loop instrumentation**
   - Without detailed logs of agent decisions and tool calls, debugging is nearly impossible.

5. **Overly generic planner prompts**
   - Planner not told when/how to rely on RAG, leading to erratic or suboptimal plans.

### 4.8. Typical use cases

- Complex research tasks requiring synthesis across many documents and data sources.
- Audits, diagnostics, multi‑stage decision support.
- Workflows that must combine RAG with other tools (code execution, SQL, external APIs).

**Sources**

- DEPS, Reflexion, multi‑step agent methods: `large-language-model.pdf / p.7`
- Generative Agents and memory: `large-language-model.pdf / p.28`
- LangChain agents & tools: `langchain.pdf / p.0–3`

---

## 5. Comparative analysis of naive, advanced, and agentic RAG

### 5.1. High‑level comparison table

| Dimension                     | Naive RAG                                  | Advanced RAG                                                                 | Agentic RAG                                                                                          |
|------------------------------|--------------------------------------------|------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| Core idea                    | Simple retrieval + single LLM call         | Better retrieval pipeline + single LLM call                                  | Agent loop that calls RAG (and other tools) over multiple steps                                     |
| Architecture                 | 1 embedding index + LLM                     | Hybrid/specialized indices, re‑rankers, evaluation pipeline                  | LLM agent + planner + tools (RAG + others) + memory + orchestrator                                  |
| Interaction pattern          | One‑shot: query → retrieve → answer        | One‑shot: query → advanced retrieve → answer                                 | Multi‑step: plan → tool calls (RAG, etc.) → observe → refine → answer                               |
| Retrieval                    | Single dense (or sparse) search            | Hybrid (dense + sparse), re‑ranking, optimized similarity                    | RAG tool internally using naive/advanced retrieval                                                  |
| Planning                     | None                                       | None (fixed pipeline)                                                        | Explicit (DEPS‑like planning, agent loop)                                                           |
| Memory                       | Prompt context only                         | Prompt context + logs/metrics                                                | Short‑term conversation state + possible long‑term episodic memory                                  |
| Infra complexity             | Low                                        | Medium                                                                       | High                                                                                                 |
| Latency                      | Low                                        | Medium (extra retrieval steps)                                               | High (multiple LLM + RAG/tool calls)                                                                |
| Cost per complex task        | Low                                        | Medium                                                                       | High                                                                                                 |
| Answer quality: simple Q&A   | Usually sufficient                         | High, but may be overkill                                                    | Can be high but often unnecessary                                                                   |
| Answer quality: hard tasks   | Limited (no planning, weak retrieval)      | Better retrieval but still one‑shot                                          | Best: can decompose tasks and iterate                                                               |
| Typical use cases            | FAQ, small doc chat                        | Large/heterogeneous corpora, compliance, high‑accuracy knowledge access      | Research, audits, diagnostics, complex workflows with tools                                         |
| Course support focus         | Intro RAG, basic pipeline                  | Retrieval optimization, hybrid, reranking, RAG evaluation, LangChain chains  | Agentic methods (DEPS, Reflexion, Generative Agents), LangChain agents & tools                      |

### 5.2. Trade‑offs

1. **Simplicity vs capability**
   - **Naive RAG** – minimal engineering, fast to build, good baseline; limited for complex queries and large corpora.
   - **Advanced RAG** – significantly better retrieval quality; moderate engineering overhead; still simpler than full agents.
   - **Agentic RAG** – maximally capable for complex problems, but most demanding in terms of infra, cost, and monitoring.

2. **Retrieval optimization vs reasoning optimization**
   - Advanced RAG focuses on **improving retrieval**: hybrid search, re‑ranking, similarity optimization.  
     Source: `retrieval-augmented-generation.pdf / p.2–3`
   - Agentic RAG focuses on **improving reasoning and control**: planning, multi‑step interaction, reflection.  
     Source: `large-language-model.pdf / p.7, p.28`

3. **Control and debuggability**
   - Naive RAG is easiest to debug: a single retrieval call and a single prompt.
   - Advanced RAG introduces more knobs, but they are standard IR components with clear metrics (recall, precision, NDCG).
   - Agentic RAG requires tooling for step‑level traces, tool call logs, and plan inspection.

4. **Scalability drivers**
   - If the bottleneck is **data scale/heterogeneity** (huge, diverse corpora) → invest in **advanced RAG** (retriever quality).
   - If the bottleneck is **task complexity/workflows** (multi‑stage decisions, cross‑tool usage) → invest in **agentic RAG**.

---

## 6. Practical recommendations

### 6.1. When to use naive RAG

Choose naive RAG if:

- You have a **small to medium, relatively homogeneous corpus**.
- Tasks are mostly **FAQ‑style** or simple “document lookup + explanation”.
- You need **low latency, low cost**, and minimal engineering effort.
- You are building an MVP to validate the usefulness of RAG at all.

This aligns with the basic course framing of RAG as “LLM + external documents” without sophisticated retrieval machinery.  
Source: `retrieval-augmented-generation.pdf / p.0–1`

### 6.2. When to move to advanced RAG

Upgrade to advanced RAG if:

- You already have naive RAG but see:
  - relevant documents present in the corpus yet **not showing up** in retrieved context,
  - frequent **irrelevant or incomplete answers**.
- Your corpus becomes **large and diverse**, requiring:
  - hybrid search (sparse + dense),
  - better chunking and metadata‑aware retrieval.
- You want **measurable, tunable retrieval quality** using RAG benchmarks (BEIR, NQ).  
  Source: `retrieval-augmented-generation.pdf / p.2–3`
- You are ready to invest in:
  - hybrid retrievers,
  - re‑ranking models,
  - retrieval evaluation pipelines.

### 6.3. When to invest in agentic RAG

Consider agentic RAG if:

- Tasks are **inherently multi‑step and open‑ended**:
  - research, audits, complex diagnostics,
  - decision support requiring multiple data sources and tools.
- You need the system to **adapt interactively**:
  - ask clarifying questions,
  - change plan mid‑way,
  - combine RAG with code, SQL, external APIs.
- You benefit from **agent memory** and cross‑episode learning (inspired by Generative Agents, Reflexion).  
  Source: `large-language-model.pdf / p.7, p.28`
- Your team can support:
  - an agent runtime/orchestrator,
  - robust monitoring/logging of agent behaviour,
  - design and maintenance of multiple tools (including RAG retrievers).  
  Source: `langchain.pdf / p.0–3`

### 6.4. Recommended evolution path

1. **Start with naive RAG**
   - Build a thin “LLM + vector DB” layer.
   - Validate usefulness and gather user feedback.

2. **Strengthen retrieval to advanced RAG**
   - Improve chunking and indexing.
   - Add hybrid retrieval and re‑ranking.
   - Introduce retrieval metrics and RAG evaluation.

3. **Wrap RAG into agents (agentic RAG)**
   - When tasks and users demand more complex, adaptive workflows:
     - expose RAG as a tool,
     - add other tools as needed,
     - implement an agent loop with planning and memory.

This staged approach is consistent with how the course presents RAG as a basic building block, then deepens retrieval patterns, and finally introduces agentic methods that can orchestrate RAG among other tools.

---

## Sources

### Core RAG and retrieval patterns

- `retrieval-augmented-generation.pdf / p.0–1` – definition of RAG, motivation, basic pipeline (naive RAG), high‑level architecture.
- `retrieval-augmented-generation.pdf / p.2` – similarity metrics, ANN, Late Interactions, re‑ranking techniques for better retrieval.
- `retrieval-augmented-generation.pdf / p.3` – hybrid retrieval, file‑format‑based chunking, RAG evaluation benchmarks (BEIR, Natural Questions).

### LangChain, chains, and agents

- `langchain.pdf / p.0–3` – LangChain as SDK to connect LLMs to data (RAG), build chains and agents, and define tools.

### Agents, planning, and memory

- `large-language-model.pdf / p.7` – DEPS (Describe, Explain, Plan and Select), Reflexion method, interactive multi‑step planning with LLMs.
- `large-language-model.pdf / p.28` – Generative Agents and multi‑task agents, role of memory and long‑term behaviour in agent systems.