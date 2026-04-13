# Comparing Naive RAG and Sentence Window Retrieval

> **⚠️ Best-effort draft:** this report was saved after reaching the maximum number of revise cycles and may still contain unresolved gaps noted by the Critic.

# Comparing Naive RAG and Sentence-Window Retrieval

## Executive summary

Retrieval-Augmented Generation (RAG) systems typically start with a **naive, chunk-based retrieval** setup: documents are split into fixed-size chunks, embedded, and top-k chunks are fed directly to the LLM. This is simple, cheap, and widely supported, but it wastes context and struggles with fine-grained questions.

**Sentence-window retrieval** instead indexes at (roughly) sentence granularity, retrieves individual sentences, then expands each hit into a small *window* of neighboring sentences before sending them to the LLM. This improves local relevance, grounding, and token efficiency, at the cost of a larger index, extra post-processing, and more engineering complexity.

Empirical research on retrieval granularity (e.g., Dense X Retrieval, SentGraph/MultiHop-RAG) and practitioner reports (LlamaIndex’s `SentenceWindowRetriever`, legal/enterprise RAG blogs) broadly support this trade-off: sentence-level/windowed retrieval improves retrieval metrics and perceived answer quality for complex, fine-grained QA, but increases index size and latency.

For most teams:
- Start with **naive chunk-based RAG** for speed and simplicity.
- Add **sentence-window or hybrid approaches** for long, dense, or high-stakes domains (legal, policy, technical specs) where precise grounding and token efficiency matter.

---

## 1. Definitions

### 1.1 Naive RAG (chunk-based retrieval)

Naive RAG refers to the standard, tutorial-style pattern for retrieval-augmented generation:

1. **Chunking**: Split documents into fixed-size text chunks (e.g., 512–2,000 tokens), often with small overlaps.
2. **Indexing**: Embed each chunk and store it in a vector database, along with metadata.
3. **Retrieval**: Embed the user query and run similarity search over chunk embeddings.
4. **Context construction**: Concatenate the top-k retrieved chunks and feed them to the LLM as context.

Granularity of both retrieval and context is the **chunk**. This is the pattern described in many RAG overviews and courses.

- Local KB: `retrieval-augmented-generation.pdf`, p.0–5 – describes standard RAG: chunk documents, embed, store in a vector DB, retrieve top-k chunks as context.

### 1.2 Sentence-window retrieval

Sentence-window retrieval decouples **what you retrieve** from **what you show the LLM**:

1. Index **sentences** (or small micro-chunks of 1–3 sentences) instead of large chunks.
2. At query time, retrieve the most relevant **sentences** by embedding similarity.
3. For each retrieved sentence, build a **window** of neighboring sentences (e.g., the sentence itself ±N neighbors) to supply enough local context.
4. Merge and deduplicate overlapping windows, then pass the resulting spans to the LLM.

Granularity of retrieval is **sentence-level**; the context boundary is a **window** around each relevant sentence.

This pattern is implemented explicitly in modern RAG frameworks (e.g., LlamaIndex’s `SentenceWindowRetriever`) and described in advanced RAG tutorials and blogs.

---

## 2. Mechanics: pipelines step-by-step

### 2.1 Naive RAG pipeline

**Preprocessing**
- Split documents into chunks by tokens or characters (e.g., 512–1,000 tokens), sometimes with 10–20% overlap.
- Embed each chunk with a sentence/embedding model.
- Store vectors in a vector DB (FAISS, Milvus, Qdrant, Weaviate, etc.) with metadata (document id, page, section).

**Query-time retrieval**
- Embed the user query.
- Perform approximate nearest-neighbor (ANN) search over chunk embeddings.
- Optionally apply metadata filters and/or re-rank with a cross-encoder.

**Context assembly**
- Concatenate top-k chunks (e.g., k=3–10) in rank order.
- Truncate to the LLM’s context limit.
- Feed to the LLM along with instructions and the user query.

This is exactly the “embed chunks → retrieve → augment prompt” loop covered in the local RAG material (`retrieval-augmented-generation.pdf`, p.0–5).

### 2.2 Sentence-window retrieval pipeline

**Preprocessing**
- **Sentence segmentation**: split each document into sentences via an NLP library (spaCy, NLTK, etc.), or into short micro-chunks (1–3 sentences) if segmentation is noisy.
- For each sentence (or micro-chunk):
  - Compute an embedding.
  - Store text, embedding, doc metadata, and its position (e.g., sentence index).

Framework examples:
- LlamaIndex (2023–2024): `SentenceWindowRetriever` retrieves sentence-level “nodes” then attaches neighbor nodes using a `SentenceWindowNodePostprocessor`.
  - Docs: <https://docs.llamaindex.ai>
- LangChain: while it doesn’t brand it as “sentence-window”, its `ParentDocumentRetriever` (2023–2024) approximates a coarse+fine pattern: index small chunks but retrieve via larger parent docs.
  - Docs: <https://python.langchain.com>

**Query-time retrieval**
- Embed the user query.
- Run similarity search over **sentence embeddings**.
- Retrieve top-M sentences (e.g., M=20–50).

**Window expansion and post-processing**
- For each hit, build a window (e.g., ±2 sentences around the hit) or up to a token budget per window.
- Merge overlapping windows from the same document.
- Optionally re-rank windows (e.g., with a cross-encoder) and enforce diversity across documents.

**Context assembly**
- Concatenate the top windows (e.g., 3–8 windows) until hitting a context token budget.
- Provide these windows as context to the LLM.

---

## 3. Detailed pipeline comparison

| Aspect                         | Naive RAG (chunk-based)                                      | Sentence-window retrieval                                          |
|--------------------------------|--------------------------------------------------------------|----------------------------------------------------------------------|
| Indexing unit                  | Chunk (e.g., 512–2,000 tokens)                              | Sentence or micro-chunk (1–3 sentences)                             |
| Context unit to LLM            | Entire chunk                                                 | Window: hit sentence ± neighboring sentences                        |
| Preprocessing                  | Simple chunking; fewer embeddings                           | Sentence segmentation; many more embeddings                         |
| Index size                     | Smaller (1 vector per chunk)                                | Larger (≈2–5× more vectors, depending on granularity)               |
| Retrieval granularity          | Coarse (mixed relevant + irrelevant content)                | Fine-grained (centered on highly relevant sentences)                |
| Context relevance              | Often mixed; many irrelevant tokens per retrieved block     | High; most tokens in windows are directly relevant                  |
| Handling of chunk boundaries   | Vulnerable; important info can straddle boundaries          | Natural; boundaries at sentence level, windows reconstruct context   |
| Post-processing complexity     | Low (maybe dedup + rerank)                                  | Higher (window expansion, dedup, merging, possible rerank)          |
| Token efficiency               | Lower; significant context waste                             | Higher; more “useful tokens per call”                               |
| Latency                        | Lower for small k; simple pipeline                          | Higher: more hits + window building + optional reranking           |
| Robustness to noisy text       | Often cushioned by including surrounding text                | More sensitive; noisy sentences can pull irrelevant windows         |
| Implementation complexity      | Low; widely documented and supported                        | Medium–high; more moving parts and tuning knobs                    |

---

## 4. Pros and cons of naive RAG

### 4.1 Advantages

1. **Simplicity and ecosystem support**
   - Directly supported in almost all RAG stacks (LangChain, LlamaIndex, Haystack, Milvus/Zilliz, Qdrant, Weaviate).
   - Easy to explain, implement, and debug.

2. **Lower preprocessing and storage cost**
   - Fewer embeddings per document → smaller index size.
   - Faster indexing and lower long-term storage requirements.

3. **Good baseline for many use cases**
   - When docs are already well-structured (e.g., sections, chapters), chunking at section or paragraph boundaries works reasonably well.
   - For broad, high-level questions, coarse context is usually sufficient.

4. **Lower latency and operational complexity**
   - Single-stage vector search and simple context concatenation.
   - Fewer components to monitor and tune.

### 4.2 Disadvantages

1. **Overly coarse retrieval granularity**
   - Answer-relevant information may be a few sentences, but retrieval drags in entire chunks (hundreds–thousands of tokens).
   - Wastes context and may distract the model with off-topic details.

2. **Context pollution and hallucinations**
   - Local RAG notes highlight that mixing multiple partially relevant sources can lead to incorrect or hallucinated answers.
   - Large chunks increase the chance of including conflicting or misleading context.
   - Source: `retrieval-augmented-generation.pdf`, p.4 (discussion of incorrect conclusions and hallucinations from retrieved context).

3. **Chunk-boundary problems**
   - Important paragraphs may straddle chunk boundaries.
   - Even with overlap, the retrieved chunk may miss crucial preceding or following sentences.

4. **Poor fit for fine-grained or multi-hop questions**
   - For questions that depend on a single clause or specific numeric value, relevant sentences can be buried inside big chunks.
   - Multi-hop reasoning over many small pieces of evidence is harder when retrieval units are coarse.

5. **Token inefficiency at scale**
   - To cover all relevant information, you may increase k (more chunks), which compounds token usage and latency.

---

## 5. Pros and cons of sentence-window retrieval

### 5.1 Advantages

1. **Higher retrieval precision and token efficiency**
   - Retrieval is focused on sentences most semantically similar to the query.
   - Windows add only nearby sentences, so most tokens in context are relevant.
   - GenerativeAI.pub (2024) describes sentence-window retrieval as providing “more accurate, relevant, and comprehensive information retrieval” than base RAG in enterprise QA, largely by avoiding large irrelevant spans.
     - Source: *Advanced RAG Retrieval Strategies: Sentence-Window Retrieval*, GenerativeAI.pub, ~2024. <https://generativeai.pub/advanced-rag-retrieval-strategies-sentence-window-retrieval-b6964b6e56f7>

2. **Better grounding and explainability**
   - Each answer can be grounded in specific sentences; easy to highlight them to users.
   - Supports compliance and auditing workflows (e.g., “show the clause this answer came from”).

3. **Improved performance on fine-grained and multi-hop tasks**
   - **Dense X Retrieval** (EMNLP 2024) shows that sentence/proposition-level indices can outperform passage-level indices in retrieval metrics (e.g., Recall@k, MRR) for QA tasks requiring precise evidence alignment.
     - Source: *Dense X Retrieval: What Retrieval Granularity Should We Use?*, EMNLP 2024. <https://aclanthology.org/2024.emnlp-main.845.pdf>
   - **SentGraph** (2024) and related work (MultiHop-RAG, arXiv:2401.15391) find that sentence-level retrieval significantly outperforms passage-level retrieval in multi-hop QA retrieval-only evaluations, supporting the intuition that fine-grained retrieval helps compositional reasoning.
     - Source: *SentGraph: Hierarchical Sentence Graph for Multi-hop QA*, 2024. <https://arxiv.org/html/2601.03014v2>

4. **Good fit for dense, regulatory, or highly detailed domains**
   - Legal contracts, financial regulations, technical standards, and detailed policies often encode critical facts in short sentences or clauses.
   - Sentence-window retrieval is naturally suited to retrieve exactly these spans plus minimal necessary context.

5. **Growing framework support**
   - LlamaIndex’s `SentenceWindowRetriever` (2023–2024) offers a ready-made implementation that retrieves sentence-level nodes and automatically attaches a configurable window of neighbors; example docs show improved grounded QA with fewer hallucinations.
     - Source: LlamaIndex docs – *SentenceWindowRetriever* (2023–2024). <https://docs.llamaindex.ai>

### 5.2 Disadvantages

1. **Larger index and higher memory footprint**
   - Many more vectors: indexing at sentence level or micro-chunk level can yield roughly 2–5× the number of vectors vs moderate-sized chunking, depending on corpus and settings.
   - Dense X Retrieval notes that finer granularity (sentence/proposition) increases index size and computational cost compared with passage-level indexing.

2. **More complex query-time pipeline**
   - Adds window expansion, merging, deduplication, and often reranking.
   - Each component introduces hyperparameters (window size, M hits, overlap thresholds) and edge cases.

3. **Latency overhead**
   - Retrieving more small items plus post-processing them can be slower than retrieving a few big chunks, particularly if you:
     - Use cross-encoder reranking over many candidate windows, or
     - Implement sophisticated diversification or clustering.

4. **Sensitivity to segmentation quality and noisy sentences**
   - Sentence segmentation in messy data (OCRed PDFs, code comments, bullet-heavy docs) is error-prone.
   - A single noisy or spuriously similar sentence can pull in an irrelevant window if you don’t apply good filtering or reranking.

5. **Risk of missing long-range dependencies**
   - Windows are local; if an answer depends on scattered context (e.g., figure captions + distant explanations, or cross-section constraints), small windows might not capture everything.
   - You may need complementary coarse-grained retrieval or hierarchical designs to recover long-range context.

6. **Higher engineering and tuning cost**
   - More knobs to tune (window size, number of hits, reranker configuration, dedup strategies).
   - Requires evaluation infrastructure to optimize trade-offs between accuracy, latency, and cost.

---

## 6. Empirical evidence and benchmarks

Direct, fully controlled **end-to-end RAG** benchmarks that isolate “naive chunk RAG vs sentence-window RAG” are still limited. Available evidence comes from:

1. **Retrieval-only academic studies on granularity**.
2. **RAG-focused frameworks and blogs**, which report internal experiments.

### 6.1 Retrieval granularity studies

1. **Dense X Retrieval: What Retrieval Granularity Should We Use? (EMNLP 2024)**
   - URL: <https://aclanthology.org/2024.emnlp-main.845.pdf>
   - Compares passage-, sentence-, and proposition-level indexing across QA benchmarks.
   - Findings (summarized):
     - Sentence/proposition-level indices can yield higher retrieval performance (Recall@k, MRR) for questions requiring precise evidence alignment.
     - Finer granularity increases index size and retrieval cost.
     - Passage-level often remains competitive and more efficient on simpler tasks.
   - Relevance for RAG: supports using sentence-level retrieval when questions require fine-grained evidence, but highlights the cost trade-off.

2. **SentGraph & MultiHop-RAG (2024)**
   - SentGraph paper: <https://arxiv.org/html/2601.03014v2>
   - Cites **MultiHop-RAG** (arXiv:2401.15391) and experiments where sentence-level retrieval significantly outperforms passage-level retrieval for multi-hop QA.
   - Relevance: strengthens the case that sentence-level retrieval improves retrieval quality for compositional, multi-hop reasoning tasks.

### 6.2 Practitioner and framework reports

1. **GenerativeAI.pub – Advanced RAG Retrieval Strategies: Sentence-Window Retrieval (~2024)**
   - URL: <https://generativeai.pub/advanced-rag-retrieval-strategies-sentence-window-retrieval-b6964b6e56f7>
   - Claims (for enterprise KBs, internal data):
     - Sentence-window RAG reduced hallucinations and improved answer accuracy vs base chunk-based RAG, especially for detailed policy/technical questions.
     - Better token efficiency (smaller, more focused context).
   - Evidence type: before/after internal comparisons, largely qualitative; some reports of “double-digit percentage” reductions in hallucination-related support tickets.

2. **LlamaIndex – SentenceWindowRetriever examples (2023–2024)**
   - Docs: <https://docs.llamaindex.ai>
   - Example notebooks show:
     - More faithful answers when using sentence windows vs coarse chunks.
     - Easier citation and explanation of answers based on retrieved sentences.
   - Evidence type: qualitative examples; some internal benchmarks (unpublished) reported by community/blog posts.

3. **Legal and fine-grained QA case studies (2023–2025)**
   - Legal-tech and enterprise RAG blogs (various vendors) describe moving from 1,000–2,000-token chunks to clause/sentence-level retrieval:
     - Improved interpretability (linking answers to specific clauses).
     - Fewer user-reported misinterpretations of contracts and policies.
   - Metrics are typically proprietary; evidence is anecdotal but consistent across reports.

**Synthesis:**
- Controlled retrieval-only studies support the idea that **finer granularity (sentence/proposition)** improves retrieval quality for complex QA.
- Production and framework reports suggest **sentence-window RAG improves grounded QA and reduces hallucinations** for fine-grained tasks.
- Costs: **larger indices** and **higher latency**, in line with the theory.

---

## 7. When to use which

### 7.1 Prefer naive chunk-based RAG when

1. **You’re building an MVP or prototype**
   - You want a working system quickly with minimal complexity.
   - Naive chunking is sufficiently accurate for early testing.

2. **Documents are short or naturally sectional**
   - FAQs, short KB articles, simple API docs, product pages.
   - Each chunk (page/section) already forms a coherent unit.

3. **Queries are high-level or approximate**
   - “Explain our refund policy.”
   - “What does this product do?”
   - Coarse retrieval is fine; the LLM can read an entire section.

4. **Infrastructure constraints are tight**
   - Limited memory or CPU/GPU; need to minimize index size and retrieval latency.
   - Edge deployments or cost-sensitive environments.

5. **Grounding can be approximate**
   - Conversational assistants, onboarding bots, or low-stakes Q&A where minor inaccuracies are acceptable.

### 7.2 Prefer sentence-window retrieval when

1. **You need precise, auditable answers**
   - Legal contracts, HR policies, compliance docs, clinical guidelines, or financial regulations.
   - Answers must be traceable to specific sentences/clauses.

2. **Documents are long, dense, and technical**
   - RFQs, standards (e.g., RFCs, ISO docs), technical manuals, research papers.
   - Key facts are often encapsulated in single sentences or short paragraphs.

3. **Queries are fine-grained or multi-hop**
   - Factoid questions (exact thresholds, penalties, default values).
   - Multi-step reasoning that composes multiple locally relevant facts.

4. **Tokens are scarce or expensive**
   - Smaller context windows (e.g., 8–16k tokens) or cost-sensitive usage patterns.
   - Sentence-window RAG maximizes the proportion of useful tokens.

5. **You have bandwidth for infra and evaluation**
   - You can collect evaluation sets, tune window size and k, and monitor latency/cost.

### 7.3 Hybrid or mixed scenarios

- Large, heterogeneous corpora (long PDFs + short notes + emails) often benefit from **query- or doc-type–based routing**:
  - Simple docs → naive chunk RAG.
  - Long or regulatory docs → sentence-window or hybrid retrieval.

---

## 8. Practical recommendations and hybrid patterns

### 8.1 Index size, memory, and latency

- Expect **sentence-level indices** to be roughly **2–5× larger** than chunk-based indices, depending on average sentence length and micro-chunk strategy.
- Use:
  - Lower-dimensional embeddings (e.g., 384-d instead of 768-d),
  - Vector compression (PQ, scalar quantization), and
  - ANN indices tuned for your recall/latency needs.

Latency management strategies:
- Limit the number of retrieved sentences (M) and windows (k).
- Precompute adjacency (neighbor sentence IDs) to make window expansion O(1).
- Cache windows for frequent queries.
- Use light cross-encoders or bi-encoder rerankers, not heavy models, for reranking.

### 8.2 Hybrid designs

1. **Hierarchical (coarse + fine) retrieval**

- Build:
  - A **coarse index** over larger chunks (e.g., sections/paragraphs).
  - A **fine index** over sentences or micro-chunks.

- Workflow:
  1. Coarse retrieval: retrieve top-M chunks.
  2. Fine retrieval: within those chunks, retrieve top-N sentences.
  3. Window assembly: build windows around those sentences and merge.

- Benefits:
  - Limits the search space for sentence-level retrieval.
  - Retains robustness and long-range context from chunk-level retrieval.

- Related pattern: LangChain’s `ParentDocumentRetriever` (2023–2024) indexes child chunks but retrieves via parent documents, then selects children as context.

2. **Two-stage retrieval with reranking**

- Stage 1: high-recall dense retrieval (chunks or sentences).
- Stage 2: cross-encoder or hybrid (BM25 + dense) reranking over candidate chunks/windows.
- Choose top-k windows for the LLM.

- Benefits:
  - Balances recall and precision.
  - Allows coarser first-stage retrieval for speed, with fine-grained second-stage selection.

3. **Dynamic window sizing**

- Adjust window size based on:
  - Query type (definition vs explanation vs multi-step reasoning).
  - Document structure (e.g., expand to fill a subsection if sentences fall inside the same heading).

- Heuristic example:
  - Start with ±1–2 sentences.
  - For long/complex queries or narrative docs, allow ±3–5 sentences.

4. **Mixed-granularity micro-chunking**

- Use micro-chunks of 2–3 sentences as the basic index unit.
- For most queries, treat micro-chunks like small chunks (simpler pipeline).
- For harder queries (detected by query classifier or low confidence), apply an additional sentence-level or windowing refinement within top micro-chunks.

5. **Post-processing and dedup best practices**

- Merge or drop highly overlapping windows (e.g., >60–70% token overlap).
- Enforce document diversity for exploratory queries (top results should come from multiple docs, unless one doc is clearly dominant).
- Use lightweight rerankers to down-weight spurious sentence matches.

### 8.3 Operational guidance

- **Instrumentation**
  - Track retrieval metrics (e.g., % of retrieved text actually referenced in answers, via user feedback or heuristic overlap).
  - Monitor hallucination reports and where they originate (particular docs, query types).

- **Iteration path**
  1. Deploy naive chunk-based RAG with good chunking (semantic or heading-based where possible).
  2. Collect evaluation data on difficult, fine-grained tasks.
  3. Introduce sentence-window retrieval or a hierarchical variant only for the hardest domains or query types.
  4. Tune k, M, window size, and rerankers based on offline evals and latency/cost constraints.

---

## 9. Summary

- **Naive RAG (chunk-based)** is the default pattern: simple, inexpensive, and good enough for many short, well-structured documents and high-level questions. Its main weaknesses are coarse retrieval granularity, context pollution, and inefficiency for fine-grained or multi-hop tasks.

- **Sentence-window retrieval** increases retrieval granularity by indexing sentences and building small windows around relevant ones. It improves token efficiency, grounding, and retrieval precision, particularly for long, dense, or high-stakes documents. The trade-offs are a larger index, more complex post-processing, and higher latency/engineering cost.

- **Empirical work on retrieval granularity** (Dense X Retrieval, SentGraph/MultiHop-RAG) supports finer granularity for complex QA, while **framework and practitioner reports** (LlamaIndex’s SentenceWindowRetriever, GenerativeAI.pub, legal/enterprise RAG blogs) corroborate improvements in grounded QA quality and hallucination reduction.

- **Practical guidance**:
  - Start with a well-chosen chunking strategy and naive RAG.
  - Instrument your system and identify where it fails: fine-grained, multi-hop, or high-stakes questions over long documents.
  - Introduce sentence-window retrieval or hierarchical hybrids selectively for those pain points, and tune for your latency and cost budgets.

---

## Sources

**Local knowledge base**
- `retrieval-augmented-generation.pdf`, p.0–5 – Overview of RAG, chunking strategies, retrieval loop, and limitations (e.g., incorrect conclusions and hallucinations due to context misuse).
- `large-language-model.pdf`, p.5–7 – Background on LLM context windows and architectures (used to reason about token budgets and context usage).

**Web sources (2023–2025)**
- GenerativeAI.pub – *Advanced RAG Retrieval Strategies: Sentence-Window Retrieval* (~2024). <https://generativeai.pub/advanced-rag-retrieval-strategies-sentence-window-retrieval-b6964b6e56f7>
- Dense X Retrieval: *What Retrieval Granularity Should We Use?* EMNLP 2024. <https://aclanthology.org/2024.emnlp-main.845.pdf>
- *SentGraph: Hierarchical Sentence Graph for Multi-hop QA*, 2024. <https://arxiv.org/html/2601.03014v2> (and references to MultiHop-RAG, arXiv:2401.15391).
- LlamaIndex documentation – `SentenceWindowRetriever` & advanced retrievers (2023–2024). <https://docs.llamaindex.ai>
- LangChain documentation – `ParentDocumentRetriever` and text-splitting strategies (2023–2024). <https://python.langchain.com>
- Zilliz / Milvus blog – *Building RAG with Milvus, vLLM, and Meta’s Llama 3.1* (2024). <https://zilliz.com/blog/building-rag-milvus-vllm-llama-3-1>