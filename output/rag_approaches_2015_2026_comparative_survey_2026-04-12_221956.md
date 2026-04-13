# Retrieval Augmented Generation Approaches from 2015 to 2026: A Comparative Survey

## Executive summary

This report synthesizes research on Retrieval-Augmented Generation (RAG) and closely related approaches from roughly 2015 through early 2026. It does **not** enumerate every individual model (which would be infeasible), but instead provides an **exhaustive taxonomy** of families and design patterns, with representative systems and a chronological perspective. It covers:

- Historical evolution from differentiable memory networks and DrQA-style pipelines to DPR/REALM/RAG/FiD, RETRO/Atlas, and modern LLM+tools agents.
- Core **architecture families**: pipeline RAG, end-to-end / external-memory LMs, agentic/tool-based retrieval, long-context + memory hybrids, multimodal and code-focused RAG.
- **Retrieval mechanisms and index structures**: sparse, dense, late interaction, hybrid retrieval, cross-encoder and LLM-based reranking.
- **Training paradigms**: contrastive retriever training, joint retriever–generator optimization, RLHF for tools, and groundedness-driven alignment.
- **Benchmarks and evaluation**: QA/IR benchmarks and newer groundedness/faithfulness metrics (RAG Triad, TRUST-SCORE, TREC RAG track, industry evaluators).
- Comparative **trade-offs** and practical guidance for selecting and deploying RAG systems in 2024–2026.

Throughout, we explicitly distinguish between evidence-backed developments up to ~early 2026 and more speculative trends.

---

## 1. Scope and definitions

### 1.1 What counts as RAG?

Following Gao et al. (2023) and later surveys, we treat **Retrieval-Augmented Generation (RAG)** broadly as:

> Any architecture where a generative language model conditions on **retrieved external information** (non-parametric memory) at inference time to produce its output.

This includes but is not limited to:

- Classic **retriever–reader/generator pipelines** (e.g., DPR + FiD, RAG (Lewis et al.)).
- **End-to-end / external-memory LMs** where retrieval is integrated during pretraining or decoding (REALM, RETRO, kNN-LM).
- **Tool-augmented LLMs** that call search/web APIs (WebGPT, GPT-4 with browsing) when retrieval is explicitly used to ground answers.
- **Agentic RAG** where LLM agents orchestrate retrieval, tools, and memory over multiple steps (ReAct, agent frameworks).
- **Multimodal and code RAG**, where retrieved units may be images, video segments, or code/documentation.

We also discuss adjacent areas (long-context LMs, memory-augmented agents) insofar as they compete with or complement RAG, but we mark them when they do *not* use explicit IR-style retrieval.

### 1.2 Timeline and scope

- **Temporal scope**: roughly 2015–early 2026.
- **Task scope**: knowledge-intensive QA, factual dialogue, document and code assistants, multimodal QA, and more general LLM+tools systems where retrieval plays a central role.

Rather than listing all models, we aim for **exhaustive coverage of categories** and representative examples.

---

## 2. Historical evolution (2015–2026)

### 2.1 Precursors: differentiable memory & early open-domain QA (2014–2017)

**Differentiable external memory**

- **Neural Turing Machines (NTM)** (Graves et al., 2014)
  - RNN controller + differentiable memory matrix; learned read/write heads with content- and location-based addressing.
  - Retrieval over memory is fully differentiable, trained end-to-end by gradient descent.
  - Anticipates external memory but is small-scale and not IR-like (no large corpus, no ANN index).

- **End-to-End Memory Networks / Key-Value Memory Networks** (Sukhbaatar et al., 2015; Miller et al., 2016)
  - Store sentences or key–value pairs as memory slots.
  - Query attends to slots via softmax over similarity, possibly in multiple hops.
  - Demonstrate multi-hop reasoning over external evidence (bAbI, simple QA), but memory size is limited and jointly trained; there is no separation between model parameters and a large, independently updatable corpus.

These works established concepts of **external memory** and **multi-hop attention**, but not the IR-scale, non-parametric retrieval that characterizes modern RAG.

**Early open-domain QA pipeline: DrQA**

- **DrQA** (Chen et al., 2017, “Reading Wikipedia to Answer Open-Domain Questions”)
  - Retriever: TF–IDF/bigram hashing over Wikipedia articles.
  - Reader: RNN-based span extractor (later swapped with BERT-like readers).
  - Pipeline: retrieve top-N documents → run neural reader → answer.
  - Significance: first strong open-domain QA pipeline on Wikipedia and the conceptual ancestor of retriever–reader RAG systems.

### 2.2 Early neural retriever–reader pipelines (2017–2019)

- **BERTserini** (Yang et al., 2019)
  - Combines Anserini (BM25) retriever with BERT reader.
  - Shows that simply replacing RNN readers with BERT yields strong gains, establishing the **sparse-retriever + neural-reader** template.

- Emerging dual-encoder retrievers
  - Dense dual-encoder models for QA and response selection appear (2018–2019), learning query and passage embeddings with contrastive losses and in-batch/heuristic negatives.
  - Still mostly evaluated as IR components, not yet tightly integrated with generators.

### 2.3 Canonical RAG era (2019–2021)

**Dense Passage Retrieval (DPR)** – Karpukhin et al., 2020

- Dense dual-encoder retriever (BERT encoders for questions and passages) trained with contrastive learning.
- Embeddings indexed via FAISS or similar ANN systems; significantly outperforms BM25 on various open-domain QA datasets.
- Becomes the standard retriever for many RAG systems.

**REALM** – Guu et al., 2020

- Retrieval-augmented language model pretraining.
- Encoder-only transformer trained with masked language modeling where, for each masked position, relevant passages are retrieved from a large corpus.
- Uses REINFORCE-like methods to update the retriever based on downstream MLM loss.
- First influential **end-to-end retrieval-augmented LM**; complex and compute-heavy but conceptually unifies retrieval and language modeling.

**RAG (Lewis et al., 2020)**

- Popularizes the term **Retrieval-Augmented Generation**.
- Architecture:
  - Retriever: typically DPR over Wikipedia.
  - Generator: BART (seq2seq) model.
  - Variants:
    - RAG-Sequence: retrieve once, condition generator on top-k passages concatenated with the query.
    - RAG-Token: per-token marginalization over passages.
- Usually trains/fine-tunes generator with retriever frozen, though joint variants exist.
- Establishes the canonical **retrieve-then-generate** pattern for knowledge-intensive tasks.

**FiD (Fusion-in-Decoder)** – Izacard & Grave, 2020/21

- Retrieve k passages (via BM25 or DPR); encode each with T5 encoder and then **fuse** all encodings in the decoder’s cross-attention.
- Allows the decoder to attend over multiple documents simultaneously.
- Often outperforms RAG-Sequence; becomes the de facto “reader” for many later RAG LMs (e.g., Atlas).

**kNN-LM** – Khandelwal et al., 2020

- Builds a **datastore of training hidden states and next-token distributions**.
- At inference, retrieves nearest neighbors in hidden-state space and interpolates their distributions with the base LM.
- Retrieval is over internal parametric training traces rather than external documents, but conceptually bridges LMs with non-parametric memory.

### 2.4 Scaling and specialization (2021–2023)

**RETRO** – Borgeaud et al., 2022

- Large transformer LM with explicit retrieval blocks.
- Chunks training corpus into fixed-size segments; encodes them into dense vectors and indexes them.
- During pretraining, at designated layers, the LM retrieves nearest-neighbor chunks and attends to them.
- Achieves GPT-3-level performance with fewer parameters by offloading factual knowledge to the external database.

**Atlas** – Izacard et al., 2022

- Retrieval-augmented LM for knowledge-intensive NLP.
- Uses DPR-like retriever and FiD-based reader (large T5).
- Multi-task, multi-dataset training (QA, fact verification, entity disambiguation) to build a robust general RAG LM.

**Citation- and tool-based systems: WebGPT, GopherCite**

- **WebGPT** (OpenAI, 2021)
  - GPT-3 with a browser tool that calls Bing search.
  - Trained with imitation and RLHF to query the web, read pages, and provide answers with citations.
- **GopherCite** (DeepMind, ~2022)
  - Gopher LM with retrieval and explicit citations from a curated knowledge base.
- These systems extend **RAG from static corpora to live web search** and emphasize **answer attribution**.

**ColBERT and late interaction**

- **ColBERT** (Khattab & Zaharia, 2020) and later ColBERTv2
  - Encode queries and documents into token-level embeddings.
  - Compute relevance via max-sim across query and document tokens (late interaction).
  - Provide better ranking than single-vector dense retrievers at the cost of index size and compute.
- Often used as a high-quality retriever or reranker in RAG stacks.

**Multi-hop and hybrid retrieval**

- Multi-hop QA (HotpotQA) spurs techniques that chain multiple retrieval steps.
- Hybrid retrieval (sparse + dense) emerges as a strong pattern: combine BM25 with DPR/Contriever/GTR; rerank with cross-encoders.

### 2.5 LLM + tools, multimodal, and agentic RAG (2023–2026)

**LLMs with tools (GPT-4, Claude, Gemini)**

- Commercial LLMs expose APIs for **tool calling**, including web browsing, vector DB search, and code execution.
- Retrieval tools implement RAG under the hood (often hybrid sparse+dense retrieval plus caching), while the LLM learns (via RLHF and supervised fine-tuning) *when* and *how* to call them.

**RAG frameworks: LangChain, LlamaIndex, etc.**

- Provide standardized components for chunking, indexing, retrieval (FAISS, HNSW, vector DBs), reranking, and LLM orchestration.
- Embody the **system-level RAG pipeline** described in later surveys (e.g., “RAG and RAU” 2024; “RAG and Beyond” 2024).

**Agentic retrieval patterns: ReAct, Self-Ask**

- **ReAct** (Yao et al., 2022): interleaves “Thought” (reasoning) and “Action” (tool calls, often retrieval) in a single prompt; effective for multi-step question answering and planning.
- **Self-Ask with Search** (Press et al., 2022): decomposes questions into sub-questions, answering each sub-question with web search assistance.
- These methods turn RAG into a **multi-step, adaptive process**, not a single retrieval call.

**Multimodal RAG (2024–2025)**

- **MMed-RAG** (NeurIPS 2024 / arXiv:2410.13085)
  - Versatile multimodal RAG for medical vision–language models.
  - Domain-aware retrieval: specialized indexing for different medical domains (radiology, pathology, etc.).
  - Retrieves both textual and image-associated information.
  - Adaptive context selection: selects a subset of retrieved contexts to feed the LVLM, avoiding context overload.
  - RAG-based preference fine-tuning: uses retrieved evidence in preference optimization to improve factuality.

- **RULE** (EMNLP 2024) – Reliable Multimodal RAG for Factuality in Medical VLMs
  - Focuses on factual reliability of multimodal (image + text) answers.
  - Uses multimodal retrieval over medical images and reports; introduces mechanisms to reduce hallucinated findings.

- **“Ask in Any Modality: A Comprehensive Survey on Multimodal Retrieval-Augmented Generation”** (2025)
  - Surveys text–image, text–video, and audio–text RAG.
  - Identifies common patterns:
    - CLIP-like shared encoders to embed images and text in a common space for retrieval.
    - Cascaded pipelines: retrieve images/documents, then feed them alongside text into a VLM.
    - Document AI multimodal RAG: scanned pages, charts, and OCR text.

- Industry patterns (e.g., Medium 2025 case studies): multimodal RAG for **document AI**, where VLMs index and retrieve page images + OCR text and answer layout-aware questions about tables and figures.

**Code RAG (2022–2025)**

- IDE assistants (GitHub Copilot, Amazon CodeWhisperer, etc.)
  - Index project code, documentation, and sometimes tickets/wikis in vector stores.
  - At query time, retrieve relevant files/snippets and include them in prompts for code LMs like GPT-4, Code Llama, or CodeGemma.
  - Provide repository- and organization-aware code suggestions: de facto RAG for code.

- **Rescue: Retrieval-Augmented Secure Code Generation** (2025)
  - ArXiv preprint (2025).
  - RAG pipeline retrieves security-relevant documents (e.g., CWE, internal security guidelines) and conditions code generation on them, aiming to reduce insecure patterns.

- Enterprise “documentation-aware” code assistants (2023–2025)
  - Use dense retrievers over code, design docs, runbooks; cross-encoder or LLM rerankers; code LMs for generation.

**Long-context models vs RAG; emerging memory architectures (2024–2026)**

- Long-context LMs (e.g., context windows from 32K up to ~1M tokens in models like Gemini 1.5 Pro, Claude 3.5, and academic long-context transformers) allow **in-context ingestion of large documents**.
- They challenge the need for RAG for tasks where all relevant context can fit in the window.
- However, they do not eliminate:
  - The need for **search** across very large corpora.
  - The importance of efficient indexing and retrieval for billions of documents.

Emerging research (2025–2026) explores **memory-augmented LMs** that unify aspects of RAG and long-context memory:

- **BudgetMem** (2025): learns **selective memory policies** to store and recall a subset of past tokens or experiences, saving cost vs naive full-context replay.
- **Multi-layered memory architectures for LLM agents** (2026): propose hierarchical short-term, episodic, and long-term memories; agents retrieve from these stores as they reason.
- These are not classic document RAG but share the core idea of **non-parametric retrieval and external memory**, and they increasingly blend with RAG-style retrieval over documents.

---

## 3. Key model families and architectures

We group RAG-related methods into several **architectural families**, each with representative models.

### 3.1 Pipeline RAG: retriever + reader/generator

**Definition**: A separate retrieval module selects documents/passages, then a generator conditions on them to produce an answer.

Representative systems:

- **DrQA** (2017): BM25 + RNN reader (span extractor).
- **BERTserini** (2019): BM25 + BERT reader.
- **DPR + FiD** (2020–2022): dense retrieval plus Fusion-in-Decoder.
- **RAG (Lewis et al., 2020)**: DPR retriever + BART generator; RAG-Sequence and RAG-Token.
- **Atlas** (2022): DPR retriever + FiD reader, multi-task training.

Architecture patterns:

- Typically retrieve top-k passages via sparse or dense retrievers.
- Concatenate or separately encode passages and query; decoder cross-attends all retrieved representations.
- Retrieval is often frozen or trained separately from the generator.

### 3.2 End-to-end differentiable RAG and external-memory LMs

**Definition**: LMs where retrieval is integrated into pretraining or decoding, often with gradient flow (or approximate gradients) through retrieval.

Representative systems:

- **REALM**: integrates DPR-like retriever into MLM pretraining; updates retriever with RL-style gradients.
- **RETRO**: retrieval blocks at specific transformer layers; retrieval index built from pretraining corpus.
- **kNN-LM**: retrieval over internal hidden-state datastore at inference time.
- Early **memory networks** and **NTM** can be seen as extreme small-scale versions.

Architecture patterns:

- Retrieval is invoked **inside** the LM, not just before it.
- The retriever and LM are often co-trained; index may be built over the LM’s own training data.
- Enables **parameter-efficient knowledge**: smaller LM with large external store.

### 3.3 Tool- and agent-based RAG

**Definition**: LLMs that call retrieval as a *tool* (e.g., “search”) within multi-step reasoning or agent frameworks.

Representative systems and patterns:

- **WebGPT**: GPT-3 uses a browser tool to search Bing, read web pages, and answer with citations.
- **ReAct**: prompt-based agent that alternates thought and action (tool calls including search).
- **Self-Ask with Search**: decomposes questions into sub-questions and uses search between steps.
- Commercial LLMs (GPT-4, Claude, Gemini) with browsing or vector-search tools.

Architecture patterns:

- LLM acts as a **controller** deciding when to retrieve, how to decompose tasks, and how to combine multiple retrieval calls.
- Retrieval is often web search or vector search; RAG is part of a broader **agentic workflow** with additional tools.

### 3.4 Multimodal RAG

**Definition**: Retrieval over and generation conditioned on multimodal evidence: images, videos, scanned documents, etc.

Representative systems:

- **MMed-RAG**: multimodal RAG for medical vision–language; domain-aware retrieval, multimodal context, preference fine-tuning.
- **RULE**: focuses on factuality for medical VLMs with multimodal retrieval.
- Systems surveyed in **“Ask in Any Modality”** (2025): text–image, text–video, document QA.

Architecture patterns:

- Use multimodal encoders (e.g., CLIP-like) to embed visual and textual content into a shared space.
- Index embeddings of images, captions, OCR text, and sometimes video segments.
- Retrieval returns multimodal chunks (e.g., page image + text) that are passed to a VLM capable of cross-modal attention.

### 3.5 Code RAG

**Definition**: RAG where documents are source code and related artifacts.

Representative patterns and systems:

- **IDE assistants**: index repositories and docs; retrieve relevant files/snippets for context-aware code completion.
- **Rescue**: retrieval-augmented secure code generation; retrieves security guidelines before generation.

Architecture patterns:

- Code, comments, and docs are embedded with code- or text-embedding models.
- Vector index supports retrieval by symbol names, semantic functions, or docstrings.
- Generator (code LM) takes **local code context + retrieved artifacts** as input.

### 3.6 Long-context + RAG and memory-augmented hybrids

Representative trends:

- **Long-context LMs**: large context windows reduce reliance on external retrieval for some tasks.
- **Memory-augmented LMs** (BudgetMem, multi-layered agent memory): maintain learned or external memories that support selective recall, approximating a more dynamic, policy-driven RAG.

Architecture patterns:

- Combine a **search component** (vector or other) to find candidate memories/docs.
- Use memory policies or attention mechanisms to select which information to integrate at different stages of reasoning.

---

## 4. Retrieval mechanisms and indexing

### 4.1 Retrieval mechanism types

**Table 1 – Retrieval mechanisms for RAG**

| Type | Mechanism | Pros | Cons | Typical RAG use cases |
|------|-----------|------|------|------------------------|
| Sparse (BM25, TF–IDF, SPLADE++, uniCOIL) | Inverted index over terms; SPLADE++/uniCOIL learn sparse term weights | Fast, interpretable, easy updates, good for exact matches | Weak semantic matching, struggles with paraphrases, language-dependent tokenization | Baselines; hybrid retrieval; compliance/legal search where term matching matters |
| Dense (DPR, Contriever, GTR, OpenAI/other embeddings) | Dual-encoder or embedding model; ANN search in vector space | Strong semantic similarity, language-agnostic, good recall | Requires supervised or distillation training; vector DB infra; index refresh cost | Open-domain QA; enterprise doc RAG; code/documentation retrieval |
| Late interaction (ColBERT, ColBERTv2) | Store token-level embeddings; max-sim over token pairs | High relevance, nuanced phrase matching, improved ranking | Larger index, higher compute, more complex infra | High-precision domains (legal, medical, scientific search); reranking stage in RAG |
| Hybrid (sparse + dense) | Merge candidate sets and scores from sparse and dense retrieval | Balances lexical and semantic recall; robust to query types | Complex scoring; hyperparameter tuning; more infra | Production RAG over heterogeneous corpora; search portals |
| Cross-encoder reranking | Jointly encode query+doc, output relevance score | Very strong precision, captures complex interactions | Expensive; limited to reranking top-N candidates | Second-stage reranking in RAG; small/medium-scale high-stakes tasks |
| LLM-based reranking/judging | LLM evaluates relevance/groundedness for candidate contexts | Flexible, task-aware; can reason about multi-sentence context | High cost; variance; need calibration/prompting | RAG Triad evaluation; quality monitoring; high-value queries |

### 4.2 Index structures and vector databases

Common index structures:

- **FAISS**: GPU/CPU ANN with IVF, PQ, HNSW; widely used in research DPR/RETRO/Atlas setups.
- **HNSW**: graph-based ANN; high recall at low latency; common in production vector DBs.
- **ScaNN**: Google’s ANN; optimized for TPU/CPU search at scale.

Vector databases (Pinecone, Weaviate, Qdrant, Milvus, Chroma, etc.) integrate these structures with features:

- Metadata filters (e.g., user/tenant, time, language).
- Hybrid search (BM25 + vector scores).
- Multi-tenant management, replication, and durability for production RAG.

---

## 5. Training paradigms

### 5.1 Contrastive training for dense retrievers

- Objective: for each query q, maximize similarity to positive passage p⁺ and minimize similarity to negatives p⁻.
- Negatives:
  - In-batch negatives: other positives in batch.
  - Hard negatives: retrieved by BM25 or other methods but not containing the answer.
- Loss: InfoNCE / softmax over positives+negatives, or margin-based losses.
- Datasets: NQ, TriviaQA, MS MARCO, BEIR subsets, etc.

### 5.2 Joint training of retriever and generator

- **REALM**: uses RL-style gradient estimates to update retriever based on LM loss.
- Variants of **RAG/Atlas**: fine-tune both DPR-like retriever and FiD reader jointly on QA tasks.
- Pros: tailor retrieval to downstream tasks; potential improvements in EM/F1.
- Cons: high compute; risk of overfitting to specific tasks; retriever less reusable.

### 5.3 RLHF and tool-use learning

- **WebGPT**: RLHF reward signals encourage useful browsing behavior and high-quality, cited answers.
- **Toolformer** (Schick et al., 2023): self-supervised fine-tuning where LLM annotates beneficial tool calls and is trained to imitate them, including retrieval tools.
- Commercial LLMs: combine supervised fine-tuning on tool-usage logs with RLHF for correctness and helpfulness.

### 5.4 Groundedness-driven training (2023–2025)

- **TRUST-ALIGN**: uses a dataset of grounded vs ungrounded RAG outputs (~19k examples) to train models to prefer grounded responses (via RL or preference fine-tuning).
- Medical RAG systems (MMed-RAG, RULE): incorporate retrieved evidence into training objectives to penalize hallucinated medical statements.

### 5.5 Continual learning and dynamic corpora

- Incremental retriever updates: periodic re-embedding and re-indexing new documents.
- Adapter-based or LoRA fine-tuning of retrievers to new domains.
- Benchmarks like **CRAG** (KDD Cup 2024) emphasize dynamic/streaming corpora and motivate **continuous improvement loops** for RAG.

---

## 6. Benchmarks and evaluation

### 6.1 Classic QA and IR benchmarks

- **Open-domain QA**: Natural Questions, TriviaQA, WebQuestions, CuratedTREC, HotpotQA, KILT tasks.
- **Retrieval**: MS MARCO (passage ranking & QA), BEIR (multi-domain IR), MTEB retrieval tasks.
- Metrics: EM, token-level F1, Recall@k, MRR, nDCG.

### 6.2 RAG-specific benchmarks and tracks

- **KILT**: knowledge-intensive benchmarks aligned to fixed Wikipedia snapshots, enabling consistent evaluation of retriever–reader systems.
- **CRAG** (KDD Cup 2024): Comprehensive RAG Benchmark
  - Evaluates retrieval and generation simultaneously on dynamic datasets.
  - Emphasizes: answer correctness, retrieval coverage, and evidence grounding.
- **TREC 2024 RAG Track**
  - NIST track focusing on retrieval-augmented systems.
  - Metrics include standard IR metrics on retrieval and human/automatic assessment of answer quality and support.

### 6.3 Groundedness, faithfulness, hallucination metrics

**Concepts**

- **Groundedness**: extent to which model output is supported by provided context.
- **Faithfulness**: consistency of output with context and world facts; lack of contradictions.
- **Hallucination**: unsupported or incorrect statements produced by the model.

**RAG Triad (TruLens/Snowflake)**

- **Context relevance**: how relevant retrieved context is to the question.
- **Groundedness**: whether the answer is justified by context.
- **Answer relevance**: whether the answer addresses the question.
- Implementation often uses **LLM-as-a-judge** scoring.

**TRUST-SCORE and TRUST-ALIGN** (2024)

- TRUST-SCORE: a composite metric quantifying grounding errors in RAG outputs; considers unsupported claims and contradictions.
- TRUST-ALIGN: alignment technique using curated groundedness examples to fine-tune models toward faithful outputs.

**Industry evaluators (2024–2026)**

- **Openlayer** (2026 guide): groundedness evaluator that labels answer spans as supported/unsupported, aggregates per-query scores, and recommends workflows for monitoring.
- **deepset**: RAG evaluation framework with groundedness dashboards; uses LLM-as-judge metrics.
- **Athina**: API that scores groundedness between query, context, and answer (0–1 scale).

**Attribution and evidence-based metrics**

- Attribution precision/recall and F1 on evidence sentences.
- Entailment-based scores using NLI models or LLMs: does context entail the answer?
- These metrics are widely used in recent RAG papers, especially in sensitive domains (medical, legal).

---

## 7. Comparative strengths, weaknesses, and trade-offs

### 7.1 Chronological overview (2015–2026)

**Table 2 – Major RAG-related milestones (representative, not exhaustive)**

| Period | Representative systems/families | Key contributions to RAG |
|--------|---------------------------------|---------------------------|
| 2014–2017 | NTM, Memory Networks, DrQA | External differentiable memory; first open-domain QA with explicit retriever–reader separation |
| 2018–2019 | BERTserini, early dual-encoders | Strong sparse+neural pipelines; initial dense retrievers |
| 2020 | DPR, REALM, RAG, FiD, kNN-LM, ColBERT | Establish dense retrieval; name and formalize RAG; fusion-in-decoder; non-parametric LM extensions; late interaction retrieval |
| 2021–2022 | RETRO, Atlas, WebGPT, GopherCite, ReAct | Large external-memory LMs; general-purpose RAG LMs; web-based retrieval with RLHF; agentic retrieval patterns |
| 2022–2023 | HyDE, production RAG stacks | Query expansion via hypothetical docs; widespread adoption of vector DBs and hybrid retrieval in industry |
| 2023–2024 | GPT-4/Claude/Gemini tools, RAG Triad, CRAG, TREC RAG | Tool APIs for retrieval; standardized triad metrics; dedicated RAG benchmarks/tracks |
| 2024–2025 | Multimodal RAG (MMed-RAG, RULE), Rescue (code RAG), multimodal RAG survey | RAG extended to images/video and code; focus on factuality and security; broad multimodal taxonomy |
| 2025–2026 | Memory-augmented LMs (BudgetMem, multi-layered agent memory), dynamic/parametric RAG | Convergence of RAG with long-context and memory architectures; learned retrieval/memory policies; tighter integration in agentic frameworks |

### 7.2 RAG vs alternative approaches

**Table 3 – High-level trade-offs**

| Approach | Quality (factual/knowledge-intensive) | Latency | Infra complexity | Updateability | Evidence/faithfulness |
|----------|----------------------------------------|---------|------------------|---------------|------------------------|
| Pure parametric LMs (no retrieval) | Strong on well-covered training data; weaker on niche/new info | Low | Low | Low (requires retraining/fine-tuning) | Prone to hallucinations; hard to inspect evidence |
| Classic text-only RAG | Strong on factual/document-backed tasks; sensitive to retrieval quality | Medium | Medium (index + retriever) | High (update index) | Better groundedness; can still hallucinate beyond context |
| Multimodal RAG | Strong when visual/layout evidence is crucial; depends on multimodal alignment | Medium–High | High (VLMs + multimodal indices) | High | Potentially high, but evaluation is harder |
| Code RAG | Strong for repo/documentation-grounded code; helps adhere to internal APIs | Medium–High | Medium–High (repo indexing) | High (auto index updates) | Better adherence to actual code; still requires careful evaluation |
| Long-context LMs (no external retrieval) | Good when all relevant context fits window; no retrieval errors | Medium–High (grows with context) | Medium (needs long-context-capable infra) | Medium (can stream new context but core knowledge fixed) | Still can hallucinate; evidence attribution less explicit |
| Memory-augmented/dynamic RAG | Aims to combine RAG strengths with learned memory policies | Variable | High | High | Potentially strong if trained with groundedness signals |
| Tool-agentic systems with RAG | Often best on complex, multi-step tasks | High | Very high (orchestration, tools, RAG, memory) | Very high (access live data/tools) | Can be highly faithful if carefully evaluated; behavior more complex |

### 7.3 Strengths and weaknesses by family

- **Pipeline RAG**
  - Strengths: modularity; ease of swapping retrievers/generators; good engineering fit; strong baselines.
  - Weaknesses: decoupled training; retrieval errors irrecoverable; sensitive to chunking and k.

- **End-to-end / external-memory LMs**
  - Strengths: theoretically optimal integration; parameter-efficient knowledge storage.
  - Weaknesses: complex training; difficult to update knowledge without re-pretraining; heavy infra.

- **Tool/agent-based RAG**
  - Strengths: flexibility; suitability for reasoning-heavy tasks; ability to use multiple data sources and tools.
  - Weaknesses: high latency; debugging complexity; dependence on prompt engineering and RLHF.

- **Multimodal RAG**
  - Strengths: can answer questions requiring visual/layout understanding; powerful for domains like radiology, charts, and scanned documents.
  - Weaknesses: costly; requires high-quality multimodal encoders and alignment; evaluation still immature.

- **Code RAG**
  - Strengths: repository awareness; better use of organizational knowledge; supports safety/security and compliance.
  - Weaknesses: scaling indices for large monorepos; keeping indices in sync with rapidly changing code; risk of leaking sensitive code if misconfigured.

---

## 8. Practical guidance for system designers (2024–2026)

### 8.1 When to use which approach

- Prefer **classic RAG** when:
  - Your corpus is relatively static or slowly changing (docs, manuals, wikis).
  - You need traceable evidence and controllable knowledge updates.
  - Latency/cost must be kept moderate.

- Prefer **multimodal RAG** when:
  - Answers depend on images, PDFs, diagrams, or scanned documents.
  - You have (or can obtain) high-quality multimodal encoders and labeled data for evaluation.

- Prefer **code RAG** when:
  - You assist developers on large internal codebases.
  - You want adherence to internal libraries, APIs, and patterns.

- Prefer **long-context LMs** when:
  - Relevant information fits into a single (or few) long context windows (e.g., a single book, long report, or small set of logs).
  - You prefer simpler infra and are willing to pay for larger context windows.

- Prefer **tool-agentic RAG** when:
  - Tasks are complex, multi-step, or require mixing static docs, web search, databases, and computation.
  - Latency and infra complexity are acceptable trade-offs for quality.

### 8.2 Design patterns for modern RAG

- **Chunking & indexing**
  - Use semantically coherent chunks (paragraph/sentence-level) rather than arbitrary token windows.
  - Apply some overlap to reduce boundary issues.
  - Store rich metadata (source, timestamps, access control) for filtering.

- **Retrieval & reranking pipeline**
  - Start with hybrid retrieval (BM25 + dense) to generate 50–100 candidates.
  - Rerank with cross-encoder or LLM-based scoring.
  - For difficult queries, consider **HyDE**-style synthetic document generation to improve retrieval.

- **Groundedness checking**
  - Use LLM-as-a-judge metrics (RAG Triad, groundedness evaluators like Athina) in offline evaluation and online monitoring.
  - For high-risk domains (medical, legal, security), include human-in-the-loop review and run robust groundedness benchmarks (TRUST-SCORE-style analyses).

- **Adaptation and maintenance**
  - Design for continuous index and embedding updates (cron jobs, streaming pipelines).
  - Monitor retrieval quality (Recall@k) and groundedness over time; watch for corpus drift.

### 8.3 Open problems and near-future directions

- **Better retriever–generator co-training** without prohibitive compute.
- **Dynamic retrieval policies**: learning when and how much to retrieve during generation (dynamic/parametric RAG).
- **Privacy-preserving RAG**: encrypted or on-device retrieval, secure code RAG (Rescue-like systems), and access control-aware indexing.
- **Robust multimodal groundedness**: metrics and datasets that test consistency between visual and textual evidence.
- **Unified memory architectures** that combine document RAG with agent memories and long-context traces.

---

## 9. Summary

From 2015 to early 2026, retrieval-augmented generation has evolved from simple BM25 + RNN pipelines and differentiable toy memories to a spectrum of sophisticated systems:

- **Classic RAG** and **external-memory LMs** show that much factual knowledge can be offloaded to external corpora while keeping parametric models smaller.
- **Dense, late-interaction, and hybrid retrieval** engines, combined with vector databases, have made high-quality retrieval practical at scale.
- **LLM+tools and agentic RAG** expand RAG into multi-step interactions with the web, APIs, and memory stores.
- **Multimodal and code RAG** bring RAG into images, video, and large codebases, emphasizing domain-specific groundedness and safety.
- **Groundedness and trustworthiness metrics** (RAG Triad, TRUST-SCORE, industry evaluators) have matured evaluation beyond simple EM/F1.

The line between **RAG**, **memory-augmented LMs**, and **tool-using agents** is increasingly blurred. Going into 2026, the central research and engineering challenge is to design systems that **retrieve the right information at the right time**, integrate it efficiently into large models, and provide **faithful, controllable, and auditable** outputs across modalities and domains.

---

## Sources

### Local knowledge-base

- retrieval-augmented-generation.pdf, pp.0–5 – Internal notes on RAG pipelines, late interactions (ColBERT), ANN search, hybrid retrieval, and references to TREC 2024 RAG Track and Microsoft 2025 RAG architecture guide.
- large-language-model.pdf, p.13 – Discussion of hallucinations and motivation for RAG.

### Key surveys and overviews

- Gao et al., "Retrieval-Augmented Generation for Large Language Models: A Survey", arXiv:2312.10997 (2023).
- "RAG and RAU: A Survey on Retrieval-Augmented Language Model", arXiv:2404.19543 (2024).
- "Retrieval-Augmented Generation and Beyond", arXiv:2409.14924 (2024).
- "Ask in Any Modality: A Comprehensive Survey on Multimodal Retrieval-Augmented Generation", arXiv:2502.08826 (2025).
- Trustworthiness in RAG survey, arXiv:2409.10102 (2024).

### Representative model and system papers

- Chen et al., "Reading Wikipedia to Answer Open-Domain Questions", ACL 2017 (DrQA).
- Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering", EMNLP 2020 (DPR).
- Guu et al., "REALM: Retrieval-Augmented Language Model Pre-Training", ICML 2020.
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", NeurIPS 2020 (RAG).
- Izacard & Grave, "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering", arXiv:2007.01282 (FiD).
- Khandelwal et al., "kNN-LM: Generalization through Memorization", ICLR 2021.
- Borgeaud et al., "Improving Language Models by Retrieving from Trillions of Tokens", ICML 2022 (RETRO).
- Izacard et al., "Atlas: Few-shot Learning with Retrieval-Augmented Language Models", ICML 2023.
- Khattab & Zaharia, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT", SIGIR 2020.
- WebGPT, OpenAI, 2021 (blog and arXiv preprint on web-browsing GPT-3).
- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models", ICLR 2023.
- Press et al., "Measuring and Narrowing the Compositionality Gap in Language Models", includes Self-Ask with Search pattern, 2022.

### Multimodal and code RAG

- "MMed-RAG: Versatile Multimodal RAG System for Medical Vision-Language Models", arXiv:2410.13085, NeurIPS 2024.
- "RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models", EMNLP 2024.
- "Rescue: Retrieval Augmented Secure Code Generation", arXiv:2510.18204 (2025).

### Groundedness, trust, and evaluation

- Snowflake/TruLens blog, "Benchmarking LLM-as-a-Judge for the RAG Triad Metrics", 2023.
- Openlayer, "Measuring RAG Groundedness – Complete Evaluation Guide", 2026.
- deepset.ai, "Evaluating RAG LLMs: Groundedness and Answer Quality", 2024.
- Athina, Groundedness evaluator API docs, 2024–2025.
- "Measuring and Enhancing Trustworthiness of LLMs in RAG Through Grounded ...", arXiv:2409.11242 (TRUST-SCORE/TRUST-ALIGN).

### Benchmarks and tracks

- KILT benchmark suite, 2020.
- BEIR: "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models", SIGIR 2021.
- MTEB: "The Massive Text Embedding Benchmark", 2023.
- CRAG (Comprehensive RAG Benchmark), KDD Cup 2024.
- TREC 2024 RAG Track documentation.