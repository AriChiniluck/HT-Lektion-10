# FAISS for Fast Similarity Search in RAG Systems

## Executive summary

FAISS is a high‑performance library for similarity search on dense vectors. In Retrieval‑Augmented Generation (RAG), it is typically the engine behind the “vector database” that maps embeddings of text chunks to their nearest neighbors. FAISS enables fast retrieval by:

- Using specialized index structures (Flat, IVF, PQ, HNSW) so queries do not need to scan all vectors.
- Compressing vectors with Product Quantization (PQ/OPQ) so very large corpora fit into RAM/VRAM.
- Exploiting SIMD and GPUs to accelerate distance computations.

The main knobs you tune are: which index type to use; whether and how strongly to quantize; and runtime search parameters (e.g., nprobe, efSearch). These jointly control recall, latency, and memory use.

---

## 1. Conceptual overview: FAISS in a RAG pipeline

A typical RAG flow:

1. Offline
   - Split documents into chunks.
   - Encode each chunk into a d‑dimensional vector (e.g., d=768) using an embedding model.
   - Build a FAISS index over these vectors.

2. Online query
   - Encode the user query (and possibly conversation context) into a d‑dimensional vector.
   - Use FAISS to find top‑k similar chunk vectors.
   - Fetch the corresponding texts/metadata and feed them to the LLM.

Where FAISS helps:

- **Brute‑force (Flat) search** uses optimized C++/SIMD/BLAS and optional GPUs to compute exact distances to all vectors quickly.
- **Approximate nearest‑neighbor (ANN) indexes** (IVF, IVFPQ, HNSW) examine only a small subset of vectors, giving huge speedups at the cost of small recall loss.
- **Quantization/compression** lets you store many more vectors in the same memory while keeping distances approximate but usable for retrieval.

RAG requirements (latency ~ tens of ms, recall "good enough", memory constrained) align very well with FAISS’s design.

---

## 2. How main FAISS indexes store and search vectors

### 2.1 Flat indexes (IndexFlatL2 / IndexFlatIP)

**Idea**

- Store all vectors as a dense matrix and compute distances to every vector at query time.

**Storage**

- Matrix `X` of shape (N, d), typically float32.
- Memory ≈ `N * d * 4` bytes + small overhead (e.g., d=768 → ~3 KB/vector).

**Search**

- Given query q, compute distance/similarity to all rows of X.
- Exact k‑NN; complexity O(N·d).

**Usage**

```python
import faiss

d = 768
index = faiss.IndexFlatIP(d)   # or IndexFlatL2
index.add(xb)                  # xb: (N, d) float32

D, I = index.search(xq, k)     # xq: (nq, d)
```

**When useful in RAG**

- N up to ~100k–500k on CPU, or a few million on GPU.
- As a ground‑truth baseline to evaluate approximate indexes.

---

### 2.2 IVF (IndexIVFFlat): inverted file with full‑precision vectors

**Idea**

Partition the space into nlist regions via a coarse quantizer (k‑means centroids). At search time, only probe the nprobe nearest regions and scan their contents.

**Storage**

- Coarse quantizer: an index of centroids (typically `IndexFlatL2/IP` with `nlist` centroids).
- Inverted lists: for each centroid, a list of the assigned full‑precision vectors.

**Build**

```python
import faiss

d = 768
nlist = 4096
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

index.train(xtrain)   # sample of your vectors
index.add(xb)         # assigns each xb[i] to its nearest centroid
```

**Search**

1. Use quantizer to find the nprobe closest centroids for q.
2. Only scan vectors in those lists.
3. Return global top‑k.

**Key knobs**

- `nlist` (index granularity): more lists → smaller lists, more centroids to maintain.
- `nprobe` (search breadth): higher → better recall, more latency.

**In RAG**

- Good for mid‑to‑large corpora (≈1M–50M vectors) when you can store full‑precision vectors but need lower latency than Flat.

---

### 2.3 IVF + PQ / OPQ (IndexIVFPQ)

**Idea**

Combine IVF (search only a subset of lists) with Product Quantization (PQ) to compress each vector into a small code, enabling very large indexes.

**PQ basics**

- Split each d‑dimensional vector into m sub‑vectors of size d/m.
- For each sub‑space, learn a codebook of size 2^nbits.
- Store only the index of the nearest codeword per sub‑vector.
- Memory per vector: `m * nbits` bits (e.g., m=64, nbits=8 ⇒ 64 bytes/vector).

**OPQ (Optimized PQ)**

- Learn a rotation of the input space before PQ to reduce quantization error, especially helpful at high dimension (e.g., 768).

**Storage**

- IVF centroids as in IVFFlat.
- For each vector in a list: its PQ (or OPQ+PQ) code instead of the raw float32 vector.

**Search**

1. As with IVF: find nprobe nearest centroids.
2. For each visited list, compute approximate distances from q to codes using precomputed lookup tables (no full reconstruction needed).
3. Optionally refine top‑R candidates with exact distances if original vectors are stored separately.

**Usage sketch**

```python
import faiss

d = 768
nlist = 4096
m = 64
nbits = 8

quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)

index.train(xtrain)
index.add(xb)
index.nprobe = 32
D, I = index.search(xq, k)
```

**In RAG**

- Enables storing tens to hundreds of millions of chunks on a single machine or GPU.
- You trade memory and throughput against some recall loss vs IVFFlat.

---

### 2.4 HNSW (IndexHNSWFlat / IndexHNSWPQ)

**Idea**

HNSW (Hierarchical Navigable Small World) builds a multi‑layer proximity graph where each vector connects to M neighbors. Queries walk the graph greedily, exploring a subset of nodes.

**Storage**

- Graph structure: neighbor lists for each node at multiple levels.
- Vector storage:
  - Full precision (`IndexHNSWFlat`), or
  - Quantized codes (e.g., `IndexHNSWPQ`).

**Build & search**

```python
import faiss

d = 768
M = 32
index = faiss.IndexHNSWFlat(d, M)
index.hnsw.efConstruction = 200
index.add(xb)               # builds the graph online

index.hnsw.efSearch = 80
D, I = index.search(xq, k)
```

- `M`: number of neighbors per node (higher → better recall, more memory and build time).
- `efConstruction`: search breadth during build (higher → higher quality graph).
- `efSearch`: search breadth at query time (higher → better recall, more latency).

**In RAG**

- Especially attractive on CPU‑only deployments where you want high recall and sub‑linear latency, and can afford more RAM.

---

### 2.5 GPU indexes

FAISS has GPU analogues for many CPU indexes:

- Flat: `GpuIndexFlatL2`, `GpuIndexFlatIP`.
- IVF: `GpuIndexIVFFlat`, `GpuIndexIVFPQ` (via `index_cpu_to_gpu`).

Typical pattern:

```python
import faiss

res = faiss.StandardGpuResources()

d = 768
cpu_index = faiss.IndexFlatIP(d)

# Move to GPU 0
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

gpu_index.add(xb)
D, I = gpu_index.search(xq, k)
```

- The algorithm is the same as the CPU index; the difference is that distance computations and code lookups run on GPU.
- GPU speedups (5–20×) are largest when N is big and you can batch queries.

---

## 3. Trade‑offs: recall, latency, memory, complexity

### 3.1 Comparison table

For 768‑dim embeddings, tuned reasonably for RAG‑style workloads:

| Index         | Recall (typical)                          | Latency behavior                                  | Memory vs Flat            | Build complexity                              |
|--------------|--------------------------------------------|--------------------------------------------------|---------------------------|-----------------------------------------------|
| Flat (L2/IP) | ~100% (exact)                             | Linear in N; predictable; very fast on GPU       | 1.0× baseline             | Very low (no training)                        |
| IVFFlat      | ~90–99% (depends on nlist/nprobe & data)  | Sublinear; tune via nprobe                       | ~1.0–1.2× (centroids)     | Moderate (k‑means train; choose nlist)        |
| IVFPQ/OPQ    | ~80–97% (PQ/OPQ, higher bits → higher end)| Sublinear; per‑candidate cheaper than IVFFlat    | ~4–16× smaller than Flat | High (train IVF+PQ; more hyperparameters)     |
| HNSW         | ~95–99% (high efSearch)                   | Logarithmic-ish; good recall–latency tradeoff    | ~1.2–3× Flat (graph)     | High (graph build O(N log N), tuning needed)  |

### 3.2 Key takeaways

- **Flat** is ideal as a correctness baseline and for small indexes; it becomes too slow or memory‑heavy for tens of millions of vectors unless heavily parallelized.
- **IVFFlat** is the first scalable step: similar memory but much lower latency at large N, with a smooth recall/latency trade‑off via `nprobe`.
- **IVFPQ/OPQ** is the tool for *very large* RAG corpora where RAM/VRAM is the main constraint; you accept some recall loss in exchange for 4–16× memory savings.
- **HNSW** offers strong recall–latency trade‑offs on CPU at the cost of higher memory and more complex construction.
- All ANN methods need empirical tuning on your data; use Flat as a ground‑truth reference when possible.

---

## 4. Practical RAG recipes (768‑dim embeddings)

These are opinionated starting points; always validate on your own distribution.

### 4.1 By corpus size (N = #chunks)

#### N ≈ 50k

- **Index**: `IndexFlatIP` or `IndexFlatL2` on CPU.
- **GPU**: Not required for latency; use only if you already have it and care about QPS.
- **Rationale**: Brute‑force over 50k vectors is cheap; exact search is simplest and most robust.

#### N ≈ 500k

- **Baseline**: Try `IndexFlat` (CPU). If per‑query latency is too high:
  - **Option A (CPU)**: `IndexIVFFlat`.
    - `nlist`: 1k–4k (e.g., 2048).
    - `nprobe`: 8–32 (start around 16).
  - **Option B (CPU)**: `IndexHNSWFlat`.
    - `M`: 16–32.
    - `efSearch`: 64–128.
- **GPU**: `GpuIndexFlatIP` is often enough; it keeps things simple and exact.

#### N ≈ 5M

- **CPU**:
  - If RAM is sufficient: `IndexIVFFlat`.
    - `nlist`: 4k–16k (e.g., 8192).
    - `nprobe`: 16–64.
  - If RAM constrained: `IndexIVFPQ` (optionally with OPQ).
    - `nlist`: 4k–16k.
    - PQ: `m = 64`, `nbits = 8` (64 B/vector).
- **GPU**:
  - Recommended if you want low latency.
  - Use GPU IVFFlat if vectors fit in VRAM; otherwise IVFPQ/OPQ or sharding.

#### N ≈ 50M+

- **Memory is the main constraint**.
- **Preferred**: IVF + PQ/OPQ, possibly combined with sharding.
  - `nlist`: 32k–128k (e.g., 65,536).
  - `nprobe`: 32–128 (more for recall, less for latency).
  - PQ: `m ≈ 48–64`, `nbits = 8` (48–64 B/vector).
- **HNSW**: viable if you have a lot of RAM and prefer graph ANN, but memory overhead is high.
- **GPU**: strongly recommended; likely multiple GPUs, each holding one shard of the index.

---

## 5. Operational patterns: IDs, persistence, and updates

### 5.1 Using `add_with_ids` and ID mapping in RAG

RAG requires mapping neighbors back to documents and chunks.

Pattern:

1. Assign each chunk a stable FAISS ID (e.g., row index in your chunk table).
2. Call `add_with_ids` so FAISS stores your IDs directly.
3. Maintain an external mapping (DB or key–value store):
   
   `faiss_id → {doc_id, chunk_id, metadata_ref}`

Example (simplified):

```python
import faiss
import numpy as np

# chunks: list of (doc_id, chunk_id, embedding)
embs = np.vstack([c[2] for c in chunks]).astype('float32')
faiss_ids = np.arange(len(chunks), dtype='int64')

index = faiss.IndexFlatIP(embs.shape[1])
index.add_with_ids(embs, faiss_ids)

id_map = {
    int(fid): {"doc_id": doc_id, "chunk_id": chunk_id}
    for fid, (doc_id, chunk_id, _) in zip(faiss_ids, chunks)
}

D, I = index.search(query_embs, k=10)
results = [[id_map[int(fid)] for fid in row if fid != -1] for row in I]
```

- For IVF/HNSW/PQ, the usage is the same; just construct those indexes instead of `IndexFlat`.
- Deletions: Often handled logically (e.g., by marking deleted in metadata) and cleaned up in periodic rebuilds.

### 5.2 Saving and loading indexes

- Save:

```python
faiss.write_index(index, "faiss.index")
```

- Load:

```python
index = faiss.read_index("faiss.index")
```

Recommendations:

- Persist both the FAISS index and your external ID→metadata mapping.
- Typically persist *CPU* indexes; then use `index_cpu_to_gpu` on startup to copy to GPU.
- Keep FAISS versions consistent across build and deploy environments to avoid compatibility issues, especially for complex indexes.

### 5.3 Updating indexes

Common patterns:

- **Append‑only**
  - Use `add`/`add_with_ids` for new chunks.
  - Works well for Flat and HNSW; IVF/PQ can handle moderate growth.

- **Periodic rebuild**
  - When data distribution changes significantly or you accumulate a lot of new data:
    - For IVF/PQ, retrain centroids and PQ codebooks on a fresh sample.
    - Rebuild the index offline, then swap it in atomically.

- **Deletions/updates**
  - FAISS support is limited and index‑type‑specific.
  - Practical RAG approach:
    - Maintain an "active" flag in metadata; filter out inactive entries after retrieval.
    - Periodically rebuild to physically remove them.

---

## 6. GPU‑focused guidance for RAG

### 6.1 When to use GPU Flat vs GPU IVF/IVFPQ

- **GPU Flat**
  - Use when:
    - N ≲ 5–10M and vectors fit in VRAM with margin.
    - You want exact recall.
  - Pros: Simple, no training, easy to reason about.

- **GPU IVFFlat**
  - Use when:
    - N is large (millions to tens of millions) and GPU Flat is too slow.
    - You can afford full‑precision vectors but need sublinear search.
  - Pros: Tunable recall/latency via `nprobe`.

- **GPU IVFPQ/OPQ**
  - Use when:
    - Vectors do **not** fit as float32 in available VRAM.
    - You want large indexes on a single or few GPUs.
  - Pros: Massive memory savings (e.g., ~50× vs float32 with suitable PQ) and good QPS.

### 6.2 Batching

- Each GPU call has fixed overhead (PCIe transfer + kernel launches).
- To make good use of GPU compute:
  - Aim for batch sizes of ≈32–512 queries per search call.
  - For interactive apps, use service‑level or micro‑batching across concurrent users with a small time window (e.g., 5–20 ms), if latency budget allows.

### 6.3 Memory sizing examples

- **10M × 768‑dim float32 (Flat)**
  - Raw: `10M * 768 * 4 B ≈ 30.7 GB`.
  - With overhead: expect ≈32–36 GB.
  - Fits only on high‑memory GPUs (e.g., 40–80 GB) and leaves limited headroom.

- **10M vectors with PQ (m=64, nbits=8)**
  - Codes: 64 B/vector → ≈640 MB codes total, plus centroids/codebooks.
  - ~50× smaller than float32 storage.

Implications for RAG:

- As `N * d * 4B` approaches 50–70% of VRAM, consider:
  - Moving from Flat/IVFFlat to IVFPQ/OPQ.
  - Sharding across multiple GPUs.
- Similar logic applies to CPU RAM, just at larger absolute sizes.

---

## 7. Concluding summary: how FAISS enables fast RAG retrieval

- FAISS accelerates RAG by replacing naive full‑scan similarity search with optimized exact (Flat) and approximate (IVF, PQ, HNSW) indexes, plus GPU acceleration.
- Flat indexes provide a simple, exact baseline that works well for small corpora and as a reference to evaluate ANN recall.
- IVF indexes (IVFFlat) give sublinear search time by restricting distance computations to a small subset of vectors, tuned via `nlist` and `nprobe`.
- PQ/OPQ‑based indexes (IVFPQ/OPQ) compress vectors dramatically, enabling hundred‑million‑scale RAG corpora at manageable memory cost with acceptable recall.
- HNSW offers a graph‑based alternative with strong recall–latency trade‑offs on CPU, at the cost of extra memory and more complex construction.
- GPU support multiplies throughput and reduces latency, especially when combined with batching, making even large‑scale RAG retrieval practical.
- In practice, you choose and configure a FAISS index based on corpus size, memory budget, and latency/recall requirements, then tune a handful of parameters (`nprobe`, `m`, `nbits`, `M`, `efSearch`) using measurements on your own data.

---

## Sources

**Local knowledge base**
- retrieval-augmented-generation.pdf (for general RAG context)

**Web**
- FAISS GitHub wiki: Index types and usage (Flat, IVF, PQ, HNSW, GPU) — facebookresearch/faiss.
- FAISS API docs: `IndexFlat`, `IndexIVFFlat`, `IndexIVFPQ`, `IndexHNSW*`, GPU wrappers.
- Subhajit Bhar, "FAISS Index Types for Production RAG" (configuration tips and rules of thumb for IVF/HNSW in RAG‑like use cases).
- Siddharth Jain, "Deep Dive into Faiss IndexIVFPQ" (PQ/OPQ internals and trade‑offs).
- Fuzzypoint, "Memory‑Squeezed Vector Search" (practical IVF and quantization heuristics).
- Pinecone and other vector DB blogs on FAISS/HNSW parameters and benchmarks (for recall/latency/memory trade‑off intuition).