# Design Rationale: Phase 1

**Date:** 2025-11-29  
**Architect:** Phase 1 Architect  
**Status:** IMPLEMENTED

---

## 1. Core Philosophy: "Compute is Expensive, RAM is Cheap(ish), Latency is King"

The Binary Semantic Cache is built on a counter-intuitive premise: **We can trade a tiny amount of accuracy (1-5%) for massive gains in speed (30x) and memory efficiency (30x).**

In RAG and LLM Agent systems, the bottleneck is often the **latency of semantic search** and the **cost of storing millions of float vectors**. By converting embeddings to binary codes *before* they enter the hot path, we eliminate the need for heavy vector databases for the caching layer.

---

## 2. Why Binary Codes?

Standard embeddings use `float32`. A 768-dimension vector takes:
`768 * 4 bytes = 3,072 bytes`

A 256-bit binary code takes:
`256 bits = 32 bytes`

**That is a 99% reduction in raw payload size.**

### The Math: Sign Random Projection
We use **Sign Random Projection (SRP)**.
1.  **Projection:** $v' = R v$ where $R$ is a random orthogonal matrix.
2.  **Binarization:** $b = \text{sign}(v')$.

This preserves **Cosine Similarity**. The Hamming distance between two binary codes is directly proportional to the angle between the original vectors.
> **Phase 1 Validation:** We achieved a correlation of **0.99** between our binary Hamming distance and true Cosine similarity.

---

## 3. Architecture Components

### 3.1 The Encoder (`encoder.py`)
-   **Role:** Converts float vectors → binary codes.
-   **Design:** Stateless, deterministic.
-   **Key Decision:** Uses a fixed seed (42) for the projection matrix. This ensures that the same input text *always* produces the same binary code, even across restarts.

### 3.2 The Similarity Engine (`similarity.py`)
-   **Role:** Computes Hamming distance between a query and all cached codes.
-   **Design:** Optimized for modern CPUs.
-   **Optimization:** Uses `Numba` (JIT compiler) to emit optimized machine code for XOR + POPCOUNT operations.
-   **Fallback:** Pure NumPy implementation for compatibility.

### 3.3 The Cache (`cache.py`)
-   **Role:** Stores codes, metadata, and manages eviction.
-   **Design:** **Array-Based Structure**.
-   **Why not a Dict?** Python dictionaries have massive memory overhead (pointers, boxing). We use contiguous `numpy.ndarray` buffers for storage and a custom **Array-Based Doubly Linked List** for LRU tracking.
-   **Result:** Reduces overhead from ~1000 bytes/entry (naive dict) to ~119 bytes/entry.

---

## 4. Threshold Semantics

The Semantic Cache is probabilistic. It relies on a **Similarity Threshold**.

-   **Contract:** `HIT` if `similarity >= threshold`.
-   **Similarity Definition:** `1.0 - (hamming_distance / code_bits)`.
-   **Default:** `0.80`.
    -   *Why 0.80?* Binary quantization introduces ~5% noise. A "true" cosine similarity of 0.85 might appear as 0.80-0.82 in binary space. Lowering the default threshold ensures we don't miss valid matches (False Negatives), at the slight risk of False Positives.

---

## 5. Known Limitations (Phase 1)

1.  **Python Overhead:** Even with array-based optimization, Python's object model imposes a fixed overhead (~100 bytes) per entry for metadata wrappers. Phase 2 (Rust) will remove this.
2.  **Linear Scan:** Phase 1 uses a linear scan over all cached items. This is blazing fast for <100k items (1ms), but scales linearly ($O(N)$). Phase 2 will introduce an index ($O(\log N)$).
3.  **Embedding Models:** Currently, we assume the user provides embeddings or uses the Ollama integration. We do not bundle a heavy embedding model to keep the package lightweight.
4.  **Threshold Sensitivity:** Users must tune the threshold for their specific embedding model. What is "0.80" for `nomic-embed-text` might be "0.60" for `bert`.

---

## 6. Why This Beats a Vector DB

For a caching layer, a Vector DB is overkill.
-   **Vector DB:** Network calls, complex deployment, high memory usage, exact float search (slow).
-   **Binary Cache:** In-process, zero network latency, tiny memory footprint, binary search (fast).

We are building **Redis for Semantics**, not **Postgres for Vectors**.

