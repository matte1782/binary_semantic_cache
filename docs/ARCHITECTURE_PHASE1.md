# Binary Semantic Cache — Phase 1 Architecture

**Version:** 1.0  
**Date:** 2025-11-28  
**Status:** APPROVED  
**Prerequisite:** PHASE_1_PREFLIGHT_CHECK.md verdict = GO

---

## 1. Overview

The Binary Semantic Cache is a high-performance caching layer for LLM API calls that uses binary hashing of semantic embeddings to detect similar queries and return cached responses.

**Core Value Proposition:**
- Cache hits avoid expensive LLM API calls ($0.002-0.06 per call)
- Sub-millisecond lookup for 100K cached entries
- Memory-efficient: ~3 MB per 100K entries
- Drop-in replacement via OpenAI-compatible proxy

---

## 2. Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT APPLICATION                          │
│                    (unchanged, uses OpenAI SDK)                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTP POST /v1/chat/completions
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          PROXY LAYER                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │   FastAPI    │  │  Middleware  │  │  OpenAI-Compatible       │  │
│  │   Server     │──│  (caching)   │──│  Request/Response Models │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│   EMBEDDING LAYER   │ │     CORE LAYER      │ │   UPSTREAM LLM      │
│                     │ │                     │ │                     │
│ ┌─────────────────┐ │ │ ┌─────────────────┐ │ │ ┌─────────────────┐ │
│ │  BaseEmbedder   │ │ │ │  BinaryEncoder  │ │ │ │  OpenAI API     │ │
│ │  (abstract)     │ │ │ │  (frozen algo)  │ │ │ │  (passthrough)  │ │
│ └─────────────────┘ │ │ └─────────────────┘ │ │ └─────────────────┘ │
│         │           │ │         │           │ │                     │
│         ▼           │ │         ▼           │ └─────────────────────┘
│ ┌─────────────────┐ │ │ ┌─────────────────┐ │
│ │  OpenAIEmbedder │ │ │ │ HammingSimilar  │ │
│ │  LocalEmbedder  │ │ │ │ (Numba JIT)     │ │
│ └─────────────────┘ │ │ └─────────────────┘ │
└─────────────────────┘ │         │           │
                        │         ▼           │
                        │ ┌─────────────────┐ │
                        │ │ BinarySemanticC │ │
                        │ │     Cache       │ │
                        │ │ (in-memory)     │ │
                        │ └─────────────────┘ │
                        │         │           │
                        │         ▼           │
                        │ ┌─────────────────┐ │
                        │ │  LRU Eviction   │ │
                        │ └─────────────────┘ │
                        └─────────────────────┘
```

---

## 3. Data Flow

### 3.1 Cache Lookup Flow (Hot Path)

```
INPUT: User query (text string)
                │
                ▼
┌───────────────────────────────────┐
│ 1. EMBEDDING                      │
│    text → float[384] embedding    │
│    (OpenAI or local model)        │
│    Latency: ~50-200ms (external)  │
└───────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│ 2. BINARY ENCODING                │
│    float[384] → RandomProjection  │
│    → binarize_sign → pack_codes   │
│    → uint64[4] binary code        │
│    Latency: <1ms                  │
└───────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│ 3. SIMILARITY SEARCH              │
│    query_code XOR all_codes       │
│    popcount → Hamming distance    │
│    → normalized similarity        │
│    Latency: <1ms (Numba)          │
└───────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│ 4. THRESHOLD CHECK                │
│    if max_similarity >= 0.85:     │
│      CACHE HIT → return cached    │
│    else:                          │
│      CACHE MISS → call upstream   │
└───────────────────────────────────┘
                │
         ┌──────┴──────┐
         ▼             ▼
    [CACHE HIT]   [CACHE MISS]
    Return cached  Call OpenAI API
    response       Cache response
    ~1ms           ~500-2000ms
```

### 3.2 Binary Encoding Pipeline (FROZEN)

```
FROZEN FORMULA — DO NOT MODIFY WITHOUT RE-VALIDATION

Input:  embedding ∈ ℝ^d  (d=384, unit normalized)
Output: code ∈ {0,1}^256 packed as uint64[4]

Step 1: Random Projection
    P = GaussianMatrix(d × 256, seed=42)  # Frozen seed
    projected = embedding @ P             # → ℝ^256

Step 2: Sign Binarization
    binary = (projected >= 0).astype(int) # → {0,1}^256

Step 3: Bit Packing
    code = pack_codes(binary)             # → uint64[4] (LSB-first)

VALIDATION REQUIREMENT:
    Any change requires re-running similarity_correlation_test.py
    Must achieve correlation r ≥ 0.93
```

---

## 4. Module Responsibilities

### 4.1 Core Module (`src/binary_semantic_cache/core/`)

| File | Class/Function | Responsibility |
|------|----------------|----------------|
| `encoder.py` | `BinaryEncoder` | Wraps BinaryLLM projection + binarization + packing |
| `similarity.py` | `hamming_similarity_numba()` | Numba-JIT optimized batch Hamming distance |
| `cache.py` | `BinarySemanticCache` | Main cache: get/put, similarity search, eviction |
| `eviction.py` | `LRUEvictionPolicy` | LRU eviction using OrderedDict |

### 4.2 Proxy Module (`src/binary_semantic_cache/proxy/`)

| File | Responsibility |
|------|----------------|
| `server.py` | FastAPI application factory |
| `routes.py` | OpenAI-compatible endpoints (`/v1/chat/completions`) |
| `middleware.py` | Cache lookup/store middleware |

### 4.3 Embedding Module (`src/binary_semantic_cache/embedding/`)

| File | Class | Responsibility |
|------|-------|----------------|
| `base.py` | `BaseEmbedder` (ABC) | Abstract interface for all embedders |
| `openai.py` | `OpenAIEmbedder` | OpenAI API embeddings |
| `local.py` | `LocalEmbedder` | sentence-transformers local embeddings |

---

## 5. Frozen Interfaces from Validation

These interfaces are validated and MUST NOT be modified:

### 5.1 BinaryLLM Components (External)

```python
# From binary_llm.src.quantization.binarization
class RandomProjection:
    def __init__(self, input_dim: int, output_dim: int, seed: int = 42): ...
    def project(self, x: np.ndarray) -> np.ndarray: ...

def binarize_sign(x: np.ndarray) -> np.ndarray: ...

# From binary_llm.src.quantization.packing
def pack_codes(codes: np.ndarray) -> np.ndarray: ...
def unpack_codes(packed: np.ndarray, code_bits: int) -> np.ndarray: ...
```

### 5.2 Validated Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| `code_bits` | 256 | DECISION_LOG D2 |
| `similarity_threshold` | 0.85 | DECISION_LOG D6, validated in S3 |
| `projection_seed` | 42 | Determinism requirement |
| `embedding_dim` | 384 | Default (configurable) |

---

## 6. Performance Targets

From PHASE_1_PREFLIGHT_CHECK.md:

| Metric | Target | Kill Trigger | Measured (PoC) |
|--------|--------|--------------|----------------|
| Encode latency (per embedding) | <1 ms | >5 ms | 0.72 ms ✓ |
| Lookup latency (100K entries) | <1 ms (Numba) | >2 ms (Numba) | 13.5 ms (Python) |
| Memory (100K entries) | <4 MB | >10 MB | 3.05 MB ✓ |
| Similarity correlation | ≥0.93 | <0.90 | 0.9378 ✓ |

**CRITICAL:** Lookup latency requires Numba optimization. Python-only PoC achieved 13.5ms. Production MUST implement Numba path to achieve <1ms.

---

## 7. Security Boundaries

### 7.1 Input Validation

```
All external inputs MUST be validated:
- Text queries: max length, encoding check
- Embeddings (if accepting raw): shape, dtype, normalization
- API keys: never logged, never cached
- Upstream responses: sanitize before caching
```

### 7.2 No Arbitrary Code Execution

```
FORBIDDEN:
- eval() or exec() on any user input
- Dynamic imports based on user input
- Pickle of untrusted data (use JSON for cache)
- Shell commands with user-provided arguments
```

### 7.3 Resource Limits

```
- max_entries: configurable, default 100K
- max_query_length: 10000 characters
- max_cache_entry_size: 1 MB
- request_timeout: 30s upstream
```

---

## 8. Error Handling Strategy

### 8.1 Graceful Degradation

```
If cache fails → forward to upstream (no caching)
If Numba unavailable → fall back to NumPy with warning
If embedding fails → return error (don't cache)
If upstream fails → return upstream error (don't cache)
```

### 8.2 Never Cache Errors

```
ONLY cache responses where:
- upstream_status == 200
- response_body is valid JSON
- response contains expected fields
```

---

## 9. Configuration Schema

```python
class CacheConfig(BaseSettings):
    # Core settings
    max_entries: int = 100_000
    similarity_threshold: float = 0.85
    code_bits: int = 256
    embedding_dim: int = 384
    
    # Embedding provider
    embedding_provider: Literal["openai", "local"] = "local"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Proxy settings (optional)
    upstream_base_url: str = "https://api.openai.com/v1"
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    
    # Performance tuning
    numba_enabled: bool = True
    numba_threads: int = 4
    
    class Config:
        env_prefix = "BSC_"
```

---

## 10. Dependency Graph

```
Phase 1 Implementation Order:

     [pyproject.toml]
           │
           ▼
    ┌──────────────┐
    │   encoder    │ ← BinaryLLM (external)
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │  similarity  │ ← Numba
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │   eviction   │
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │    cache     │ ← encoder, similarity, eviction
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │  embedding   │ ← Optional (OpenAI, local)
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │    proxy     │ ← cache, embedding, FastAPI
    └──────────────┘
```

---

## 11. Test Strategy

### 11.1 Unit Tests (tests/unit/)

```
test_encoder.py      → Encoding determinism, batch consistency
test_similarity.py   → Numba vs NumPy accuracy, benchmark
test_cache.py        → Get/put, eviction, thread safety
test_eviction.py     → LRU correctness
```

### 11.2 Integration Tests (tests/integration/)

```
test_proxy.py        → Full HTTP round-trip
test_end_to_end.py   → Real embeddings + cache + proxy
```

### 11.3 Regression Tests

```
Mandatory before merge:
- similarity_correlation_test.py (r ≥ 0.93)
- benchmark_latency.py (encode <1ms, lookup <1ms with Numba)
- benchmark_memory.py (<4MB per 100K)
```

---

## 12. References

| Document | Purpose |
|----------|---------|
| `PHASE_1_PREFLIGHT_CHECK.md` | GO/NO-GO decision, guardrails |
| `docs/DECISION_LOG_v1.md` | 9 architectural decisions (D1-D9) |
| `PHASE_1_ROADMAP.md` | Week-by-week implementation plan |
| `validation/results/FINAL_VALIDATION_REPORT.md` | PoC validation results |

---

**Document Control:**
- Author: Architect Agent
- Reviewed: Hostile Reviewer (PASS)
- Status: APPROVED for Phase 1 implementation

