# Phase 2 Architecture: Binary Semantic Cache (Rust-Native)

**Status:** LIVE  
**Based on:** Phase 1 Architecture (Frozen)  
**Backend:** Rust (Mandatory)

---

## 1. System Overview

The Phase 2 architecture transitions the `BinarySemanticCache` from a pure Python prototype to a high-performance hybrid system. The core computational heavy lifting (encoding, similarity search) is offloaded to a Rust extension (`binary_semantic_cache_rs`), while Python retains control over high-level orchestration, LRU management, and API surfaces.

### 1.1 Core Components

```mermaid
graph TD
    Client[Client Application] --> API[Python API (Cache.get/put)]
    API --> LRU[LRU Manager (Python/NumPy)]
    API --> RustExt[Rust Extension (PyO3)]
    
    subgraph "Python Layer"
        LRU
        Storage[Numpy Arrays (Codes, Metadata)]
    end
    
    subgraph "Rust Layer (binary_semantic_cache_rs)"
        RustExt --> Encoder[RustBinaryEncoder]
        RustExt --> Sim[HammingSimilarity]
        Encoder --> Model[OnnxRuntime/Candle (Future)]
        Sim --> AVX[AVX2/NEON Intrinsics]
    end
    
    Storage -- Zero-Copy View --> Sim
```

---

## 2. Component Design

### 2.1 BinarySemanticCache (Python)
- **Role:** Orchestrator.
- **Responsibility:** 
  - Manages `numpy` arrays for storage (codes, responses, metadata).
  - Implements LRU eviction policy (array-based doubly linked list).
  - Thread safety (`RLock`).
  - **NEW:** Delegates encoding and search to Rust.
- **Constraint:** Must explicitly import and use Rust backend.

### 2.2 RustBinaryEncoder (Rust)
- **Role:** Deterministic Quantizer.
- **Responsibility:** 
  - Converts float embeddings (vector) → binary codes (uint64 array).
  - Maintains `seed=42` determinism identical to Phase 1 Python encoder.
- **Interface:** `encode(embedding: np.ndarray) -> np.ndarray[uint64]`

### 2.3 HammingSimilarity (Rust)
- **Role:** High-Speed Search Engine.
- **Responsibility:** 
  - `find_nearest`: Scans active codes for best match > threshold.
  - `batch_similarity`: Computes all-pairs similarity (for bulk ops).
- **Optimization:** Uses POPCNT instructions (AVX2/SSE4.2) and parallel iteration (Rayon) where applicable.
- **Interface:** 
  - `find_nearest(query: np.ndarray, codes: np.ndarray, threshold: float) -> Optional[(index, score)]`

---

## 3. Integration Strategy

### 3.1 Data Flow (Get)
1. **Input:** Float embedding `E` (Python).
2. **Encode:** `RustBinaryEncoder.encode(E)` → Code `C` (Rust, zero-copy return).
3. **View:** `cache._get_active_codes()` returns slice of `self._codes` (Python).
4. **Search:** `HammingSimilarity.find_nearest(C, codes_view, threshold)` (Rust).
   - Rust accepts NumPy array as `PyReadonlyArray2`.
   - Releases GIL for long computations (if needed, though <1ms is fast).
5. **Result:** Returns `(index, similarity)` or `None`.
6. **Return:** Python constructs `CacheEntry`.

### 3.2 Fallback & Safety
- **Mandatory Rust:** The system strictly requires the Rust extension.
- **Failure Mode:** If `binary_semantic_cache_rs` cannot be imported, the application MUST crash at startup with a descriptive error ("Build required").
- **Rationale:** Prevents silent performance degradation.
- **Testing:** Phase 1 Python components (`BinaryEncoder`, `similarity.py`) are retained **strictly for testing** (Oracles) to verify Rust correctness.

---

## 4. Persistence
- **Format:** `np.savez` (Compressed NumPy Archives).
- **Compatibility:** Identical to Phase 1. Rust implementation reads/writes compatible binary layouts (uint64 arrays).

---

## 5. Future Considerations (Phase 3)
- **MMap:** `np.load(mmap_mode='r')` allows zero-copy loading from disk, which Rust can consume directly.
- **SIMD:** Explicit AVX-512 support in Rust if available.

