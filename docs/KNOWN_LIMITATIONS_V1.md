# Known Limitations (v1.0)

This document outlines the known limitations, constraints, and trade-offs of the `binary_semantic_cache` v1.0 release.

## 1. Performance Limitations

### 1.1 Full Cache Load Time
While the *index* (binary codes) loads instantly (<10ms for 1M entries), the *response objects* are stored in a Python pickle file. Loading 1 million response objects takes approximately **300ms**. This is an O(n) operation dominated by Python's unpickling overhead.

### 1.2 Linear Scan Scaling
The cache uses a linear scan (O(N)) search. This allows for zero-index-build-time and perfect accuracy relative to the quantized codes, but it scales linearly.
- **N < 100k:** Sub-millisecond (0.16ms).
- **N = 1M:** ~1.6ms.
- **N > 1M:** Performance will degrade linearly. Use cases requiring >1M entries should consider sharding or wait for Phase 3 (HNSW).

### 1.3 Concurrency
The system uses a global `RLock` to serialize writes to ensure thread safety between the Python LRU and Rust storage.
- **Reads:** Fast and release the GIL in Rust.
- **Writes:** Serialized. High write contention may cause blocking.

## 2. Memory Limitations

### 2.1 In-Memory Storage
The entire index and all response objects must fit in RAM. There is no disk-based serving mode.
- **Index Overhead:** Fixed at ~52 bytes per entry.
- **Response Overhead:** Depends entirely on your data size (e.g., large JSON strings).

### 2.2 Memory Estimation Accuracy
The `cache.memory_usage()` method reports the exact memory used by the Rust index and Python internal structures. It **does not** (and cannot accurately) account for the memory consumed by the actual response string objects, as Python's object overhead varies. Always provision extra RAM.

## 3. Correctness & Semantics

### 3.1 Delete Behavior
The `delete(key)` method immediately removes the entry from the Python lookup table (making it inaccessible). However, the underlying slot in the Rust vector is **not freed** immediately. It remains "orphaned" until the LRU mechanism eventually recycles that slot for a new entry. This is a design choice to avoid O(N) shifts in the compact Rust vector.

### 3.2 Timestamp Resolution
LRU timestamps are stored as `u32` seconds since the epoch (2020-01-01). Access patterns occurring within the same second may not strictly preserve LRU order. This is considered acceptable for a semantic cache where "approximate LRU" is sufficient.

### 3.3 Quantization Drift
Binary quantization introduces a small amount of information loss. The Hamming distance is a proxy for Cosine Similarity.
- **Implication:** A threshold of `0.80` in binary space is not mathematically identical to `0.80` in float space. Users must tune thresholds empirically (see `THRESHOLD_TUNING_GUIDE.md`).

## 4. Platform & Distribution

### 4.1 Build Requirements
v1.0 requires a **Rust toolchain** (Cargo) to be installed to build the package from source. There are currently no pre-built binary wheels on PyPI.

### 4.2 Windows Unicode
While Unicode path handling was improved in v0.2.0, extensive testing on non-English Windows locales has not been performed in CI.

### 4.3 Pickle Security
The `responses.pkl` file uses Python's `pickle` module. **Do not load cache files from untrusted sources.** Pickle deserialization can execute arbitrary code. Only load caches that you or your organization created.

