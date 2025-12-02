# API Stability Guarantees (v1.0)

**Date:** 2025-12-02  
**Version:** 1.0.0  
**Status:** OFFICIAL

This document defines the API stability guarantees for `binary_semantic_cache` v1.x releases. It categorizes all public APIs into three tiers: **Stable**, **Deprecated**, and **Unstable/Internal**.

---

## Stability Tiers

| Tier | Meaning |
| :--- | :--- |
| **Stable** | Will not have breaking changes in any v1.x release. Safe to depend on. |
| **Deprecated** | Will be removed in v2.0.0. Use the documented replacement. |
| **Unstable/Internal** | May change or be removed in any release without notice. Do not depend on. |

---

## 1. Stable APIs (Will Not Break in v1.x)

### 1.1 `BinarySemanticCache`

The primary cache class. All methods and properties listed below are stable.

| Member | Signature | Notes |
| :--- | :--- | :--- |
| `__init__` | `(encoder, max_entries=100000, similarity_threshold=0.80)` | Constructor |
| `get` | `(embedding: np.ndarray) -> Optional[CacheEntry]` | Lookup by similarity |
| `put` | `(embedding: np.ndarray, response: Any, store_embedding: bool = False) -> int` | Store entry, returns index |
| `delete` | `(entry_id: int) -> bool` | Delete by index |
| `clear` | `() -> None` | Remove all entries |
| `stats` | `() -> CacheStats` | Get statistics |
| `memory_bytes` | `() -> int` | Estimate memory usage |
| `get_all_entries` | `() -> List[CacheEntry]` | Get all entries |
| `save_mmap_v3` | `(path: str) -> None` | Save to v3 format |
| `load_mmap_v3` | `(path: str, skip_checksum: bool = False) -> None` | Load from v3 format |
| `__len__` | `() -> int` | Number of entries |
| `__repr__` | `() -> str` | String representation |
| `encoder` | `property -> Encoder` | Get encoder instance |
| `max_entries` | `property -> int` | Maximum capacity |
| `similarity_threshold` | `property -> float` | Hit threshold |

### 1.2 `CacheEntry`

Immutable result object returned by `get()`. All fields are stable.

| Field | Type | Notes |
| :--- | :--- | :--- |
| `id` | `int` | Entry index |
| `code` | `np.ndarray` | Binary code (uint64) |
| `response` | `Any` | Cached response object |
| `created_at` | `float` | Unix timestamp (creation) |
| `last_accessed` | `float` | Unix timestamp (last access) |
| `access_count` | `int` | Number of accesses |
| `similarity` | `float` | Similarity score (default 1.0) |

### 1.3 `CacheStats`

Statistics dataclass returned by `stats()`. All fields and properties are stable.

| Member | Type | Notes |
| :--- | :--- | :--- |
| `size` | `int` | Current entry count |
| `max_size` | `int` | Maximum capacity |
| `hits` | `int` | Total cache hits |
| `misses` | `int` | Total cache misses |
| `evictions` | `int` | Total evictions |
| `memory_bytes` | `int` | Estimated memory usage |
| `hit_rate` | `property -> float` | hits / (hits + misses) |
| `memory_mb` | `property -> float` | memory_bytes / 1MB |

### 1.4 `RustBinaryEncoder`

The production encoder (Rust backend). All methods and properties listed below are stable.

| Member | Signature | Notes |
| :--- | :--- | :--- |
| `__init__` | `(embedding_dim: int, code_bits: int = 256, seed: int = 42)` | Constructor |
| `encode` | `(embedding: np.ndarray) -> np.ndarray` | Encode single/batch |
| `embedding_dim` | `property -> int` | Input dimension |
| `code_bits` | `property -> int` | Output bits (256) |
| `n_words` | `property -> int` | Number of uint64 words |

### 1.5 `PythonBinaryEncoder` (Test Oracle)

The Python encoder retained for testing. Same interface as `RustBinaryEncoder`.

**Note:** This class is stable for testing purposes only. Production code should use `RustBinaryEncoder`.

### 1.6 Exception Classes

All exception classes in the error hierarchy are stable.

| Exception | Base | Purpose |
| :--- | :--- | :--- |
| `CacheError` | `Exception` | Base class for all cache errors |
| `ChecksumError` | `CacheError` | SHA-256 checksum mismatch |
| `FormatVersionError` | `CacheError` | Unsupported persistence format |
| `CorruptFileError` | `CacheError` | Invalid or truncated cache file |
| `UnsupportedPlatformError` | `CacheError` | Platform incompatibility (e.g., endianness) |

### 1.7 Utility Functions

| Function | Signature | Notes |
| :--- | :--- | :--- |
| `detect_format_version` | `(path: str) -> int` | Returns 2 (v2) or 3 (v3) |

### 1.8 Constants

| Constant | Value | Notes |
| :--- | :--- | :--- |
| `DEFAULT_MAX_ENTRIES` | `100_000` | Default cache capacity |
| `DEFAULT_THRESHOLD` | `0.80` | Default similarity threshold |
| `DEFAULT_CODE_BITS` | `256` | Fixed binary code size |
| `MMAP_FORMAT_VERSION` | `2` | v2 format identifier |
| `MMAP_FORMAT_VERSION_V3` | `3` | v3 format identifier |

---

## 2. Deprecated APIs (Will Be Removed in v2.0)

These methods emit `DeprecationWarning` when called. Use the documented replacements.

| Method | Replacement | Removal Version |
| :--- | :--- | :--- |
| `BinarySemanticCache.save(path)` | `save_mmap_v3(path)` | v2.0.0 |
| `BinarySemanticCache.load(path)` | `load_mmap_v3(path)` | v2.0.0 |
| `BinarySemanticCache.save_mmap(path)` | `save_mmap_v3(path)` | v2.0.0 |
| `BinarySemanticCache.load_mmap(path)` | `load_mmap_v3(path)` | v2.0.0 |

**Migration Example:**

```python
# Old (deprecated)
cache.save("cache.npz")
cache.load("cache.npz")

# New (stable)
cache.save_mmap_v3("cache_v3/")
cache.load_mmap_v3("cache_v3/")
```

---

## 3. Unstable/Internal APIs (May Change)

The following are internal implementation details and are **not** part of the public API. They may change or be removed without notice.

### 3.1 Internal Methods (Prefixed with `_`)

All methods starting with `_` are internal:

- `_set_response(idx, response)`
- `_get_response(idx)`
- `_delete_response(idx)`
- `_compute_checksum(data)`
- `_validate_single(embedding)`
- `_validate_batch(embeddings)`
- `_encode_single(embedding)`
- `_encode_batch(embeddings)`

### 3.2 Internal Attributes

- `_encoder`
- `_storage` (RustCacheStorage instance)
- `_responses` (Python list)
- `_lock` (RLock)
- `_hits`, `_misses`, `_evictions`

### 3.3 Rust Internals

The following Rust bindings are internal and may change:

- `RustCacheStorage` (use `BinarySemanticCache` instead)
- `HammingSimilarity` (use `BinarySemanticCache.get()` instead)
- `hamming_distance` (internal utility)
- `rust_version` (informational only)

### 3.4 Protocol Classes

- `EncoderProtocol` — Type hint only, not for subclassing.

### 3.5 File Format Internals

The following constants define the v3 file format. They are stable in terms of format compatibility but should not be used directly:

- `V3_HEADER_FILE`, `V3_ENTRIES_FILE`, `V3_RESPONSES_FILE`
- `V3_ENTRY_SIZE` (44 bytes)
- `EPOCH_2020`

---

## 4. Semantic Contracts (Frozen)

The following semantic behaviors are guaranteed and will not change in v1.x:

| Contract | Definition |
| :--- | :--- |
| **Encoder Determinism** | `RustBinaryEncoder(seed=42)` produces identical codes for identical inputs across all v1.x releases. |
| **Threshold Semantics** | `HIT` if and only if `similarity >= threshold`. |
| **Similarity Formula** | `similarity = 1.0 - (hamming_distance / code_bits)` |
| **LRU Eviction** | When `len(cache) >= max_entries`, the least-recently-used entry is evicted. |

---

## 5. Breaking Change Policy

For v1.x releases:

1. **Stable APIs** will not have breaking changes.
2. **Deprecated APIs** will continue to work but emit warnings.
3. **Unstable APIs** may change at any time.

For v2.0.0:

1. **Deprecated APIs** will be removed.
2. **Stable APIs** may have breaking changes (with migration guide).

---

*This document is the authoritative source for API stability in v1.x.*

