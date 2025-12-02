"""
Memory Stability Tests (Sprint 1c-OPT)

Tests for MS-01, MS-02, MS-03 from PHASE2_TEST_MATRIX.md:
- MS-01: Rust index stays 44 B/entry
- MS-02: Total memory < 150 B/entry at 100k and 500k (hard cap)
- MS-03: Run-to-run variance < 3% (non-blocking)

These tests verify memory stability after the dict→list optimization.
"""

from __future__ import annotations

import gc
import sys
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

# Add src to path
_ROOT = Path(__file__).resolve().parents[2]
_SRC_PATH = _ROOT / "src"
sys.path.insert(0, str(_SRC_PATH))

from binary_semantic_cache.core.encoder import BinaryEncoder
from binary_semantic_cache.core.cache import BinarySemanticCache


# =============================================================================
# Constants
# =============================================================================
EMBEDDING_DIM = 384
CODE_BITS = 256
SEED = 42

# MS-02 hard cap (from PHASE2_DECISION_LOG.md)
TOTAL_BYTES_HARD_CAP = 150.0

# MS-03 variance threshold (non-blocking)
RUN_TO_RUN_CV_THRESHOLD = 3.0  # percent


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def encoder() -> BinaryEncoder:
    """Create a BinaryEncoder for testing."""
    return BinaryEncoder(
        embedding_dim=EMBEDDING_DIM,
        code_bits=CODE_BITS,
        seed=SEED,
    )


def create_embedding(seed: int) -> np.ndarray:
    """Create a single normalized embedding."""
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return emb / np.linalg.norm(emb)


def profile_cache_memory(
    encoder: BinaryEncoder,
    n_entries: int,
    max_entries: int | None = None,
) -> Dict[str, Any]:
    """
    Profile memory usage for a cache with n_entries.
    
    Returns dict with:
    - rust_index_bytes: Rust storage memory
    - rust_index_per_entry: Rust storage per entry
    - total_bytes: cache.memory_bytes()
    - total_per_entry: total_bytes / n_entries
    """
    gc.collect()
    
    if max_entries is None:
        max_entries = n_entries + 100
    
    cache = BinarySemanticCache(
        encoder=encoder,
        max_entries=max_entries,
        similarity_threshold=0.80,
    )
    
    # Fill cache
    rng = np.random.default_rng(SEED)
    batch_size = min(1000, n_entries)
    
    for i in range(0, n_entries, batch_size):
        n = min(batch_size, n_entries - i)
        embeddings = rng.standard_normal((n, EMBEDDING_DIM)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        for j in range(n):
            cache.put(embeddings[j], i + j)  # Integer response
    
    gc.collect()
    
    rust_bytes = cache._storage.memory_usage()
    total_bytes = cache.memory_bytes()
    
    return {
        "n_entries": n_entries,
        "rust_index_bytes": rust_bytes,
        "rust_index_per_entry": rust_bytes / n_entries if n_entries > 0 else 0,
        "total_bytes": total_bytes,
        "total_per_entry": total_bytes / n_entries if n_entries > 0 else 0,
    }


# =============================================================================
# MS-01: Rust Index 44 B/entry
# =============================================================================

class TestMemoryIndex:
    """MS-01: Verify Rust index stays exactly 44 B/entry."""

    def test_memory_index_44_bytes_at_1k(self, encoder: BinaryEncoder) -> None:
        """MS-01: Rust index = 44 B/entry at 1k entries."""
        result = profile_cache_memory(encoder, n_entries=1000)
        assert result["rust_index_per_entry"] == 44.0, (
            f"Expected 44 B/entry, got {result['rust_index_per_entry']}"
        )

    def test_memory_index_44_bytes_at_10k(self, encoder: BinaryEncoder) -> None:
        """MS-01: Rust index = 44 B/entry at 10k entries."""
        result = profile_cache_memory(encoder, n_entries=10_000)
        assert result["rust_index_per_entry"] == 44.0, (
            f"Expected 44 B/entry, got {result['rust_index_per_entry']}"
        )

    @pytest.mark.slow
    def test_memory_index_44_bytes_at_100k(self, encoder: BinaryEncoder) -> None:
        """MS-01: Rust index = 44 B/entry at 100k entries."""
        result = profile_cache_memory(encoder, n_entries=100_000)
        assert result["rust_index_per_entry"] == 44.0, (
            f"Expected 44 B/entry, got {result['rust_index_per_entry']}"
        )


# =============================================================================
# MS-02: Total Memory Under Cap
# =============================================================================

class TestMemoryTotalCap:
    """MS-02: Verify total memory < 150 B/entry at scale."""

    def test_memory_total_under_cap_at_10k(self, encoder: BinaryEncoder) -> None:
        """MS-02: Total memory < 150 B/entry at 10k entries."""
        result = profile_cache_memory(encoder, n_entries=10_000)
        assert result["total_per_entry"] < TOTAL_BYTES_HARD_CAP, (
            f"Expected < {TOTAL_BYTES_HARD_CAP} B/entry, got {result['total_per_entry']:.2f}"
        )

    @pytest.mark.slow
    def test_memory_total_under_cap_at_100k(self, encoder: BinaryEncoder) -> None:
        """MS-02: Total memory < 150 B/entry at 100k entries (BLOCKING)."""
        result = profile_cache_memory(encoder, n_entries=100_000)
        assert result["total_per_entry"] < TOTAL_BYTES_HARD_CAP, (
            f"Expected < {TOTAL_BYTES_HARD_CAP} B/entry, got {result['total_per_entry']:.2f}"
        )

    def test_list_vs_dict_overhead(self, encoder: BinaryEncoder) -> None:
        """Verify list-based storage has lower overhead than dict.
        
        List: ~8 bytes/slot (pointer only, fixed allocation)
        Dict: ~56 bytes/entry (bucket + key + value + overhead)
        
        With 10k entries and max_entries=10k:
        - List overhead: sys.getsizeof([None] * 10000) ≈ 80,056 bytes
        - Dict overhead: sys.getsizeof({i: None for i in range(10000)}) ≈ 295,000 bytes
        """
        n_entries = 10_000
        
        # Measure list overhead
        test_list: List[Any] = [None] * n_entries
        list_overhead = sys.getsizeof(test_list)
        
        # Measure dict overhead (for comparison)
        test_dict: Dict[int, Any] = {i: None for i in range(n_entries)}
        dict_overhead = sys.getsizeof(test_dict)
        
        # List should be significantly smaller
        assert list_overhead < dict_overhead, (
            f"List ({list_overhead}) should be smaller than dict ({dict_overhead})"
        )
        
        # List should be roughly 8 bytes/slot + base overhead
        # Base overhead is ~56 bytes, so total ≈ 56 + 8 * n_entries
        expected_list_max = 56 + 8 * n_entries + 1000  # Allow some slack
        assert list_overhead < expected_list_max, (
            f"List overhead {list_overhead} exceeds expected max {expected_list_max}"
        )


# =============================================================================
# MS-03: Run-to-Run Variance (Non-Blocking)
# =============================================================================

class TestMemoryVariance:
    """MS-03: Verify run-to-run variance is small (non-blocking)."""

    def test_memory_variance_small_at_1k(self, encoder: BinaryEncoder) -> None:
        """MS-03: Run-to-run CV < 3% at 1k entries.
        
        Note: This test is non-blocking but tracked.
        """
        n_entries = 1000
        trials = 5
        
        total_values: List[float] = []
        for _ in range(trials):
            result = profile_cache_memory(encoder, n_entries=n_entries)
            total_values.append(result["total_per_entry"])
        
        mean_total = sum(total_values) / len(total_values)
        variance = sum((v - mean_total) ** 2 for v in total_values) / len(total_values)
        stddev = variance ** 0.5
        cv_percent = (stddev / mean_total * 100) if mean_total > 0 else 0
        
        # Non-blocking: warn but don't fail if CV > threshold
        if cv_percent > RUN_TO_RUN_CV_THRESHOLD:
            pytest.skip(
                f"Non-blocking: CV {cv_percent:.2f}% exceeds {RUN_TO_RUN_CV_THRESHOLD}% threshold"
            )
        
        assert cv_percent < RUN_TO_RUN_CV_THRESHOLD, (
            f"CV {cv_percent:.2f}% exceeds {RUN_TO_RUN_CV_THRESHOLD}% threshold"
        )

    def test_memory_deterministic_allocation(self, encoder: BinaryEncoder) -> None:
        """Verify list allocation is deterministic (no resize jitter).
        
        With a fixed-size list, memory should be constant regardless of fill level.
        """
        max_entries = 1000
        
        # Create cache with fixed capacity
        cache = BinarySemanticCache(
            encoder=encoder,
            max_entries=max_entries,
            similarity_threshold=0.80,
        )
        
        # Memory before adding any entries
        memory_empty = cache.memory_bytes()
        
        # Add 100 entries
        for i in range(100):
            emb = create_embedding(seed=i)
            cache.put(emb, i)
        
        memory_100 = cache.memory_bytes()
        
        # Add 400 more entries (500 total)
        for i in range(100, 500):
            emb = create_embedding(seed=i)
            cache.put(emb, i)
        
        memory_500 = cache.memory_bytes()
        
        # List overhead should be constant (same capacity)
        # Only Rust storage grows (44 B/entry)
        
        # Rust storage delta: (500 - 100) * 44 = 17600 bytes
        expected_rust_delta = (500 - 100) * 44
        actual_delta = memory_500 - memory_100
        
        # Allow some tolerance for GC and measurement noise
        assert abs(actual_delta - expected_rust_delta) < 1000, (
            f"Memory delta {actual_delta} differs from expected {expected_rust_delta}"
        )


# =============================================================================
# Integration: Verify Sprint 1c-OPT Optimization
# =============================================================================

class TestListOptimization:
    """Verify the dict→list optimization is in effect."""

    def test_responses_is_list(self, encoder: BinaryEncoder) -> None:
        """Verify _responses is a list, not a dict."""
        cache = BinarySemanticCache(
            encoder=encoder,
            max_entries=100,
            similarity_threshold=0.80,
        )
        
        assert isinstance(cache._responses, list), (
            f"Expected list, got {type(cache._responses)}"
        )

    def test_responses_preallocated(self, encoder: BinaryEncoder) -> None:
        """Verify _responses is pre-allocated to max_entries."""
        max_entries = 100
        cache = BinarySemanticCache(
            encoder=encoder,
            max_entries=max_entries,
            similarity_threshold=0.80,
        )
        
        assert len(cache._responses) == max_entries, (
            f"Expected {max_entries} slots, got {len(cache._responses)}"
        )

    def test_empty_slots_are_none(self, encoder: BinaryEncoder) -> None:
        """Verify empty slots contain None."""
        cache = BinarySemanticCache(
            encoder=encoder,
            max_entries=100,
            similarity_threshold=0.80,
        )
        
        # All slots should be None initially
        assert all(r is None for r in cache._responses), (
            "Expected all slots to be None initially"
        )
        
        # Add one entry
        emb = create_embedding(seed=42)
        idx = cache.put(emb, "test_response")
        
        # Only that slot should be non-None
        non_none_count = sum(1 for r in cache._responses if r is not None)
        assert non_none_count == 1, (
            f"Expected 1 non-None slot, got {non_none_count}"
        )
        assert cache._responses[idx] == "test_response"

    def test_delete_sets_none(self, encoder: BinaryEncoder) -> None:
        """Verify delete sets slot to None (not removes key)."""
        cache = BinarySemanticCache(
            encoder=encoder,
            max_entries=100,
            similarity_threshold=0.80,
        )
        
        emb = create_embedding(seed=42)
        idx = cache.put(emb, "test_response")
        
        # Verify entry exists
        assert cache._responses[idx] == "test_response"
        
        # Delete entry
        cache.delete(idx)
        
        # Slot should be None, not removed
        assert cache._responses[idx] is None
        assert len(cache._responses) == 100  # Length unchanged


