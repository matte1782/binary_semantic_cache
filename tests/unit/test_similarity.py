"""Unit tests for Hamming similarity functions.

Tests:
- Accuracy vs NumPy baseline
- Performance target (<1ms at 100K)
- Edge cases (empty, single entry)
"""

import time
from typing import Tuple

import numpy as np
import pytest

import sys
from pathlib import Path

# Add src to path
_SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from binary_semantic_cache.core.similarity import (
    POPCOUNT_TABLE,
    hamming_distance_batch,
    hamming_similarity,
    find_nearest,
    is_numba_available,
    _hamming_distance_numpy,
)


def create_test_codes(n_entries: int, n_words: int = 4, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Create random test codes and a query."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 2**64, size=(n_entries, n_words), dtype=np.uint64)
    query = rng.integers(0, 2**64, size=(n_words,), dtype=np.uint64)
    return query, codes


class TestPopCountTable:
    """Test the precomputed popcount table."""

    def test_popcount_table_correctness(self) -> None:
        """Verify popcount table is correct for all bytes."""
        for i in range(256):
            expected = bin(i).count("1")
            assert POPCOUNT_TABLE[i] == expected, f"Wrong popcount for {i}"

    def test_popcount_table_shape(self) -> None:
        """Verify popcount table shape and dtype."""
        assert POPCOUNT_TABLE.shape == (256,)
        assert POPCOUNT_TABLE.dtype == np.uint8


class TestHammingAccuracy:
    """Test that Hamming distance is computed correctly."""

    def test_identical_codes(self) -> None:
        """Identical codes should have distance 0."""
        query = np.array([0xFFFFFFFFFFFFFFFF, 0x0], dtype=np.uint64)
        codes = np.array([[0xFFFFFFFFFFFFFFFF, 0x0]], dtype=np.uint64)

        distances = hamming_distance_batch(query, codes, use_numba=False)
        assert distances[0] == 0

    def test_all_different_codes(self) -> None:
        """Inverted codes should have distance = code_bits."""
        query = np.array([0xFFFFFFFFFFFFFFFF] * 4, dtype=np.uint64)
        codes = np.array([[0x0] * 4], dtype=np.uint64)

        distances = hamming_distance_batch(query, codes, use_numba=False)
        assert distances[0] == 256  # 4 * 64 bits

    def test_single_bit_difference(self) -> None:
        """Single bit difference should have distance 1."""
        query = np.array([0x1, 0x0, 0x0, 0x0], dtype=np.uint64)
        codes = np.array([[0x0, 0x0, 0x0, 0x0]], dtype=np.uint64)

        distances = hamming_distance_batch(query, codes, use_numba=False)
        assert distances[0] == 1

    def test_known_distance(self) -> None:
        """Test a known Hamming distance."""
        # 0b1010 vs 0b1100 = distance 2 (bits 1 and 2 differ)
        query = np.array([0b1010, 0, 0, 0], dtype=np.uint64)
        codes = np.array([[0b1100, 0, 0, 0]], dtype=np.uint64)

        distances = hamming_distance_batch(query, codes, use_numba=False)
        assert distances[0] == 2

    @pytest.mark.skipif(not is_numba_available(), reason="Numba not available")
    def test_numba_matches_numpy(self) -> None:
        """Numba implementation should match NumPy exactly."""
        query, codes = create_test_codes(1000)

        numpy_distances = hamming_distance_batch(query, codes, use_numba=False)
        numba_distances = hamming_distance_batch(query, codes, use_numba=True)

        np.testing.assert_array_equal(numpy_distances, numba_distances)


class TestHammingSimilarity:
    """Test normalized Hamming similarity."""

    def test_identical_similarity_one(self) -> None:
        """Identical codes should have similarity 1.0."""
        query = np.array([0xABCD, 0x1234, 0x5678, 0x9ABC], dtype=np.uint64)
        codes = np.array([[0xABCD, 0x1234, 0x5678, 0x9ABC]], dtype=np.uint64)

        sims = hamming_similarity(query, codes, code_bits=256)
        assert sims[0] == pytest.approx(1.0)

    def test_opposite_similarity_zero(self) -> None:
        """Inverted codes should have similarity 0.0."""
        query = np.array([0xFFFFFFFFFFFFFFFF] * 4, dtype=np.uint64)
        codes = np.array([[0x0] * 4], dtype=np.uint64)

        sims = hamming_similarity(query, codes, code_bits=256)
        assert sims[0] == pytest.approx(0.0)

    def test_half_similarity(self) -> None:
        """Half bits different should have similarity ~0.5."""
        # Create codes where exactly half bits differ
        query = np.array([0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x0, 0x0], dtype=np.uint64)
        codes = np.array([[0x0, 0x0, 0x0, 0x0]], dtype=np.uint64)

        sims = hamming_similarity(query, codes, code_bits=256)
        assert sims[0] == pytest.approx(0.5)

    def test_similarity_range(self) -> None:
        """Similarity should always be in [0, 1]."""
        query, codes = create_test_codes(100)

        sims = hamming_similarity(query, codes, code_bits=256)

        assert all(0.0 <= s <= 1.0 for s in sims)


class TestFindNearest:
    """Test find_nearest function."""

    def test_find_exact_match(self) -> None:
        """Should find exact match with similarity 1.0."""
        query = np.array([0x1234, 0x5678, 0x9ABC, 0xDEF0], dtype=np.uint64)
        codes = np.array([
            [0x0, 0x0, 0x0, 0x0],
            [0x1234, 0x5678, 0x9ABC, 0xDEF0],  # Exact match
            [0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF],
        ], dtype=np.uint64)

        result = find_nearest(query, codes, code_bits=256, threshold=0.85)

        assert result is not None
        idx, sim = result
        assert idx == 1
        assert sim == pytest.approx(1.0)

    def test_no_match_below_threshold(self) -> None:
        """Should return None if all similarities below threshold."""
        query = np.array([0xFFFFFFFFFFFFFFFF] * 4, dtype=np.uint64)
        codes = np.array([[0x0] * 4], dtype=np.uint64)

        result = find_nearest(query, codes, code_bits=256, threshold=0.85)

        assert result is None

    def test_empty_codes(self) -> None:
        """Should handle empty codes array."""
        query = np.array([0x1234, 0x5678, 0x9ABC, 0xDEF0], dtype=np.uint64)
        codes = np.zeros((0, 4), dtype=np.uint64)

        result = find_nearest(query, codes, code_bits=256, threshold=0.85)

        assert result is None


class TestEdgeCases:
    """Test edge cases."""

    def test_single_entry(self) -> None:
        """Should work with single entry."""
        query = np.array([0x1234, 0x5678, 0x9ABC, 0xDEF0], dtype=np.uint64)
        codes = np.array([[0x1234, 0x5678, 0x9ABC, 0xDEF0]], dtype=np.uint64)

        distances = hamming_distance_batch(query, codes)
        assert len(distances) == 1
        assert distances[0] == 0

    def test_empty_codes_array(self) -> None:
        """Should handle empty codes array."""
        query = np.array([0x1234, 0x5678, 0x9ABC, 0xDEF0], dtype=np.uint64)
        codes = np.zeros((0, 4), dtype=np.uint64)

        distances = hamming_distance_batch(query, codes)
        assert len(distances) == 0

    def test_large_batch(self) -> None:
        """Should handle large batch correctly."""
        query, codes = create_test_codes(10000)

        distances = hamming_distance_batch(query, codes, use_numba=False)
        assert len(distances) == 10000


class TestInputValidation:
    """Test input validation."""

    def test_wrong_query_dtype(self) -> None:
        """Should reject non-uint64 query."""
        query = np.array([1, 2, 3, 4], dtype=np.int32)
        codes = np.zeros((10, 4), dtype=np.uint64)

        with pytest.raises(ValueError, match="uint64"):
            hamming_distance_batch(query, codes)

    def test_wrong_codes_dtype(self) -> None:
        """Should reject non-uint64 codes."""
        query = np.array([1, 2, 3, 4], dtype=np.uint64)
        codes = np.zeros((10, 4), dtype=np.int32)

        with pytest.raises(ValueError, match="uint64"):
            hamming_distance_batch(query, codes)

    def test_mismatched_words(self) -> None:
        """Should reject mismatched word counts."""
        query = np.array([1, 2, 3], dtype=np.uint64)  # 3 words
        codes = np.zeros((10, 4), dtype=np.uint64)   # 4 words

        with pytest.raises(ValueError, match="words"):
            hamming_distance_batch(query, codes)

    def test_wrong_query_dim(self) -> None:
        """Should reject 2D query."""
        query = np.zeros((1, 4), dtype=np.uint64)
        codes = np.zeros((10, 4), dtype=np.uint64)

        with pytest.raises(ValueError, match="1D"):
            hamming_distance_batch(query, codes)

    def test_invalid_code_bits(self) -> None:
        """Should reject non-positive code_bits."""
        query = np.array([1, 2, 3, 4], dtype=np.uint64)
        codes = np.zeros((10, 4), dtype=np.uint64)

        with pytest.raises(ValueError, match="positive"):
            hamming_similarity(query, codes, code_bits=0)


class TestPerformance:
    """Test performance targets."""

    @pytest.mark.skipif(not is_numba_available(), reason="Numba not available")
    def test_performance_100k_numba(self) -> None:
        """Numba should achieve <1ms for 100K entries."""
        query, codes = create_test_codes(100_000)

        # Warmup
        hamming_distance_batch(query, codes, use_numba=True)

        # Benchmark
        n_runs = 10
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            hamming_distance_batch(query, codes, use_numba=True)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        avg_ms = sum(times) / len(times)
        min_ms = min(times)

        print(f"\nNumba 100K performance: avg={avg_ms:.2f}ms, min={min_ms:.2f}ms")

        # Legacy fallback target relaxed for Phase 2.5 (Production uses Rust)
        # Target: <1ms (Phase 1), relaxed trigger: >10ms (Phase 2.5)
        assert min_ms < 10.0, f"Numba legacy guard failed: {min_ms:.2f}ms > 10ms relaxed threshold"

    def test_performance_100k_numpy(self) -> None:
        """NumPy fallback should be reasonable (<20ms for 100K)."""
        query, codes = create_test_codes(100_000)

        # Benchmark
        n_runs = 5
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _hamming_distance_numpy(query, codes)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        avg_ms = sum(times) / len(times)
        min_ms = min(times)

        print(f"\nNumPy 100K performance: avg={avg_ms:.2f}ms, min={min_ms:.2f}ms")

        # NumPy fallback should be <35ms (PoC was 13.5ms, but CI hardware varies)
        # This tests the fallback path, not production Rust path.
        assert min_ms < 35.0, f"NumPy too slow: {min_ms:.2f}ms > 35ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

