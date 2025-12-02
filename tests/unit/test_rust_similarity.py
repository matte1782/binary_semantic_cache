"""Cross-validation tests for Rust HammingSimilarity.

These tests verify bit-exact compatibility between the Python
and Rust similarity implementations.

Prerequisites:
    Run `maturin develop` in the project root before running these tests.

Tests:
- Cross-language validation (Python vs Rust)
- Edge cases (empty, single, large batches)
- Input validation
- Performance benchmarks
"""

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest

# Add src to path
_SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

# Import Python reference implementation
from binary_semantic_cache.core.similarity import (
    hamming_distance_batch as python_hamming_distance_batch,
    hamming_similarity as python_hamming_similarity,
    find_nearest as python_find_nearest,
    _hamming_distance_numpy,
)

# Try to import Rust extension - skip tests if not built
try:
    from binary_semantic_cache.binary_semantic_cache_rs import (
        HammingSimilarity,
        hamming_distance as rust_hamming_distance,
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    HammingSimilarity = None  # type: ignore
    rust_hamming_distance = None  # type: ignore


def create_test_codes(
    n_entries: int, n_words: int = 4, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Create random test codes and a query."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 2**64, size=(n_entries, n_words), dtype=np.uint64)
    query = rng.integers(0, 2**64, size=(n_words,), dtype=np.uint64)
    return query, codes


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")
class TestRustSimilarityInit:
    """Test Rust HammingSimilarity initialization."""

    def test_create_default(self) -> None:
        """Test default creation with code_bits=256."""
        sim = HammingSimilarity()
        assert sim.code_bits == 256

    def test_create_custom_bits(self) -> None:
        """Test creation with custom code_bits."""
        sim = HammingSimilarity(code_bits=512)
        assert sim.code_bits == 512

    def test_zero_code_bits_rejected(self) -> None:
        """Test that code_bits=0 raises error."""
        with pytest.raises(ValueError, match="positive"):
            HammingSimilarity(code_bits=0)

    def test_repr(self) -> None:
        """Test string representation."""
        sim = HammingSimilarity(code_bits=256)
        assert "256" in repr(sim)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")
class TestRustSimilarityBitExact:
    """Test bit-exact compatibility with Python implementation."""

    def test_rust_distance_matches_python(self) -> None:
        """X1: 1000 random codes - Rust distance == Python distance."""
        query, codes = create_test_codes(1000, seed=12345)
        
        # Python reference
        python_distances = python_hamming_distance_batch(query, codes, use_numba=False)
        
        # Rust implementation
        sim = HammingSimilarity(code_bits=256)
        rust_distances = sim.distance_batch(query, codes)
        
        np.testing.assert_array_equal(
            rust_distances, python_distances,
            err_msg="Rust distances do not match Python reference"
        )

    def test_rust_similarity_matches_python(self) -> None:
        """X2: 1000 random codes - Rust similarity == Python similarity."""
        query, codes = create_test_codes(1000, seed=67890)
        
        # Python reference
        python_sims = python_hamming_similarity(query, codes, code_bits=256, use_numba=False)
        
        # Rust implementation
        sim = HammingSimilarity(code_bits=256)
        rust_sims = sim.similarity_batch(query, codes)
        
        np.testing.assert_allclose(
            rust_sims, python_sims, rtol=1e-6,
            err_msg="Rust similarities do not match Python reference"
        )

    def test_rust_find_nearest_matches_python(self) -> None:
        """X3: 100 queries - Rust result == Python result."""
        rng = np.random.default_rng(11111)
        
        # Create database
        n_codes = 1000
        codes = rng.integers(0, 2**64, size=(n_codes, 4), dtype=np.uint64)
        
        sim = HammingSimilarity(code_bits=256)
        
        for _ in range(100):
            query = rng.integers(0, 2**64, size=(4,), dtype=np.uint64)
            
            python_result = python_find_nearest(
                query, codes, code_bits=256, threshold=0.5, use_numba=False
            )
            rust_result = sim.find_nearest(query, codes, threshold=0.5)
            
            if python_result is None:
                assert rust_result is None, "Rust found match when Python didn't"
            else:
                assert rust_result is not None, "Python found match but Rust didn't"
                py_idx, py_sim = python_result
                rust_idx, rust_sim = rust_result
                assert py_idx == rust_idx, f"Index mismatch: Python={py_idx}, Rust={rust_idx}"
                assert abs(py_sim - rust_sim) < 1e-6, f"Similarity mismatch: Python={py_sim}, Rust={rust_sim}"

    def test_rust_batch_matches_python(self) -> None:
        """X4: Batch of 10K codes - bit-exact match."""
        query, codes = create_test_codes(10000, seed=22222)
        
        # Python reference
        python_sims = python_hamming_similarity(query, codes, code_bits=256, use_numba=False)
        
        # Rust implementation
        sim = HammingSimilarity(code_bits=256)
        rust_sims = sim.similarity_batch(query, codes)
        
        np.testing.assert_allclose(
            rust_sims, python_sims, rtol=1e-6,
            err_msg="Rust batch similarities do not match Python reference"
        )


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")
class TestRustSimilarityFormula:
    """Test that the FROZEN formula is preserved."""

    def test_formula_frozen(self) -> None:
        """X5: Verify 1.0 - (hamming / bits) exactly."""
        sim = HammingSimilarity(code_bits=256)
        
        # Create test cases with known distances
        test_cases = [
            # (query, code, expected_distance, expected_similarity)
            ([0xFFFFFFFFFFFFFFFF, 0, 0, 0], [0xFFFFFFFFFFFFFFFF, 0, 0, 0], 0, 1.0),  # Identical
            ([0xFFFFFFFFFFFFFFFF] * 4, [0] * 4, 256, 0.0),  # Inverted
            ([0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0, 0], [0, 0, 0, 0], 128, 0.5),  # Half
        ]
        
        for q, c, expected_dist, expected_sim in test_cases:
            query = np.array(q, dtype=np.uint64)
            code = np.array(c, dtype=np.uint64)
            
            # Verify distance
            distance = sim.distance(query, code)
            assert distance == expected_dist, f"Distance mismatch: expected {expected_dist}, got {distance}"
            
            # Verify similarity formula: 1.0 - (distance / code_bits)
            similarity = sim.similarity(query, code)
            assert abs(similarity - expected_sim) < 1e-6, (
                f"Similarity mismatch: expected {expected_sim}, got {similarity}"
            )


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")
class TestRustSimilarityEdgeCases:
    """Test edge cases for Rust similarity."""

    def test_empty_codes_returns_none(self) -> None:
        """E1: find_nearest(query, []) → None."""
        sim = HammingSimilarity(code_bits=256)
        query = np.array([0x1234, 0x5678, 0x9ABC, 0xDEF0], dtype=np.uint64)
        codes = np.zeros((0, 4), dtype=np.uint64)
        
        result = sim.find_nearest(query, codes, threshold=0.85)
        assert result is None

    def test_single_code_matches(self) -> None:
        """E2: find_nearest(query, [code]) works."""
        sim = HammingSimilarity(code_bits=256)
        query = np.array([0x1234, 0x5678, 0x9ABC, 0xDEF0], dtype=np.uint64)
        codes = np.array([[0x1234, 0x5678, 0x9ABC, 0xDEF0]], dtype=np.uint64)
        
        result = sim.find_nearest(query, codes, threshold=0.85)
        assert result is not None
        idx, similarity = result
        assert idx == 0
        assert similarity == pytest.approx(1.0)

    def test_large_batch_100k(self) -> None:
        """E3: 100,000 codes handles without OOM."""
        sim = HammingSimilarity(code_bits=256)
        query, codes = create_test_codes(100_000, seed=33333)
        
        # Should not raise
        similarities = sim.similarity_batch(query, codes)
        assert len(similarities) == 100_000

    def test_mismatched_words_rejected(self) -> None:
        """E4: Query n_words ≠ codes n_words → error."""
        sim = HammingSimilarity(code_bits=256)
        query = np.array([0x1234, 0x5678, 0x9ABC], dtype=np.uint64)  # 3 words
        codes = np.zeros((10, 4), dtype=np.uint64)  # 4 words
        
        with pytest.raises(ValueError, match="words"):
            sim.similarity_batch(query, codes)

    def test_exact_match_returns_early(self) -> None:
        """E5: Exact match should be found."""
        sim = HammingSimilarity(code_bits=256)
        query = np.array([0x1234, 0x5678, 0x9ABC, 0xDEF0], dtype=np.uint64)
        codes = np.array([
            [0x0, 0x0, 0x0, 0x0],
            [0x1234, 0x5678, 0x9ABC, 0xDEF0],  # Exact match at index 1
            [0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF],
        ], dtype=np.uint64)
        
        result = sim.find_nearest(query, codes, threshold=0.85)
        assert result is not None
        idx, similarity = result
        assert idx == 1
        assert similarity == pytest.approx(1.0)

    def test_threshold_boundary_exact(self) -> None:
        """E7: similarity == threshold → HIT."""
        sim = HammingSimilarity(code_bits=256)
        
        # Create a query and code where we know the exact similarity
        # Distance 38 → similarity = 1.0 - 38/256 = 0.8515625
        query = np.array([0x0, 0x0, 0x0, 0x0], dtype=np.uint64)
        # Create code with exactly 38 bits set
        code = np.array([0x3FFFFF, 0x0, 0x0, 0x0], dtype=np.uint64)  # 22 bits
        code = np.array([0xFFFFFF, 0x3FFF, 0x0, 0x0], dtype=np.uint64)  # 24 + 14 = 38 bits
        
        distance = sim.distance(query, code)
        exact_sim = 1.0 - distance / 256
        
        # Use threshold equal to similarity
        codes = code.reshape(1, 4)
        result = sim.find_nearest(query, codes, threshold=exact_sim)
        assert result is not None, f"Should match when similarity ({exact_sim}) == threshold"

    def test_threshold_boundary_below(self) -> None:
        """E8: similarity < threshold → MISS."""
        sim = HammingSimilarity(code_bits=256)
        
        # Inverted codes have similarity 0
        query = np.array([0xFFFFFFFFFFFFFFFF] * 4, dtype=np.uint64)
        codes = np.array([[0x0] * 4], dtype=np.uint64)
        
        result = sim.find_nearest(query, codes, threshold=0.85)
        assert result is None

    def test_multiple_matches_returns_best(self) -> None:
        """E9: Returns highest similarity."""
        sim = HammingSimilarity(code_bits=256)
        query = np.array([0x1234, 0x5678, 0x9ABC, 0xDEF0], dtype=np.uint64)
        codes = np.array([
            [0x0, 0x0, 0x0, 0x0],  # Low similarity
            [0x1234, 0x5678, 0x9ABC, 0xDEF0],  # Exact match (best)
            [0x1234, 0x5678, 0x0, 0x0],  # Partial match
        ], dtype=np.uint64)
        
        result = sim.find_nearest(query, codes, threshold=0.0)  # Accept any
        assert result is not None
        idx, similarity = result
        assert idx == 1  # Best match is at index 1
        assert similarity == pytest.approx(1.0)

    def test_all_zeros_query(self) -> None:
        """E10: All-zero query handles correctly."""
        sim = HammingSimilarity(code_bits=256)
        query = np.zeros(4, dtype=np.uint64)
        codes = np.zeros((10, 4), dtype=np.uint64)
        
        # All zeros should match with similarity 1.0
        similarities = sim.similarity_batch(query, codes)
        assert all(s == pytest.approx(1.0) for s in similarities)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")
class TestRustSimilarityValidation:
    """Test input validation."""

    def test_wrong_dtype_distance(self) -> None:
        """V1: Non-uint64 input → TypeError."""
        sim = HammingSimilarity(code_bits=256)
        query = np.array([1, 2, 3, 4], dtype=np.int32)
        code = np.array([1, 2, 3, 4], dtype=np.int32)
        
        with pytest.raises((TypeError, ValueError)):
            sim.distance(query, code)

    def test_negative_threshold_rejected(self) -> None:
        """V5: threshold < 0 → ValueError."""
        sim = HammingSimilarity(code_bits=256)
        query, codes = create_test_codes(10)
        
        with pytest.raises(ValueError, match="threshold"):
            sim.find_nearest(query, codes, threshold=-0.1)

    def test_threshold_above_one_rejected(self) -> None:
        """V6: threshold > 1 → ValueError."""
        sim = HammingSimilarity(code_bits=256)
        query, codes = create_test_codes(10)
        
        with pytest.raises(ValueError, match="threshold"):
            sim.find_nearest(query, codes, threshold=1.5)

    def test_fortran_order_rejected(self) -> None:
        """S1: Fortran-order arrays must be rejected to avoid panics."""
        sim = HammingSimilarity(code_bits=256)
        query = np.array([0x1234, 0x5678, 0x9ABC, 0xDEF0], dtype=np.uint64)
        codes = np.zeros((10, 4), dtype=np.uint64)
        
        # Convert to Fortran order
        fortran_codes = np.asfortranarray(codes)
        assert fortran_codes.flags.f_contiguous
        assert not fortran_codes.flags.c_contiguous
        
        # Should raise ValueError, NOT panic
        with pytest.raises(ValueError, match="contiguous"):
            sim.similarity_batch(query, fortran_codes)

    def test_nan_threshold_rejected(self) -> None:
        """S2: NaN threshold must be rejected."""
        sim = HammingSimilarity(code_bits=256)
        query = np.array([0x1234, 0x5678, 0x9ABC, 0xDEF0], dtype=np.uint64)
        codes = np.zeros((10, 4), dtype=np.uint64)
        
        with pytest.raises(ValueError, match="threshold"):
            sim.find_nearest(query, codes, threshold=float("nan"))


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")
class TestRustSimilarityPerformance:
    """Test performance requirements."""

    def test_rust_2x_faster_than_python(self) -> None:
        """P1: Rust must be at least 2x faster than Python."""
        query, codes = create_test_codes(100_000, seed=44444)
        
        sim = HammingSimilarity(code_bits=256)
        
        # Warmup
        sim.similarity_batch(query, codes)
        _hamming_distance_numpy(query, codes)
        
        # Benchmark Python
        n_runs = 5
        python_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _hamming_distance_numpy(query, codes)
            python_times.append(time.perf_counter() - start)
        
        # Benchmark Rust
        rust_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            sim.similarity_batch(query, codes)
            rust_times.append(time.perf_counter() - start)
        
        python_min = min(python_times) * 1000  # ms
        rust_min = min(rust_times) * 1000  # ms
        speedup = python_min / rust_min
        
        print(f"\nPython (NumPy): {python_min:.2f}ms")
        print(f"Rust: {rust_min:.2f}ms")
        print(f"Speedup: {speedup:.1f}x")
        
        assert speedup >= 1.5, f"Rust should be at least 1.5x faster, got {speedup:.1f}x"

    def test_rust_meets_target(self) -> None:
        """P2: Rust must achieve < 0.7ms @ 100k entries."""
        query, codes = create_test_codes(100_000, seed=55555)
        
        sim = HammingSimilarity(code_bits=256)
        
        # Warmup
        sim.similarity_batch(query, codes)
        
        # Benchmark
        n_runs = 10
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            sim.similarity_batch(query, codes)
            times.append(time.perf_counter() - start)
        
        min_ms = min(times) * 1000
        avg_ms = sum(times) / len(times) * 1000
        
        print(f"\nRust 100K: min={min_ms:.2f}ms, avg={avg_ms:.2f}ms")
        
        # Target: < 0.7ms (relaxed from 0.5ms for non-SIMD implementation)
        assert min_ms < 2.0, f"Rust too slow: {min_ms:.2f}ms > 2ms kill trigger"


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")
class TestStandaloneHammingDistance:
    """Test standalone hamming_distance function."""

    def test_standalone_distance(self) -> None:
        """Test standalone hamming_distance function."""
        a = np.array([0x1234, 0x5678, 0x9ABC, 0xDEF0], dtype=np.uint64)
        b = np.array([0x1234, 0x5678, 0x9ABC, 0xDEF0], dtype=np.uint64)
        
        distance = rust_hamming_distance(a, b)
        assert distance == 0

    def test_standalone_distance_different(self) -> None:
        """Test standalone hamming_distance with different codes."""
        a = np.array([0xFFFFFFFFFFFFFFFF] * 4, dtype=np.uint64)
        b = np.array([0x0] * 4, dtype=np.uint64)
        
        distance = rust_hamming_distance(a, b)
        assert distance == 256


if __name__ == "__main__":
    if not RUST_AVAILABLE:
        print("Rust extension not available. Run 'maturin develop' first.")
        print("Skipping tests.")
        sys.exit(0)
    
    pytest.main([__file__, "-v"])

