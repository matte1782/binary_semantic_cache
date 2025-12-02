"""
Hamming Similarity Module

Computes normalized Hamming similarity between binary codes.
Optimized with Numba JIT for sub-millisecond performance at 100K entries.

Performance Targets (from PHASE_1_PREFLIGHT_CHECK.md):
- Lookup latency: <1ms for 100K entries (Numba)
- Kill trigger: >2ms with Numba

See: docs/DECISION_LOG_v1.md (D3, D5)
"""

from __future__ import annotations

import logging
from typing import Final, Optional, Tuple

import numpy as np

try:
    import numba
    from numba import prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    prange = range  # type: ignore

logger = logging.getLogger(__name__)

# Precomputed popcount table for bytes (0-255)
POPCOUNT_TABLE: Final[np.ndarray] = np.array(
    [bin(i).count("1") for i in range(256)], dtype=np.uint8
)


def _hamming_distance_numpy(query: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """
    NumPy baseline implementation of Hamming distance.

    Used as fallback when Numba is not available.

    Args:
        query: Single binary code as uint64 array of shape (n_words,).
        codes: Database of binary codes as uint64 array of shape (n_entries, n_words).

    Returns:
        Hamming distances as int array of shape (n_entries,).
    """
    # XOR query with all codes
    xored = np.bitwise_xor(codes, query)

    # Count bits using precomputed table
    # Reshape to view as bytes
    bytes_view = xored.view(np.uint8)

    # Lookup popcount for each byte and sum
    distances = POPCOUNT_TABLE[bytes_view].reshape(codes.shape[0], -1).sum(axis=1)

    return distances.astype(np.int32)


if NUMBA_AVAILABLE:

    @numba.jit(nopython=True, parallel=True, cache=True)
    def _hamming_distance_numba(query: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """
        Numba-optimized Hamming distance computation.

        Uses parallel execution for maximum throughput.

        Args:
            query: Single binary code as uint64 array of shape (n_words,).
            codes: Database of binary codes as uint64 array of shape (n_entries, n_words).

        Returns:
            Hamming distances as int32 array of shape (n_entries,).
        """
        n_entries = codes.shape[0]
        n_words = codes.shape[1]
        distances = np.zeros(n_entries, dtype=np.int32)

        for i in prange(n_entries):
            dist = np.int32(0)
            for w in range(n_words):
                # Explicit uint64 cast to avoid Numba type inference issues
                xored = np.uint64(codes[i, w]) ^ np.uint64(query[w])
                # Count bits using Brian Kernighan's algorithm
                while xored != np.uint64(0):
                    xored = xored & (xored - np.uint64(1))
                    dist += np.int32(1)
            distances[i] = dist

        return distances

    # Warm up Numba JIT
    def _warmup_numba() -> None:
        """Warm up Numba JIT compilation."""
        dummy_query = np.zeros(4, dtype=np.uint64)
        dummy_codes = np.zeros((10, 4), dtype=np.uint64)
        _hamming_distance_numba(dummy_query, dummy_codes)

    try:
        _warmup_numba()
        logger.debug("Numba JIT warmed up successfully")
    except Exception as e:
        logger.warning("Numba warmup failed: %s", e)
        NUMBA_AVAILABLE = False


def hamming_distance_batch(
    query: np.ndarray,
    codes: np.ndarray,
    use_numba: Optional[bool] = None,
) -> np.ndarray:
    """
    Compute Hamming distance between query and all codes.

    Args:
        query: Single binary code as uint64 array of shape (n_words,).
        codes: Database of binary codes as uint64 array of shape (n_entries, n_words).
        use_numba: Force Numba (True) or NumPy (False). None = auto-detect.

    Returns:
        Hamming distances as int array of shape (n_entries,).

    Raises:
        ValueError: If inputs have incompatible shapes or dtypes.
    """
    # Validate inputs
    if query.dtype != np.uint64:
        raise ValueError(f"query must be uint64, got {query.dtype}")
    if codes.dtype != np.uint64:
        raise ValueError(f"codes must be uint64, got {codes.dtype}")
    if query.ndim != 1:
        raise ValueError(f"query must be 1D, got shape {query.shape}")
    if codes.ndim != 2:
        raise ValueError(f"codes must be 2D, got shape {codes.shape}")
    if query.shape[0] != codes.shape[1]:
        raise ValueError(
            f"query has {query.shape[0]} words but codes have {codes.shape[1]}"
        )

    # Handle empty codes
    if codes.shape[0] == 0:
        return np.array([], dtype=np.int32)

    # Choose implementation
    if use_numba is None:
        use_numba = NUMBA_AVAILABLE

    if use_numba and NUMBA_AVAILABLE:
        return _hamming_distance_numba(query, codes)
    else:
        if use_numba and not NUMBA_AVAILABLE:
            logger.warning("Numba requested but not available, falling back to NumPy")
        return _hamming_distance_numpy(query, codes)


def hamming_similarity(
    query: np.ndarray,
    codes: np.ndarray,
    code_bits: int = 256,
    use_numba: Optional[bool] = None,
) -> np.ndarray:
    """
    Compute normalized Hamming similarity between query and all codes.

    Similarity is computed as: 1.0 - (hamming_distance / code_bits)

    Args:
        query: Single binary code as uint64 array of shape (n_words,).
        codes: Database of binary codes as uint64 array of shape (n_entries, n_words).
        code_bits: Total number of bits in each code (default: 256).
        use_numba: Force Numba (True) or NumPy (False). None = auto-detect.

    Returns:
        Similarity scores as float array of shape (n_entries,).
        Values range from 0.0 (completely different) to 1.0 (identical).

    Example:
        >>> similarities = hamming_similarity(query_code, all_codes)
        >>> best_match_idx = np.argmax(similarities)
        >>> if similarities[best_match_idx] >= 0.85:
        ...     print("Cache hit!")
    """
    if code_bits <= 0:
        raise ValueError(f"code_bits must be positive, got {code_bits}")

    distances = hamming_distance_batch(query, codes, use_numba=use_numba)
    similarities = 1.0 - (distances.astype(np.float64) / code_bits)

    return similarities.astype(np.float32)


def find_nearest(
    query: np.ndarray,
    codes: np.ndarray,
    code_bits: int = 256,
    threshold: float = 0.85,
    use_numba: Optional[bool] = None,
) -> Optional[Tuple[int, float]]:
    """
    Find the nearest code above similarity threshold.

    Args:
        query: Single binary code.
        codes: Database of binary codes.
        code_bits: Total number of bits.
        threshold: Minimum similarity for a match.
        use_numba: Force Numba (True) or NumPy (False). None = auto-detect.

    Returns:
        Tuple of (index, similarity) if found, None if no match above threshold.
    """
    if codes.shape[0] == 0:
        return None

    similarities = hamming_similarity(query, codes, code_bits, use_numba=use_numba)
    best_idx = int(np.argmax(similarities))
    best_sim = float(similarities[best_idx])

    if best_sim >= threshold:
        return (best_idx, best_sim)
    return None


def is_numba_available() -> bool:
    """Check if Numba is available and working."""
    return NUMBA_AVAILABLE
