"""
Core module for Binary Semantic Cache.

Contains:
- BinaryEncoder: Converts float embeddings to binary codes
- hamming_similarity: Numba-optimized similarity computation
- BinarySemanticCache: Main cache implementation
- LRUEvictionPolicy: Least Recently Used eviction
"""

from .cache import BinarySemanticCache, CacheEntry, CacheStats
from .encoder import BinaryEncoder
from .eviction import EvictionPolicy, LRUEvictionPolicy
from .similarity import (
    hamming_distance_batch,
    hamming_similarity,
    find_nearest,
    is_numba_available,
)

__all__ = [
    "BinaryEncoder",
    "BinarySemanticCache",
    "CacheEntry",
    "CacheStats",
    "EvictionPolicy",
    "LRUEvictionPolicy",
    "hamming_distance_batch",
    "hamming_similarity",
    "find_nearest",
    "is_numba_available",
]
