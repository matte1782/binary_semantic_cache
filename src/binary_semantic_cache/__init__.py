"""
Binary Semantic Cache

A high-performance semantic caching layer for LLM APIs using binary embeddings.

Phase 2: Rust backend is MANDATORY for performance targets.
See: docs/ARCHITECTURE_PHASE2.md

Author: Binary Semantic Cache Team
Version: 0.2.0
"""

__version__ = "0.2.0"

# =============================================================================
# Rust Backend Import (MANDATORY)
# =============================================================================
# Phase 2 requires Rust backend. If import fails, crash with clear instructions.
try:
    from binary_semantic_cache.binary_semantic_cache_rs import (
        RustBinaryEncoder,
        HammingSimilarity,
        rust_version,
        hamming_distance,
    )
    RUST_BACKEND_AVAILABLE = True
except ImportError as e:
    raise ImportError(
        "Rust extension not available. Build required:\n"
        "  cd src/binary_semantic_cache_rs && cargo build --release && cd ../..\n"
        "  maturin develop --release\n"
        f"Original error: {e}"
    ) from e

# =============================================================================
# Core Components
# =============================================================================
from binary_semantic_cache.core.cache import (
    BinarySemanticCache,
    CacheEntry,
    CacheStats,
    # Error classes (CacheError is base class for all)
    CacheError,
    ChecksumError,
    FormatVersionError,
    CorruptFileError,
    UnsupportedPlatformError,
    # Utility functions
    detect_format_version,
    # Constants
    DEFAULT_MAX_ENTRIES,
    DEFAULT_THRESHOLD,
    DEFAULT_CODE_BITS,
    MMAP_FORMAT_VERSION,
    MMAP_FORMAT_VERSION_V3,
)

# Python encoder retained as test oracle (not for production use)
from binary_semantic_cache.core.encoder import BinaryEncoder as PythonBinaryEncoder

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Version
    "__version__",
    # Rust backend
    "RustBinaryEncoder",
    "HammingSimilarity",
    "rust_version",
    "hamming_distance",
    "RUST_BACKEND_AVAILABLE",
    # Cache
    "BinarySemanticCache",
    "CacheEntry",
    "CacheStats",
    # Exceptions (CacheError is base class)
    "CacheError",
    "ChecksumError",
    "FormatVersionError",
    "CorruptFileError",
    "UnsupportedPlatformError",
    # Utility functions
    "detect_format_version",
    # Constants
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_THRESHOLD",
    "DEFAULT_CODE_BITS",
    "MMAP_FORMAT_VERSION",
    "MMAP_FORMAT_VERSION_V3",
    # Python encoder (test oracle)
    "PythonBinaryEncoder",
]
