"""
Binary Semantic Cache Module

Main cache implementation that combines:
- Binary encoding (RustBinaryEncoder - Rust backend)
- Hamming similarity search (RustCacheStorage - Rust backend)
- LRU eviction (RustCacheStorage - Rust backend)

Thread-safe for concurrent access.
Memory-optimized: Rust-backed storage (44 bytes/entry for codes+metadata).

Phase 2.5 Sprint 1b: Uses RustCacheStorage for codes/metadata storage.
Phase 2.5 Sprint 1c-OPT: Responses stored in fixed-size Python list (pre-allocated)
to eliminate dict resize jitter and stabilize memory overhead.

See: docs/DECISION_LOG_v1.md (D4, D6)
See: docs/ARCHITECTURE_PHASE2.md
See: docs/phase2_specs/SPRINT1B_INTEGRATION_SPEC.md
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import shutil
import struct
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Final, List, NamedTuple, Optional, Protocol, Tuple, Union

import numpy as np


# =============================================================================
# Mmap Persistence Constants
# =============================================================================
# v2 format (deprecated, retained for backward compatibility)
MMAP_FORMAT_VERSION: Final[int] = 2
MMAP_HEADER_FILE: Final[str] = "header.json"
MMAP_CODES_FILE: Final[str] = "codes.bin"
MMAP_CREATED_AT_FILE: Final[str] = "created_at.bin"
MMAP_ACCESS_COUNT_FILE: Final[str] = "access_count.bin"
MMAP_RESPONSES_FILE: Final[str] = "responses.pkl"

# v3 format (Sprint 2a - directory-based with 44-byte packed entries)
MMAP_FORMAT_VERSION_V3: Final[int] = 3
V3_HEADER_FILE: Final[str] = "header.json"
V3_ENTRIES_FILE: Final[str] = "entries.bin"
V3_RESPONSES_FILE: Final[str] = "responses.pkl"
V3_ENTRY_SIZE: Final[int] = 44  # 32 (codes) + 4 (created_at) + 4 (last_accessed) + 4 (access_count)
# 2020-01-01 00:00:00 UTC epoch for timestamp compression (matches Rust)
EPOCH_2020: Final[int] = 1577836800


# =============================================================================
# Error Hierarchy (Sprint 2a)
# =============================================================================

class CacheError(Exception):
    """Base class for all cache-related errors."""
    pass


class ChecksumError(CacheError):
    """Raised when cache file integrity check fails (SHA-256 mismatch)."""
    pass


class FormatVersionError(CacheError):
    """Raised when cache file format version is unsupported (e.g., version != 3)."""
    pass


class CorruptFileError(CacheError):
    """Raised when cache file is corrupted (missing files, invalid JSON, truncated data, entry_size != 44)."""
    pass


class UnsupportedPlatformError(CacheError):
    """Raised when platform is incompatible (e.g., big-endian system loading little-endian data)."""
    pass


def detect_format_version(path: str) -> int:
    """
    Detect the persistence format version of a cache file/directory.
    
    Args:
        path: Path to the cache file (.npz) or directory (v3).
    
    Returns:
        2: v2 format (.npz file)
        3: v3 format (directory with header.json)
    
    Raises:
        ValueError: If format cannot be determined.
        FileNotFoundError: If path does not exist.
    """
    p = Path(path)
    
    if not p.exists():
        raise FileNotFoundError(f"Cache path not found: {path}")
    
    # v3 format: directory with header.json
    if p.is_dir():
        header_path = p / V3_HEADER_FILE
        if header_path.exists():
            try:
                with open(header_path, "r", encoding="utf-8") as f:
                    header = json.load(f)
                version = header.get("version", 0)
                if version == 3:
                    return 3
            except (json.JSONDecodeError, IOError):
                pass
        # Check for v2 directory format (has codes.bin but no version 3 header)
        if (p / MMAP_CODES_FILE).exists():
            return 2
        raise ValueError(f"Unknown directory format at: {path}")
    
    # v2 format: .npz file
    if p.is_file() and (p.suffix == ".npz" or str(p).endswith(".npz")):
        return 2
    
    raise ValueError(f"Cannot determine cache format for: {path}")


# =============================================================================
# Rust Backend Import (MANDATORY)
# =============================================================================
# Phase 2 requires Rust backend. If import fails, crash with clear instructions.
try:
    from binary_semantic_cache.binary_semantic_cache_rs import (
        RustBinaryEncoder,
        HammingSimilarity,
        RustCacheStorage,
    )
    RUST_BACKEND_AVAILABLE = True
except ImportError as e:
    raise ImportError(
        "Rust extension not available. Build required:\n"
        "  cd src/binary_semantic_cache_rs && cargo build --release && cd ../..\n"
        "  maturin develop --release\n"
        f"Original error: {e}"
    ) from e

# Python implementations retained as test oracles (not used at runtime)
from .encoder import BinaryEncoder as PythonBinaryEncoder
from .similarity import find_nearest as python_find_nearest


# =============================================================================
# Encoder Protocol (Type Compatibility)
# =============================================================================
class EncoderProtocol(Protocol):
    """Protocol for encoder implementations (Python or Rust)."""
    
    @property
    def embedding_dim(self) -> int: ...
    
    @property
    def code_bits(self) -> int: ...
    
    @property
    def n_words(self) -> int: ...
    
    def encode(self, embedding: np.ndarray) -> np.ndarray: ...


# Type alias for encoder (accepts both Python and Rust implementations)
Encoder = Union[RustBinaryEncoder, PythonBinaryEncoder]

logger = logging.getLogger(__name__)

# Frozen constants
DEFAULT_MAX_ENTRIES: Final[int] = 100_000
# NOTE: Default threshold lowered from 0.85 to 0.80 to compensate for
# binary quantization error (~5%). Cosine 0.85 → Hamming ~0.82.
DEFAULT_THRESHOLD: Final[float] = 0.80
DEFAULT_CODE_BITS: Final[int] = 256


class CacheEntry(NamedTuple):
    """
    Lightweight cache entry for API compatibility.
    Uses NamedTuple for minimal memory overhead.
    """
    id: int  # Integer ID
    code: np.ndarray
    response: Any
    created_at: float
    last_accessed: float
    access_count: int
    similarity: float = 1.0  # Similarity score (1.0 for exact match or when not computed)


@dataclass
class CacheStats:
    """Cache statistics for monitoring."""
    size: int
    max_size: int
    hits: int
    misses: int
    evictions: int
    memory_bytes: int

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a ratio (0.0 to 1.0)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def memory_mb(self) -> float:
        """Memory usage in MB."""
        return self.memory_bytes / (1024 * 1024)


class BinarySemanticCache:
    """
    Memory-efficient binary semantic cache with LRU eviction.
    
    Phase 2.5 Sprint 1b: Uses RustCacheStorage for codes and metadata.
    Phase 2.5 Sprint 1c-OPT: Responses stored in fixed-size Python list
    (pre-allocated to capacity) to eliminate dict resize jitter.
    
    Target: <10MB for 100K entries (codes+metadata only).
    
    Phase 2: Uses Rust backend (RustBinaryEncoder + RustCacheStorage) for
    performance targets (< 0.5ms lookup @ 100k entries).
    """

    __slots__ = (
        "_encoder",
        "_storage",          # RustCacheStorage (Rust) for codes/metadata
        "_responses",        # List[Optional[Any]]: response objects indexed by storage slot
        "_max_entries",
        "_threshold",
        "_code_bits",
        "_n_words",
        # Thread safety
        "_lock",
        # Statistics
        "_hits",
        "_misses",
        "_evictions",
    )

    def __init__(
        self,
        encoder: Encoder,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        similarity_threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        """
        Initialize the binary semantic cache.
        
        Args:
            encoder: Binary encoder (RustBinaryEncoder recommended for Phase 2).
            max_entries: Maximum number of cache entries.
            similarity_threshold: Minimum similarity for cache hit (0.0 to 1.0).
        
        Raises:
            ValueError: If max_entries <= 0 or threshold not in [0, 1].
        """
        if max_entries <= 0:
            raise ValueError(f"max_entries must be positive, got {max_entries}")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(f"similarity_threshold must be in [0, 1], got {similarity_threshold}")

        self._encoder = encoder
        self._max_entries = max_entries
        self._threshold = similarity_threshold
        self._code_bits = encoder.code_bits
        self._n_words = encoder.n_words
        
        # Initialize Rust storage backend (44 bytes/entry for codes+metadata)
        # LRU eviction is handled by RustCacheStorage (timestamp-based)
        self._storage = RustCacheStorage(capacity=max_entries, code_bits=self._code_bits)
        
        # Response storage: Fixed-size Python list indexed by Rust slot index
        # Pre-allocated to capacity to eliminate dict resize jitter (Sprint 1c-OPT)
        # None indicates an empty slot (no response stored)
        self._responses: List[Optional[Any]] = [None] * max_entries
        
        self._lock = threading.RLock()
        
        # Stats
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        logger.debug(
            "BinarySemanticCache initialized: max=%d, threshold=%.2f, backend=RustCacheStorage",
            max_entries,
            similarity_threshold,
        )

    # =========================================================================
    # Internal Response Storage API (Sprint 1c)
    # =========================================================================

    def _set_response(self, idx: int, response: Any) -> None:
        """
        Store a response at the given Rust storage index.
        
        Called inside the RLock critical section.
        
        Args:
            idx: Rust storage index (must be within [0, max_entries)).
            response: Response object to store.
        """
        self._responses[idx] = response

    def _get_response(self, idx: int) -> Any:
        """
        Retrieve a response by Rust storage index.
        
        Called inside the RLock critical section.
        Returns None if slot is empty (defensive miss).
        
        Args:
            idx: Rust storage index.
        
        Returns:
            Response object or None if slot is empty.
        """
        if 0 <= idx < len(self._responses):
            return self._responses[idx]
        return None

    def _delete_response(self, idx: int) -> None:
        """
        Delete a response by Rust storage index.
        
        Called inside the RLock critical section.
        Idempotent: sets slot to None (no error if already None).
        
        Args:
            idx: Rust storage index.
        """
        if 0 <= idx < len(self._responses):
            self._responses[idx] = None

    # --- Public API ---

    @property
    def encoder(self) -> Encoder:
        """Get the encoder instance (Rust or Python)."""
        return self._encoder

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def similarity_threshold(self) -> float:
        return self._threshold

    def get(self, embedding: np.ndarray) -> Optional[CacheEntry]:
        """
        Look up a cached response by embedding similarity.
        
        Uses Rust backend (RustCacheStorage.search) for high-performance
        similarity search. FROZEN FORMULA: similarity = 1.0 - (hamming_distance / code_bits)
        
        Args:
            embedding: Float32 array of shape (embedding_dim,).
        
        Returns:
            CacheEntry if similarity >= threshold, None otherwise.
        """
        with self._lock:
            if len(self._storage) == 0:
                self._misses += 1
                return None

            # Encode query using Rust encoder
            query_code = self._encoder.encode(embedding)

            # Find nearest above threshold using Rust storage search
            # Threshold semantics enforced in Rust: HIT iff similarity >= threshold
            result = self._storage.search(query_code, threshold=self._threshold)

            if result is None:
                self._misses += 1
                return None

            idx, similarity = result

            # Defensive check: response must exist at this index
            # If missing, treat as miss to avoid partial CacheEntry
            resp = self._get_response(idx)
            if resp is None:
                logger.warning(
                    "Response missing for index %d (storage/response desync). Treating as miss.",
                    idx
                )
                self._misses += 1
                return None

            # Update access timestamp in Rust BEFORE getting metadata
            # so the returned CacheEntry reflects the updated state
            now = int(time.time())
            try:
                self._storage.mark_used(idx, now)
            except IndexError:
                logger.warning(
                    "IndexError marking index %d as used. Treating as miss.",
                    idx
                )
                self._misses += 1
                return None

            # Get metadata and code from Rust storage (now reflects updated access)
            try:
                meta = self._storage.get_metadata(idx)
                code_tuple = self._storage.get_code(idx)
            except IndexError:
                logger.warning(
                    "IndexError accessing storage index %d. Treating as miss.",
                    idx
                )
                self._misses += 1
                return None

            self._hits += 1

            # Convert code tuple to numpy array
            code = np.array(code_tuple, dtype=np.uint64)

            return CacheEntry(
                id=idx,
                code=code.copy(),
                response=resp,
                created_at=float(meta["created_at"]),
                last_accessed=float(meta["last_accessed"]),
                access_count=int(meta["access_count"]),
                similarity=float(similarity),
            )

    def put(
        self,
        embedding: np.ndarray,
        response: Any,
        store_embedding: bool = False,
    ) -> int:
        """
        Store an embedding-response pair.
        
        Uses Rust encoder (RustBinaryEncoder.encode) for high-performance encoding.
        
        Args:
            embedding: Float32 array of shape (embedding_dim,).
            response: Response object to cache.
            store_embedding: Ignored (kept for API compatibility).
        
        Returns:
            Entry index (integer).
        """
        with self._lock:
            # Encode embedding using Rust encoder
            code = self._encoder.encode(embedding)
            now = int(time.time())

            # Check if we need to evict
            if len(self._storage) >= self._max_entries:
                # Eviction flow per SPRINT1B_INTEGRATION_SPEC.md §5:
                # 1. Rust selects victim index (timestamp-based LRU)
                idx = self._storage.evict_lru()
                
                # 2. Python removes the response (idempotent)
                self._delete_response(idx)
                
                # 3. Rust overwrites the slot in place
                self._storage.replace(idx, code, now)
                
                # 4. Python stores the new response
                self._set_response(idx, response)
                
                self._evictions += 1
                return idx
            else:
                # Not full: append new entry
                idx = self._storage.add(code, now)
                self._set_response(idx, response)
                return idx

    def delete(self, entry_id: int) -> bool:
        """
        Delete entry by index.
        
        Note: With RustCacheStorage, deletion is not directly supported.
        We remove the response and let the slot be reused on next eviction.
        For now, we mark this as a limitation and return False if index invalid.
        """
        with self._lock:
            if entry_id < 0 or entry_id >= len(self._storage):
                return False
            
            # Remove response from Python list (set to None)
            # The Rust slot remains but will be evicted naturally
            if self._responses[entry_id] is not None:
                self._responses[entry_id] = None
                return True
            return False

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            # Recreate storage to clear all entries
            self._storage = RustCacheStorage(
                capacity=self._max_entries, 
                code_bits=self._code_bits
            )
            # Reset response list to all None (keep same capacity)
            self._responses = [None] * self._max_entries

    def stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                size=len(self._storage),
                max_size=self._max_entries,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                memory_bytes=self.memory_bytes(),
            )

    def memory_bytes(self) -> int:
        """
        Estimate total memory usage in bytes.
        
        Returns:
            Sum of Rust index (44 B/entry) + Python list overhead.
            
        Note:
            - Rust storage: exactly 44 bytes/entry (codes + metadata).
            - Python responses: Fixed-size list with ~8 bytes/slot (pointer).
              List container overhead is constant regardless of fill level.
            - Does NOT include the size of response objects themselves.
        """
        rust_bytes = self._storage.memory_usage()
        # sys.getsizeof returns list container + pointer array overhead
        # For a pre-allocated list, this is: base + 8 * capacity (pointers)
        # No per-entry dict bucket overhead since we use a list
        response_overhead = sys.getsizeof(self._responses)
        return rust_bytes + response_overhead

    def get_all_entries(self) -> List[CacheEntry]:
        """Get all cache entries."""
        with self._lock:
            entries = []
            for idx in range(len(self._storage)):
                resp = self._responses[idx] if idx < len(self._responses) else None
                if resp is None:
                    # Skip entries without responses (deleted or desynced)
                    continue
                
                try:
                    meta = self._storage.get_metadata(idx)
                    code_tuple = self._storage.get_code(idx)
                except IndexError:
                    continue
                
                code = np.array(code_tuple, dtype=np.uint64)
                entries.append(CacheEntry(
                    id=idx,
                    code=code.copy(),
                    response=resp,
                    created_at=float(meta["created_at"]),
                    last_accessed=float(meta["last_accessed"]),
                    access_count=int(meta["access_count"]),
                ))
            return entries

    def save(self, path: str) -> None:
        """
        Save cache to disk (DEPRECATED).
        
        .. deprecated:: 0.2.0
            Use :meth:`save_mmap_v3` for the v3 format with integrity checks.
        """
        warnings.warn(
            "save() is deprecated and will be removed in v0.3.0. "
            "Use save_mmap_v3() for v3 format with integrity checks.",
            DeprecationWarning,
            stacklevel=2,
        )
        with self._lock:
            filepath = Path(path)
            n = len(self._storage)
            
            if n == 0:
                np.savez(
                    filepath,
                    codes=np.array([], dtype=np.uint64).reshape(0, self._n_words),
                    created_at=np.array([], dtype=np.float64),
                    access_count=np.array([], dtype=np.int32),
                )
                return

            # Extract data from Rust storage
            codes = np.zeros((n, self._n_words), dtype=np.uint64)
            created_at = np.zeros(n, dtype=np.float64)
            access_count = np.zeros(n, dtype=np.int32)
            responses = []
            
            for idx in range(n):
                try:
                    code_tuple = self._storage.get_code(idx)
                    codes[idx] = np.array(code_tuple, dtype=np.uint64)
                    
                    meta = self._storage.get_metadata(idx)
                    created_at[idx] = float(meta["created_at"])
                    access_count[idx] = int(meta["access_count"])
                    
                    responses.append(self._responses[idx] if idx < len(self._responses) else None)
                except IndexError:
                    # Should not happen, but be defensive
                    responses.append(None)

            np.savez(
                filepath,
                codes=codes,
                created_at=created_at,
                access_count=access_count,
                responses=np.array(responses, dtype=object),
            )
            logger.info("Cache saved to %s", path)

    def load(self, path: str) -> None:
        """
        Load cache from disk (DEPRECATED).
        
        .. deprecated:: 0.2.0
            Use :meth:`load_mmap_v3` for the v3 format with integrity checks.
        """
        warnings.warn(
            "load() is deprecated and will be removed in v0.3.0. "
            "Use load_mmap_v3() for v3 format with integrity checks.",
            DeprecationWarning,
            stacklevel=2,
        )
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Cache file not found: {path}")

        with self._lock:
            self.clear()
            
            data = np.load(filepath, allow_pickle=True)
            codes = data["codes"]
            
            if codes.size == 0:
                return

            n = codes.shape[0]
            now = int(time.time())
            
            # Load entries into Rust storage
            responses_data = data["responses"]
            for i in range(n):
                code = codes[i]
                created_at_val = int(data["created_at"][i])
                # Use created_at as timestamp for loading
                idx = self._storage.add(code, created_at_val)
                
                # Store response in the pre-allocated list
                if idx < len(self._responses):
                    self._responses[idx] = responses_data[i]
            
            logger.info("Cache loaded from %s: %d entries", path, n)

    # =========================================================================
    # Zero-Copy Persistence (Phase 2.4)
    # =========================================================================

    @staticmethod
    def _compute_checksum(data: bytes) -> str:
        """Compute SHA-256 checksum of data."""
        return hashlib.sha256(data).hexdigest()

    def save_mmap(self, path: str) -> None:
        """
        Save cache to disk in mmap-compatible v2 format (DEPRECATED).
        
        Creates a directory with raw binary files. Uses atomic write (temp dir +
        rename) for crash safety. This is an O(n) operation that copies data from
        Rust storage to disk.
        
        .. deprecated:: 0.3.0
            Use :meth:`save_mmap_v3` for the v3 format with 44-byte packed entries.
        
        File Format (v2)::
        
            cache_dir/
            ├── header.json        # Metadata (config, sizes, checksums)
            ├── codes.bin          # Raw uint64 array
            ├── created_at.bin     # Raw float64 array
            ├── access_count.bin   # Raw int32 array
            └── responses.pkl      # Pickled responses
        
        Args:
            path: Directory path for the cache. Created if it doesn't exist.
        
        Raises:
            OSError: If directory creation or file writing fails.
        
        Example::
        
            cache.save_mmap("/path/to/cache_dir")
            # Later:
            cache.load_mmap("/path/to/cache_dir")
        """
        with self._lock:
            target_dir = Path(path)
            temp_dir = Path(str(path) + ".tmp")
            
            # Clean up any previous failed attempt
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            
            # Create temp directory
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Get active data from Rust storage
                n = len(self._storage)
                
                if n == 0:
                    codes = np.array([], dtype=np.uint64).reshape(0, self._n_words)
                    created_at = np.array([], dtype=np.float64)
                    access_count = np.array([], dtype=np.int32)
                    responses = []
                else:
                    codes = np.zeros((n, self._n_words), dtype=np.uint64)
                    created_at = np.zeros(n, dtype=np.float64)
                    access_count = np.zeros(n, dtype=np.int32)
                    responses = []
                    
                    for idx in range(n):
                        try:
                            code_tuple = self._storage.get_code(idx)
                            codes[idx] = np.array(code_tuple, dtype=np.uint64)
                            
                            meta = self._storage.get_metadata(idx)
                            created_at[idx] = float(meta["created_at"])
                            access_count[idx] = int(meta["access_count"])
                            
                            responses.append(self._responses[idx] if idx < len(self._responses) else None)
                        except IndexError:
                            responses.append(None)
                
                # Save binary files
                codes_path = temp_dir / MMAP_CODES_FILE
                created_at_path = temp_dir / MMAP_CREATED_AT_FILE
                access_count_path = temp_dir / MMAP_ACCESS_COUNT_FILE
                responses_path = temp_dir / MMAP_RESPONSES_FILE
                
                # Save codes as raw binary (C-contiguous for mmap compatibility)
                codes_bytes = np.ascontiguousarray(codes).tobytes()
                codes_path.write_bytes(codes_bytes)
                
                # Save metadata arrays
                created_at_bytes = np.ascontiguousarray(created_at).tobytes()
                created_at_path.write_bytes(created_at_bytes)
                
                access_count_bytes = np.ascontiguousarray(access_count).tobytes()
                access_count_path.write_bytes(access_count_bytes)
                
                # Save responses as pickle (not mmap-compatible)
                with open(responses_path, "wb") as f:
                    pickle.dump(list(responses), f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Compute checksums
                checksum_codes = self._compute_checksum(codes_bytes)
                checksum_created_at = self._compute_checksum(created_at_bytes)
                checksum_access_count = self._compute_checksum(access_count_bytes)
                
                # Read responses file for checksum
                responses_bytes = responses_path.read_bytes()
                checksum_responses = self._compute_checksum(responses_bytes)
                
                # Create header
                header = {
                    "version": MMAP_FORMAT_VERSION,
                    "format": "mmap",
                    "n_entries": n,
                    "code_bits": self._code_bits,
                    "n_words": self._n_words,
                    "threshold": self._threshold,
                    "max_entries": self._max_entries,
                    "checksums": {
                        "codes": checksum_codes,
                        "created_at": checksum_created_at,
                        "access_count": checksum_access_count,
                        "responses": checksum_responses,
                    },
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                
                # Save header
                header_path = temp_dir / MMAP_HEADER_FILE
                with open(header_path, "w", encoding="utf-8") as f:
                    json.dump(header, f, indent=2)
                
                # Atomic rename: remove existing target, rename temp to target
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                temp_dir.rename(target_dir)
                
                logger.info(
                    "Cache saved (mmap format) to %s: %d entries",
                    path, n
                )
                
            except Exception:
                # Cleanup on failure
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                raise

    def load_mmap(self, path: str) -> None:
        """
        Load cache from disk using v2 format (DEPRECATED).
        
        This is an O(n) operation that reads data from disk and copies it into
        Rust storage. Despite the name "mmap", the data is fully loaded into
        memory (not lazily paged).
        
        .. deprecated:: 0.3.0
            Use :meth:`load_mmap_v3` for the v3 format with improved integrity checks.
        
        Args:
            path: Directory path containing the mmap cache files.
        
        Raises:
            FileNotFoundError: If the cache directory doesn't exist.
            ChecksumError: If file integrity verification fails.
            ValueError: If the format version is unsupported.
        
        Example::
        
            cache.load_mmap("/path/to/cache_dir")
            result = cache.get(embedding)
        """
        cache_dir = Path(path)
        if not cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {path}")
        if not cache_dir.is_dir():
            raise ValueError(f"Expected directory, got file: {path}")
        
        # Read header
        header_path = cache_dir / MMAP_HEADER_FILE
        if not header_path.exists():
            raise FileNotFoundError(f"Header file not found: {header_path}")
        
        with open(header_path, "r", encoding="utf-8") as f:
            header = json.load(f)
        
        # Validate version
        version = header.get("version", 1)
        if version != MMAP_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported mmap format version: {version}. "
                f"Expected: {MMAP_FORMAT_VERSION}"
            )
        
        n_entries = header["n_entries"]
        n_words = header["n_words"]
        checksums = header.get("checksums", {})
        
        with self._lock:
            self.clear()
            
            if n_entries == 0:
                logger.info("Cache loaded (mmap format) from %s: 0 entries", path)
                return
            
            # Verify and load codes using mmap
            codes_path = cache_dir / MMAP_CODES_FILE
            if not codes_path.exists():
                raise FileNotFoundError(f"Codes file not found: {codes_path}")
            
            # Verify checksum before mmap
            codes_bytes = codes_path.read_bytes()
            if "codes" in checksums:
                actual_checksum = self._compute_checksum(codes_bytes)
                if actual_checksum != checksums["codes"]:
                    raise ChecksumError(
                        f"Codes checksum mismatch. Expected: {checksums['codes']}, "
                        f"Got: {actual_checksum}"
                    )
            
            # Memory-map the codes array (read-only)
            codes_mmap = np.memmap(
                codes_path,
                dtype=np.uint64,
                mode='r',
                shape=(n_entries, n_words),
            )
            
            # Load and verify created_at
            created_at_path = cache_dir / MMAP_CREATED_AT_FILE
            created_at_data = None
            if created_at_path.exists():
                created_at_bytes = created_at_path.read_bytes()
                if "created_at" in checksums:
                    actual_checksum = self._compute_checksum(created_at_bytes)
                    if actual_checksum != checksums["created_at"]:
                        raise ChecksumError("created_at checksum mismatch")
                
                created_at_data = np.memmap(
                    created_at_path,
                    dtype=np.float64,
                    mode='r',
                    shape=(n_entries,),
                )
            
            # Load and verify access_count (not used for loading, but verify)
            access_count_path = cache_dir / MMAP_ACCESS_COUNT_FILE
            if access_count_path.exists():
                access_count_bytes = access_count_path.read_bytes()
                if "access_count" in checksums:
                    actual_checksum = self._compute_checksum(access_count_bytes)
                    if actual_checksum != checksums["access_count"]:
                        raise ChecksumError("access_count checksum mismatch")
            
            # Load and verify responses (pickle, not mmap)
            responses_path = cache_dir / MMAP_RESPONSES_FILE
            responses = []
            if responses_path.exists():
                responses_bytes = responses_path.read_bytes()
                if "responses" in checksums:
                    actual_checksum = self._compute_checksum(responses_bytes)
                    if actual_checksum != checksums["responses"]:
                        raise ChecksumError("responses checksum mismatch")
                
                with open(responses_path, "rb") as f:
                    responses = pickle.load(f)
            
            # Load entries into Rust storage
            now = int(time.time())
            for i in range(n_entries):
                code = codes_mmap[i].copy()  # Copy from mmap
                # Use created_at if available, otherwise use now
                timestamp = int(created_at_data[i]) if created_at_data is not None else now
                idx = self._storage.add(code, timestamp)
                
                # Store response in the pre-allocated list
                if i < len(responses) and idx < len(self._responses):
                    self._responses[idx] = responses[i]
            
            logger.info(
                "Cache loaded (mmap format) from %s: %d entries",
                path, n_entries
            )

    # =========================================================================
    # Persistence Format v3 (Sprint 2a)
    # =========================================================================

    def save_mmap_v3(self, path: str) -> None:
        """
        Save cache to disk in v3 directory format with integrity checks.
        
        Creates a directory with 44-byte packed binary entries and SHA-256
        checksums. Uses atomic write pattern (temp dir + rename) for crash safety.
        
        **Performance Characteristics (O(n) copy):**
        
        This is an O(n) operation that copies data from Rust storage to disk.
        Typical performance on reference hardware (Intel Core i7):
        
        - 100k entries: ~120ms save time
        - 1M entries: ~2.7s save time
        
        File Format v3::
        
            cache_v3/
            ├── header.json      # Metadata (version, checksums, config)
            ├── entries.bin      # Packed 44-byte structs (little-endian)
            └── responses.pkl    # Pickled responses (protocol 5)
        
        The entries.bin format uses 44-byte little-endian packed structs:
        
        - codes: [u64; 4] = 32 bytes (256-bit binary code)
        - created_at: u32 = 4 bytes (seconds since 2020-01-01 epoch)
        - last_accessed: u32 = 4 bytes (seconds since 2020-01-01 epoch)
        - access_count: u32 = 4 bytes
        
        Args:
            path: Directory path for the cache. Created if it doesn't exist.
        
        Raises:
            OSError: If directory creation or file writing fails.
        
        Example::
        
            cache.save_mmap_v3("/path/to/cache_v3")
            # Later:
            cache.load_mmap_v3("/path/to/cache_v3")
        """
        with self._lock:
            target_dir = Path(path)
            temp_dir = Path(str(path) + ".tmp")
            
            # Clean up any previous failed attempt
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            
            # Create temp directory
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                n = len(self._storage)
                
                # Pack entries into 44-byte structs
                # Format: <4Q (4 x u64 = 32 bytes) + 3I (3 x u32 = 12 bytes) = 44 bytes
                # Little-endian: '<'
                entry_struct = struct.Struct("<4Q3I")
                assert entry_struct.size == V3_ENTRY_SIZE, f"Entry struct size mismatch: {entry_struct.size} != {V3_ENTRY_SIZE}"
                
                entries_buffer = bytearray(n * V3_ENTRY_SIZE)
                responses = []
                
                for idx in range(n):
                    try:
                        code_tuple = self._storage.get_code(idx)
                        meta = self._storage.get_metadata(idx)
                        
                        # Convert Unix timestamps to 2020-epoch relative u32
                        created_at_unix = int(meta["created_at"])
                        last_accessed_unix = int(meta["last_accessed"])
                        access_count = int(meta["access_count"])
                        
                        # Clamp to 2020 epoch (saturate to 0 for pre-2020)
                        created_at_rel = max(0, created_at_unix - EPOCH_2020)
                        last_accessed_rel = max(0, last_accessed_unix - EPOCH_2020)
                        
                        # Pack into buffer
                        offset = idx * V3_ENTRY_SIZE
                        entry_struct.pack_into(
                            entries_buffer, offset,
                            code_tuple[0], code_tuple[1], code_tuple[2], code_tuple[3],
                            created_at_rel, last_accessed_rel, access_count
                        )
                        
                        responses.append(self._responses[idx] if idx < len(self._responses) else None)
                    except IndexError:
                        responses.append(None)
                
                entries_bytes = bytes(entries_buffer)
                
                # Write entries.bin
                entries_path = temp_dir / V3_ENTRIES_FILE
                with open(entries_path, "wb") as f:
                    f.write(entries_bytes)
                    f.flush()
                    os.fsync(f.fileno())
                
                entries_checksum = self._compute_checksum(entries_bytes)
                
                # Write responses.pkl
                responses_path = temp_dir / V3_RESPONSES_FILE
                with open(responses_path, "wb") as f:
                    pickle.dump(responses, f, protocol=5)
                    f.flush()
                    os.fsync(f.fileno())
                
                responses_bytes = responses_path.read_bytes()
                responses_checksum = self._compute_checksum(responses_bytes)
                
                # Create header
                header = {
                    "version": MMAP_FORMAT_VERSION_V3,
                    "code_bits": self._code_bits,
                    "n_entries": n,
                    "entry_size": V3_ENTRY_SIZE,
                    "endian": "little",
                    "checksum_algo": "sha256",
                    "entries_checksum": entries_checksum,
                    "responses_checksum": responses_checksum,
                    "max_entries": self._max_entries,
                    "threshold": self._threshold,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                
                # Write header.json
                header_path = temp_dir / V3_HEADER_FILE
                with open(header_path, "w", encoding="utf-8") as f:
                    json.dump(header, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                
                # Atomic commit using backup-rename-delete pattern
                # os.replace doesn't work for directories on all platforms
                backup_dir = Path(str(path) + ".bak")
                
                # Clean up any leftover backup from previous failed attempts
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                
                try:
                    # Step 1: If target exists, rename to backup (atomic)
                    if target_dir.exists():
                        os.rename(str(target_dir), str(backup_dir))
                    
                    # Step 2: Rename temp to target (atomic)
                    os.rename(str(temp_dir), str(target_dir))
                    
                    # Step 3: Delete backup (non-atomic, but original is safe)
                    if backup_dir.exists():
                        shutil.rmtree(backup_dir)
                        
                except Exception:
                    # Rollback: restore backup if it exists
                    if backup_dir.exists() and not target_dir.exists():
                        os.rename(str(backup_dir), str(target_dir))
                    raise
                
                logger.info(
                    "Cache saved (v3 format) to %s: %d entries, %d bytes",
                    path, n, len(entries_bytes)
                )
                
            except Exception:
                # Cleanup on failure
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                raise

    def load_mmap_v3(self, path: str, skip_checksum: bool = False) -> None:
        """
        Load cache from disk using v3 directory format.
        
        Validates header schema, checksums, and loads entries into Rust storage.
        This is a **blocking** operation that completes in two phases:
        
        **Phase A: Index Load (Search Ready)**
        
        Loads entries.bin into Rust storage via O(n) memcpy. This phase is fast
        (~10ms for 1M entries) and enables similarity search immediately after.
        
        **Phase B: Response Hydration (Full Hydration)**
        
        Unpickles responses.pkl into Python list. This phase is slower (~1.2s for
        1M entries) due to Python object creation overhead.
        
        **Performance Characteristics (O(n) copy):**
        
        Typical performance on reference hardware (Intel Core i7):
        
        - Index load @ 1M entries: ~10ms (Search Ready)
        - Full load @ 1M entries: ~1.2s (includes response hydration)
        
        The method blocks until both phases complete. Async/lazy loading of
        responses may be added in a future release.
        
        Args:
            path: Directory path containing the v3 cache files.
            skip_checksum: If True, skip checksum validation (for debugging only).
        
        Raises:
            FileNotFoundError: If the cache directory doesn't exist.
            FormatVersionError: If header version != 3.
            CorruptFileError: If files are missing, invalid, or entry_size != 44.
            ChecksumError: If file integrity verification fails.
            UnsupportedPlatformError: If endian != "little".
        
        Example::
        
            cache.load_mmap_v3("/path/to/cache_v3")
            result = cache.get(embedding)  # Cache is search-ready
        """
        cache_dir = Path(path)
        
        if not cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {path}")
        if not cache_dir.is_dir():
            raise CorruptFileError(f"Expected directory, got file: {path}")
        
        # Read header
        header_path = cache_dir / V3_HEADER_FILE
        if not header_path.exists():
            raise CorruptFileError(f"Header file not found: {header_path}")
        
        try:
            with open(header_path, "r", encoding="utf-8") as f:
                header = json.load(f)
        except json.JSONDecodeError as e:
            raise CorruptFileError(f"Invalid header JSON: {e}") from e
        
        # Validate header
        version = header.get("version", 0)
        if version != MMAP_FORMAT_VERSION_V3:
            raise FormatVersionError(
                f"Unsupported format version: {version}. Expected: {MMAP_FORMAT_VERSION_V3}"
            )
        
        entry_size = header.get("entry_size", 0)
        if entry_size != V3_ENTRY_SIZE:
            raise CorruptFileError(
                f"Invalid entry_size: {entry_size}. Expected: {V3_ENTRY_SIZE}"
            )
        
        endian = header.get("endian", "")
        if endian != "little":
            raise UnsupportedPlatformError(
                f"Unsupported endianness: {endian}. Only 'little' is supported."
            )
        
        n_entries = header.get("n_entries", 0)
        entries_checksum = header.get("entries_checksum", "")
        responses_checksum = header.get("responses_checksum", "")
        
        with self._lock:
            self.clear()
            
            if n_entries == 0:
                logger.info("Cache loaded (v3 format) from %s: 0 entries", path)
                return
            
            # Verify and load entries.bin
            entries_path = cache_dir / V3_ENTRIES_FILE
            if not entries_path.exists():
                raise CorruptFileError(f"Entries file not found: {entries_path}")
            
            entries_bytes = entries_path.read_bytes()
            
            # Validate size
            expected_size = n_entries * V3_ENTRY_SIZE
            if len(entries_bytes) != expected_size:
                raise CorruptFileError(
                    f"Entries file size mismatch: {len(entries_bytes)} bytes, "
                    f"expected {expected_size} bytes ({n_entries} entries * {V3_ENTRY_SIZE} bytes)"
                )
            
            # Verify checksum
            if not skip_checksum and entries_checksum:
                actual_checksum = self._compute_checksum(entries_bytes)
                if actual_checksum != entries_checksum:
                    raise ChecksumError(
                        f"Entries checksum mismatch. Expected: {entries_checksum}, "
                        f"Got: {actual_checksum}"
                    )
            
            # Verify and load responses.pkl
            responses_path = cache_dir / V3_RESPONSES_FILE
            if not responses_path.exists():
                raise CorruptFileError(f"Responses file not found: {responses_path}")
            
            responses_bytes = responses_path.read_bytes()
            
            if not skip_checksum and responses_checksum:
                actual_checksum = self._compute_checksum(responses_bytes)
                if actual_checksum != responses_checksum:
                    raise ChecksumError(
                        f"Responses checksum mismatch. Expected: {responses_checksum}, "
                        f"Got: {actual_checksum}"
                    )
            
            try:
                with open(responses_path, "rb") as f:
                    responses = pickle.load(f)
            except (pickle.UnpicklingError, EOFError) as e:
                raise CorruptFileError(f"Invalid responses pickle: {e}") from e
            
            # Bulk load entries into Rust storage (O(N) memcpy)
            # This replaces the slow manual unpacking loop
            try:
                self._storage = RustCacheStorage.from_bytes_v3(
                    entries_bytes,
                    self._max_entries,
                    self._code_bits
                )
            except ValueError as e:
                raise CorruptFileError(f"Failed to load entries into Rust storage: {e}") from e
            
            # Hydrate responses list (align with Rust storage indices)
            # Since we just loaded fresh storage, indices 0..n-1 map 1:1 to responses list
            n_loaded = len(responses)
            
            # Pre-allocate response list to full capacity (already done in clear(), but ensure)
            self._responses = [None] * self._max_entries
            
            # Copy loaded responses
            if n_loaded > 0:
                limit = min(n_loaded, self._max_entries)
                self._responses[:limit] = responses[:limit]
            
            logger.info(
                "Cache loaded (v3 format) from %s: %d entries",
                path, n_entries
            )

    def __len__(self) -> int:
        return len(self._storage)

    def __repr__(self) -> str:
        return (
            f"BinarySemanticCache(size={len(self._storage)}, "
            f"max_entries={self._max_entries}, "
            f"threshold={self._threshold})"
        )
