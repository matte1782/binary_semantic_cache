"""
Unit tests for RustCacheStorage PyO3 bindings.

Tests RS-10 through RS-18 from PHASE2_TEST_MATRIX.md (Sprint 1a-2).
These tests verify the correctness of the Rust storage backend exposed to Python.

Run with:
    pytest tests/unit/test_rust_storage.py -v
"""

import numpy as np
import pytest
import time

# Import the Rust extension
from binary_semantic_cache.binary_semantic_cache_rs import RustCacheStorage


class TestRustCacheStorageInit:
    """Test initialization and basic properties."""

    def test_create_storage(self):
        """Verify storage can be created with valid parameters."""
        storage = RustCacheStorage(capacity=100, code_bits=256)
        assert len(storage) == 0
        assert storage.capacity == 100
        assert storage.code_bits == 256

    def test_create_storage_default_code_bits(self):
        """Verify default code_bits is 256."""
        storage = RustCacheStorage(capacity=100)
        assert storage.code_bits == 256

    def test_create_storage_zero_capacity_rejected(self):
        """Verify zero capacity raises ValueError."""
        with pytest.raises(ValueError, match="capacity must be positive"):
            RustCacheStorage(capacity=0)

    def test_create_storage_zero_code_bits_rejected(self):
        """Verify zero code_bits raises ValueError."""
        with pytest.raises(ValueError, match="code_bits must be positive"):
            RustCacheStorage(capacity=100, code_bits=0)

    def test_repr(self):
        """Verify string representation."""
        storage = RustCacheStorage(capacity=100, code_bits=256)
        repr_str = repr(storage)
        assert "RustCacheStorage" in repr_str
        assert "capacity=100" in repr_str
        assert "code_bits=256" in repr_str
        assert "len=0" in repr_str


class TestRustCacheStorageFunctional:
    """RS-10: Functional tests for add and search."""

    def test_add_and_retrieve(self):
        """RS-10: Add entry, search for it, verify HIT with correct index and similarity."""
        storage = RustCacheStorage(capacity=10, code_bits=256)
        
        # Add an entry
        code = np.array([0x1234567890ABCDEF, 0xFEDCBA0987654321, 
                         0xAAAAAAAAAAAAAAAA, 0x5555555555555555], dtype=np.uint64)
        timestamp = 1700000000
        idx = storage.add(code, timestamp)
        
        assert idx == 0
        assert len(storage) == 1
        
        # Search for exact match
        result = storage.search(code, threshold=0.85)
        assert result is not None
        index, similarity = result
        assert index == 0
        assert similarity == 1.0  # Exact match

    def test_add_multiple_entries(self):
        """Verify multiple entries can be added and searched."""
        storage = RustCacheStorage(capacity=100, code_bits=256)
        
        # Add 10 entries
        codes = []
        for i in range(10):
            code = np.array([i, i + 1, i + 2, i + 3], dtype=np.uint64)
            codes.append(code)
            idx = storage.add(code, timestamp=1700000000 + i)
            assert idx == i
        
        assert len(storage) == 10
        
        # Search for each entry
        for i, code in enumerate(codes):
            result = storage.search(code, threshold=0.99)
            assert result is not None
            index, similarity = result
            assert index == i
            assert similarity == 1.0

    def test_search_empty_storage(self):
        """Verify search on empty storage returns None."""
        storage = RustCacheStorage(capacity=10, code_bits=256)
        query = np.array([1, 2, 3, 4], dtype=np.uint64)
        
        result = storage.search(query, threshold=0.85)
        assert result is None

    def test_search_no_match_above_threshold(self):
        """Verify search returns None when no match above threshold."""
        storage = RustCacheStorage(capacity=10, code_bits=256)
        
        # Add entry with all zeros
        code = np.array([0, 0, 0, 0], dtype=np.uint64)
        storage.add(code, timestamp=1700000000)
        
        # Search with all ones (maximum distance = 256 bits)
        query = np.array([0xFFFFFFFFFFFFFFFF] * 4, dtype=np.uint64)
        result = storage.search(query, threshold=0.5)
        
        # Similarity = 1.0 - (256/256) = 0.0, which is < 0.5
        assert result is None

    def test_search_finds_best_match(self):
        """Verify search returns the best match above threshold."""
        storage = RustCacheStorage(capacity=10, code_bits=256)
        
        # Add entries with varying distances from query
        # Entry 0: far from query
        storage.add(np.array([0xFFFFFFFFFFFFFFFF] * 4, dtype=np.uint64), 1700000000)
        # Entry 1: exact match
        storage.add(np.array([0x1234, 0x5678, 0x9ABC, 0xDEF0], dtype=np.uint64), 1700000001)
        # Entry 2: close but not exact
        storage.add(np.array([0x1234, 0x5678, 0x9ABC, 0xDEF1], dtype=np.uint64), 1700000002)
        
        query = np.array([0x1234, 0x5678, 0x9ABC, 0xDEF0], dtype=np.uint64)
        result = storage.search(query, threshold=0.5)
        
        assert result is not None
        index, similarity = result
        assert index == 1  # Exact match at index 1
        assert similarity == 1.0


class TestRustCacheStorageCapacity:
    """RS-11: Capacity limit tests."""

    def test_capacity_limit_enforced(self):
        """RS-11: Fill storage to capacity, verify len() == capacity."""
        storage = RustCacheStorage(capacity=5, code_bits=256)
        
        for i in range(5):
            code = np.array([i, 0, 0, 0], dtype=np.uint64)
            storage.add(code, timestamp=1700000000 + i)
        
        assert len(storage) == 5

    def test_add_when_full_raises_error(self):
        """RS-11: Verify add() raises ValueError when full."""
        storage = RustCacheStorage(capacity=3, code_bits=256)
        
        # Fill to capacity
        for i in range(3):
            code = np.array([i, 0, 0, 0], dtype=np.uint64)
            storage.add(code, timestamp=1700000000 + i)
        
        # Try to add one more
        code = np.array([99, 0, 0, 0], dtype=np.uint64)
        with pytest.raises(ValueError, match="Storage is full"):
            storage.add(code, timestamp=1700000000)


class TestRustCacheStorageLRU:
    """RS-12: LRU eviction tests."""

    def test_find_lru_eviction_candidate(self):
        """RS-12: Verify evict_lru() returns the entry with oldest last_accessed."""
        storage = RustCacheStorage(capacity=5, code_bits=256)
        
        # Add 5 entries with different timestamps
        # Entry 2 has the oldest timestamp
        timestamps = [1700000100, 1700000200, 1700000000, 1700000300, 1700000400]
        for i, ts in enumerate(timestamps):
            code = np.array([i, 0, 0, 0], dtype=np.uint64)
            storage.add(code, timestamp=ts)
        
        # Entry 2 (index 2) should be the LRU candidate
        lru_idx = storage.evict_lru()
        assert lru_idx == 2

    def test_find_lru_after_mark_used(self):
        """RS-12: Touch all but one entry, verify evict_lru returns the cold one."""
        storage = RustCacheStorage(capacity=5, code_bits=256)
        
        # Add 5 entries with same initial timestamp
        base_ts = 1700000000
        for i in range(5):
            code = np.array([i, 0, 0, 0], dtype=np.uint64)
            storage.add(code, timestamp=base_ts)
        
        # Touch all except entry 2
        new_ts = base_ts + 1000
        for i in [0, 1, 3, 4]:
            storage.mark_used(i, new_ts)
        
        # Entry 2 should be the LRU candidate (not touched)
        lru_idx = storage.evict_lru()
        assert lru_idx == 2

    def test_evict_lru_empty_raises_error(self):
        """Verify evict_lru() raises ValueError on empty storage."""
        storage = RustCacheStorage(capacity=5, code_bits=256)
        
        with pytest.raises(ValueError, match="Cannot evict from empty storage"):
            storage.evict_lru()

    def test_replace_after_evict(self):
        """Verify replace() works correctly after evict_lru()."""
        storage = RustCacheStorage(capacity=3, code_bits=256)
        
        # Fill storage
        for i in range(3):
            code = np.array([i, 0, 0, 0], dtype=np.uint64)
            storage.add(code, timestamp=1700000000 + i)
        
        # Find LRU (should be index 0 with oldest timestamp)
        lru_idx = storage.evict_lru()
        assert lru_idx == 0
        
        # Replace the LRU entry
        new_code = np.array([99, 99, 99, 99], dtype=np.uint64)
        storage.replace(lru_idx, new_code, timestamp=1700001000)
        
        # Verify the new entry is searchable
        result = storage.search(new_code, threshold=0.99)
        assert result is not None
        assert result[0] == 0
        assert result[1] == 1.0


class TestRustCacheStorageInterop:
    """RS-13: NumPy array conversion tests."""

    def test_numpy_array_conversion_contiguous(self):
        """RS-13: Verify contiguous arrays work correctly."""
        storage = RustCacheStorage(capacity=10, code_bits=256)
        
        # C-contiguous array
        code = np.array([1, 2, 3, 4], dtype=np.uint64)
        assert code.flags['C_CONTIGUOUS']
        
        idx = storage.add(code, timestamp=1700000000)
        assert idx == 0

    def test_numpy_array_conversion_from_slice(self):
        """RS-13: Verify sliced arrays are handled (may need copy)."""
        storage = RustCacheStorage(capacity=10, code_bits=256)
        
        # Create array and take a contiguous slice
        big_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint64)
        code = np.ascontiguousarray(big_array[0])
        
        idx = storage.add(code, timestamp=1700000000)
        assert idx == 0


class TestRustCacheStorageSafety:
    """RS-14: Index bounds and safety tests."""

    def test_index_out_of_bounds_mark_used(self):
        """RS-14: Verify mark_used with invalid index raises IndexError."""
        storage = RustCacheStorage(capacity=10, code_bits=256)
        storage.add(np.array([1, 2, 3, 4], dtype=np.uint64), 1700000000)
        
        with pytest.raises(IndexError, match="out of bounds"):
            storage.mark_used(10, 1700000000)

    def test_index_out_of_bounds_get_metadata(self):
        """RS-14: Verify get_metadata with invalid index raises IndexError."""
        storage = RustCacheStorage(capacity=10, code_bits=256)
        storage.add(np.array([1, 2, 3, 4], dtype=np.uint64), 1700000000)
        
        with pytest.raises(IndexError, match="out of bounds"):
            storage.get_metadata(10)

    def test_index_out_of_bounds_replace(self):
        """RS-14: Verify replace with invalid index raises IndexError."""
        storage = RustCacheStorage(capacity=10, code_bits=256)
        storage.add(np.array([1, 2, 3, 4], dtype=np.uint64), 1700000000)
        
        with pytest.raises(IndexError, match="out of bounds"):
            storage.replace(10, np.array([5, 6, 7, 8], dtype=np.uint64), 1700000000)

    def test_index_out_of_bounds_get_code(self):
        """Verify get_code with invalid index raises IndexError."""
        storage = RustCacheStorage(capacity=10, code_bits=256)
        
        with pytest.raises(IndexError, match="out of bounds"):
            storage.get_code(0)


class TestRustCacheStorageShapeValidation:
    """RS-15: Shape validation tests."""

    def test_invalid_shape_too_long(self):
        """RS-15: Verify code with wrong shape (too long) raises ValueError."""
        storage = RustCacheStorage(capacity=10, code_bits=256)
        
        # 8 elements instead of 4
        code = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint64)
        with pytest.raises(ValueError, match="shape"):
            storage.add(code, timestamp=1700000000)

    def test_invalid_shape_too_short(self):
        """RS-15: Verify code with wrong shape (too short) raises ValueError."""
        storage = RustCacheStorage(capacity=10, code_bits=256)
        
        # 2 elements instead of 4
        code = np.array([1, 2], dtype=np.uint64)
        with pytest.raises(ValueError, match="shape"):
            storage.add(code, timestamp=1700000000)

    def test_invalid_shape_2d(self):
        """RS-15: Verify 2D array raises appropriate error."""
        storage = RustCacheStorage(capacity=10, code_bits=256)
        
        # 2D array
        code = np.array([[1, 2, 3, 4]], dtype=np.uint64)
        # Should fail because it's not 1D with 4 elements
        with pytest.raises((ValueError, TypeError)):
            storage.add(code, timestamp=1700000000)


class TestRustCacheStorageDtypeValidation:
    """RS-16: Dtype validation tests."""

    def test_invalid_dtype_float32(self):
        """RS-16: Verify float32 array raises TypeError."""
        storage = RustCacheStorage(capacity=10, code_bits=256)
        
        code = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        with pytest.raises(TypeError):
            storage.add(code, timestamp=1700000000)

    def test_invalid_dtype_int32(self):
        """RS-16: Verify int32 array raises TypeError."""
        storage = RustCacheStorage(capacity=10, code_bits=256)
        
        code = np.array([1, 2, 3, 4], dtype=np.int32)
        with pytest.raises(TypeError):
            storage.add(code, timestamp=1700000000)

    def test_invalid_dtype_float64(self):
        """RS-16: Verify float64 array raises TypeError."""
        storage = RustCacheStorage(capacity=10, code_bits=256)
        
        code = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        with pytest.raises(TypeError):
            storage.add(code, timestamp=1700000000)


class TestRustCacheStorageMemory:
    """RS-17: Memory usage tests."""

    def test_memory_usage_reporting(self):
        """RS-17: Verify memory_usage() returns exactly 44 * n bytes."""
        storage = RustCacheStorage(capacity=100, code_bits=256)
        
        # Empty storage
        assert storage.memory_usage() == 0
        
        # Add entries and verify memory grows by 44 bytes each
        for i in range(10):
            code = np.array([i, 0, 0, 0], dtype=np.uint64)
            storage.add(code, timestamp=1700000000 + i)
            expected_memory = (i + 1) * 44
            assert storage.memory_usage() == expected_memory, \
                f"Expected {expected_memory} bytes, got {storage.memory_usage()}"

    def test_memory_usage_large_storage(self):
        """Verify memory calculation for larger storage."""
        storage = RustCacheStorage(capacity=1000, code_bits=256)
        
        for i in range(100):
            code = np.array([i, 0, 0, 0], dtype=np.uint64)
            storage.add(code, timestamp=1700000000)
        
        assert storage.memory_usage() == 100 * 44


class TestRustCacheStorageMetadata:
    """RS-18: Metadata and mark_used tests."""

    def test_metadata_and_mark_used(self):
        """RS-18: Verify get_metadata returns correct values and mark_used updates them."""
        storage = RustCacheStorage(capacity=10, code_bits=256)
        
        # Add an entry
        code = np.array([1, 2, 3, 4], dtype=np.uint64)
        creation_time = 1700000000
        storage.add(code, timestamp=creation_time)
        
        # Check initial metadata
        meta = storage.get_metadata(0)
        assert meta["created_at"] == creation_time
        assert meta["last_accessed"] == creation_time
        assert meta["access_count"] == 0
        
        # Mark as used
        access_time = creation_time + 100
        storage.mark_used(0, access_time)
        
        # Check updated metadata
        meta = storage.get_metadata(0)
        assert meta["created_at"] == creation_time  # Unchanged
        assert meta["last_accessed"] == access_time  # Updated
        assert meta["access_count"] == 1  # Incremented
        
        # Mark as used again
        access_time_2 = access_time + 100
        storage.mark_used(0, access_time_2)
        
        meta = storage.get_metadata(0)
        assert meta["last_accessed"] == access_time_2
        assert meta["access_count"] == 2

    def test_access_count_saturation(self):
        """Verify access_count saturates at u32::MAX."""
        storage = RustCacheStorage(capacity=10, code_bits=256)
        
        code = np.array([1, 2, 3, 4], dtype=np.uint64)
        storage.add(code, timestamp=1700000000)
        
        # Call mark_used many times (can't test u32::MAX easily, but verify it doesn't crash)
        for i in range(100):
            storage.mark_used(0, 1700000000 + i)
        
        meta = storage.get_metadata(0)
        assert meta["access_count"] == 100


class TestRustCacheStorageThreshold:
    """Test threshold semantics (Phase 1 contract)."""

    def test_threshold_boundary_exact(self):
        """Verify similarity == threshold returns HIT."""
        storage = RustCacheStorage(capacity=10, code_bits=256)
        
        # Add entry with known code
        code = np.array([0, 0, 0, 0], dtype=np.uint64)
        storage.add(code, timestamp=1700000000)
        
        # Query with exact match (similarity = 1.0)
        result = storage.search(code, threshold=1.0)
        assert result is not None
        assert result[1] == 1.0

    def test_threshold_boundary_below(self):
        """Verify similarity < threshold returns None."""
        storage = RustCacheStorage(capacity=10, code_bits=256)
        
        # Add entry with all zeros
        code = np.array([0, 0, 0, 0], dtype=np.uint64)
        storage.add(code, timestamp=1700000000)
        
        # Query with 128 bits different (similarity = 0.5)
        query = np.array([0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0, 0], dtype=np.uint64)
        
        # Threshold just above similarity should return None
        result = storage.search(query, threshold=0.51)
        assert result is None
        
        # Threshold at or below similarity should return HIT
        result = storage.search(query, threshold=0.5)
        assert result is not None

    def test_invalid_threshold_values(self):
        """Verify invalid threshold values raise ValueError."""
        storage = RustCacheStorage(capacity=10, code_bits=256)
        storage.add(np.array([1, 2, 3, 4], dtype=np.uint64), 1700000000)
        query = np.array([1, 2, 3, 4], dtype=np.uint64)
        
        with pytest.raises(ValueError, match="threshold"):
            storage.search(query, threshold=-0.1)
        
        with pytest.raises(ValueError, match="threshold"):
            storage.search(query, threshold=1.1)


class TestRustCacheStorageGetCode:
    """Test get_code functionality."""

    def test_get_code_returns_correct_values(self):
        """Verify get_code returns the exact code that was stored."""
        storage = RustCacheStorage(capacity=10, code_bits=256)
        
        original_code = np.array([0x1234567890ABCDEF, 0xFEDCBA0987654321,
                                  0xAAAAAAAAAAAAAAAA, 0x5555555555555555], dtype=np.uint64)
        storage.add(original_code, timestamp=1700000000)
        
        retrieved_code = storage.get_code(0)
        
        assert len(retrieved_code) == 4
        for i in range(4):
            assert retrieved_code[i] == original_code[i]

