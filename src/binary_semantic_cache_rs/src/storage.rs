//! Memory-efficient storage for cache entries.
//!
//! This module provides the `CacheEntryPacked` struct, which stores cache entry
//! data in exactly 44 bytes using `#[repr(C, packed)]` to eliminate padding.
//!
//! It also provides `RustCacheStorage`, a PyO3-exposed container that manages
//! a vector of `CacheEntryPacked` entries with timestamp-based LRU eviction.
//!
//! # Memory Layout (44 bytes total per entry)
//!
//! | Field         | Type     | Size | Offset | Description                          |
//! |---------------|----------|------|--------|--------------------------------------|
//! | `codes`       | [u64; 4] | 32 B | 0      | 256-bit binary code (4 x 64-bit)     |
//! | `created_at`  | u32      | 4 B  | 32     | Seconds since 2020-01-01 epoch       |
//! | `last_accessed`| u32     | 4 B  | 36     | Seconds since 2020-01-01 epoch       |
//! | `access_count`| u32      | 4 B  | 40     | Number of cache hits (saturating)    |
//!
//! # Epoch
//!
//! Timestamps are stored relative to 2020-01-01 00:00:00 UTC (Unix timestamp 1577836800).
//! This allows coverage until approximately 2156 AD with u32 storage.
//!
//! # LRU Strategy
//!
//! Uses timestamp-based LRU (O(N) eviction scan) instead of linked-list LRU (O(1))
//! to support > 65k entries (u16 pointer limit). Eviction scans are fast in Rust.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Unix timestamp for 2020-01-01 00:00:00 UTC.
/// All timestamps are stored relative to this epoch to save 4 bytes per entry.
pub const EPOCH_2020: u64 = 1577836800;

// Compile-time assertion: struct MUST be exactly 44 bytes.
// This check runs at compile time and will fail the build if violated.
const _: () = assert!(std::mem::size_of::<CacheEntryPacked>() == 44);

/// A memory-efficient cache entry storing binary codes and metadata.
///
/// This struct is exactly 44 bytes due to `#[repr(C, packed)]`, which:
/// - Uses C memory layout for predictable field ordering
/// - Eliminates all padding between fields
/// - Results in alignment of 1 (packed)
///
/// # Safety
///
/// The `packed` attribute can cause unaligned access issues on some platforms.
/// However, since all fields are accessed through methods that copy values,
/// this is safe. Direct field access should be avoided in performance-critical
/// code on platforms that don't support unaligned access.
///
/// # Fields
///
/// - `codes`: 256-bit binary code stored as 4 x u64 words (LSB-first packing)
/// - `created_at`: Creation timestamp (seconds since 2020-01-01)
/// - `last_accessed`: Last access timestamp (seconds since 2020-01-01)
/// - `access_count`: Number of times this entry was accessed (saturating at u32::MAX)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct CacheEntryPacked {
    codes: [u64; 4],
    created_at: u32,
    last_accessed: u32,
    access_count: u32,
}

impl CacheEntryPacked {
    /// Creates a new cache entry with the given binary codes and timestamp.
    ///
    /// # Arguments
    ///
    /// * `codes` - The 256-bit binary code as 4 x u64 words
    /// * `now_unix` - Current Unix timestamp (seconds since 1970-01-01)
    ///
    /// # Returns
    ///
    /// A new `CacheEntryPacked` with:
    /// - `created_at` and `last_accessed` set to `now_unix` (converted to 2020 epoch)
    /// - `access_count` initialized to 0
    ///
    /// # Panics
    ///
    /// Does not panic. Timestamps before 2020-01-01 will saturate to 0.
    #[inline]
    pub fn new(codes: [u64; 4], now_unix: u64) -> Self {
        let relative_time = Self::to_relative_time(now_unix);
        Self {
            codes,
            created_at: relative_time,
            last_accessed: relative_time,
            access_count: 0,
        }
    }

    /// Returns the binary codes stored in this entry.
    #[inline]
    pub fn codes(&self) -> [u64; 4] {
        self.codes
    }

    /// Returns the creation timestamp as a Unix timestamp.
    ///
    /// Converts from the internal 2020-relative format back to Unix time.
    #[inline]
    pub fn created_at_unix(&self) -> u64 {
        Self::to_unix_time(self.created_at)
    }

    /// Returns the last access timestamp as a Unix timestamp.
    ///
    /// Converts from the internal 2020-relative format back to Unix time.
    #[inline]
    pub fn last_accessed_unix(&self) -> u64 {
        Self::to_unix_time(self.last_accessed)
    }

    /// Returns the number of times this entry has been accessed.
    #[inline]
    pub fn access_count(&self) -> u32 {
        self.access_count
    }

    /// Records an access to this cache entry.
    ///
    /// Updates `last_accessed` to the given timestamp and increments `access_count`.
    /// The access count saturates at `u32::MAX` to prevent overflow.
    ///
    /// # Arguments
    ///
    /// * `now_unix` - Current Unix timestamp (seconds since 1970-01-01)
    #[inline]
    pub fn record_access(&mut self, now_unix: u64) {
        self.last_accessed = Self::to_relative_time(now_unix);
        self.access_count = self.access_count.saturating_add(1);
    }

    /// Converts a Unix timestamp to a 2020-relative timestamp.
    ///
    /// Saturates to 0 for timestamps before 2020-01-01.
    #[inline]
    fn to_relative_time(unix_time: u64) -> u32 {
        unix_time.saturating_sub(EPOCH_2020) as u32
    }

    /// Converts a 2020-relative timestamp back to a Unix timestamp.
    #[inline]
    fn to_unix_time(relative_time: u32) -> u64 {
        EPOCH_2020 + relative_time as u64
    }

    /// Returns the internal relative timestamp for LRU comparison.
    #[inline]
    pub fn last_accessed_relative(&self) -> u32 {
        self.last_accessed
    }
}

// =============================================================================
// RustCacheStorage: PyO3-exposed container for CacheEntryPacked
// =============================================================================

/// A memory-efficient cache storage container exposed to Python.
///
/// Stores binary codes and metadata in packed 44-byte entries.
/// Supports timestamp-based LRU eviction (O(N) scan).
///
/// # Example (Python)
///
/// ```python
/// import numpy as np
/// from binary_semantic_cache.binary_semantic_cache_rs import RustCacheStorage
///
/// storage = RustCacheStorage(capacity=10000, code_bits=256)
/// code = np.array([0x1234, 0x5678, 0x9ABC, 0xDEF0], dtype=np.uint64)
/// idx = storage.add(code, timestamp=1700000000)
/// print(f"Added at index {idx}, len={len(storage)}")
/// ```
#[pyclass]
pub struct RustCacheStorage {
    entries: Vec<CacheEntryPacked>,
    capacity: usize,
    code_bits: usize,
}

#[pymethods]
impl RustCacheStorage {
    /// Create a new cache storage with fixed capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of entries (e.g., 1,000,000)
    /// * `code_bits` - Size of binary codes in bits (default: 256)
    ///
    /// # Raises
    ///
    /// * `ValueError` - If capacity is zero or code_bits is zero
    #[new]
    #[pyo3(signature = (capacity, code_bits=256))]
    fn new(capacity: usize, code_bits: usize) -> PyResult<Self> {
        if capacity == 0 {
            return Err(PyValueError::new_err("capacity must be positive"));
        }
        if code_bits == 0 {
            return Err(PyValueError::new_err("code_bits must be positive"));
        }
        Ok(RustCacheStorage {
            entries: Vec::with_capacity(capacity.min(1024)), // Pre-allocate reasonably
            capacity,
            code_bits,
        })
    }

    /// Add a new entry to the storage.
    ///
    /// # Arguments
    ///
    /// * `code` - Binary code as uint64 array of shape (4,)
    /// * `timestamp` - Current Unix timestamp (seconds since 1970)
    ///
    /// # Returns
    ///
    /// Index of the stored entry.
    ///
    /// # Raises
    ///
    /// * `ValueError` - If storage is full (call evict_lru first)
    /// * `ValueError` - If code shape is incorrect
    /// * `TypeError` - If code dtype is not uint64
    fn add(&mut self, code: PyReadonlyArray1<u64>, timestamp: u64) -> PyResult<usize> {
        // Validate capacity
        if self.entries.len() >= self.capacity {
            return Err(PyValueError::new_err(
                "Storage is full. Call evict_lru() first to make space.",
            ));
        }

        // Validate and extract code
        let codes = self.validate_and_extract_code(&code)?;

        // Create entry and append
        let entry = CacheEntryPacked::new(codes, timestamp);
        let index = self.entries.len();
        self.entries.push(entry);

        Ok(index)
    }

    /// Overwrite an existing entry at the given index.
    ///
    /// Used after evict_lru() to reuse a slot.
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the entry to overwrite
    /// * `code` - Binary code as uint64 array of shape (4,)
    /// * `timestamp` - Current Unix timestamp
    ///
    /// # Raises
    ///
    /// * `IndexError` - If index is out of bounds
    /// * `ValueError` - If code shape is incorrect
    fn replace(&mut self, index: usize, code: PyReadonlyArray1<u64>, timestamp: u64) -> PyResult<()> {
        if index >= self.entries.len() {
            return Err(PyIndexError::new_err(format!(
                "index {} out of bounds for storage of length {}",
                index,
                self.entries.len()
            )));
        }

        let codes = self.validate_and_extract_code(&code)?;
        self.entries[index] = CacheEntryPacked::new(codes, timestamp);
        Ok(())
    }

    /// Find the nearest neighbor above similarity threshold.
    ///
    /// # Arguments
    ///
    /// * `query` - Binary code as uint64 array of shape (4,)
    /// * `threshold` - Minimum similarity (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// Tuple of (index, similarity) if found, None otherwise.
    ///
    /// # Raises
    ///
    /// * `ValueError` - If query shape is incorrect or threshold is invalid
    #[pyo3(signature = (query, threshold=0.85))]
    fn search(&self, query: PyReadonlyArray1<u64>, threshold: f32) -> PyResult<Option<(usize, f32)>> {
        // Validate threshold
        if threshold.is_nan() || threshold < 0.0 || threshold > 1.0 {
            return Err(PyValueError::new_err(format!(
                "threshold must be in [0.0, 1.0], got {}",
                threshold
            )));
        }

        // Handle empty storage
        if self.entries.is_empty() {
            return Ok(None);
        }

        // Validate and extract query
        let query_codes = self.validate_and_extract_code(&query)?;

        // Linear scan for best match
        let mut best_idx: Option<usize> = None;
        let mut best_sim: f32 = f32::NEG_INFINITY;

        for (i, entry) in self.entries.iter().enumerate() {
            let distance = hamming_distance_codes(&query_codes, &entry.codes());
            let similarity = 1.0 - (distance as f32 / self.code_bits as f32);

            if similarity > best_sim {
                best_sim = similarity;
                best_idx = Some(i);

                // Early termination on exact match
                if distance == 0 {
                    break;
                }
            }
        }

        match best_idx {
            Some(idx) if best_sim >= threshold => Ok(Some((idx, best_sim))),
            _ => Ok(None),
        }
    }

    /// Find the least recently used entry (oldest last_accessed timestamp).
    ///
    /// # Returns
    ///
    /// Index of the LRU entry.
    ///
    /// # Raises
    ///
    /// * `ValueError` - If storage is empty
    fn evict_lru(&self) -> PyResult<usize> {
        if self.entries.is_empty() {
            return Err(PyValueError::new_err("Cannot evict from empty storage"));
        }

        // Find entry with oldest last_accessed timestamp
        let mut lru_idx = 0;
        let mut lru_time = self.entries[0].last_accessed_relative();

        for (i, entry) in self.entries.iter().enumerate().skip(1) {
            let t = entry.last_accessed_relative();
            if t < lru_time {
                lru_time = t;
                lru_idx = i;
            }
        }

        Ok(lru_idx)
    }

    /// Update the last_accessed timestamp and increment access_count.
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the entry to update
    /// * `timestamp` - Current Unix timestamp
    ///
    /// # Raises
    ///
    /// * `IndexError` - If index is out of bounds
    fn mark_used(&mut self, index: usize, timestamp: u64) -> PyResult<()> {
        if index >= self.entries.len() {
            return Err(PyIndexError::new_err(format!(
                "index {} out of bounds for storage of length {}",
                index,
                self.entries.len()
            )));
        }

        self.entries[index].record_access(timestamp);
        Ok(())
    }

    /// Get metadata for an entry.
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the entry
    ///
    /// # Returns
    ///
    /// Dictionary with created_at, last_accessed, access_count (Unix timestamps).
    ///
    /// # Raises
    ///
    /// * `IndexError` - If index is out of bounds
    fn get_metadata<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyDict>> {
        if index >= self.entries.len() {
            return Err(PyIndexError::new_err(format!(
                "index {} out of bounds for storage of length {}",
                index,
                self.entries.len()
            )));
        }

        let entry = &self.entries[index];
        let dict = PyDict::new_bound(py);
        dict.set_item("created_at", entry.created_at_unix())?;
        dict.set_item("last_accessed", entry.last_accessed_unix())?;
        dict.set_item("access_count", entry.access_count())?;
        Ok(dict)
    }

    /// Get the binary code at the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the entry
    ///
    /// # Returns
    ///
    /// Binary code as list of 4 uint64 values.
    ///
    /// # Raises
    ///
    /// * `IndexError` - If index is out of bounds
    fn get_code(&self, index: usize) -> PyResult<[u64; 4]> {
        if index >= self.entries.len() {
            return Err(PyIndexError::new_err(format!(
                "index {} out of bounds for storage of length {}",
                index,
                self.entries.len()
            )));
        }

        Ok(self.entries[index].codes())
    }

    /// Return total memory usage in bytes for entry data.
    ///
    /// Should be exactly 44 * num_entries.
    fn memory_usage(&self) -> usize {
        self.entries.len() * std::mem::size_of::<CacheEntryPacked>()
    }

    /// Return the current number of entries.
    fn __len__(&self) -> usize {
        self.entries.len()
    }

    /// Return the capacity.
    #[getter]
    fn capacity(&self) -> usize {
        self.capacity
    }

    /// Return the code_bits.
    #[getter]
    fn code_bits(&self) -> usize {
        self.code_bits
    }

    /// String representation.
    fn __repr__(&self) -> String {
        format!(
            "RustCacheStorage(capacity={}, code_bits={}, len={})",
            self.capacity,
            self.code_bits,
            self.entries.len()
        )
    }
}

impl RustCacheStorage {
    /// Validate and extract a code array from Python.
    fn validate_and_extract_code(&self, code: &PyReadonlyArray1<u64>) -> PyResult<[u64; 4]> {
        let slice = code.as_slice().map_err(|_| {
            PyTypeError::new_err("code must be a contiguous uint64 array")
        })?;

        // Validate shape: must be exactly 4 elements for 256-bit codes
        let expected_words = self.code_bits / 64;
        if slice.len() != expected_words {
            return Err(PyValueError::new_err(format!(
                "code must have shape ({},) for {}-bit codes, got shape ({},)",
                expected_words, self.code_bits, slice.len()
            )));
        }

        // Copy to fixed-size array
        let mut codes = [0u64; 4];
        codes.copy_from_slice(slice);
        Ok(codes)
    }
}

/// Compute Hamming distance between two 4-word codes.
#[inline(always)]
fn hamming_distance_codes(a: &[u64; 4], b: &[u64; 4]) -> u32 {
    (a[0] ^ b[0]).count_ones()
        + (a[1] ^ b[1]).count_ones()
        + (a[2] ^ b[2]).count_ones()
        + (a[3] ^ b[3]).count_ones()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RS-01: Verify CacheEntryPacked is exactly 44 bytes with alignment 1.
    ///
    /// This test validates the memory layout guarantees required for:
    /// - Efficient memory usage (< 50 bytes/entry target)
    /// - Predictable serialization for mmap persistence
    /// - No hidden padding bytes
    #[test]
    fn test_entry_size_44_bytes() {
        assert_eq!(
            std::mem::size_of::<CacheEntryPacked>(),
            44,
            "CacheEntryPacked must be exactly 44 bytes"
        );
        assert_eq!(
            std::mem::align_of::<CacheEntryPacked>(),
            1,
            "CacheEntryPacked must be packed (alignment 1)"
        );
    }

    /// RS-02: Verify basic entry creation and field initialization.
    ///
    /// Tests that:
    /// - Codes are stored correctly
    /// - Timestamps are converted and stored
    /// - access_count starts at 0
    #[test]
    fn test_entry_creation() {
        let codes = [1u64, 2u64, 3u64, 4u64];
        let now = 1700000000u64; // 2023-11-14 (approximately)

        let entry = CacheEntryPacked::new(codes, now);

        assert_eq!(entry.codes(), codes, "Codes must match input");
        assert_eq!(entry.created_at_unix(), now, "created_at must match input timestamp");
        assert_eq!(entry.last_accessed_unix(), now, "last_accessed must match input timestamp");
        assert_eq!(entry.access_count(), 0, "access_count must start at 0");
    }

    /// RS-03: Verify 2020 epoch timestamp conversion is correct.
    ///
    /// Tests:
    /// - Exact epoch (2020-01-01) stores as 0
    /// - Timestamps after epoch store correctly
    /// - Round-trip conversion preserves values
    #[test]
    fn test_timestamp_conversion() {
        let codes = [0u64; 4];

        // Test exact epoch (2020-01-01 00:00:00 UTC)
        let entry_epoch = CacheEntryPacked::new(codes, EPOCH_2020);
        // Copy field to avoid unaligned reference (packed struct)
        let created_at_value = { entry_epoch.created_at };
        assert_eq!(
            created_at_value, 0,
            "Epoch timestamp must store as 0"
        );
        assert_eq!(
            entry_epoch.created_at_unix(),
            EPOCH_2020,
            "Epoch must round-trip correctly"
        );

        // Test +1 year (approximately 31,536,000 seconds)
        let one_year_later = EPOCH_2020 + 365 * 24 * 3600;
        let entry_year = CacheEntryPacked::new(codes, one_year_later);
        assert_eq!(
            entry_year.created_at_unix(),
            one_year_later,
            "One year after epoch must round-trip correctly"
        );

        // Test current-ish time (2023-11-14)
        let current_time = 1700000000u64;
        let entry_current = CacheEntryPacked::new(codes, current_time);
        assert_eq!(
            entry_current.created_at_unix(),
            current_time,
            "Current time must round-trip correctly"
        );

        // Verify internal storage is relative (copy to avoid unaligned reference)
        let expected_relative = (current_time - EPOCH_2020) as u32;
        let actual_created_at = { entry_current.created_at };
        assert_eq!(
            actual_created_at, expected_relative,
            "Internal storage must be relative to 2020 epoch"
        );
    }

    /// RS-04: Verify access counting and last_accessed updates.
    ///
    /// Tests:
    /// - record_access increments access_count
    /// - record_access updates last_accessed
    /// - created_at remains unchanged
    /// - Multiple accesses accumulate correctly
    #[test]
    fn test_access_count_increment() {
        let codes = [0u64; 4];
        let t0 = 1700000000u64;
        let mut entry = CacheEntryPacked::new(codes, t0);

        // Initial state
        assert_eq!(entry.access_count(), 0, "Initial access_count must be 0");
        assert_eq!(entry.last_accessed_unix(), t0, "Initial last_accessed must match creation time");

        // First access (+60 seconds)
        let t1 = t0 + 60;
        entry.record_access(t1);
        assert_eq!(entry.access_count(), 1, "access_count must be 1 after first access");
        assert_eq!(entry.last_accessed_unix(), t1, "last_accessed must update to t1");
        assert_eq!(entry.created_at_unix(), t0, "created_at must remain unchanged");

        // Second access (+120 seconds)
        let t2 = t0 + 120;
        entry.record_access(t2);
        assert_eq!(entry.access_count(), 2, "access_count must be 2 after second access");
        assert_eq!(entry.last_accessed_unix(), t2, "last_accessed must update to t2");
        assert_eq!(entry.created_at_unix(), t0, "created_at must remain unchanged");
    }

    /// Additional test: Verify access_count saturates at u32::MAX.
    ///
    /// This ensures no overflow panic when an entry is accessed many times.
    #[test]
    fn test_access_count_saturation() {
        let codes = [0u64; 4];
        let now = 1700000000u64;
        let mut entry = CacheEntryPacked::new(codes, now);

        // Manually set access_count near max
        entry.access_count = u32::MAX - 1;

        // First increment should work
        entry.record_access(now);
        assert_eq!(entry.access_count(), u32::MAX, "access_count must reach MAX");

        // Second increment should saturate (not overflow)
        entry.record_access(now);
        assert_eq!(entry.access_count(), u32::MAX, "access_count must saturate at MAX");
    }

    /// Additional test: Verify timestamps before 2020 saturate to 0.
    ///
    /// This ensures no underflow panic for old timestamps.
    #[test]
    fn test_timestamp_before_epoch() {
        let codes = [0u64; 4];
        let old_time = 1500000000u64; // 2017-07-14 (before 2020 epoch)

        let entry = CacheEntryPacked::new(codes, old_time);

        // Should saturate to 0 (relative time) - copy to avoid unaligned reference
        let created_at_value = { entry.created_at };
        assert_eq!(created_at_value, 0, "Pre-epoch timestamp must saturate to 0");
        // Unix conversion returns epoch (not the original time)
        assert_eq!(entry.created_at_unix(), EPOCH_2020, "Pre-epoch timestamp converts to epoch");
    }
}

