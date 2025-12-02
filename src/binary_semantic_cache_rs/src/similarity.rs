//! Hamming Similarity Module
//!
//! Computes normalized Hamming similarity between binary codes.
//! Optimized for maximum throughput using:
//! - Native `count_ones()` which compiles to POPCNT instruction
//! - Zero allocations in the hot path
//! - Cache-friendly memory access patterns
//!
//! FROZEN FORMULA: similarity = 1.0 - (hamming_distance / code_bits)
//! See: docs/DECISION_LOG_v1.md (D3, D5)

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Hamming similarity calculator for binary codes.
///
/// This Rust implementation is designed for bit-exact compatibility with
/// the Python `hamming_similarity` function while providing 2x+ speedup.
///
/// # FROZEN FORMULA
///
/// ```text
/// similarity = 1.0 - (hamming_distance / code_bits)
/// ```
///
/// # Example (Python)
///
/// ```python
/// import numpy as np
/// from binary_semantic_cache import binary_semantic_cache_rs as rs
///
/// # Create similarity calculator
/// sim = rs.HammingSimilarity(code_bits=256)
///
/// # Compute similarities
/// query = np.array([0x1234, 0x5678, 0x9ABC, 0xDEF0], dtype=np.uint64)
/// codes = np.random.randint(0, 2**64, size=(1000, 4), dtype=np.uint64)
/// similarities = sim.similarity_batch(query, codes)
///
/// # Find best match above threshold
/// result = sim.find_nearest(query, codes, threshold=0.85)
/// if result:
///     index, score = result
///     print(f"Found match at {index} with similarity {score:.3f}")
/// ```
#[pyclass]
pub struct HammingSimilarity {
    /// Number of bits in each binary code
    code_bits: usize,
}

#[pymethods]
impl HammingSimilarity {
    /// Create a new Hamming similarity calculator.
    ///
    /// # Arguments
    ///
    /// * `code_bits` - Total number of bits in each binary code (e.g., 256)
    ///
    /// # Raises
    ///
    /// * `ValueError` - If code_bits is zero or negative
    #[new]
    #[pyo3(signature = (code_bits=256))]
    fn new(code_bits: usize) -> PyResult<Self> {
        if code_bits == 0 {
            return Err(PyValueError::new_err("code_bits must be positive"));
        }
        Ok(HammingSimilarity { code_bits })
    }

    /// Get the number of code bits.
    #[getter]
    fn code_bits(&self) -> usize {
        self.code_bits
    }

    /// Compute Hamming distance between query and a single code.
    ///
    /// # Arguments
    ///
    /// * `query` - Binary code as uint64 array of shape (n_words,)
    /// * `code` - Binary code as uint64 array of shape (n_words,)
    ///
    /// # Returns
    ///
    /// Hamming distance (number of differing bits)
    fn distance(&self, query: PyReadonlyArray1<u64>, code: PyReadonlyArray1<u64>) -> PyResult<u32> {
        let q = query.as_slice()?;
        let c = code.as_slice()?;

        if q.len() != c.len() {
            return Err(PyValueError::new_err(format!(
                "query has {} words but code has {} words",
                q.len(),
                c.len()
            )));
        }

        Ok(hamming_distance_single(q, c))
    }

    /// Compute Hamming distances between query and all codes.
    ///
    /// # Arguments
    ///
    /// * `query` - Binary code as uint64 array of shape (n_words,)
    /// * `codes` - Database of binary codes as uint64 array of shape (n_entries, n_words)
    ///
    /// # Returns
    ///
    /// Hamming distances as uint32 array of shape (n_entries,)
    fn distance_batch<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<u64>,
        codes: PyReadonlyArray2<u64>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let q = query.as_slice()?;
        let codes_arr = codes.as_array();

        // Check for C-contiguous layout to avoid panics
        if !codes_arr.is_standard_layout() {
            return Err(PyValueError::new_err(
                "codes must be C-contiguous (row-major). Use np.ascontiguousarray(codes)"
            ));
        }

        // Validate dimensions
        if codes_arr.ncols() != q.len() {
            return Err(PyValueError::new_err(format!(
                "query has {} words but codes have {} words",
                q.len(),
                codes_arr.ncols()
            )));
        }

        // Compute distances
        let distances = hamming_distance_batch_impl(q, &codes_arr);

        Ok(distances.to_pyarray_bound(py))
    }

    /// Compute normalized Hamming similarity between query and a single code.
    ///
    /// FROZEN FORMULA: similarity = 1.0 - (hamming_distance / code_bits)
    ///
    /// # Arguments
    ///
    /// * `query` - Binary code as uint64 array of shape (n_words,)
    /// * `code` - Binary code as uint64 array of shape (n_words,)
    ///
    /// # Returns
    ///
    /// Similarity score in range [0.0, 1.0]
    fn similarity(
        &self,
        query: PyReadonlyArray1<u64>,
        code: PyReadonlyArray1<u64>,
    ) -> PyResult<f32> {
        let distance = self.distance(query, code)?;
        Ok(distance_to_similarity(distance, self.code_bits))
    }

    /// Compute normalized Hamming similarities between query and all codes.
    ///
    /// FROZEN FORMULA: similarity = 1.0 - (hamming_distance / code_bits)
    ///
    /// # Arguments
    ///
    /// * `query` - Binary code as uint64 array of shape (n_words,)
    /// * `codes` - Database of binary codes as uint64 array of shape (n_entries, n_words)
    ///
    /// # Returns
    ///
    /// Similarity scores as float32 array of shape (n_entries,)
    fn similarity_batch<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<u64>,
        codes: PyReadonlyArray2<u64>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let q = query.as_slice()?;
        let codes_arr = codes.as_array();

        // Check for C-contiguous layout to avoid panics
        if !codes_arr.is_standard_layout() {
            return Err(PyValueError::new_err(
                "codes must be C-contiguous (row-major). Use np.ascontiguousarray(codes)"
            ));
        }

        // Validate dimensions
        if codes_arr.ncols() != q.len() {
            return Err(PyValueError::new_err(format!(
                "query has {} words but codes have {} words",
                q.len(),
                codes_arr.ncols()
            )));
        }

        // Compute similarities
        let n_entries = codes_arr.nrows();
        let mut similarities = Vec::with_capacity(n_entries);

        for i in 0..n_entries {
            let code_row = codes_arr.row(i);
            let distance = hamming_distance_slice(q, code_row.as_slice().unwrap());
            similarities.push(distance_to_similarity(distance, self.code_bits));
        }

        Ok(similarities.to_pyarray_bound(py))
    }

    /// Find the nearest code above similarity threshold.
    ///
    /// # Arguments
    ///
    /// * `query` - Binary code as uint64 array of shape (n_words,)
    /// * `codes` - Database of binary codes as uint64 array of shape (n_entries, n_words)
    /// * `threshold` - Minimum similarity for a match (default: 0.85)
    ///
    /// # Returns
    ///
    /// Tuple of (index, similarity) if found, None if no match above threshold
    #[pyo3(signature = (query, codes, threshold=0.85))]
    fn find_nearest(
        &self,
        query: PyReadonlyArray1<u64>,
        codes: PyReadonlyArray2<u64>,
        threshold: f32,
    ) -> PyResult<Option<(usize, f32)>> {
        // Validate threshold
        if threshold.is_nan() || threshold < 0.0 || threshold > 1.0 {
            return Err(PyValueError::new_err(format!(
                "threshold must be in [0.0, 1.0], got {}",
                threshold
            )));
        }

        let q = query.as_slice()?;
        let codes_arr = codes.as_array();

        // Check for C-contiguous layout to avoid panics
        if !codes_arr.is_standard_layout() {
            return Err(PyValueError::new_err(
                "codes must be C-contiguous (row-major). Use np.ascontiguousarray(codes)"
            ));
        }

        // Handle empty codes
        if codes_arr.nrows() == 0 {
            return Ok(None);
        }

        // Validate dimensions
        if codes_arr.ncols() != q.len() {
            return Err(PyValueError::new_err(format!(
                "query has {} words but codes have {} words",
                q.len(),
                codes_arr.ncols()
            )));
        }

        // Find nearest with early termination on exact match
        let result = find_nearest_impl(q, &codes_arr, self.code_bits, threshold);

        Ok(result)
    }

    /// String representation.
    fn __repr__(&self) -> String {
        format!("HammingSimilarity(code_bits={})", self.code_bits)
    }
}

// =============================================================================
// Internal Implementation (No allocations in hot path)
// =============================================================================

/// Compute Hamming distance between two slices using count_ones().
///
/// This compiles to the POPCNT instruction on modern CPUs.
#[inline(always)]
fn hamming_distance_single(a: &[u64], b: &[u64]) -> u32 {
    let mut distance = 0u32;
    for i in 0..a.len() {
        distance += (a[i] ^ b[i]).count_ones();
    }
    distance
}

/// Compute Hamming distance between slice and row view.
#[inline(always)]
fn hamming_distance_slice(a: &[u64], b: &[u64]) -> u32 {
    let mut distance = 0u32;
    for i in 0..a.len() {
        distance += (a[i] ^ b[i]).count_ones();
    }
    distance
}

/// Convert Hamming distance to similarity using FROZEN FORMULA.
///
/// FROZEN FORMULA: similarity = 1.0 - (hamming_distance / code_bits)
#[inline(always)]
fn distance_to_similarity(distance: u32, code_bits: usize) -> f32 {
    1.0 - (distance as f32 / code_bits as f32)
}

/// Batch Hamming distance computation.
fn hamming_distance_batch_impl(
    query: &[u64],
    codes: &ndarray::ArrayView2<u64>,
) -> Vec<u32> {
    let n_entries = codes.nrows();
    let mut distances = Vec::with_capacity(n_entries);

    for i in 0..n_entries {
        let code_row = codes.row(i);
        let distance = hamming_distance_slice(query, code_row.as_slice().unwrap());
        distances.push(distance);
    }

    distances
}

/// Find nearest implementation with optional early termination.
fn find_nearest_impl(
    query: &[u64],
    codes: &ndarray::ArrayView2<u64>,
    code_bits: usize,
    threshold: f32,
) -> Option<(usize, f32)> {
    let n_entries = codes.nrows();
    
    let mut best_idx: Option<usize> = None;
    let mut best_sim: f32 = f32::NEG_INFINITY;

    for i in 0..n_entries {
        let code_row = codes.row(i);
        let distance = hamming_distance_slice(query, code_row.as_slice().unwrap());
        let similarity = distance_to_similarity(distance, code_bits);

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
        Some(idx) if best_sim >= threshold => Some((idx, best_sim)),
        _ => None,
    }
}

// =============================================================================
// Standalone Functions (for direct use without struct)
// =============================================================================

/// Compute Hamming distance between two binary codes.
///
/// This is a standalone function for simple use cases.
///
/// # Arguments
///
/// * `a` - First binary code as uint64 array
/// * `b` - Second binary code as uint64 array
///
/// # Returns
///
/// Hamming distance (number of differing bits)
#[pyfunction]
pub fn hamming_distance(a: PyReadonlyArray1<u64>, b: PyReadonlyArray1<u64>) -> PyResult<u32> {
    let a_slice = a.as_slice()?;
    let b_slice = b.as_slice()?;

    if a_slice.len() != b_slice.len() {
        return Err(PyValueError::new_err(format!(
            "arrays have different lengths: {} vs {}",
            a_slice.len(),
            b_slice.len()
        )));
    }

    Ok(hamming_distance_single(a_slice, b_slice))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_distance_identical() {
        // S1: Identical codes should have distance 0
        let a = [0xFFFFFFFFFFFFFFFFu64, 0x0];
        let b = [0xFFFFFFFFFFFFFFFFu64, 0x0];
        assert_eq!(hamming_distance_single(&a, &b), 0);
    }

    #[test]
    fn test_hamming_distance_inverted() {
        // S2: Inverted codes should have distance = all bits
        let a = [0xFFFFFFFFFFFFFFFFu64; 4];
        let b = [0x0u64; 4];
        assert_eq!(hamming_distance_single(&a, &b), 256); // 4 * 64 = 256
    }

    #[test]
    fn test_hamming_distance_single_bit() {
        // S3: Single bit difference should have distance 1
        let a = [0x1u64, 0, 0, 0];
        let b = [0x0u64, 0, 0, 0];
        assert_eq!(hamming_distance_single(&a, &b), 1);
    }

    #[test]
    fn test_hamming_distance_known() {
        // S4: Known pattern 0b1010 vs 0b1100 = distance 2
        let a = [0b1010u64, 0, 0, 0];
        let b = [0b1100u64, 0, 0, 0];
        assert_eq!(hamming_distance_single(&a, &b), 2);
    }

    #[test]
    fn test_similarity_identical() {
        // S5: Identical codes → 1.0
        let distance = 0;
        let code_bits = 256;
        let sim = distance_to_similarity(distance, code_bits);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_similarity_inverted() {
        // S6: Inverted codes → 0.0
        let distance = 256;
        let code_bits = 256;
        let sim = distance_to_similarity(distance, code_bits);
        assert!((sim - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_similarity_half() {
        // S7: Half bits different → 0.5
        let distance = 128;
        let code_bits = 256;
        let sim = distance_to_similarity(distance, code_bits);
        assert!((sim - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_similarity_range() {
        // S8: Similarity should always be in [0.0, 1.0]
        for distance in 0..=256 {
            let sim = distance_to_similarity(distance, 256);
            assert!(sim >= 0.0 && sim <= 1.0);
        }
    }

    #[test]
    fn test_count_ones_correctness() {
        // Verify count_ones matches expected popcount
        assert_eq!(0u64.count_ones(), 0);
        assert_eq!(1u64.count_ones(), 1);
        assert_eq!(0xFFu64.count_ones(), 8);
        assert_eq!(0xFFFFFFFFFFFFFFFFu64.count_ones(), 64);
        assert_eq!(0xAAAAAAAAAAAAAAAAu64.count_ones(), 32); // Alternating bits
    }

    #[test]
    fn test_formula_matches_python() {
        // FROZEN FORMULA: similarity = 1.0 - (hamming_distance / code_bits)
        // This is the exact formula used in Python
        let test_cases = [
            (0, 256, 1.0f32),
            (256, 256, 0.0f32),
            (128, 256, 0.5f32),
            (64, 256, 0.75f32),
            (32, 256, 0.875f32),
        ];

        for (distance, code_bits, expected) in test_cases {
            let actual = distance_to_similarity(distance, code_bits);
            assert!(
                (actual - expected).abs() < 1e-6,
                "distance={}, expected={}, got={}",
                distance,
                expected,
                actual
            );
        }
    }
}

