//! Binary Encoder Module
//!
//! Converts float embeddings to compact binary codes using:
//! 1. Gaussian Random Projection (dimensionality reduction)
//! 2. Sign Binarization (float → {0, 1})
//! 3. Bit Packing (bits → u64 words)
//!
//! FROZEN FORMULA - Matches Python implementation exactly.
//! See: docs/DECISION_LOG_v1.md (D1, D2)

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

/// Default embedding dimension (matches Python)
#[allow(dead_code)]
const DEFAULT_EMBEDDING_DIM: usize = 384;
/// Default number of bits in binary code (matches Python)
#[allow(dead_code)]
const DEFAULT_CODE_BITS: usize = 256;

/// Binary encoder that converts float embeddings to packed binary codes.
///
/// This Rust implementation is designed for bit-exact compatibility with
/// the Python `BinaryEncoder` class. It accepts a pre-computed projection
/// matrix from Python to guarantee deterministic output matching.
///
/// # Architecture
///
/// The encoding pipeline is:
/// 1. **Project:** `projected = embedding @ projection_matrix.T`
/// 2. **Binarize:** `bits = (projected >= 0) ? 1 : 0`
/// 3. **Pack:** Pack bits into u64 words (LSB-first)
///
/// # Example (Python)
///
/// ```python
/// import numpy as np
/// from binary_semantic_cache import binary_semantic_cache_rs as rs
///
/// # Generate projection matrix in Python for determinism
/// rng = np.random.default_rng(42)
/// projection = rng.standard_normal((256, 384)).astype(np.float32)
///
/// # Create Rust encoder with the matrix
/// encoder = rs.RustBinaryEncoder(384, 256, projection)
///
/// # Encode embeddings
/// embedding = np.random.randn(384).astype(np.float32)
/// code = encoder.encode(embedding)
/// ```
#[pyclass]
pub struct RustBinaryEncoder {
    /// Input embedding dimension
    embedding_dim: usize,
    /// Number of bits in output binary code
    code_bits: usize,
    /// Number of u64 words in packed output
    n_words: usize,
    /// Projection matrix of shape (code_bits, embedding_dim)
    /// Stored as row-major for efficient access
    projection_matrix: Array2<f32>,
}

#[pymethods]
impl RustBinaryEncoder {
    /// Create a new binary encoder.
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - Dimension of input float embeddings
    /// * `code_bits` - Number of bits in output binary code
    /// * `projection_matrix` - Optional pre-computed projection matrix of shape (code_bits, embedding_dim)
    /// * `seed` - Optional seed to generate projection matrix (if matrix not provided)
    ///
    /// # Raises
    ///
    /// * `ValueError` - If dimensions don't match or values are invalid
    #[new]
    #[pyo3(signature = (embedding_dim, code_bits, projection_matrix=None, seed=None))]
    fn new(
        py: Python<'_>,
        embedding_dim: usize,
        code_bits: usize,
        projection_matrix: Option<PyReadonlyArray2<f32>>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        // Validate embedding_dim
        if embedding_dim == 0 {
            return Err(PyValueError::new_err(
                "embedding_dim must be positive",
            ));
        }

        // Validate code_bits
        if code_bits == 0 {
            return Err(PyValueError::new_err("code_bits must be positive"));
        }

        let matrix_owned = if let Some(m) = projection_matrix {
            // Get matrix as ndarray view
            let matrix = m.as_array();

            // Validate matrix shape
            if matrix.shape() != [code_bits, embedding_dim] {
                return Err(PyValueError::new_err(format!(
                    "projection_matrix shape must be ({}, {}), got {:?}",
                    code_bits,
                    embedding_dim,
                    matrix.shape()
                )));
            }

            // Check for non-finite values
            if !matrix.iter().all(|&x| x.is_finite()) {
                return Err(PyValueError::new_err(
                    "projection_matrix contains non-finite values (NaN or Inf)",
                ));
            }
            
            matrix.to_owned()
        } else if let Some(s) = seed {
            // Generate matrix using numpy to match Python implementation exactly
            // This ensures bit-exact compatibility with Python's BinaryEncoder(seed=N)
            let numpy = py.import_bound("numpy")?;
            let np_random = numpy.getattr("random")?;
            let rng = np_random.call_method1("default_rng", (s,))?;
            
            // Generate standard normal distribution: shape (embedding_dim, code_bits)
            // CRITICAL: Must pass dtype=np.float32 directly to standard_normal()
            // because rng.standard_normal((shape), dtype=float32) produces DIFFERENT
            // values than rng.standard_normal((shape)).astype(float32).
            // Python's RandomProjection uses dtype=np.float32 directly.
            let shape = (embedding_dim, code_bits);
            let float32 = numpy.getattr("float32")?;
            
            // Create kwargs dict with dtype=np.float32
            let kwargs = pyo3::types::PyDict::new_bound(py);
            kwargs.set_item("dtype", float32)?;
            
            // Call standard_normal with dtype kwarg
            let matrix_obj = rng.call_method("standard_normal", (shape,), Some(&kwargs))?;
            
            // Transpose to (code_bits, embedding_dim) for efficient row-wise dot product in Rust
            // Python: x @ weights (1xDim @ DimxBits)
            // Rust: weights.T @ x (BitsxDim @ Dimx1)
            let matrix_t = matrix_obj.getattr("T")?;
            
            // Extract as PyReadonlyArray2 to convert to ndarray Array2
            let array_readonly: PyReadonlyArray2<f32> = matrix_t.extract()?;
            let matrix = array_readonly.as_array();
            
            matrix.to_owned()
        } else {
            return Err(PyValueError::new_err(
                "Must provide either projection_matrix or seed",
            ));
        };

        // Calculate n_words (ceiling division)
        let n_words = (code_bits + 63) / 64;

        Ok(RustBinaryEncoder {
            embedding_dim,
            code_bits,
            n_words,
            projection_matrix: matrix_owned,
        })
    }

    /// Get the embedding dimension.
    #[getter]
    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Get the number of code bits.
    #[getter]
    fn code_bits(&self) -> usize {
        self.code_bits
    }

    /// Get the number of u64 words in output.
    #[getter]
    fn n_words(&self) -> usize {
        self.n_words
    }

    /// Encode a single embedding or batch of embeddings.
    ///
    /// Automatically detects single vs batch based on input dimensions:
    /// - 1D input (embedding_dim,) → returns 1D output (n_words,)
    /// - 2D input (N, embedding_dim) → returns 2D output (N, n_words)
    ///
    /// # Arguments
    ///
    /// * `embedding` - Float32 array of shape (embedding_dim,) or (N, embedding_dim)
    ///
    /// # Returns
    ///
    /// Packed binary codes as u64 array.
    ///
    /// # Raises
    ///
    /// * `TypeError` - If input is not a numpy array
    /// * `ValueError` - If shape or dtype is wrong
    fn encode<'py>(
        &self,
        py: Python<'py>,
        embedding: &Bound<'py, PyAny>,
    ) -> PyResult<PyObject> {
        // Try to interpret as 1D array first
        if let Ok(arr1d) = embedding.extract::<PyReadonlyArray1<f32>>() {
            let result = self.encode_single_impl(arr1d.as_array())?;
            return Ok(result.to_pyarray_bound(py).into());
        }

        // Try to interpret as 2D array
        if let Ok(arr2d) = embedding.extract::<PyReadonlyArray2<f32>>() {
            let result = self.encode_batch_impl(arr2d.as_array())?;
            return Ok(result.to_pyarray_bound(py).into());
        }

        Err(PyTypeError::new_err(
            "Expected numpy array of float32 with 1 or 2 dimensions",
        ))
    }

    /// Encode a batch of embeddings.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Float32 array of shape (N, embedding_dim)
    ///
    /// # Returns
    ///
    /// Packed binary codes as u64 array of shape (N, n_words).
    fn encode_batch<'py>(
        &self,
        py: Python<'py>,
        embeddings: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<u64>>> {
        let result = self.encode_batch_impl(embeddings.as_array())?;
        Ok(result.to_pyarray_bound(py))
    }

    /// String representation.
    fn __repr__(&self) -> String {
        format!(
            "RustBinaryEncoder(embedding_dim={}, code_bits={})",
            self.embedding_dim, self.code_bits
        )
    }
}

impl RustBinaryEncoder {
    /// Internal implementation for single embedding encoding.
    fn encode_single_impl(&self, embedding: ArrayView1<f32>) -> PyResult<Array1<u64>> {
        // Validate shape
        if embedding.len() != self.embedding_dim {
            return Err(PyValueError::new_err(format!(
                "Expected embedding dim {}, got {}",
                self.embedding_dim,
                embedding.len()
            )));
        }

        // Check for non-finite values
        if !embedding.iter().all(|&x| x.is_finite()) {
            return Err(PyValueError::new_err(
                "Embedding contains non-finite values (NaN or Inf)",
            ));
        }

        // Project: projected[i] = sum(embedding[j] * projection_matrix[i, j])
        let projected = self.project_single(&embedding);

        // Binarize and pack
        let packed = self.binarize_and_pack_single(&projected);

        Ok(packed)
    }

    /// Internal implementation for batch embedding encoding.
    fn encode_batch_impl(&self, embeddings: ArrayView2<f32>) -> PyResult<Array2<u64>> {
        // Validate shape
        if embeddings.ncols() != self.embedding_dim {
            return Err(PyValueError::new_err(format!(
                "Expected embedding dim {}, got {}",
                self.embedding_dim,
                embeddings.ncols()
            )));
        }

        // Check for non-finite values
        if !embeddings.iter().all(|&x| x.is_finite()) {
            return Err(PyValueError::new_err(
                "Embeddings contain non-finite values (NaN or Inf)",
            ));
        }

        let batch_size = embeddings.nrows();
        let mut result = Array2::<u64>::zeros((batch_size, self.n_words));

        // Process each embedding
        for (i, embedding) in embeddings.axis_iter(Axis(0)).enumerate() {
            let projected = self.project_single(&embedding);
            let packed = self.binarize_and_pack_single(&projected);
            result.row_mut(i).assign(&packed);
        }

        Ok(result)
    }

    /// Project a single embedding through the projection matrix.
    ///
    /// Computes: projected = embedding @ projection_matrix.T
    /// Which is equivalent to: projected[i] = sum(embedding[j] * projection_matrix[i, j])
    #[inline]
    fn project_single(&self, embedding: &ArrayView1<f32>) -> Array1<f32> {
        // Manual matrix-vector multiply for optimal performance
        // projected[i] = dot(projection_matrix.row(i), embedding)
        let mut projected = Array1::<f32>::zeros(self.code_bits);

        for (i, proj_val) in projected.iter_mut().enumerate() {
            let row = self.projection_matrix.row(i);
            let mut sum: f32 = 0.0;
            for (a, b) in row.iter().zip(embedding.iter()) {
                sum += a * b;
            }
            *proj_val = sum;
        }

        projected
    }

    /// Binarize projected values and pack into u64 words.
    ///
    /// Binarization: bit = 1 if projected >= 0, else 0
    /// Packing: LSB-first layout (bit 0 → LSB of word 0)
    #[inline]
    fn binarize_and_pack_single(&self, projected: &Array1<f32>) -> Array1<u64> {
        let mut packed = Array1::<u64>::zeros(self.n_words);

        for (bit_idx, &proj_val) in projected.iter().enumerate() {
            // Binarize: 1 if >= 0, else 0
            if proj_val >= 0.0 {
                // Determine word index and bit position within word
                let word_idx = bit_idx / 64;
                let bit_pos = bit_idx % 64;

                // Set the bit (LSB-first layout)
                packed[word_idx] |= 1u64 << bit_pos;
            }
        }

        packed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Helper to create a simple test projection matrix
    fn make_test_projection(code_bits: usize, embedding_dim: usize) -> Array2<f32> {
        // Create a simple deterministic matrix for testing
        let mut matrix = Array2::<f32>::zeros((code_bits, embedding_dim));
        for i in 0..code_bits {
            for j in 0..embedding_dim {
                // Simple pattern: alternating positive/negative
                matrix[[i, j]] = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            }
        }
        matrix
    }

    #[test]
    fn test_encoder_creation() {
        let projection = make_test_projection(256, 384);
        let encoder = RustBinaryEncoder {
            embedding_dim: 384,
            code_bits: 256,
            n_words: 4,
            projection_matrix: projection,
        };

        assert_eq!(encoder.embedding_dim, 384);
        assert_eq!(encoder.code_bits, 256);
        assert_eq!(encoder.n_words, 4);
    }

    #[test]
    fn test_n_words_calculation() {
        // 64 bits = 1 word
        assert_eq!((64 + 63) / 64, 1);
        // 128 bits = 2 words
        assert_eq!((128 + 63) / 64, 2);
        // 256 bits = 4 words
        assert_eq!((256 + 63) / 64, 4);
        // 257 bits = 5 words
        assert_eq!((257 + 63) / 64, 5);
    }

    #[test]
    fn test_binarize_and_pack() {
        let projection = make_test_projection(64, 4);
        let encoder = RustBinaryEncoder {
            embedding_dim: 4,
            code_bits: 64,
            n_words: 1,
            projection_matrix: projection,
        };

        // Test with known projected values
        let mut projected = Array1::<f32>::zeros(64);
        // Set first 8 bits to specific pattern: 0,1,0,1,0,1,0,1 (based on sign)
        projected[0] = -1.0; // bit 0 = 0
        projected[1] = 1.0;  // bit 1 = 1
        projected[2] = -1.0; // bit 2 = 0
        projected[3] = 1.0;  // bit 3 = 1
        projected[4] = -1.0; // bit 4 = 0
        projected[5] = 1.0;  // bit 5 = 1
        projected[6] = -1.0; // bit 6 = 0
        projected[7] = 1.0;  // bit 7 = 1

        let packed = encoder.binarize_and_pack_single(&projected);

        // Expected: bits 1,3,5,7 are set = 0b10101010 = 170 in low byte
        // But rest are 0, so result should be 0xAA = 170
        assert_eq!(packed[0] & 0xFF, 0xAA);
    }

    #[test]
    fn test_encode_single_shape() {
        let projection = make_test_projection(256, 384);
        let encoder = RustBinaryEncoder {
            embedding_dim: 384,
            code_bits: 256,
            n_words: 4,
            projection_matrix: projection,
        };

        let embedding = Array1::<f32>::zeros(384);
        let result = encoder.encode_single_impl(embedding.view()).unwrap();

        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_encode_batch_shape() {
        let projection = make_test_projection(256, 384);
        let encoder = RustBinaryEncoder {
            embedding_dim: 384,
            code_bits: 256,
            n_words: 4,
            projection_matrix: projection,
        };

        let embeddings = Array2::<f32>::zeros((10, 384));
        let result = encoder.encode_batch_impl(embeddings.view()).unwrap();

        assert_eq!(result.shape(), [10, 4]);
    }

    #[test]
    fn test_deterministic_output() {
        let projection = make_test_projection(256, 384);
        let encoder = RustBinaryEncoder {
            embedding_dim: 384,
            code_bits: 256,
            n_words: 4,
            projection_matrix: projection,
        };

        // Same input should give same output
        let embedding = Array1::from_vec(vec![0.5f32; 384]);
        
        let result1 = encoder.encode_single_impl(embedding.view()).unwrap();
        let result2 = encoder.encode_single_impl(embedding.view()).unwrap();

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_wrong_embedding_dim() {
        let projection = make_test_projection(256, 384);
        let encoder = RustBinaryEncoder {
            embedding_dim: 384,
            code_bits: 256,
            n_words: 4,
            projection_matrix: projection,
        };

        let wrong_embedding = Array1::<f32>::zeros(256); // Wrong size
        let result = encoder.encode_single_impl(wrong_embedding.view());

        assert!(result.is_err());
    }

    #[test]
    fn test_nan_input_rejected() {
        let projection = make_test_projection(256, 384);
        let encoder = RustBinaryEncoder {
            embedding_dim: 384,
            code_bits: 256,
            n_words: 4,
            projection_matrix: projection,
        };

        let mut embedding = Array1::<f32>::zeros(384);
        embedding[0] = f32::NAN;
        let result = encoder.encode_single_impl(embedding.view());

        assert!(result.is_err());
    }

    #[test]
    fn test_inf_input_rejected() {
        let projection = make_test_projection(256, 384);
        let encoder = RustBinaryEncoder {
            embedding_dim: 384,
            code_bits: 256,
            n_words: 4,
            projection_matrix: projection,
        };

        let mut embedding = Array1::<f32>::zeros(384);
        embedding[0] = f32::INFINITY;
        let result = encoder.encode_single_impl(embedding.view());

        assert!(result.is_err());
    }
}

