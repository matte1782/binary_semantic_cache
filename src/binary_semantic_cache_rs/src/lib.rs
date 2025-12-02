//! Binary Semantic Cache - Rust Backend
//!
//! High-performance Rust implementation of the binary semantic cache core operations.
//! This module provides Python bindings via PyO3 for:
//! - Binary encoding with deterministic projection
//! - Hamming distance similarity (optimized with POPCNT)
//!
//! Phase 2 Goal: Replace performance-critical Python code with Rust for:
//! - Lookup < 0.5ms @ 100k entries
//! - Bit-exact compatibility with Python encoder (seed=42)

use pyo3::prelude::*;

mod encoder;
mod similarity;
pub mod storage;

pub use encoder::RustBinaryEncoder;
pub use similarity::HammingSimilarity;
pub use storage::RustCacheStorage;

/// Returns the version string of the Rust backend.
///
/// This is a placeholder function to verify that the Rust extension
/// is correctly built and can be imported from Python.
///
/// # Returns
/// A string containing the crate version.
#[pyfunction]
fn rust_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Returns a greeting message to verify the extension works.
///
/// # Arguments
/// * `name` - Optional name to include in greeting
///
/// # Returns
/// A greeting string
#[pyfunction]
#[pyo3(signature = (name=None))]
fn hello_from_rust(name: Option<&str>) -> String {
    match name {
        Some(n) => format!("Hello, {}! This is binary_semantic_cache_rs v{}", n, env!("CARGO_PKG_VERSION")),
        None => format!("Hello from binary_semantic_cache_rs v{}!", env!("CARGO_PKG_VERSION")),
    }
}

/// Binary Semantic Cache Rust extension module.
///
/// This module contains:
/// - `RustBinaryEncoder`: Binary encoding with deterministic projection
/// - `HammingSimilarity`: Hamming distance and similarity computation
/// - `RustCacheStorage`: Memory-efficient cache storage (44 bytes/entry)
/// - `hamming_distance`: Standalone Hamming distance function
/// - `rust_version`: Version info
/// - `hello_from_rust`: Test function
#[pymodule]
fn binary_semantic_cache_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_version, m)?)?;
    m.add_function(wrap_pyfunction!(hello_from_rust, m)?)?;
    m.add_function(wrap_pyfunction!(similarity::hamming_distance, m)?)?;
    m.add_class::<RustBinaryEncoder>()?;
    m.add_class::<HammingSimilarity>()?;
    m.add_class::<RustCacheStorage>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_version() {
        let version = rust_version();
        assert!(!version.is_empty());
        assert_eq!(version, "0.1.0");
    }

    #[test]
    fn test_hello_from_rust() {
        let greeting = hello_from_rust(None);
        assert!(greeting.contains("binary_semantic_cache_rs"));
        
        let named_greeting = hello_from_rust(Some("Test"));
        assert!(named_greeting.contains("Test"));
        assert!(named_greeting.contains("binary_semantic_cache_rs"));
    }
}

