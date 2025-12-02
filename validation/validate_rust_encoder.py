#!/usr/bin/env python3
"""
Rust Encoder Cross-Validation Script

PURPOSE:
    Validate that the Rust BinaryEncoder produces bit-exact output
    matching the Python BinaryEncoder for the same projection matrix.

EXPERIMENT:
    1. Generate 1,000 random embeddings
    2. Create projection matrix with seed=42
    3. Encode with both Python reference and Rust encoder
    4. Assert bit-exact match for every vector
    5. Measure and compare performance

PREREQUISITES:
    - Run `maturin develop` to build the Rust extension
    - Ensure numpy is installed

USAGE:
    python validation/validate_rust_encoder.py

OUTPUT:
    - Bit-exact match report (PASS/FAIL)
    - Speedup factor (Rust vs Python reference)
    - Detailed timing breakdown

Author: Scientist / Benchmark Specialist
Date: 2025-11-29
"""

import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np

# Add src to path
_SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))


# ==============================================================================
# CONFIGURATION
# ==============================================================================

CONFIG = {
    "n_vectors": 1000,           # Number of test vectors
    "embedding_dim": 384,        # Embedding dimension (matches Phase 1)
    "code_bits": 256,            # Binary code bits (matches Phase 1)
    "seed": 42,                  # Random seed for projection matrix
    "test_seed": 12345,          # Seed for generating test embeddings
    "warmup_iterations": 10,     # Warmup iterations before timing
}


# ==============================================================================
# PYTHON REFERENCE IMPLEMENTATION
# ==============================================================================

def generate_projection_matrix(
    code_bits: int, 
    embedding_dim: int, 
    seed: int = 42
) -> np.ndarray:
    """
    Generate Gaussian random projection matrix.
    
    Matches numpy's default_rng behavior for determinism.
    
    Args:
        code_bits: Number of output bits
        embedding_dim: Input embedding dimension
        seed: Random seed
        
    Returns:
        Matrix of shape (code_bits, embedding_dim), dtype float32
    """
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((code_bits, embedding_dim)).astype(np.float32)
    return matrix


def python_encode_reference(
    embeddings: np.ndarray,
    projection_matrix: np.ndarray,
    code_bits: int,
) -> np.ndarray:
    """
    Reference Python implementation of the encoding pipeline.
    
    FROZEN FORMULA: project → binarize → pack
    
    Args:
        embeddings: Float32 array of shape (N, embedding_dim)
        projection_matrix: Float32 matrix of shape (code_bits, embedding_dim)
        code_bits: Number of bits in output
        
    Returns:
        Packed uint64 codes of shape (N, n_words)
    """
    # Project: (N, embedding_dim) @ (embedding_dim, code_bits).T -> (N, code_bits)
    projected = embeddings @ projection_matrix.T
    
    # Binarize: 1 if >= 0, else 0
    binary = (projected >= 0).astype(np.uint8)
    
    # Pack into uint64 (LSB-first layout)
    n_words = (code_bits + 63) // 64
    n_samples = binary.shape[0]
    packed = np.zeros((n_samples, n_words), dtype=np.uint64)
    
    for sample_idx in range(n_samples):
        for bit_idx in range(code_bits):
            if binary[sample_idx, bit_idx]:
                word_idx = bit_idx // 64
                bit_pos = bit_idx % 64
                packed[sample_idx, word_idx] |= np.uint64(1) << np.uint64(bit_pos)
    
    return packed


# ==============================================================================
# VALIDATION LOGIC
# ==============================================================================

def check_rust_available() -> bool:
    """Check if Rust extension is available."""
    try:
        from binary_semantic_cache.binary_semantic_cache_rs import RustBinaryEncoder
        return True
    except ImportError:
        return False


def run_validation() -> Dict[str, Any]:
    """
    Run the full validation experiment.
    
    Returns:
        Dictionary with validation results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": CONFIG,
        "status": "UNKNOWN",
        "bit_exact_match": None,
        "n_matched": 0,
        "n_total": CONFIG["n_vectors"],
        "mismatches": [],
        "timing": {},
        "speedup": None,
    }
    
    print("=" * 70)
    print("RUST ENCODER CROSS-VALIDATION")
    print("=" * 70)
    print()
    
    # Check Rust availability
    if not check_rust_available():
        print("❌ FAILED: Rust extension not available.")
        print("   Run 'maturin develop' to build the extension first.")
        results["status"] = "FAILED"
        results["error"] = "Rust extension not available"
        return results
    
    print("✓ Rust extension available")
    
    # Import Rust encoder
    from binary_semantic_cache.binary_semantic_cache_rs import RustBinaryEncoder
    
    # Generate projection matrix (same for both encoders)
    print(f"\n[1/5] Generating projection matrix (seed={CONFIG['seed']})...")
    projection_matrix = generate_projection_matrix(
        CONFIG["code_bits"], 
        CONFIG["embedding_dim"], 
        CONFIG["seed"]
    )
    print(f"      Shape: {projection_matrix.shape}, dtype: {projection_matrix.dtype}")
    
    # Generate test embeddings
    print(f"\n[2/5] Generating {CONFIG['n_vectors']} test embeddings...")
    np.random.seed(CONFIG["test_seed"])
    embeddings = np.random.randn(CONFIG["n_vectors"], CONFIG["embedding_dim"]).astype(np.float32)
    # Normalize embeddings (realistic scenario)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    print(f"      Shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    
    # Create Rust encoder
    print(f"\n[3/5] Creating Rust encoder...")
    rust_encoder = RustBinaryEncoder(
        CONFIG["embedding_dim"], 
        CONFIG["code_bits"], 
        projection_matrix
    )
    print(f"      {rust_encoder}")
    
    # Warmup
    print(f"\n[4/5] Warmup ({CONFIG['warmup_iterations']} iterations)...")
    warmup_emb = embeddings[:10]
    for _ in range(CONFIG["warmup_iterations"]):
        _ = python_encode_reference(warmup_emb, projection_matrix, CONFIG["code_bits"])
        _ = rust_encoder.encode(warmup_emb)
    print("      Done")
    
    # Run validation
    print(f"\n[5/5] Running validation...")
    print("-" * 70)
    
    # Time Python reference
    start_py = time.perf_counter()
    python_codes = python_encode_reference(embeddings, projection_matrix, CONFIG["code_bits"])
    end_py = time.perf_counter()
    python_time = (end_py - start_py) * 1000  # ms
    
    results["timing"]["python_total_ms"] = python_time
    results["timing"]["python_per_vector_us"] = (python_time * 1000) / CONFIG["n_vectors"]
    
    print(f"      Python: {python_time:.2f} ms total, {results['timing']['python_per_vector_us']:.2f} µs/vector")
    
    # Time Rust encoder
    start_rs = time.perf_counter()
    rust_codes = rust_encoder.encode(embeddings)
    end_rs = time.perf_counter()
    rust_time = (end_rs - start_rs) * 1000  # ms
    
    results["timing"]["rust_total_ms"] = rust_time
    results["timing"]["rust_per_vector_us"] = (rust_time * 1000) / CONFIG["n_vectors"]
    
    print(f"      Rust:   {rust_time:.2f} ms total, {results['timing']['rust_per_vector_us']:.2f} µs/vector")
    
    # Calculate speedup
    if rust_time > 0:
        speedup = python_time / rust_time
        results["speedup"] = speedup
        print(f"\n      Speedup: {speedup:.1f}x")
    
    # Compare outputs
    print("\n" + "-" * 70)
    print("BIT-EXACT COMPARISON")
    print("-" * 70)
    
    n_matched = 0
    mismatches = []
    
    for i in range(CONFIG["n_vectors"]):
        py_code = python_codes[i]
        rs_code = rust_codes[i]
        
        if np.array_equal(py_code, rs_code):
            n_matched += 1
        else:
            mismatches.append({
                "index": i,
                "python": py_code.tolist(),
                "rust": rs_code.tolist(),
            })
            if len(mismatches) <= 5:  # Only print first 5 mismatches
                print(f"  MISMATCH at index {i}:")
                print(f"    Python: {py_code}")
                print(f"    Rust:   {rs_code}")
    
    results["n_matched"] = n_matched
    results["mismatches"] = mismatches[:10]  # Keep first 10 for report
    results["bit_exact_match"] = (n_matched == CONFIG["n_vectors"])
    
    print(f"\n      Matched: {n_matched}/{CONFIG['n_vectors']}")
    
    # Final status
    print("\n" + "=" * 70)
    if results["bit_exact_match"]:
        results["status"] = "PASSED"
        print("✅ VALIDATION PASSED")
        print(f"   - 100% bit-exact match ({n_matched}/{CONFIG['n_vectors']} vectors)")
        print(f"   - Rust speedup: {results['speedup']:.1f}x faster than Python reference")
    else:
        results["status"] = "FAILED"
        print("❌ VALIDATION FAILED")
        print(f"   - {len(mismatches)} mismatches found")
        print(f"   - Match rate: {100*n_matched/CONFIG['n_vectors']:.2f}%")
    print("=" * 70)
    
    return results


def save_results(results: Dict[str, Any]) -> Path:
    """Save results to JSON file."""
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "rust_encoder_validation.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    
    return output_file


def generate_report(results: Dict[str, Any]) -> str:
    """Generate human-readable report."""
    lines = [
        "# Rust Encoder Validation Report",
        "",
        f"**Date:** {results['timestamp']}",
        f"**Status:** {'✅ PASSED' if results['status'] == 'PASSED' else '❌ FAILED'}",
        "",
        "## Configuration",
        "",
        f"- Test Vectors: {results['config']['n_vectors']}",
        f"- Embedding Dim: {results['config']['embedding_dim']}",
        f"- Code Bits: {results['config']['code_bits']}",
        f"- Projection Seed: {results['config']['seed']}",
        "",
        "## Results",
        "",
        "### Bit-Exact Match",
        "",
        f"| Metric | Value |",
        f"|:---|:---|",
        f"| Matched | {results['n_matched']}/{results['n_total']} |",
        f"| Match Rate | {100*results['n_matched']/results['n_total']:.2f}% |",
        f"| Status | {'✅ PASS' if results['bit_exact_match'] else '❌ FAIL'} |",
        "",
        "### Performance",
        "",
        f"| Encoder | Total Time | Per Vector |",
        f"|:---|:---|:---|",
        f"| Python | {results['timing'].get('python_total_ms', 0):.2f} ms | {results['timing'].get('python_per_vector_us', 0):.2f} µs |",
        f"| Rust | {results['timing'].get('rust_total_ms', 0):.2f} ms | {results['timing'].get('rust_per_vector_us', 0):.2f} µs |",
        "",
        f"**Speedup:** {results.get('speedup', 0):.1f}x",
        "",
        "## Conclusion",
        "",
    ]
    
    if results["status"] == "PASSED":
        lines.extend([
            "The Rust encoder produces **bit-exact** output matching the Python reference.",
            "All Phase 1 contracts are preserved.",
            "",
            f"Rust is **{results.get('speedup', 0):.1f}x faster** than the Python reference implementation.",
        ])
    else:
        lines.extend([
            "**FAILURE:** The Rust encoder does NOT produce bit-exact output.",
            "",
            "Investigation required:",
            "1. Check projection matrix transfer",
            "2. Verify binarization threshold (>= 0)",
            "3. Verify LSB-first packing order",
        ])
    
    return "\n".join(lines)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print()
    
    # Run validation
    results = run_validation()
    
    # Save results
    output_file = save_results(results)
    print(f"\nResults saved to: {output_file}")
    
    # Generate report
    report = generate_report(results)
    report_file = output_file.with_suffix(".md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to: {report_file}")
    
    # Exit with appropriate code
    sys.exit(0 if results["status"] == "PASSED" else 1)

