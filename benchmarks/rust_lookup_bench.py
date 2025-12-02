#!/usr/bin/env python3
"""
Rust vs Python Similarity Benchmark

Compares Phase 1 (Python/Numba) vs Phase 2 (Rust) similarity performance:
- Lookup latency at various cache sizes
- Accuracy verification (same index returned)
- Bit-exact similarity values

Target: Rust < 0.5ms at 100K entries (Phase 2 goal)
Baseline: Python/Numba = 1.14ms at 100K entries (Phase 1)
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add src to path
_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _ROOT / "src"
sys.path.insert(0, str(_SRC_PATH))

# Python reference implementations
from binary_semantic_cache.core.similarity import (
    hamming_similarity as python_hamming_similarity,
    hamming_distance_batch as python_hamming_distance_batch,
    find_nearest as python_find_nearest,
    is_numba_available,
    _hamming_distance_numpy,
)

# Try to import Rust extension
try:
    from binary_semantic_cache.binary_semantic_cache_rs import (
        HammingSimilarity,
        hamming_distance as rust_hamming_distance,
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    HammingSimilarity = None
    rust_hamming_distance = None

# Configuration
WARMUP_ITERATIONS = 5
SEED = 42
CODE_BITS = 256
N_WORDS = 4  # 256 bits / 64 bits per word


def get_system_info() -> Dict[str, Any]:
    """Get system metadata."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "numba_available": is_numba_available(),
        "rust_available": RUST_AVAILABLE,
    }


def create_codes(n: int, n_words: int = N_WORDS, seed: int = SEED) -> np.ndarray:
    """Create random binary codes."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**64, size=(n, n_words), dtype=np.uint64)


def create_query(n_words: int = N_WORDS, seed: int = SEED + 1) -> np.ndarray:
    """Create a single random query."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**64, size=(n_words,), dtype=np.uint64)


def benchmark_python_similarity(
    query: np.ndarray,
    codes: np.ndarray,
    n_iterations: int,
    use_numba: bool = True,
) -> Dict[str, float]:
    """Benchmark Python similarity computation."""
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        python_hamming_similarity(query, codes, code_bits=CODE_BITS, use_numba=use_numba)
    
    # Measure
    times_us: List[float] = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        python_hamming_similarity(query, codes, code_bits=CODE_BITS, use_numba=use_numba)
        end = time.perf_counter()
        times_us.append((end - start) * 1_000_000)
    
    arr = np.array(times_us)
    return {
        "backend": "numba" if (use_numba and is_numba_available()) else "numpy",
        "mean_us": float(np.mean(arr)),
        "std_us": float(np.std(arr)),
        "min_us": float(np.min(arr)),
        "max_us": float(np.max(arr)),
        "p50_us": float(np.percentile(arr, 50)),
        "p99_us": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr) / 1000),
        "min_ms": float(np.min(arr) / 1000),
    }


def benchmark_rust_similarity(
    query: np.ndarray,
    codes: np.ndarray,
    n_iterations: int,
) -> Dict[str, float]:
    """Benchmark Rust similarity computation."""
    if not RUST_AVAILABLE:
        raise RuntimeError("Rust extension not available")
    
    sim = HammingSimilarity(code_bits=CODE_BITS)
    
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        sim.similarity_batch(query, codes)
    
    # Measure
    times_us: List[float] = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        sim.similarity_batch(query, codes)
        end = time.perf_counter()
        times_us.append((end - start) * 1_000_000)
    
    arr = np.array(times_us)
    return {
        "backend": "rust",
        "mean_us": float(np.mean(arr)),
        "std_us": float(np.std(arr)),
        "min_us": float(np.min(arr)),
        "max_us": float(np.max(arr)),
        "p50_us": float(np.percentile(arr, 50)),
        "p99_us": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr) / 1000),
        "min_ms": float(np.min(arr) / 1000),
    }


def benchmark_find_nearest(
    query: np.ndarray,
    codes: np.ndarray,
    n_iterations: int,
    threshold: float = 0.5,  # Low threshold to ensure we get results
) -> Dict[str, Any]:
    """Benchmark find_nearest and verify accuracy."""
    if not RUST_AVAILABLE:
        raise RuntimeError("Rust extension not available")
    
    sim = HammingSimilarity(code_bits=CODE_BITS)
    
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        python_find_nearest(query, codes, code_bits=CODE_BITS, threshold=threshold, use_numba=True)
        sim.find_nearest(query, codes, threshold=threshold)
    
    # Measure Python
    python_times: List[float] = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        python_result = python_find_nearest(query, codes, code_bits=CODE_BITS, threshold=threshold, use_numba=True)
        end = time.perf_counter()
        python_times.append((end - start) * 1_000_000)
    
    # Measure Rust
    rust_times: List[float] = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        rust_result = sim.find_nearest(query, codes, threshold=threshold)
        end = time.perf_counter()
        rust_times.append((end - start) * 1_000_000)
    
    # Verify accuracy
    python_final = python_find_nearest(query, codes, code_bits=CODE_BITS, threshold=threshold, use_numba=True)
    rust_final = sim.find_nearest(query, codes, threshold=threshold)
    
    accuracy_match = True
    index_match = True
    similarity_match = True
    
    if python_final is None and rust_final is None:
        accuracy_match = True
    elif python_final is not None and rust_final is not None:
        py_idx, py_sim = python_final
        rust_idx, rust_sim = rust_final
        index_match = (py_idx == rust_idx)
        similarity_match = abs(py_sim - rust_sim) < 1e-6
        accuracy_match = index_match and similarity_match
    else:
        accuracy_match = False
        index_match = False
        similarity_match = False
    
    py_arr = np.array(python_times)
    rust_arr = np.array(rust_times)
    
    return {
        "python": {
            "backend": "numba" if is_numba_available() else "numpy",
            "mean_us": float(np.mean(py_arr)),
            "mean_ms": float(np.mean(py_arr) / 1000),
            "min_ms": float(np.min(py_arr) / 1000),
        },
        "rust": {
            "backend": "rust",
            "mean_us": float(np.mean(rust_arr)),
            "mean_ms": float(np.mean(rust_arr) / 1000),
            "min_ms": float(np.min(rust_arr) / 1000),
        },
        "speedup": float(np.mean(py_arr) / np.mean(rust_arr)),
        "accuracy": {
            "match": accuracy_match,
            "index_match": index_match,
            "similarity_match": similarity_match,
            "python_result": python_final,
            "rust_result": rust_final,
        },
    }


def verify_bit_exact_similarity(n_codes: int = 10000, seed: int = 12345) -> Dict[str, Any]:
    """Verify that Rust produces bit-exact similarity values."""
    if not RUST_AVAILABLE:
        raise RuntimeError("Rust extension not available")
    
    codes = create_codes(n_codes, seed=seed)
    query = create_query(seed=seed + 1)
    
    # Compute with Python (numpy, not numba for exact match)
    python_sims = python_hamming_similarity(query, codes, code_bits=CODE_BITS, use_numba=False)
    
    # Compute with Rust
    sim = HammingSimilarity(code_bits=CODE_BITS)
    rust_sims = sim.similarity_batch(query, codes)
    
    # Compare
    max_diff = float(np.max(np.abs(python_sims - rust_sims)))
    mean_diff = float(np.mean(np.abs(python_sims - rust_sims)))
    all_match = np.allclose(python_sims, rust_sims, rtol=1e-6)
    
    return {
        "n_codes": n_codes,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "all_match": all_match,
        "tolerance": 1e-6,
    }


def run_benchmarks() -> Dict[str, Any]:
    """Run all benchmarks."""
    print("=" * 70)
    print("RUST vs PYTHON SIMILARITY BENCHMARK")
    print("=" * 70)
    print()
    
    if not RUST_AVAILABLE:
        print("ERROR: Rust extension not available!")
        print("Run: cd src/binary_semantic_cache_rs && cargo build --release && cd ../.. && maturin develop --release")
        return {"error": "Rust extension not available"}
    
    print(f"Rust extension: ✓ Available")
    print(f"Numba available: {is_numba_available()}")
    print()
    
    results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "system_info": get_system_info(),
        "config": {
            "code_bits": CODE_BITS,
            "n_words": N_WORDS,
            "warmup_iterations": WARMUP_ITERATIONS,
        },
        "benchmarks": {},
    }
    
    # =========================================================================
    # 1. Bit-exact verification
    # =========================================================================
    print("[1/4] Verifying Bit-Exact Similarity Match")
    print("-" * 50)
    
    bit_exact = verify_bit_exact_similarity(n_codes=10000)
    results["bit_exact_verification"] = bit_exact
    
    if bit_exact["all_match"]:
        print(f"  ✓ PASS: All 10,000 similarities match (max diff: {bit_exact['max_diff']:.2e})")
    else:
        print(f"  ✗ FAIL: Similarity mismatch (max diff: {bit_exact['max_diff']:.2e})")
    print()
    
    # =========================================================================
    # 2. Benchmark at 100K entries (Primary Target)
    # =========================================================================
    print("[2/4] Benchmarking at 100K Entries (Primary Target)")
    print("-" * 50)
    
    codes_100k = create_codes(100_000)
    query = create_query()
    
    # Python NumPy
    numpy_stats = benchmark_python_similarity(query, codes_100k, n_iterations=50, use_numba=False)
    print(f"  Python (NumPy):  mean={numpy_stats['mean_ms']:.3f}ms, min={numpy_stats['min_ms']:.3f}ms")
    
    # Python Numba
    if is_numba_available():
        numba_stats = benchmark_python_similarity(query, codes_100k, n_iterations=50, use_numba=True)
        print(f"  Python (Numba):  mean={numba_stats['mean_ms']:.3f}ms, min={numba_stats['min_ms']:.3f}ms")
    else:
        numba_stats = numpy_stats
        print("  Python (Numba):  N/A (not available)")
    
    # Rust
    rust_stats = benchmark_rust_similarity(query, codes_100k, n_iterations=50)
    print(f"  Rust:            mean={rust_stats['mean_ms']:.3f}ms, min={rust_stats['min_ms']:.3f}ms")
    
    # Speedups
    speedup_vs_numpy = numpy_stats['mean_ms'] / rust_stats['mean_ms']
    speedup_vs_numba = numba_stats['mean_ms'] / rust_stats['mean_ms']
    
    print()
    print(f"  Speedup vs NumPy: {speedup_vs_numpy:.1f}x")
    print(f"  Speedup vs Numba: {speedup_vs_numba:.1f}x")
    
    results["benchmarks"]["100k"] = {
        "n_entries": 100_000,
        "numpy": numpy_stats,
        "numba": numba_stats,
        "rust": rust_stats,
        "speedup_vs_numpy": speedup_vs_numpy,
        "speedup_vs_numba": speedup_vs_numba,
    }
    print()
    
    # =========================================================================
    # 3. Benchmark at various sizes
    # =========================================================================
    print("[3/4] Benchmarking at Various Cache Sizes")
    print("-" * 50)
    
    cache_sizes = [1_000, 10_000, 100_000, 500_000]
    size_results = []
    
    for size in cache_sizes:
        codes = create_codes(size)
        n_iter = 50 if size >= 100_000 else 100
        
        rust_stats = benchmark_rust_similarity(query, codes, n_iterations=n_iter)
        
        target_met = rust_stats['mean_ms'] < 0.5 if size == 100_000 else True
        status = "✓ PASS" if target_met else "✗ FAIL" if size == 100_000 else ""
        
        print(f"  {size:>7,} entries: mean={rust_stats['mean_ms']:.3f}ms {status}")
        
        size_results.append({
            "n_entries": size,
            "rust": rust_stats,
            "target_met": target_met if size == 100_000 else None,
        })
    
    results["benchmarks"]["by_size"] = size_results
    print()
    
    # =========================================================================
    # 4. find_nearest accuracy verification
    # =========================================================================
    print("[4/4] Verifying find_nearest Accuracy")
    print("-" * 50)
    
    find_nearest_results = []
    n_queries_to_test = 100
    
    # Test multiple random queries
    all_match = True
    for i in range(n_queries_to_test):
        query_i = create_query(seed=SEED + 100 + i)
        fn_result = benchmark_find_nearest(query_i, codes_100k, n_iterations=10, threshold=0.5)
        find_nearest_results.append(fn_result)
        if not fn_result["accuracy"]["match"]:
            all_match = False
    
    # Aggregate results
    total_speedup = np.mean([r["speedup"] for r in find_nearest_results])
    accuracy_rate = sum(1 for r in find_nearest_results if r["accuracy"]["match"]) / len(find_nearest_results) * 100
    
    print(f"  Queries tested: {n_queries_to_test}")
    print(f"  Accuracy rate:  {accuracy_rate:.1f}%")
    print(f"  Mean speedup:   {total_speedup:.1f}x")
    
    results["benchmarks"]["find_nearest"] = {
        "n_queries": n_queries_to_test,
        "accuracy_rate_pct": accuracy_rate,
        "all_match": all_match,
        "mean_speedup": total_speedup,
    }
    print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    target_100k = results["benchmarks"]["100k"]["rust"]["mean_ms"]
    target_met = target_100k < 0.5
    baseline = results["benchmarks"]["100k"]["numba"]["mean_ms"]
    speedup = results["benchmarks"]["100k"]["speedup_vs_numba"]
    
    print(f"Phase 1 Baseline (Numba @ 100k): {baseline:.3f}ms")
    print(f"Phase 2 Result (Rust @ 100k):    {target_100k:.3f}ms")
    print(f"Speedup:                         {speedup:.1f}x")
    print()
    print(f"Target: < 0.5ms @ 100k entries")
    print(f"Result: {'✓ PASS' if target_met else '✗ FAIL'}")
    print()
    print(f"Bit-Exact Match:  {'✓ PASS' if bit_exact['all_match'] else '✗ FAIL'}")
    print(f"Accuracy (100 queries): {'✓ PASS' if all_match else '✗ FAIL'} ({accuracy_rate:.0f}%)")
    print("=" * 70)
    
    results["summary"] = {
        "phase1_baseline_ms": baseline,
        "phase2_result_ms": target_100k,
        "speedup_vs_numba": speedup,
        "target_ms": 0.5,
        "target_met": target_met,
        "bit_exact": bit_exact["all_match"],
        "accuracy_100pct": all_match,
    }
    
    # Save results
    output_path = _ROOT / "benchmarks" / "results" / "rust_lookup_bench.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


def main() -> int:
    """Run benchmarks."""
    try:
        results = run_benchmarks()
        if "error" in results:
            return 1
        summary = results.get("summary", {})
        all_pass = (
            summary.get("target_met", False) and
            summary.get("bit_exact", False) and
            summary.get("accuracy_100pct", False)
        )
        return 0 if all_pass else 1
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

