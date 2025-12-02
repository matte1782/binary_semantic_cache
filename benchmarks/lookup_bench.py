#!/usr/bin/env python3
"""
Lookup Benchmark (Phase 1 Legacy / Baseline)

Benchmarks Hamming similarity lookup performance using the Python/Numba backend:
- Various cache sizes (1K to 500K)
- Single vs batch queries
- Numba vs NumPy comparison

LEGACY TARGET: Numba mean < 1ms at 100K entries (Phase 1 baseline)

NOTE: This benchmark measures the LEGACY Python/Numba backend.
For Phase 2 production performance, use `rust_lookup_bench.py` which
measures the Rust backend (target: < 0.5ms @ 100k entries).

As of Phase 2.5, Rust is the mandatory production backend.
This benchmark is retained for:
1. Regression testing of the Numba fallback path
2. Historical comparison with Phase 1 baseline
3. Validating that the Python path remains functional

The < 1ms target may not be met on all hardware; this is ACCEPTED
as long as the Rust backend meets its < 0.5ms target.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Add src to path
_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _ROOT / "src"
sys.path.insert(0, str(_SRC_PATH))

from binary_semantic_cache.core.encoder import BinaryEncoder
from binary_semantic_cache.core.similarity import (
    hamming_similarity,
    is_numba_available,
    POPCOUNT_TABLE,
)

# Configuration
WARMUP_ITERATIONS = 5
SEED = 42
CODE_BITS = 256


def get_system_info() -> Dict[str, Any]:
    """Get system metadata."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "numba_available": is_numba_available(),
    }


def create_codes(n: int, n_words: int = 4, seed: int = SEED) -> np.ndarray:
    """Create random binary codes."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**64, size=(n, n_words), dtype=np.uint64)


def numpy_hamming_baseline(query: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """NumPy baseline for comparison (no Numba)."""
    xored = codes ^ query
    xored_bytes = xored.view(np.uint8)
    distances = POPCOUNT_TABLE[xored_bytes].sum(axis=1)
    return distances.astype(np.int32)


def benchmark_lookup(
    codes: np.ndarray,
    n_queries: int,
    measure_iterations: int,
    use_numba: bool = True,
) -> Dict[str, float]:
    """Benchmark lookup performance."""
    n_entries = codes.shape[0]
    n_words = codes.shape[1]
    
    # Create queries
    queries = create_codes(n_queries, n_words, seed=SEED + 1)
    
    # Select function
    if use_numba and is_numba_available():
        lookup_fn = lambda q: hamming_similarity(q, codes, code_bits=CODE_BITS)
    else:
        lookup_fn = lambda q: 1.0 - (numpy_hamming_baseline(q, codes) / CODE_BITS)
    
    # Warmup
    for i in range(WARMUP_ITERATIONS):
        lookup_fn(queries[i % n_queries])
    
    # Measure
    times_us: List[float] = []
    for i in range(measure_iterations):
        query = queries[i % n_queries]
        start = time.perf_counter()
        lookup_fn(query)
        end = time.perf_counter()
        times_us.append((end - start) * 1_000_000)  # microseconds
    
    arr = np.array(times_us)
    return {
        "n_entries": n_entries,
        "n_queries": n_queries,
        "iterations": measure_iterations,
        "backend": "numba" if (use_numba and is_numba_available()) else "numpy",
        "mean_us": float(np.mean(arr)),
        "std_us": float(np.std(arr)),
        "min_us": float(np.min(arr)),
        "max_us": float(np.max(arr)),
        "mean_ms": float(np.mean(arr) / 1000),
    }


def run_benchmarks() -> Dict[str, Any]:
    """Run all lookup benchmarks."""
    print("=" * 60)
    print("LOOKUP BENCHMARK")
    print("=" * 60)
    
    numba_available = is_numba_available()
    print(f"Numba available: {numba_available}")
    
    results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "system_info": get_system_info(),
        "config": {
            "code_bits": CODE_BITS,
            "warmup_iterations": WARMUP_ITERATIONS,
        },
        "by_cache_size": [],
        "numba_vs_numpy": [],
    }
    
    # Benchmark different cache sizes
    print("\n[1/2] Lookup by Cache Size (Numba)")
    print("-" * 40)
    
    cache_sizes = [1_000, 10_000, 100_000, 500_000]
    
    for size in cache_sizes:
        # Adjust iterations for larger sizes
        measure_iterations = 50 if size >= 100_000 else 100
        
        print(f"  Creating {size:,} entry database...")
        codes = create_codes(size, n_words=4)
        
        stats = benchmark_lookup(
            codes, 
            n_queries=100, 
            measure_iterations=measure_iterations,
            use_numba=True,
        )
        results["by_cache_size"].append(stats)
        
        # NOTE: < 1.0ms is the Phase 1 legacy target. Failure is ACCEPTED on hardware
        # slower than the Phase 1 baseline machine. See rust_lookup_bench.py for
        # the authoritative Phase 2 target (< 0.5ms via Rust backend).
        status = "(PASS legacy)" if stats["mean_ms"] < 1.0 else "(baseline)" if size == 100_000 else ""
        print(f"  {size:>7,} entries: mean={stats['mean_ms']:.3f}ms {status}")
    
    # Numba vs NumPy comparison at 100K
    print("\n[2/2] Numba vs NumPy Comparison (100K entries)")
    print("-" * 40)
    
    codes_100k = create_codes(100_000, n_words=4)
    
    # NumPy baseline
    numpy_stats = benchmark_lookup(
        codes_100k,
        n_queries=100,
        measure_iterations=50,
        use_numba=False,
    )
    results["numba_vs_numpy"].append(numpy_stats)
    print(f"  NumPy:  mean={numpy_stats['mean_ms']:.3f}ms")
    
    # Numba
    if numba_available:
        numba_stats = benchmark_lookup(
            codes_100k,
            n_queries=100,
            measure_iterations=50,
            use_numba=True,
        )
        results["numba_vs_numpy"].append(numba_stats)
        
        speedup = numpy_stats["mean_ms"] / numba_stats["mean_ms"]
        print(f"  Numba:  mean={numba_stats['mean_ms']:.3f}ms")
        print(f"  Speedup: {speedup:.1f}x")
        
        results["speedup"] = speedup
    
    # Summary
    target_result = next(
        (r for r in results["by_cache_size"] if r["n_entries"] == 100_000),
        None
    )
    
    if target_result:
        target_met = target_result["mean_ms"] < 1.0
        
        print("\n" + "=" * 60)
        print("PHASE 1 LEGACY BENCHMARK (Python/Numba)")
        print("=" * 60)
        print(f"100K LOOKUP: mean={target_result['mean_ms']:.3f}ms "
              f"{'(PASS legacy)' if target_met else '(baseline)'}")
        print(f"LEGACY TARGET: < 1.0 ms (Phase 1)")
        print()
        print("NOTE: For Phase 2 production target, run rust_lookup_bench.py")
        print("      Phase 2 target: < 0.5 ms @ 100k (Rust backend)")
        print("=" * 60)
        
        results["summary"] = {
            "lookup_100k_mean_ms": target_result["mean_ms"],
            "target_ms": 1.0,
            "target_met": target_met,
        }
    
    # Save results
    output_path = _ROOT / "benchmarks" / "results" / "lookup_bench.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


def main() -> int:
    """Run benchmarks."""
    try:
        results = run_benchmarks()
        return 0 if results.get("summary", {}).get("target_met", False) else 1
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

