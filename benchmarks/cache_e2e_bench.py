#!/usr/bin/env python3
"""
Cache End-to-End Benchmark (Phase 2.3)

Benchmarks the FULL cache.get() path with Rust backend integrated:
- Encoding (RustBinaryEncoder)
- Similarity search (HammingSimilarity.find_nearest)
- LRU management
- Response retrieval

This measures REAL end-to-end latency, not just inner functions.

Target: < 0.5ms @ 100k entries (Phase 2 goal)
Baseline: 1.14ms @ 100k entries (Phase 1 Python/Numba)
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

# Import Rust backend (mandatory in Phase 2)
from binary_semantic_cache.binary_semantic_cache_rs import (
    RustBinaryEncoder,
    HammingSimilarity,
    rust_version,
)
from binary_semantic_cache.core.cache import BinarySemanticCache
from binary_semantic_cache.core.similarity import is_numba_available

# Configuration
WARMUP_ITERATIONS = 10
SEED = 42
CODE_BITS = 256
EMBEDDING_DIM = 384

# Phase 1 Baseline (FROZEN)
PHASE1_BASELINE_MS = 1.14  # Lookup latency @ 100k entries (Python/Numba)

# Phase 2 Target
PHASE2_TARGET_MS = 0.5  # Lookup latency @ 100k entries (Rust)


def get_system_info() -> Dict[str, Any]:
    """Get system metadata."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "numba_available": is_numba_available(),
        "rust_version": rust_version(),
    }


def create_projection_matrix(seed: int = SEED) -> np.ndarray:
    """Create deterministic projection matrix (seed=42)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((CODE_BITS, EMBEDDING_DIM)).astype(np.float32)


def create_embeddings(n: int, seed: int = SEED) -> np.ndarray:
    """Create random embeddings."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, EMBEDDING_DIM)).astype(np.float32)


def create_cache_with_entries(
    n_entries: int,
    projection_matrix: np.ndarray,
    seed: int = SEED,
) -> BinarySemanticCache:
    """Create a cache and fill it with n_entries."""
    encoder = RustBinaryEncoder(EMBEDDING_DIM, CODE_BITS, projection_matrix)
    cache = BinarySemanticCache(
        encoder=encoder,
        max_entries=max(n_entries, 100),  # At least 100 entries capacity
        similarity_threshold=0.80,
    )
    
    # Fill cache with random embeddings
    embeddings = create_embeddings(n_entries, seed=seed)
    for i, emb in enumerate(embeddings):
        cache.put(emb, {"response": f"answer_{i}", "index": i})
    
    return cache


def benchmark_cache_get(
    cache: BinarySemanticCache,
    queries: np.ndarray,
    n_iterations: int,
) -> Dict[str, Any]:
    """Benchmark cache.get() end-to-end latency."""
    n_queries = queries.shape[0]
    
    # Warmup
    for i in range(WARMUP_ITERATIONS):
        cache.get(queries[i % n_queries])
    
    # Measure
    times_us: List[float] = []
    hits = 0
    misses = 0
    
    for i in range(n_iterations):
        query = queries[i % n_queries]
        start = time.perf_counter()
        result = cache.get(query)
        end = time.perf_counter()
        times_us.append((end - start) * 1_000_000)  # microseconds
        
        if result is not None:
            hits += 1
        else:
            misses += 1
    
    arr = np.array(times_us)
    return {
        "n_entries": len(cache),
        "n_iterations": n_iterations,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / n_iterations if n_iterations > 0 else 0.0,
        "mean_us": float(np.mean(arr)),
        "std_us": float(np.std(arr)),
        "min_us": float(np.min(arr)),
        "max_us": float(np.max(arr)),
        "p50_us": float(np.percentile(arr, 50)),
        "p95_us": float(np.percentile(arr, 95)),
        "p99_us": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr) / 1000),
        "min_ms": float(np.min(arr) / 1000),
        "p50_ms": float(np.percentile(arr, 50) / 1000),
        "p99_ms": float(np.percentile(arr, 99) / 1000),
    }


def benchmark_cache_put(
    projection_matrix: np.ndarray,
    n_entries: int,
    n_iterations: int,
) -> Dict[str, Any]:
    """Benchmark cache.put() latency."""
    encoder = RustBinaryEncoder(EMBEDDING_DIM, CODE_BITS, projection_matrix)
    cache = BinarySemanticCache(
        encoder=encoder,
        max_entries=n_entries + n_iterations + 100,
        similarity_threshold=0.80,
    )
    
    # Pre-fill cache
    embeddings = create_embeddings(n_entries, seed=SEED)
    for i, emb in enumerate(embeddings):
        cache.put(emb, {"response": f"prefill_{i}"})
    
    # Generate new embeddings for put benchmark
    new_embeddings = create_embeddings(n_iterations, seed=SEED + 1000)
    
    # Warmup
    for i in range(min(WARMUP_ITERATIONS, n_iterations)):
        cache.put(new_embeddings[i], {"response": f"warmup_{i}"})
    
    # Measure
    times_us: List[float] = []
    for i in range(n_iterations):
        emb = new_embeddings[i % len(new_embeddings)]
        start = time.perf_counter()
        cache.put(emb, {"response": f"bench_{i}"})
        end = time.perf_counter()
        times_us.append((end - start) * 1_000_000)
    
    arr = np.array(times_us)
    return {
        "n_entries_before": n_entries,
        "n_iterations": n_iterations,
        "mean_us": float(np.mean(arr)),
        "std_us": float(np.std(arr)),
        "min_us": float(np.min(arr)),
        "max_us": float(np.max(arr)),
        "p50_us": float(np.percentile(arr, 50)),
        "p99_us": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr) / 1000),
        "min_ms": float(np.min(arr) / 1000),
    }


def run_benchmarks() -> Dict[str, Any]:
    """Run all cache end-to-end benchmarks."""
    print("=" * 70)
    print("CACHE END-TO-END BENCHMARK (Phase 2.3)")
    print("=" * 70)
    print()
    print(f"Rust extension: ✓ Available (v{rust_version()})")
    print(f"Numba available: {is_numba_available()}")
    print()
    
    results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "system_info": get_system_info(),
        "config": {
            "code_bits": CODE_BITS,
            "embedding_dim": EMBEDDING_DIM,
            "warmup_iterations": WARMUP_ITERATIONS,
            "phase1_baseline_ms": PHASE1_BASELINE_MS,
            "phase2_target_ms": PHASE2_TARGET_MS,
        },
        "benchmarks": {},
    }
    
    # Create projection matrix
    projection_matrix = create_projection_matrix()
    
    # =========================================================================
    # 1. Primary Target: 100K entries
    # =========================================================================
    print("[1/4] cache.get() at 100K Entries (PRIMARY TARGET)")
    print("-" * 50)
    
    cache_100k = create_cache_with_entries(100_000, projection_matrix, seed=SEED)
    
    # Query with embeddings that ARE in the cache (should hit)
    stored_embeddings = create_embeddings(100, seed=SEED)  # Same seed = same embeddings
    get_stats_hit = benchmark_cache_get(cache_100k, stored_embeddings, n_iterations=100)
    
    # Query with embeddings that are NOT in the cache (should miss)
    random_embeddings = create_embeddings(100, seed=SEED + 9999)
    get_stats_miss = benchmark_cache_get(cache_100k, random_embeddings, n_iterations=100)
    
    print(f"  Hit queries:  mean={get_stats_hit['mean_ms']:.3f}ms, p50={get_stats_hit['p50_ms']:.3f}ms, p99={get_stats_hit['p99_ms']:.3f}ms")
    print(f"  Miss queries: mean={get_stats_miss['mean_ms']:.3f}ms, p50={get_stats_miss['p50_ms']:.3f}ms, p99={get_stats_miss['p99_ms']:.3f}ms")
    print(f"  Hit rate (stored): {get_stats_hit['hit_rate']:.1%}")
    print(f"  Hit rate (random): {get_stats_miss['hit_rate']:.1%}")
    
    # Combined stats (realistic mix)
    combined_mean = (get_stats_hit['mean_ms'] + get_stats_miss['mean_ms']) / 2
    target_met = combined_mean < PHASE2_TARGET_MS
    
    print()
    print(f"  Combined mean: {combined_mean:.3f}ms")
    print(f"  Phase 1 baseline: {PHASE1_BASELINE_MS:.3f}ms")
    print(f"  Phase 2 target: < {PHASE2_TARGET_MS:.3f}ms")
    print(f"  Speedup vs Phase 1: {PHASE1_BASELINE_MS / combined_mean:.1f}x")
    print(f"  Result: {'✓ PASS' if target_met else '✗ FAIL'}")
    
    results["benchmarks"]["100k"] = {
        "n_entries": 100_000,
        "hit_queries": get_stats_hit,
        "miss_queries": get_stats_miss,
        "combined_mean_ms": combined_mean,
        "target_met": target_met,
        "speedup_vs_phase1": PHASE1_BASELINE_MS / combined_mean,
    }
    print()
    
    # =========================================================================
    # 2. Scaling: Various cache sizes
    # =========================================================================
    print("[2/4] cache.get() at Various Cache Sizes")
    print("-" * 50)
    
    cache_sizes = [1_000, 10_000, 50_000, 100_000]
    size_results = []
    
    for size in cache_sizes:
        cache = create_cache_with_entries(size, projection_matrix, seed=SEED)
        queries = create_embeddings(50, seed=SEED + 5000)
        n_iter = 50 if size >= 50_000 else 100
        
        stats = benchmark_cache_get(cache, queries, n_iterations=n_iter)
        size_results.append({"n_entries": size, "stats": stats})
        
        status = ""
        if size == 100_000:
            status = "✓ PASS" if stats['mean_ms'] < PHASE2_TARGET_MS else "✗ FAIL"
        
        print(f"  {size:>7,} entries: mean={stats['mean_ms']:.3f}ms, p50={stats['p50_ms']:.3f}ms {status}")
    
    results["benchmarks"]["by_size"] = size_results
    print()
    
    # =========================================================================
    # 3. cache.put() latency
    # =========================================================================
    print("[3/4] cache.put() Latency")
    print("-" * 50)
    
    put_stats = benchmark_cache_put(projection_matrix, n_entries=10_000, n_iterations=100)
    put_target_met = put_stats['mean_ms'] < 2.0
    
    print(f"  Mean: {put_stats['mean_ms']:.3f}ms")
    print(f"  P99:  {put_stats['p99_us'] / 1000:.3f}ms")
    print(f"  Target: < 2.0ms")
    print(f"  Result: {'✓ PASS' if put_target_met else '✗ FAIL'}")
    
    results["benchmarks"]["put"] = {
        "stats": put_stats,
        "target_met": put_target_met,
    }
    print()
    
    # =========================================================================
    # 4. Hit Rate Verification
    # =========================================================================
    print("[4/4] Hit Rate Verification")
    print("-" * 50)
    
    # Create cache with known embeddings
    cache_verify = create_cache_with_entries(1000, projection_matrix, seed=SEED)
    
    # Query with same embeddings (should all hit)
    same_embeddings = create_embeddings(100, seed=SEED)[:100]
    verify_stats = benchmark_cache_get(cache_verify, same_embeddings, n_iterations=100)
    
    hit_rate = verify_stats['hit_rate']
    hit_rate_target = hit_rate >= 0.90  # At least 90% hit rate for same embeddings
    
    print(f"  Same embeddings: {hit_rate:.1%} hit rate")
    print(f"  Target: >= 90%")
    print(f"  Result: {'✓ PASS' if hit_rate_target else '✗ FAIL'}")
    
    results["benchmarks"]["hit_rate_verification"] = {
        "hit_rate": hit_rate,
        "target_met": hit_rate_target,
    }
    print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    primary_result = results["benchmarks"]["100k"]["combined_mean_ms"]
    primary_target_met = results["benchmarks"]["100k"]["target_met"]
    speedup = results["benchmarks"]["100k"]["speedup_vs_phase1"]
    
    print(f"Phase 1 Baseline (Python/Numba @ 100k): {PHASE1_BASELINE_MS:.3f}ms")
    print(f"Phase 2 Result (Rust @ 100k):           {primary_result:.3f}ms")
    print(f"Speedup:                                {speedup:.1f}x")
    print()
    print(f"Target: < {PHASE2_TARGET_MS:.1f}ms @ 100k entries")
    print(f"Result: {'✓ PASS' if primary_target_met else '✗ FAIL'}")
    print()
    
    all_pass = (
        primary_target_met and
        put_target_met and
        hit_rate_target
    )
    
    print(f"cache.get() @ 100k: {'✓ PASS' if primary_target_met else '✗ FAIL'}")
    print(f"cache.put():        {'✓ PASS' if put_target_met else '✗ FAIL'}")
    print(f"Hit Rate:           {'✓ PASS' if hit_rate_target else '✗ FAIL'}")
    print()
    print(f"Overall: {'✓ ALL PASS' if all_pass else '✗ SOME FAILED'}")
    print("=" * 70)
    
    results["summary"] = {
        "phase1_baseline_ms": PHASE1_BASELINE_MS,
        "phase2_result_ms": primary_result,
        "speedup_vs_phase1": speedup,
        "target_ms": PHASE2_TARGET_MS,
        "target_met": primary_target_met,
        "put_target_met": put_target_met,
        "hit_rate_target_met": hit_rate_target,
        "all_pass": all_pass,
    }
    
    # Save results
    output_path = _ROOT / "benchmarks" / "results" / "cache_e2e_bench.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


def main() -> int:
    """Run benchmarks."""
    try:
        results = run_benchmarks()
        return 0 if results.get("summary", {}).get("all_pass", False) else 1
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

