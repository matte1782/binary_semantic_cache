#!/usr/bin/env python3
"""
Sprint 1a-2 Micro-Benchmarks: RustCacheStorage

This script measures the performance of the RustCacheStorage PyO3 bindings
to validate Phase 2.5 targets from PHASE_2_5_TECHNICAL_PLAN.md.

Targets:
    - add() latency: < 1μs (single entry)
    - memory_usage(): exactly 44 * N bytes
    - evict_lru(): < 1ms @ 100k entries (O(N) scan acceptable)
    - search(): < 0.5ms @ 100k entries

Run with:
    python benchmarks/micro_bench_storage.py

Prerequisites:
    cd src/binary_semantic_cache_rs && cargo build --release && cd ../..
    maturin develop --release
"""

import json
import time
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

# Import Rust extension
try:
    from binary_semantic_cache.binary_semantic_cache_rs import RustCacheStorage
    RUST_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Rust extension not available: {e}")
    print("Build required:")
    print("  cd src/binary_semantic_cache_rs && cargo build --release && cd ../..")
    print("  maturin develop --release")
    RUST_AVAILABLE = False


def generate_random_code(rng: np.random.Generator) -> np.ndarray:
    """Generate a random 256-bit binary code as uint64 array."""
    return rng.integers(0, 2**64, size=4, dtype=np.uint64)


def benchmark_add_latency(sizes: List[int], warmup: int = 100, trials: int = 1000) -> Dict[str, Any]:
    """
    Benchmark add() latency for single entry insertion.
    
    Target: < 1μs per add() call
    """
    results = {}
    rng = np.random.default_rng(42)
    
    for size in sizes:
        # Create storage with enough capacity
        storage = RustCacheStorage(capacity=size + trials + warmup, code_bits=256)
        
        # Pre-fill to target size
        base_timestamp = 1700000000
        for i in range(size):
            code = generate_random_code(rng)
            storage.add(code, base_timestamp + i)
        
        # Warmup
        for _ in range(warmup):
            code = generate_random_code(rng)
            storage.add(code, base_timestamp)
        
        # Benchmark
        latencies_ns = []
        for i in range(trials):
            code = generate_random_code(rng)
            start = time.perf_counter_ns()
            storage.add(code, base_timestamp + size + i)
            end = time.perf_counter_ns()
            latencies_ns.append(end - start)
        
        latencies_us = [ns / 1000.0 for ns in latencies_ns]
        
        results[size] = {
            "mean_us": statistics.mean(latencies_us),
            "median_us": statistics.median(latencies_us),
            "min_us": min(latencies_us),
            "max_us": max(latencies_us),
            "p99_us": sorted(latencies_us)[int(trials * 0.99)],
            "trials": trials,
            "target_us": 1.0,
            "pass": statistics.mean(latencies_us) < 1.0,
        }
    
    return results


def benchmark_search_latency(sizes: List[int], warmup: int = 10, trials: int = 100) -> Dict[str, Any]:
    """
    Benchmark search() latency for nearest neighbor lookup.
    
    Target: < 0.5ms @ 100k entries
    """
    results = {}
    rng = np.random.default_rng(42)
    
    for size in sizes:
        # Create and fill storage
        storage = RustCacheStorage(capacity=size, code_bits=256)
        base_timestamp = 1700000000
        
        for i in range(size):
            code = generate_random_code(rng)
            storage.add(code, base_timestamp + i)
        
        # Generate query codes
        queries = [generate_random_code(rng) for _ in range(warmup + trials)]
        
        # Warmup
        for i in range(warmup):
            storage.search(queries[i], threshold=0.85)
        
        # Benchmark
        latencies_ms = []
        for i in range(trials):
            query = queries[warmup + i]
            start = time.perf_counter()
            storage.search(query, threshold=0.85)
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000.0)
        
        target_ms = 0.5 if size >= 100_000 else 1.0
        
        results[size] = {
            "mean_ms": statistics.mean(latencies_ms),
            "median_ms": statistics.median(latencies_ms),
            "min_ms": min(latencies_ms),
            "max_ms": max(latencies_ms),
            "p99_ms": sorted(latencies_ms)[int(trials * 0.99)],
            "trials": trials,
            "target_ms": target_ms,
            "pass": statistics.mean(latencies_ms) < target_ms,
        }
    
    return results


def benchmark_evict_lru_latency(sizes: List[int], warmup: int = 10, trials: int = 100) -> Dict[str, Any]:
    """
    Benchmark evict_lru() latency (O(N) scan).
    
    Target: < 1ms @ 100k entries
    """
    results = {}
    rng = np.random.default_rng(42)
    
    for size in sizes:
        # Create and fill storage
        storage = RustCacheStorage(capacity=size, code_bits=256)
        base_timestamp = 1700000000
        
        for i in range(size):
            code = generate_random_code(rng)
            storage.add(code, base_timestamp + i)
        
        # Warmup
        for _ in range(warmup):
            storage.evict_lru()
        
        # Benchmark
        latencies_ms = []
        for _ in range(trials):
            start = time.perf_counter()
            storage.evict_lru()
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000.0)
        
        target_ms = 1.0
        
        results[size] = {
            "mean_ms": statistics.mean(latencies_ms),
            "median_ms": statistics.median(latencies_ms),
            "min_ms": min(latencies_ms),
            "max_ms": max(latencies_ms),
            "p99_ms": sorted(latencies_ms)[int(trials * 0.99)],
            "trials": trials,
            "target_ms": target_ms,
            "pass": statistics.mean(latencies_ms) < target_ms,
        }
    
    return results


def verify_memory_usage(sizes: List[int]) -> Dict[str, Any]:
    """
    Verify memory_usage() reports exactly 44 * N bytes.
    
    Target: memory_usage() == 44 * len(storage)
    """
    results = {}
    rng = np.random.default_rng(42)
    
    for size in sizes:
        storage = RustCacheStorage(capacity=size, code_bits=256)
        base_timestamp = 1700000000
        
        for i in range(size):
            code = generate_random_code(rng)
            storage.add(code, base_timestamp + i)
        
        expected_bytes = 44 * size
        actual_bytes = storage.memory_usage()
        
        results[size] = {
            "expected_bytes": expected_bytes,
            "actual_bytes": actual_bytes,
            "bytes_per_entry": actual_bytes / size if size > 0 else 0,
            "pass": actual_bytes == expected_bytes,
        }
    
    return results


def print_results(
    add_results: Dict[str, Any],
    search_results: Dict[str, Any],
    evict_results: Dict[str, Any],
    memory_results: Dict[str, Any],
) -> None:
    """Print formatted benchmark results."""
    print("=" * 70)
    print("SPRINT 1a-2 MICRO-BENCHMARKS: RustCacheStorage")
    print("=" * 70)
    print()
    
    # Add latency
    print("[1/4] add() Latency (Target: < 1μs)")
    print("-" * 50)
    for size, data in add_results.items():
        status = "✓ PASS" if data["pass"] else "✗ FAIL"
        print(f"  {size:>7,} entries: mean={data['mean_us']:.3f}μs, "
              f"p99={data['p99_us']:.3f}μs {status}")
    print()
    
    # Search latency
    print("[2/4] search() Latency (Target: < 0.5ms @ 100k)")
    print("-" * 50)
    for size, data in search_results.items():
        status = "✓ PASS" if data["pass"] else "✗ FAIL"
        print(f"  {size:>7,} entries: mean={data['mean_ms']:.3f}ms, "
              f"p99={data['p99_ms']:.3f}ms {status}")
    print()
    
    # Evict LRU latency
    print("[3/4] evict_lru() Latency (Target: < 1ms @ 100k)")
    print("-" * 50)
    for size, data in evict_results.items():
        status = "✓ PASS" if data["pass"] else "✗ FAIL"
        print(f"  {size:>7,} entries: mean={data['mean_ms']:.3f}ms, "
              f"p99={data['p99_ms']:.3f}ms {status}")
    print()
    
    # Memory verification
    print("[4/4] memory_usage() Verification (Target: exactly 44 * N)")
    print("-" * 50)
    for size, data in memory_results.items():
        status = "✓ PASS" if data["pass"] else "✗ FAIL"
        print(f"  {size:>7,} entries: expected={data['expected_bytes']:,}B, "
              f"actual={data['actual_bytes']:,}B ({data['bytes_per_entry']:.1f}B/entry) {status}")
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_pass = True
    
    # Check primary targets
    if 100_000 in search_results:
        search_pass = search_results[100_000]["pass"]
        all_pass = all_pass and search_pass
        status = "✓ PASS" if search_pass else "✗ FAIL"
        print(f"  search() @ 100k: {search_results[100_000]['mean_ms']:.3f}ms "
              f"(target: < 0.5ms) {status}")
    
    if 100_000 in evict_results:
        evict_pass = evict_results[100_000]["pass"]
        all_pass = all_pass and evict_pass
        status = "✓ PASS" if evict_pass else "✗ FAIL"
        print(f"  evict_lru() @ 100k: {evict_results[100_000]['mean_ms']:.3f}ms "
              f"(target: < 1ms) {status}")
    
    # Check memory target (any size)
    memory_pass = all(data["pass"] for data in memory_results.values())
    all_pass = all_pass and memory_pass
    status = "✓ PASS" if memory_pass else "✗ FAIL"
    print(f"  memory_usage(): 44 bytes/entry {status}")
    
    # Check add latency (any size)
    add_pass = all(data["pass"] for data in add_results.values())
    all_pass = all_pass and add_pass
    status = "✓ PASS" if add_pass else "✗ FAIL"
    print(f"  add() latency: < 1μs {status}")
    
    print()
    if all_pass:
        print("✅ ALL TARGETS MET")
    else:
        print("❌ SOME TARGETS FAILED")
    print("=" * 70)


def save_results(
    add_results: Dict[str, Any],
    search_results: Dict[str, Any],
    evict_results: Dict[str, Any],
    memory_results: Dict[str, Any],
) -> str:
    """Save results to JSON file."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "benchmark": "micro_bench_storage",
        "sprint": "1a-2",
        "add_latency": {str(k): v for k, v in add_results.items()},
        "search_latency": {str(k): v for k, v in search_results.items()},
        "evict_lru_latency": {str(k): v for k, v in evict_results.items()},
        "memory_usage": {str(k): v for k, v in memory_results.items()},
    }
    
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"micro_bench_storage_{timestamp}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    return str(output_file)


def main():
    """Run all micro-benchmarks."""
    if not RUST_AVAILABLE:
        return
    
    print()
    print("Rust extension: ✓ Available")
    print()
    
    # Define test sizes
    sizes = [1_000, 10_000, 100_000]
    
    print("Running benchmarks...")
    print()
    
    # Run benchmarks
    print("  [1/4] Benchmarking add() latency...")
    add_results = benchmark_add_latency(sizes)
    
    print("  [2/4] Benchmarking search() latency...")
    search_results = benchmark_search_latency(sizes)
    
    print("  [3/4] Benchmarking evict_lru() latency...")
    evict_results = benchmark_evict_lru_latency(sizes)
    
    print("  [4/4] Verifying memory_usage()...")
    memory_results = verify_memory_usage(sizes)
    
    print()
    
    # Print results
    print_results(add_results, search_results, evict_results, memory_results)
    
    # Save results
    output_file = save_results(add_results, search_results, evict_results, memory_results)
    print()
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()

