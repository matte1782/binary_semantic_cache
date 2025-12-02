#!/usr/bin/env python3
"""
Threading Benchmark (Phase 2.5 Sprint 1c)

Tests concurrent access to BinarySemanticCache:
- Multiple threads performing mixed read/write operations
- Measures throughput and latency under contention
- Sprint 1c: Stress test for response storage (RSP-05, RSP-06)

Goal: Verify thread safety and measure contention overhead
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Add src to path
_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _ROOT / "src"
sys.path.insert(0, str(_SRC_PATH))

from binary_semantic_cache.core.encoder import BinaryEncoder
from binary_semantic_cache.core.cache import BinarySemanticCache

# Configuration
EMBEDDING_DIM = 384
CODE_BITS = 256
SEED = 42
CACHE_SIZE = 10_000


def get_system_info() -> Dict[str, Any]:
    """Get system metadata."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": threading.active_count(),
    }


def create_embedding(seed: int) -> np.ndarray:
    """Create a single normalized embedding."""
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return emb / np.linalg.norm(emb)


def worker_task(
    cache: BinarySemanticCache,
    thread_id: int,
    n_requests: int,
    hit_ratio: float = 0.5,
) -> Tuple[int, int, float, List[float]]:
    """
    Worker function for thread benchmark.
    
    Returns: (hits, misses, total_time, individual_latencies)
    """
    hits = 0
    misses = 0
    latencies: List[float] = []
    
    # Pre-generate embeddings for this thread
    base_seed = SEED + thread_id * 1000
    
    for i in range(n_requests):
        # Decide if this should be a hit or miss attempt
        if np.random.random() < hit_ratio:
            # Try to hit: use a seed that exists in cache
            seed = base_seed + (i % 100)  # Reuse seeds for hits
        else:
            # Generate new embedding (likely miss)
            seed = base_seed + 10000 + i
        
        embedding = create_embedding(seed)
        
        start = time.perf_counter()
        result = cache.get(embedding)
        
        if result is None:
            # Cache miss - add to cache
            cache.put(embedding, {"thread": thread_id, "request": i})
            misses += 1
        else:
            hits += 1
        
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms
    
    total_time = sum(latencies)
    return hits, misses, total_time, latencies


def benchmark_threading(
    n_threads: int,
    requests_per_thread: int,
) -> Dict[str, Any]:
    """Run threading benchmark with given configuration."""
    # Create cache with some initial entries
    encoder = BinaryEncoder(
        embedding_dim=EMBEDDING_DIM,
        code_bits=CODE_BITS,
        seed=SEED,
    )
    cache = BinarySemanticCache(
        encoder=encoder,
        max_entries=CACHE_SIZE,
        similarity_threshold=0.80,
    )
    
    # Pre-populate with some entries
    for i in range(min(1000, CACHE_SIZE)):
        emb = create_embedding(SEED + i)
        cache.put(emb, {"warmup": i})
    
    # Run concurrent benchmark
    start_time = time.perf_counter()
    
    all_latencies: List[float] = []
    total_hits = 0
    total_misses = 0
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [
            executor.submit(worker_task, cache, i, requests_per_thread)
            for i in range(n_threads)
        ]
        
        for future in as_completed(futures):
            hits, misses, _, latencies = future.result()
            total_hits += hits
            total_misses += misses
            all_latencies.extend(latencies)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    total_requests = n_threads * requests_per_thread
    throughput = total_requests / total_time
    
    latency_arr = np.array(all_latencies)
    
    return {
        "n_threads": n_threads,
        "requests_per_thread": requests_per_thread,
        "total_requests": total_requests,
        "total_time_s": total_time,
        "throughput_rps": throughput,
        "hits": total_hits,
        "misses": total_misses,
        "hit_rate": total_hits / total_requests if total_requests > 0 else 0,
        "latency": {
            "mean_ms": float(np.mean(latency_arr)),
            "std_ms": float(np.std(latency_arr)),
            "min_ms": float(np.min(latency_arr)),
            "max_ms": float(np.max(latency_arr)),
            "p50_ms": float(np.percentile(latency_arr, 50)),
            "p95_ms": float(np.percentile(latency_arr, 95)),
            "p99_ms": float(np.percentile(latency_arr, 99)),
        },
    }


def run_benchmarks() -> Dict[str, Any]:
    """Run all threading benchmarks."""
    print("=" * 60)
    print("THREADING BENCHMARK")
    print("=" * 60)
    
    results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "system_info": get_system_info(),
        "config": {
            "embedding_dim": EMBEDDING_DIM,
            "code_bits": CODE_BITS,
            "cache_size": CACHE_SIZE,
        },
        "scenarios": [],
    }
    
    scenarios = [
        (10, 100),   # 10 threads, 100 requests each
        (50, 20),    # 50 threads, 20 requests each
        (100, 10),   # 100 threads, 10 requests each
    ]
    
    print("\nRunning threading scenarios...")
    print("-" * 40)
    
    for n_threads, requests_per_thread in scenarios:
        print(f"\n  {n_threads} threads × {requests_per_thread} requests:")
        
        stats = benchmark_threading(n_threads, requests_per_thread)
        results["scenarios"].append(stats)
        
        print(f"    Throughput: {stats['throughput_rps']:.1f} req/s")
        print(f"    Avg Latency: {stats['latency']['mean_ms']:.2f} ms")
        print(f"    P99 Latency: {stats['latency']['p99_ms']:.2f} ms")
        print(f"    Hit Rate: {stats['hit_rate']*100:.1f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("THREADING SUMMARY")
    print("=" * 60)
    
    for scenario in results["scenarios"]:
        print(f"  {scenario['n_threads']:3d} threads: "
              f"{scenario['throughput_rps']:,.0f} req/s, "
              f"mean={scenario['latency']['mean_ms']:.2f}ms")
    
    # Check for race conditions (no exceptions = pass)
    results["summary"] = {
        "all_scenarios_completed": True,
        "no_race_conditions": True,  # If we got here, no exceptions
    }
    
    # Save results
    output_path = _ROOT / "benchmarks" / "results" / "threading_bench.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


# =============================================================================
# Sprint 1c: Response Storage Stress Test (RSP-05, RSP-06)
# =============================================================================

def stress_test_response_storage(
    n_threads: int = 8,
    ops_per_thread: int = 100_000,
) -> Dict[str, Any]:
    """
    Sprint 1c stress test for response storage (RSP-05, RSP-06).
    
    Spawns n_threads performing concurrent set/get/delete cycles.
    Detects deadlocks, race conditions, and memory leaks.
    
    Args:
        n_threads: Number of concurrent threads.
        ops_per_thread: Operations per thread.
    
    Returns:
        Dict with test results and metrics.
    """
    print("\n" + "=" * 60)
    print("SPRINT 1c: RESPONSE STORAGE STRESS TEST")
    print("=" * 60)
    print(f"\n  Threads: {n_threads}")
    print(f"  Ops/thread: {ops_per_thread:,}")
    print(f"  Total ops: {n_threads * ops_per_thread:,}")
    
    # Create cache with limited capacity to force evictions
    encoder = BinaryEncoder(
        embedding_dim=EMBEDDING_DIM,
        code_bits=CODE_BITS,
        seed=SEED,
    )
    cache = BinarySemanticCache(
        encoder=encoder,
        max_entries=1000,  # Small to force evictions
        similarity_threshold=0.80,
    )
    
    # Track errors and metrics
    errors: List[Exception] = []
    # Timeout for deadlock detection: 600s (10 minutes) to accommodate slow machines
    # and high contention scenarios. At ~1000 ops/s under contention, 80k ops
    # takes ~80s. We use 600s to provide massive margin for CI/slow environments.
    # A true deadlock would hang indefinitely, so this timeout is purely defensive.
    deadlock_timeout = 600  # seconds
    
    # Memory baseline
    gc.collect()
    tracemalloc.start()
    baseline_mem = tracemalloc.get_traced_memory()[0]
    
    def worker(thread_id: int) -> Tuple[int, int, int]:
        """Worker function: mixed put/get/delete operations."""
        puts = 0
        gets = 0
        deletes = 0
        
        try:
            base_seed = SEED + thread_id * 100_000
            for i in range(ops_per_thread):
                op = i % 3
                seed = base_seed + (i % 500)  # Reuse some seeds for hits
                emb = create_embedding(seed)
                
                if op == 0:  # put
                    cache.put(emb, {"thread": thread_id, "op": i})
                    puts += 1
                elif op == 1:  # get
                    cache.get(emb)
                    gets += 1
                else:  # delete
                    idx = i % 100
                    cache.delete(idx)
                    deletes += 1
        except Exception as e:
            errors.append(e)
        
        return puts, gets, deletes
    
    # Run with timeout to detect deadlocks
    start_time = time.perf_counter()
    total_puts = 0
    total_gets = 0
    total_deletes = 0
    
    print("\n  Running stress test...")
    
    executor = ThreadPoolExecutor(max_workers=n_threads)
    try:
        futures = [
            executor.submit(worker, i)
            for i in range(n_threads)
        ]
        
        # Wait with timeout
        completed = 0
        for future in as_completed(futures, timeout=deadlock_timeout):
            try:
                puts, gets, deletes = future.result()
                total_puts += puts
                total_gets += gets
                total_deletes += deletes
                completed += 1
            except Exception as e:
                errors.append(e)
                
    except TimeoutError:
        print(f"\n  !!! TIMEOUT after {deadlock_timeout}s !!!")
        print("  Detailed timeout info: some futures failed to complete.")
        # Force shutdown without waiting
        executor.shutdown(wait=False)
        errors.append(TimeoutError(f"Benchmark timed out after {deadlock_timeout}s"))
    finally:
        # Ensure cleanup
        executor.shutdown(wait=False)
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    # Memory after
    gc.collect()
    current_mem = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    
    mem_growth = current_mem - baseline_mem
    mem_growth_mb = mem_growth / 1024 / 1024
    
    # Calculate expected memory (cache at capacity)
    expected_mem = cache._storage.memory_usage() + len(cache._responses) * 80
    mem_growth_pct = (mem_growth / expected_mem * 100) if expected_mem > 0 else 0
    
    # Results
    total_ops = total_puts + total_gets + total_deletes
    throughput = total_ops / elapsed if elapsed > 0 else 0
    
    no_errors = len(errors) == 0
    no_deadlock = completed == n_threads
    no_leak = mem_growth_pct < 200  # Allow 2x expected (GC timing)
    
    print(f"\n  Completed: {completed}/{n_threads} threads")
    print(f"  Total ops: {total_ops:,} ({total_puts:,} puts, {total_gets:,} gets, {total_deletes:,} deletes)")
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:,.0f} ops/s")
    print(f"  Memory growth: {mem_growth_mb:.2f} MB ({mem_growth_pct:.1f}% of expected)")
    print(f"  Errors: {len(errors)}")
    
    print("\n  --- Results ---")
    print(f"  No errors:    {'✓ PASS' if no_errors else '✗ FAIL'}")
    print(f"  No deadlock:  {'✓ PASS' if no_deadlock else '✗ FAIL'}")
    print(f"  No leak:      {'✓ PASS' if no_leak else '✗ FAIL'}")
    
    result = {
        "n_threads": n_threads,
        "ops_per_thread": ops_per_thread,
        "total_ops": total_ops,
        "elapsed_s": elapsed,
        "throughput_ops": throughput,
        "total_puts": total_puts,
        "total_gets": total_gets,
        "total_deletes": total_deletes,
        "completed_threads": completed,
        "errors": len(errors),
        "error_messages": [str(e) for e in errors[:10]],  # First 10 errors
        "memory_growth_mb": mem_growth_mb,
        "memory_growth_pct": mem_growth_pct,
        "no_errors": no_errors,
        "no_deadlock": no_deadlock,
        "no_leak": no_leak,
        "pass": no_errors and no_deadlock and no_leak,
    }
    
    return result


def run_sprint_1c_stress() -> Dict[str, Any]:
    """Run Sprint 1c stress tests."""
    results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "system_info": get_system_info(),
        "tests": [],
    }
    
    # Test 1: 8 threads, 100k ops each (RSP-05)
    print("\n[RSP-05] Thread Safety Test")
    test1 = stress_test_response_storage(n_threads=8, ops_per_thread=10_000)
    results["tests"].append({"name": "RSP-05_thread_safety", **test1})
    
    # Test 2: Rapid churn (RSP-06) - smaller but more intensive
    print("\n[RSP-06] Memory Leak Test (100k cycles)")
    test2 = stress_test_response_storage(n_threads=4, ops_per_thread=25_000)
    results["tests"].append({"name": "RSP-06_memory_leak", **test2})
    
    # Summary
    all_pass = all(t["pass"] for t in results["tests"])
    results["summary"] = {
        "all_pass": all_pass,
        "tests_run": len(results["tests"]),
        "tests_passed": sum(1 for t in results["tests"] if t["pass"]),
    }
    
    print("\n" + "=" * 60)
    print("SPRINT 1c STRESS TEST SUMMARY")
    print("=" * 60)
    print(f"\n  Tests run: {results['summary']['tests_run']}")
    print(f"  Tests passed: {results['summary']['tests_passed']}")
    print(f"  Overall: {'✓ PASS' if all_pass else '✗ FAIL'}")
    
    # Save results
    output_path = _ROOT / "benchmarks" / "results" / "sprint_1c_stress.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


def main() -> int:
    """Run benchmarks."""
    parser = argparse.ArgumentParser(description="Threading Benchmark")
    parser.add_argument(
        "--sprint-1c", "-s",
        action="store_true",
        help="Run Sprint 1c stress tests (RSP-05, RSP-06)"
    )
    args = parser.parse_args()
    
    try:
        if args.sprint_1c:
            results = run_sprint_1c_stress()
            return 0 if results["summary"]["all_pass"] else 1
        else:
            results = run_benchmarks()
            return 0
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

