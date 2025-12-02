"""
Sprint 2b Migration Benchmark: v2 → v3 Cache Migration Performance

Measures:
1. Migration time vs baseline (Load v2 + Save v3)
2. Throughput (entries/sec)
3. Overhead analysis

Usage:
    python benchmarks/migration_bench.py
    python benchmarks/migration_bench.py --entries 10000 50000 100000
    python benchmarks/migration_bench.py --quick  # 10k entries only
"""

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from binary_semantic_cache.core.cache import BinarySemanticCache, detect_format_version
from binary_semantic_cache.core.encoder import BinaryEncoder
from binary_semantic_cache.tools.migrate import migrate_v2_to_v3


def create_v2_cache(path: str, num_entries: int, encoder: BinaryEncoder) -> float:
    """
    Create a v2 cache with the specified number of entries.
    
    Returns:
        Time taken to create and save the cache (seconds).
    """
    cache = BinarySemanticCache(encoder=encoder, max_entries=num_entries + 1000)
    
    # Generate random embeddings and add to cache
    rng = np.random.default_rng(42)  # Deterministic for reproducibility
    
    for i in range(num_entries):
        emb = rng.standard_normal(encoder.embedding_dim).astype(np.float32)
        cache.put(emb, f"response_{i}")
    
    # Save as v2 (using deprecated save())
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cache.save(path)
    save_time = time.perf_counter() - start
    
    return save_time


def measure_load_v2(path: str, encoder: BinaryEncoder) -> float:
    """Measure time to load a v2 cache."""
    cache = BinarySemanticCache(encoder=encoder)
    
    gc.collect()
    start = time.perf_counter()
    cache.load(path)
    elapsed = time.perf_counter() - start
    
    return elapsed


def measure_save_v3(cache: BinarySemanticCache, path: str) -> float:
    """Measure time to save a cache as v3."""
    gc.collect()
    start = time.perf_counter()
    cache.save_mmap_v3(path)
    elapsed = time.perf_counter() - start
    
    return elapsed


def measure_migration(src_path: str, dst_path: str) -> float:
    """Measure time to migrate using the migration tool."""
    # Clean up destination if it exists
    if Path(dst_path).exists():
        shutil.rmtree(dst_path)
    
    gc.collect()
    start = time.perf_counter()
    migrate_v2_to_v3(src_path, dst_path)
    elapsed = time.perf_counter() - start
    
    return elapsed


def run_benchmark(num_entries: int, trials: int = 3) -> Dict[str, Any]:
    """
    Run migration benchmark for a given number of entries.
    
    Returns:
        Dictionary with benchmark results.
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARKING {num_entries:,} ENTRIES")
    print(f"{'='*70}")
    
    encoder = BinaryEncoder(code_bits=256, seed=42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        v2_path = os.path.join(tmpdir, "cache_v2.npz")
        v3_path = os.path.join(tmpdir, "cache_v3")
        v3_baseline_path = os.path.join(tmpdir, "cache_v3_baseline")
        
        # Step 1: Create v2 cache
        print(f"\n[1/{4 + trials}] Creating v2 cache with {num_entries:,} entries...")
        create_time = create_v2_cache(v2_path, num_entries, encoder)
        print(f"    Created in {create_time*1000:.1f}ms")
        
        # Step 2: Measure baseline (Load v2 + Save v3)
        print(f"\n[2/{4 + trials}] Measuring baseline (Load v2 + Save v3)...")
        
        # Load v2
        cache = BinarySemanticCache(encoder=encoder)
        load_v2_time = measure_load_v2(v2_path, encoder)
        cache.load(v2_path)
        print(f"    Load v2: {load_v2_time*1000:.1f}ms")
        
        # Save v3
        save_v3_time = measure_save_v3(cache, v3_baseline_path)
        print(f"    Save v3: {save_v3_time*1000:.1f}ms")
        
        baseline_time = load_v2_time + save_v3_time
        print(f"    Baseline total: {baseline_time*1000:.1f}ms")
        
        # Step 3: Run migration trials
        print(f"\n[3/{4 + trials}] Running migration trials (n={trials})...")
        migration_times = []
        
        for trial in range(trials):
            # Clean up v3 path
            if Path(v3_path).exists():
                shutil.rmtree(v3_path)
            
            migration_time = measure_migration(v2_path, v3_path)
            migration_times.append(migration_time)
            print(f"    Trial {trial+1}: {migration_time*1000:.1f}ms")
        
        migration_mean = np.mean(migration_times)
        migration_min = np.min(migration_times)
        migration_max = np.max(migration_times)
        migration_std = np.std(migration_times)
        
        # Step 4: Verify migration
        print(f"\n[4/{4 + trials}] Verifying migrated cache...")
        
        # Load v3 and verify
        cache_v3 = BinarySemanticCache(encoder=encoder)
        cache_v3.load_mmap_v3(v3_path)
        
        entries_match = len(cache_v3) == num_entries
        print(f"    Entries: {len(cache_v3):,} (expected: {num_entries:,}) {'✓' if entries_match else '✗'}")
        
        # Calculate metrics
        overhead_ratio = migration_mean / baseline_time
        throughput = num_entries / migration_mean
        
        print(f"\n{'='*70}")
        print(f"RESULTS ({num_entries:,} entries)")
        print(f"{'='*70}")
        print(f"    Migration time (mean): {migration_mean*1000:.1f}ms (σ={migration_std*1000:.1f})")
        print(f"    Migration time (min):  {migration_min*1000:.1f}ms")
        print(f"    Migration time (max):  {migration_max*1000:.1f}ms")
        print(f"    Baseline (Load+Save):  {baseline_time*1000:.1f}ms")
        print(f"    Overhead ratio:        {overhead_ratio:.2f}x")
        print(f"    Throughput:            {throughput:,.0f} entries/sec")
        
        # Check target
        target_overhead = 2.0
        passed = overhead_ratio < target_overhead
        print(f"\n    Target: Overhead < {target_overhead}x baseline")
        print(f"    Result: {'✓ PASS' if passed else '✗ FAIL'}")
        
        return {
            "entries": num_entries,
            "migration_mean_ms": migration_mean * 1000,
            "migration_min_ms": migration_min * 1000,
            "migration_max_ms": migration_max * 1000,
            "migration_std_ms": migration_std * 1000,
            "baseline_ms": baseline_time * 1000,
            "load_v2_ms": load_v2_time * 1000,
            "save_v3_ms": save_v3_time * 1000,
            "overhead_ratio": overhead_ratio,
            "throughput_entries_per_sec": throughput,
            "target_overhead": target_overhead,
            "passed": passed,
            "entries_verified": entries_match,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Sprint 2b Migration Benchmark: v2 → v3 Cache Migration Performance"
    )
    parser.add_argument(
        "--entries",
        type=int,
        nargs="+",
        default=[10000, 50000, 100000],
        help="Number of entries to benchmark (default: 10000 50000 100000)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: only benchmark 10k entries"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials per size (default: 3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/migration_bench.json",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        entry_sizes = [10000]
    else:
        entry_sizes = args.entries
    
    print("=" * 70)
    print("SPRINT 2b MIGRATION BENCHMARK: v2 → v3")
    print("=" * 70)
    print(f"\nEntry sizes to benchmark: {entry_sizes}")
    print(f"Trials per size: {args.trials}")
    print(f"\nTarget: Migration overhead < 2x (Load v2 + Save v3)")
    
    # Check Rust extension
    try:
        from binary_semantic_cache.binary_semantic_cache_rs import RustCacheStorage
        print(f"\nRust extension: ✓ Available")
    except ImportError:
        print(f"\nRust extension: ✗ NOT AVAILABLE (using Python fallback)")
    
    results = []
    all_passed = True
    
    for num_entries in entry_sizes:
        result = run_benchmark(num_entries, trials=args.trials)
        results.append(result)
        if not result["passed"]:
            all_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Entries':>12} | {'Migration (ms)':>15} | {'Baseline (ms)':>15} | {'Overhead':>10} | {'Throughput':>15} | Status")
    print("-" * 90)
    
    for r in results:
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        print(f"{r['entries']:>12,} | {r['migration_mean_ms']:>15.1f} | {r['baseline_ms']:>15.1f} | {r['overhead_ratio']:>10.2f}x | {r['throughput_entries_per_sec']:>12,.0f}/s | {status}")
    
    print("-" * 90)
    print(f"\nOverall: {'✓ ALL PASS' if all_passed else '✗ SOME FAILED'}")
    
    # Save results
    output_data = {
        "benchmark": "migration_bench",
        "timestamp": datetime.now().isoformat(),
        "target": "Migration overhead < 2x (Load v2 + Save v3)",
        "results": results,
        "overall_passed": all_passed,
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

