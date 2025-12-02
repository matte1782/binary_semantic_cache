#!/usr/bin/env python3
"""
Memory Profile Benchmark (Phase 2.5 Sprint 1c)

Profiles memory usage at various cache sizes:
- 10K, 50K, 100K, 500K, 1M entries
- Measures total memory and per-entry overhead
- Verifies linear scaling

Targets (Sprint 1c):
- Rust index: 44 bytes/entry (BLOCKING)
- Response dict: ~80 bytes/entry
- Total: ~124 bytes/entry (informational)
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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

# Stability benchmark configuration (Sprint 1c-OPT)
STABILITY_TRIALS = 5
STABILITY_SIZES = [10_000, 50_000, 100_000, 500_000]

# Derived from empirical results in benchmarks/results/memory_profile.json
# - 50k and 100k currently sit on the same plateau (~152 B/entry)
# - 500k converges down to ~142 B/entry
# These thresholds formalize "stable" behavior rather than exact byte values.
MAX_MIDSCALE_VARIATION_PERCENT = 5.0   # 50k vs 100k total bytes/entry
MAX_RUN_CV_PERCENT = 3.0               # Per-size run-to-run coefficient of variation
TOTAL_BYTES_HARD_CAP = 150.0           # Hard cap @ 100k and 500k entries (Phase 2.5 constraint)


def get_system_info() -> Dict[str, Any]:
    """Get system metadata."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }


def create_embedding(seed: int) -> np.ndarray:
    """Create a single normalized embedding."""
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return emb / np.linalg.norm(emb)


def profile_cache_size(n_entries: int, detailed: bool = False) -> Dict[str, Any]:
    """Profile memory usage for a given cache size.
    
    Args:
        n_entries: Number of entries to profile.
        detailed: If True, include Sprint 1c breakdown (Rust index, response dict).
    """
    gc.collect()
    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()
    
    # Create encoder (shared)
    encoder = BinaryEncoder(
        embedding_dim=EMBEDDING_DIM,
        code_bits=CODE_BITS,
        seed=SEED,
    )
    
    # Create cache
    cache = BinarySemanticCache(
        encoder=encoder,
        max_entries=n_entries + 100,  # Slight extra to avoid eviction
        similarity_threshold=0.80,
    )
    
    # Fill cache with entries (use integers as responses)
    batch_size = min(1000, n_entries)
    rng = np.random.default_rng(SEED)
    
    for i in range(0, n_entries, batch_size):
        n = min(batch_size, n_entries - i)
        embeddings = rng.standard_normal((n, EMBEDDING_DIM)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        for j in range(n):
            cache.put(embeddings[j], i + j)  # Integer response
    
    # Force GC
    gc.collect()
    
    # Get memory stats
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Calculate
    memory_mb = (current - baseline[0]) / 1024 / 1024
    peak_mb = (peak - baseline[1]) / 1024 / 1024
    bytes_per_entry = (current - baseline[0]) / n_entries if n_entries > 0 else 0
    
    # Theoretical minimum (codes only)
    theoretical_bytes = n_entries * CODE_BITS / 8
    theoretical_mb = theoretical_bytes / 1024 / 1024
    
    result = {
        "n_entries": n_entries,
        "memory_mb": memory_mb,
        "peak_mb": peak_mb,
        "bytes_per_entry": bytes_per_entry,
        "theoretical_mb": theoretical_mb,
        "overhead_ratio": memory_mb / theoretical_mb if theoretical_mb > 0 else 0,
    }
    
    # Sprint 1c: Detailed breakdown
    if detailed:
        # Rust index: reported by RustCacheStorage.memory_usage()
        rust_index_bytes = cache._storage.memory_usage()
        rust_index_per_entry = rust_index_bytes / n_entries if n_entries > 0 else 0
        
        # Total memory via cache.memory_bytes()
        total_cache_bytes = cache.memory_bytes()
        total_per_entry = total_cache_bytes / n_entries if n_entries > 0 else 0
        
        # Response dict overhead = total - rust_index
        response_overhead_bytes = total_cache_bytes - rust_index_bytes
        response_per_entry = response_overhead_bytes / n_entries if n_entries > 0 else 0
        
        result["sprint_1c"] = {
            "rust_index_bytes": rust_index_bytes,
            "rust_index_per_entry": rust_index_per_entry,
            "response_overhead_bytes": response_overhead_bytes,
            "response_per_entry": response_per_entry,
            "total_cache_bytes": total_cache_bytes,
            "total_per_entry": total_per_entry,
            # Targets
            "rust_index_target": 44,
            "rust_index_pass": rust_index_per_entry == 44,
            "total_target": 124,
            "total_within_bounds": total_per_entry <= 150,
        }
    
    return result


def run_benchmarks(detailed: bool = False, quick: bool = False) -> Dict[str, Any]:
    """Run all memory profiling benchmarks.
    
    Args:
        detailed: If True, include Sprint 1c breakdown (Rust index, response dict).
        quick: If True, only profile 10k and 100k entries.
    """
    print("=" * 60)
    print("MEMORY PROFILE BENCHMARK (Phase 2.5 Sprint 1c)")
    print("=" * 60)
    
    results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "system_info": get_system_info(),
        "config": {
            "embedding_dim": EMBEDDING_DIM,
            "code_bits": CODE_BITS,
        },
        "profiles": [],
    }
    
    if quick:
        sizes = [10_000, 100_000]
    else:
        sizes = [10_000, 50_000, 100_000, 500_000]
    
    print("\nProfiling cache sizes...")
    print("-" * 40)
    
    for size in sizes:
        print(f"\n  Profiling {size:,} entries...")
        
        stats = profile_cache_size(size, detailed=detailed)
        results["profiles"].append(stats)
        
        print(f"    Memory: {stats['memory_mb']:.2f} MB")
        print(f"    Per-entry: {stats['bytes_per_entry']:.0f} bytes")
        print(f"    Overhead: {stats['overhead_ratio']:.1f}x theoretical")
        
        # Sprint 1c detailed breakdown
        if detailed and "sprint_1c" in stats:
            s1c = stats["sprint_1c"]
            print(f"    --- Sprint 1c Breakdown ---")
            print(f"    Rust index:    {s1c['rust_index_per_entry']:.1f} B/entry "
                  f"(target: 44) {'✓' if s1c['rust_index_pass'] else '✗'}")
            print(f"    Response dict: {s1c['response_per_entry']:.1f} B/entry "
                  f"(target: ~80)")
            print(f"    Total:         {s1c['total_per_entry']:.1f} B/entry "
                  f"(target: ~124) {'✓' if s1c['total_within_bounds'] else '✗'}")
    
    # Check linear scaling
    print("\n" + "=" * 60)
    print("LINEAR SCALING CHECK")
    print("=" * 60)
    
    # Linear scaling means memory = Fixed + Variable × N
    # We verify:
    # 1. Bytes/entry CONVERGES (decreases) at larger scales (fixed overhead amortizes)
    # 2. Memory grows AT MOST linearly (no memory leak / quadratic growth)
    # 3. Large-scale bytes/entry is reasonable (<200 bytes for Python)
    
    bytes_per_entry_list = [p["bytes_per_entry"] for p in results["profiles"]]
    memory_list = [p["memory_mb"] for p in results["profiles"]]
    size_list = [p["n_entries"] for p in results["profiles"]]
    
    # Check 1: Bytes/entry should decrease or stabilize (not increase)
    converging = all(
        bytes_per_entry_list[i] >= bytes_per_entry_list[i+1] * 0.9
        for i in range(len(bytes_per_entry_list) - 1)
    )
    
    # Check 2: Memory should not grow faster than O(n)
    # Compare 100K to 500K: 5x entries should give <6x memory
    if len(results["profiles"]) >= 4:
        mem_100k = memory_list[2]
        mem_500k = memory_list[3]
        growth_ratio = mem_500k / mem_100k if mem_100k > 0 else 0
        sublinear = growth_ratio < 6  # 5x entries, <6x memory = OK
    else:
        sublinear = True
        growth_ratio = 0
    
    # Check 3: Large-scale efficiency
    large_scale_bpe = bytes_per_entry_list[-1]
    efficient = large_scale_bpe < 200  # <200 bytes/entry is good for Python
    
    linear_scaling = bool(converging and sublinear and efficient)
    
    print(f"\n  Bytes/entry convergence: {'✓ YES' if converging else '✗ NO'}")
    print(f"  100K→500K growth: {growth_ratio:.1f}x for 5x entries {'✓ OK' if sublinear else '✗ LEAK'}")
    print(f"  Large-scale efficiency: {large_scale_bpe:.0f} bytes/entry {'✓ GOOD' if efficient else '✗ HIGH'}")
    print(f"  Overall scaling: {'✓ PASS' if linear_scaling else '✗ FAIL'}")
    
    # Summary table
    print("\n" + "-" * 60)
    print(f"{'Size':>10} | {'Memory (MB)':>12} | {'Bytes/Entry':>12} | {'Overhead':>10}")
    print("-" * 60)
    
    for p in results["profiles"]:
        print(f"{p['n_entries']:>10,} | {p['memory_mb']:>12.2f} | "
              f"{p['bytes_per_entry']:>12.0f} | {p['overhead_ratio']:>10.1f}x")
    
    print("-" * 60)
    
    results["summary"] = {
        "large_scale_bytes_per_entry": float(large_scale_bpe),
        "linear_scaling": linear_scaling,
        "converging": converging,
        "sublinear_growth": sublinear,
        "efficient": efficient,
        "target_bytes_per_entry": 32,  # Codes only (theoretical minimum)
        "growth_100k_to_500k": float(growth_ratio) if len(results["profiles"]) >= 4 else None,
    }
    
    # Sprint 1c summary (if detailed mode)
    if detailed and results["profiles"]:
        last_profile = results["profiles"][-1]
        if "sprint_1c" in last_profile:
            s1c = last_profile["sprint_1c"]
            results["sprint_1c_summary"] = {
                "rust_index_per_entry": s1c["rust_index_per_entry"],
                "rust_index_target": 44,
                "rust_index_pass": s1c["rust_index_pass"],
                "response_per_entry": s1c["response_per_entry"],
                "total_per_entry": s1c["total_per_entry"],
                "total_target": 124,
                "total_within_bounds": s1c["total_within_bounds"],
            }
            
            print("\n" + "=" * 60)
            print("SPRINT 1c MEMORY VALIDATION")
            print("=" * 60)
            print(f"\n  Rust index:    {s1c['rust_index_per_entry']:.1f} B/entry "
                  f"(target: 44 B) {'✓ PASS' if s1c['rust_index_pass'] else '✗ FAIL'}")
            print(f"  Response dict: {s1c['response_per_entry']:.1f} B/entry "
                  f"(target: ~80 B)")
            print(f"  Total:         {s1c['total_per_entry']:.1f} B/entry "
                  f"(target: ~124 B) {'✓ PASS' if s1c['total_within_bounds'] else '✗ FAIL'}")
            
            if s1c["rust_index_pass"]:
                print("\n  ✅ RUST INDEX TARGET MET (44 B/entry)")
            else:
                print(f"\n  ❌ RUST INDEX TARGET MISSED ({s1c['rust_index_per_entry']:.1f} B/entry)")
    
    # Save results
    output_path = _ROOT / "benchmarks" / "results" / "memory_profile.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _stddev(values: List[float], mean: float) -> float:
    if not values:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return float(var ** 0.5)


def run_stability_benchmarks(trials: int = STABILITY_TRIALS) -> Dict[str, Any]:
    """Run stability benchmarks across multiple trials.

    Measures run-to-run variance and mid-scale stability for total bytes/entry.
    Always runs in detailed mode to capture Sprint 1c breakdown.
    """
    print("=" * 60)
    print("MEMORY STABILITY BENCHMARK (Phase 2.5 Sprint 1c-OPT)")
    print("=" * 60)

    results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "system_info": get_system_info(),
        "config": {
            "embedding_dim": EMBEDDING_DIM,
            "code_bits": CODE_BITS,
            "trials": trials,
        },
        "sizes": STABILITY_SIZES,
        "profiles": [],
    }

    size_summaries: Dict[int, Dict[str, float]] = {}

    for n_entries in STABILITY_SIZES:
        print(f"\nProfiling stability at {n_entries:,} entries "
              f"({trials} trials)...")

        index_values: List[float] = []
        response_values: List[float] = []
        total_values: List[float] = []

        for trial in range(trials):
            print(f"  Trial {trial + 1}/{trials}...")
            stats = profile_cache_size(n_entries=n_entries, detailed=True)
            s1c = stats.get("sprint_1c", {})

            index_values.append(float(s1c.get("rust_index_per_entry", 0.0)))
            response_values.append(float(s1c.get("response_per_entry", 0.0)))
            total_values.append(float(s1c.get("total_per_entry", 0.0)))

        index_mean = _mean(index_values)
        index_std = _stddev(index_values, index_mean)

        response_mean = _mean(response_values)
        response_std = _stddev(response_values, response_mean)

        total_mean = _mean(total_values)
        total_std = _stddev(total_values, total_mean)

        size_summaries[n_entries] = {
            "index_mean": index_mean,
            "index_std": index_std,
            "response_mean": response_mean,
            "response_std": response_std,
            "total_mean": total_mean,
            "total_std": total_std,
        }

        results["profiles"].append(
            {
                "n_entries": n_entries,
                "trials": trials,
                "index_bytes_per_entry": {
                    "values": index_values,
                    "mean": index_mean,
                    "stddev": index_std,
                },
                "response_bytes_per_entry": {
                    "values": response_values,
                    "mean": response_mean,
                    "stddev": response_std,
                },
                "total_bytes_per_entry": {
                    "values": total_values,
                    "mean": total_mean,
                    "stddev": total_std,
                },
            }
        )

    # Mid-scale stability check: 50k vs 100k total bytes/entry
    mid_50k = size_summaries.get(50_000, {})
    mid_100k = size_summaries.get(100_000, {})
    if mid_50k and mid_100k:
        m50 = mid_50k["total_mean"]
        m100 = mid_100k["total_mean"]
        midscale_variation = abs(m50 - m100)
        midscale_variation_pct = (
            midscale_variation / ((m50 + m100) / 2.0) * 100.0
            if (m50 + m100) > 0
            else 0.0
        )
    else:
        midscale_variation_pct = 0.0

    # Run-to-run stability: coefficient of variation per size
    run_cv_percent: Dict[int, float] = {}
    for n_entries, summary in size_summaries.items():
        mean_total = summary["total_mean"]
        std_total = summary["total_std"]
        cv = (std_total / mean_total * 100.0) if mean_total > 0 else 0.0
        run_cv_percent[n_entries] = cv

    # Hard cap check for total bytes/entry at 100k and 500k
    total_cap_scales = [100_000, 500_000]
    total_cap_pass = True
    total_cap_measured: Dict[int, float] = {}
    for scale in total_cap_scales:
        summary = size_summaries.get(scale)
        if not summary:
            continue
        mean_total = summary["total_mean"]
        total_cap_measured[scale] = mean_total
        if mean_total > TOTAL_BYTES_HARD_CAP:
            total_cap_pass = False

    # Index must remain at 44 B/entry (hard contract)
    index_target_pass = all(
        abs(summary["index_mean"] - 44.0) < 1e-6 for summary in size_summaries.values()
    )

    midscale_pass = midscale_variation_pct <= MAX_MIDSCALE_VARIATION_PERCENT
    run_to_run_pass = all(
        cv <= MAX_RUN_CV_PERCENT for cv in run_cv_percent.values()
    )

    overall_pass = bool(
        index_target_pass and midscale_pass and run_to_run_pass and total_cap_pass
    )

    results["stability_summary"] = {
        "midscale_variation_percent": midscale_variation_pct,
        "midscale_variation_threshold_percent": MAX_MIDSCALE_VARIATION_PERCENT,
        "midscale_pass": midscale_pass,
        "run_to_run_cv_percent": {
            str(k): v for k, v in sorted(run_cv_percent.items())
        },
        "run_to_run_threshold_percent": MAX_RUN_CV_PERCENT,
        "run_to_run_pass": run_to_run_pass,
        "total_cap_bytes_per_entry": TOTAL_BYTES_HARD_CAP,
        "total_cap_scales": total_cap_scales,
        "total_cap_measured": {str(k): v for k, v in total_cap_measured.items()},
        "total_cap_pass": total_cap_pass,
        "index_target": 44.0,
        "index_target_pass": index_target_pass,
        "overall_pass": overall_pass,
    }

    output_path = _ROOT / "benchmarks" / "results" / "memory_profile_stability.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("STABILITY SUMMARY")
    print("=" * 60)
    print(
        f"  Mid-scale variation (50k vs 100k, total B/entry): "
        f"{midscale_variation_pct:.2f}% "
        f"(threshold: {MAX_MIDSCALE_VARIATION_PERCENT:.1f}%)"
    )
    print(
        f"  Run-to-run CV per size (total B/entry, threshold "
        f"{MAX_RUN_CV_PERCENT:.1f}%):"
    )
    for n_entries in sorted(run_cv_percent.keys()):
        print(
            f"    {n_entries:>10,}: {run_cv_percent[n_entries]:5.2f}%"
        )
    for scale in total_cap_scales:
        if str(scale) in results["stability_summary"]["total_cap_measured"]:
            mean_total = results["stability_summary"]["total_cap_measured"][str(scale)]
            print(
                f"  Total cap @{scale:>7,}: {mean_total:.2f} B/entry "
                f"(cap: {TOTAL_BYTES_HARD_CAP:.1f})"
            )
    print(f"\n  Overall stability: {'✓ PASS' if overall_pass else '✗ FAIL'}")

    print(f"\nStability results saved to: {output_path}")

    return results


def main() -> int:
    """Run benchmarks."""
    parser = argparse.ArgumentParser(description="Memory Profile Benchmark")
    parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="Include Sprint 1c breakdown (Rust index, response dict)"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: only profile 10k and 100k entries"
    )
    parser.add_argument(
        "--stability",
        action="store_true",
        help=(
            "Run stability benchmark (multiple trials, mean/stddev per size, "
            "writes memory_profile_stability.json)"
        ),
    )
    parser.add_argument(
        "--stability-trials",
        type=int,
        default=STABILITY_TRIALS,
        help="Number of trials per size for --stability (default: %(default)s)",
    )
    args = parser.parse_args()
    
    try:
        if args.stability:
            stability_results = run_stability_benchmarks(trials=args.stability_trials)
            summary = stability_results.get("stability_summary", {})
            return 0 if summary.get("overall_pass", False) else 1

        results = run_benchmarks(detailed=args.detailed, quick=args.quick)

        # Check Sprint 1c targets if detailed mode
        if args.detailed and "sprint_1c_summary" in results:
            s1c = results["sprint_1c_summary"]
            if not s1c["rust_index_pass"]:
                print("\n❌ BENCHMARK FAILED: Rust index target not met")
                return 1

        return 0 if results["summary"]["linear_scaling"] else 1
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

