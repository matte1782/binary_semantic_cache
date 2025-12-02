#!/usr/bin/env python3
"""
Zero-Copy Persistence Benchmark (Phase 2.4 → Sprint 2a v3)

Benchmarks persistence format v3 vs legacy npz:
1. Save latency (save_mmap_v3 vs save)
2. Load latency (codes-only vs full load including responses) — PRIMARY TARGET: < 100ms @ 1M entries
3. Checksum overhead (SHA-256 streaming)
4. Peak resident set size (RSS) during fill/save

Target: Load 1M entries in < 100ms using persistence v3.

See: docs/PHASE2_BENCHMARK_PLAN.md
"""

from __future__ import annotations

import gc
import hashlib
import json
import platform
import shutil
import sys
import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

# Add src to path
_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _ROOT / "src"
sys.path.insert(0, str(_SRC_PATH))

from binary_semantic_cache import BinarySemanticCache
from binary_semantic_cache.binary_semantic_cache_rs import RustBinaryEncoder
from binary_semantic_cache.core.encoder import BinaryEncoder as PythonBinaryEncoder


# =============================================================================
# Configuration
# =============================================================================

EMBEDDING_DIM = 384
CODE_BITS = 256
SEED = 42

# Persistence v3 file names
PERSISTENCE_V3_HEADER = "header.json"
PERSISTENCE_V3_ENTRIES = "entries.bin"
PERSISTENCE_V3_RESPONSES = "responses.pkl"

# Test sizes (Sprint 2a requirement)
SIZES = [100_000, 1_000_000]

# Primary target
PRIMARY_TARGET_SIZE = 1_000_000
PRIMARY_TARGET_LOAD_MS = 100.0  # < 100ms @ 1M entries


# =============================================================================
# Helpers
# =============================================================================

def get_system_info() -> Dict[str, Any]:
    """Get system metadata."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }


def get_projection_matrix() -> np.ndarray:
    """Get projection matrix from Python encoder (deterministic seed=42)."""
    python_encoder = PythonBinaryEncoder(
        embedding_dim=EMBEDDING_DIM,
        code_bits=CODE_BITS,
        seed=SEED,
    )
    weights = python_encoder._projection._weights  # Shape: (384, 256)
    return weights.T.astype(np.float32)  # Transpose to (256, 384)


def create_rust_encoder(projection_matrix: np.ndarray) -> RustBinaryEncoder:
    """Create Rust encoder with given projection matrix."""
    return RustBinaryEncoder(EMBEDDING_DIM, CODE_BITS, projection_matrix)


def fill_cache(
    cache: BinarySemanticCache,
    n_entries: int,
    seed: int = SEED,
) -> List[np.ndarray]:
    """Fill cache with n_entries and return sample embeddings."""
    rng = np.random.default_rng(seed)
    sample_embeddings = []
    
    batch_size = min(10_000, n_entries)
    
    for i in range(0, n_entries, batch_size):
        n = min(batch_size, n_entries - i)
        embeddings = rng.standard_normal((n, EMBEDDING_DIM)).astype(np.float32)
        
        for j in range(n):
            cache.put(embeddings[j], {"id": i + j})
        
        # Keep first 10 for query testing
        if i == 0:
            sample_embeddings = [embeddings[k].copy() for k in range(min(10, n))]
        
        # Progress indicator
        if (i + n) % 100_000 == 0 or i + n == n_entries:
            print(f"    Filled {i + n:,}/{n_entries:,} entries...", end="\r")
    
    print()  # Newline after progress
    return sample_embeddings


def _bytes_to_mb(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return value / (1024 * 1024)


def _get_rss_bytes() -> Optional[int]:
    """Best-effort process RSS (bytes)."""
    try:
        import psutil  # type: ignore

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass

    if sys.platform != "win32":
        try:
            import resource  # type: ignore

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux returns KB, macOS returns bytes
            if sys.platform == "darwin":
                return int(rss)
            return int(rss * 1024)
        except Exception:
            return None

    try:
        import ctypes
        import ctypes.wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.wintypes.DWORD),
                ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.wintypes.SIZE_T),
                ("WorkingSetSize", ctypes.wintypes.SIZE_T),
                ("QuotaPeakPagedPoolUsage", ctypes.wintypes.SIZE_T),
                ("QuotaPagedPoolUsage", ctypes.wintypes.SIZE_T),
                ("QuotaPeakNonPagedPoolUsage", ctypes.wintypes.SIZE_T),
                ("QuotaNonPagedPoolUsage", ctypes.wintypes.SIZE_T),
                ("PagefileUsage", ctypes.wintypes.SIZE_T),
                ("PeakPagefileUsage", ctypes.wintypes.SIZE_T),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        ):
            return int(counters.WorkingSetSize)
    except Exception:
        return None
    return None


def _stream_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute SHA-256 using streaming chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def measure_checksum_cost(paths: Sequence[Path]) -> float:
    """Return elapsed milliseconds to compute SHA-256 for provided files."""
    start = time.perf_counter()
    for path in paths:
        _stream_sha256(path)
    return (time.perf_counter() - start) * 1000.0


def copy_entries_file(entries_path: Path, chunk_size: int = 16 * 1024 * 1024) -> float:
    """
    Approximate codes-only load time by streaming entries.bin into memory.

    This simulates the O(n) memcpy step into Rust storage.
    """
    total_size = entries_path.stat().st_size
    buffer = bytearray(min(total_size, chunk_size))
    start = time.perf_counter()
    with open(entries_path, "rb") as handle:
        while True:
            read = handle.readinto(buffer)
            if not read:
                break
    return (time.perf_counter() - start) * 1000.0


def ensure_v3_api(cache: BinarySemanticCache) -> None:
    missing = [
        method
        for method in ("save_mmap_v3", "load_mmap_v3")
        if not hasattr(cache, method)
    ]
    if missing:
        raise RuntimeError(
            f"BinarySemanticCache missing v3 persistence APIs: {', '.join(missing)}"
        )


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_save_mmap(
    cache: BinarySemanticCache,
    path: Path,
    n_trials: int = 3,
) -> Dict[str, float]:
    """Benchmark save_mmap() latency."""
    times = []
    
    for _ in range(n_trials):
        # Clean up
        if path.exists():
            shutil.rmtree(path)
        gc.collect()
        
        start = time.perf_counter()
        cache.save_mmap(str(path))
        end = time.perf_counter()
        
        times.append((end - start) * 1000)  # ms
    
    return {
        "mean_ms": float(np.mean(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "std_ms": float(np.std(times)),
    }


def benchmark_save_v3(
    cache: BinarySemanticCache,
    path: Path,
    n_trials: int = 3,
) -> Dict[str, float]:
    """Benchmark save_mmap_v3() latency and checksum cost."""
    ensure_v3_api(cache)
    times: List[float] = []
    checksum_ms: Optional[float] = None

    for _ in range(n_trials):
        if path.exists():
            shutil.rmtree(path)
        gc.collect()

        start = time.perf_counter()
        cache.save_mmap_v3(str(path))
        elapsed = (time.perf_counter() - start) * 1000.0
        times.append(elapsed)

        if checksum_ms is None:
            entries_path = path / PERSISTENCE_V3_ENTRIES
            responses_path = path / PERSISTENCE_V3_RESPONSES
            checksum_ms = measure_checksum_cost([entries_path, responses_path])

    return {
        "mean_ms": float(np.mean(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "std_ms": float(np.std(times)),
        "checksum_ms": float(checksum_ms or 0.0),
    }


def benchmark_load_mmap(
    projection_matrix: np.ndarray,
    path: Path,
    max_entries: int,
    n_trials: int = 5,
) -> Dict[str, float]:
    """Benchmark load_mmap() latency."""
    times = []
    
    for _ in range(n_trials):
        gc.collect()
        
        # Create fresh cache
        encoder = create_rust_encoder(projection_matrix)
        cache = BinarySemanticCache(
            encoder=encoder,
            max_entries=max_entries,
            similarity_threshold=0.80,
        )
        
        start = time.perf_counter()
        cache.load_mmap(str(path))
        end = time.perf_counter()
        
        times.append((end - start) * 1000)  # ms
        
        # Verify loaded
        assert len(cache) > 0, "Cache should have entries after load"
    
    return {
        "mean_ms": float(np.mean(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "std_ms": float(np.std(times)),
    }


def benchmark_load_v3(
    projection_matrix: np.ndarray,
    path: Path,
    max_entries: int,
    n_trials: int = 5,
) -> Dict[str, float]:
    """Benchmark load_mmap_v3() latency."""
    times: List[float] = []

    for _ in range(n_trials):
        gc.collect()
        encoder = create_rust_encoder(projection_matrix)
        cache = BinarySemanticCache(
            encoder=encoder,
            max_entries=max_entries,
            similarity_threshold=0.80,
        )
        ensure_v3_api(cache)

        start = time.perf_counter()
        cache.load_mmap_v3(str(path))
        elapsed = (time.perf_counter() - start) * 1000.0
        times.append(elapsed)

        assert len(cache) > 0, "Cache should have entries after load"

    return {
        "mean_ms": float(np.mean(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "std_ms": float(np.std(times)),
    }


def benchmark_legacy_save(
    cache: BinarySemanticCache,
    path: Path,
    n_trials: int = 3,
) -> Dict[str, float]:
    """Benchmark legacy save() latency (for comparison)."""
    times = []
    
    for _ in range(n_trials):
        # Clean up
        if path.exists():
            path.unlink()
        gc.collect()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            
            start = time.perf_counter()
            cache.save(str(path))
            end = time.perf_counter()
        
        times.append((end - start) * 1000)  # ms
    
    return {
        "mean_ms": float(np.mean(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "std_ms": float(np.std(times)),
    }


def benchmark_legacy_load(
    projection_matrix: np.ndarray,
    path: Path,
    max_entries: int,
    n_trials: int = 3,
) -> Dict[str, float]:
    """Benchmark legacy load() latency (for comparison)."""
    times = []
    
    for _ in range(n_trials):
        gc.collect()
        
        # Create fresh cache
        encoder = create_rust_encoder(projection_matrix)
        cache = BinarySemanticCache(
            encoder=encoder,
            max_entries=max_entries,
            similarity_threshold=0.80,
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            
            start = time.perf_counter()
            cache.load(str(path))
            end = time.perf_counter()
        
        times.append((end - start) * 1000)  # ms
        
        # Verify loaded
        assert len(cache) > 0, "Cache should have entries after load"
    
    return {
        "mean_ms": float(np.mean(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "std_ms": float(np.std(times)),
    }


def benchmark_first_query(
    projection_matrix: np.ndarray,
    path: Path,
    max_entries: int,
    sample_embeddings: List[np.ndarray],
    persistence_format: str = "v2",
) -> Dict[str, float]:
    """Benchmark first query latency after load_mmap/load_mmap_v3."""
    gc.collect()
    
    # Create fresh cache
    encoder = create_rust_encoder(projection_matrix)
    cache = BinarySemanticCache(
        encoder=encoder,
        max_entries=max_entries,
        similarity_threshold=0.80,
    )
    
    # Load
    if persistence_format == "v3":
        ensure_v3_api(cache)
        cache.load_mmap_v3(str(path))
    else:
        cache.load_mmap(str(path))
    
    # First query
    start = time.perf_counter()
    result = cache.get(sample_embeddings[0])
    first_query_ms = (time.perf_counter() - start) * 1000
    
    # Subsequent queries (should be faster due to caching)
    subsequent_times = []
    for emb in sample_embeddings[1:]:
        start = time.perf_counter()
        cache.get(emb)
        subsequent_times.append((time.perf_counter() - start) * 1000)
    
    return {
        "first_query_ms": float(first_query_ms),
        "mean_subsequent_ms": float(np.mean(subsequent_times)) if subsequent_times else 0,
        "hit": result is not None,
    }


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_benchmarks(
    sizes: Optional[Iterable[int]] = None,
    persistence_format: str = "v3",
    include_legacy: bool = False,
) -> Dict[str, Any]:
    """Run persistence benchmarks for the requested format."""
    if sizes is None:
        sizes = SIZES
    sizes = list(sizes)

    print("=" * 70)
    print(f"ZERO-COPY PERSISTENCE BENCHMARK ({persistence_format.upper()})")
    print("=" * 70)
    print(f"Target: Load {PRIMARY_TARGET_SIZE:,} entries in < {PRIMARY_TARGET_LOAD_MS:.0f}ms")
    print()

    results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "system_info": get_system_info(),
        "config": {
            "embedding_dim": EMBEDDING_DIM,
            "code_bits": CODE_BITS,
            "primary_target_size": PRIMARY_TARGET_SIZE,
            "primary_target_load_ms": PRIMARY_TARGET_LOAD_MS,
        },
        "persistence_format": persistence_format,
        "benchmarks": [],
    }

    print("[0/4] Generating projection matrix...")
    projection_matrix = get_projection_matrix()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        for size in sizes:
            print(f"\n{'=' * 70}")
            print(f"BENCHMARKING {size:,} ENTRIES")
            print("=" * 70)

            encoder = create_rust_encoder(projection_matrix)
            cache = BinarySemanticCache(
                encoder=encoder,
                max_entries=size + 1_000,
                similarity_threshold=0.80,
            )

            print(f"\n[1/4] Creating cache with {size:,} entries...")
            sample_embeddings = fill_cache(cache, size)
            peak_rss_mb = _bytes_to_mb(_get_rss_bytes())
            n_words = CODE_BITS // 64
            codes_size_mb = (size * n_words * 8) / (1024 * 1024)
            print(f"    Codes size (approx): {codes_size_mb:.1f} MB")

            bench_result: Dict[str, Any] = {
                "n_entries": size,
                "peak_rss_mb": peak_rss_mb,
            }

            cache_path = tmp_path / f"{persistence_format}_cache_{size}"
            legacy_npz_path = tmp_path / f"legacy_{size}.npz"

            if persistence_format == "v3":
                print(f"\n[2/4] Benchmarking save_mmap_v3()...")
                save_stats = benchmark_save_v3(cache, cache_path, n_trials=3)
                bench_result["save_v3"] = save_stats
                print(
                    "    save_mmap_v3: mean={:.1f}ms (checksum {:.1f}ms)".format(
                        save_stats["mean_ms"], save_stats["checksum_ms"]
                    )
                )

                print(f"\n[3/4] Benchmarking load_mmap_v3() (PRIMARY TARGET)...")
                load_stats = benchmark_load_v3(
                    projection_matrix, cache_path, size + 1_000, n_trials=5
                )
                entries_path = cache_path / PERSISTENCE_V3_ENTRIES
                codes_only_ms = copy_entries_file(entries_path)
                bench_result["load_v3"] = {
                    "full_load": load_stats,
                    "codes_only_ms": codes_only_ms,
                }
                status = "✓ PASS" if load_stats["mean_ms"] < PRIMARY_TARGET_LOAD_MS else "✗ FAIL"
                print(
                    f"    load_mmap_v3: mean={load_stats['mean_ms']:.1f}ms "
                    f"(codes-only {codes_only_ms:.1f}ms) {status}"
                )

                print(f"\n[4/4] Benchmarking first query after load_mmap_v3()...")
                first_query_stats = benchmark_first_query(
                    projection_matrix,
                    cache_path,
                    size + 1_000,
                    sample_embeddings,
                    persistence_format="v3",
                )
                bench_result["first_query"] = first_query_stats
                print(
                    f"    First query: {first_query_stats['first_query_ms']:.3f}ms "
                    f"(hit={first_query_stats['hit']})"
                )
                print(
                    f"    Subsequent (mean): "
                    f"{first_query_stats['mean_subsequent_ms']:.3f}ms"
                )
            else:
                print(f"\n[2/5] Benchmarking save_mmap()...")
                save_stats = benchmark_save_mmap(cache, cache_path, n_trials=3)
                bench_result["save_mmap"] = save_stats
                print(
                    f"    save_mmap: mean={save_stats['mean_ms']:.1f}ms "
                    f"(min {save_stats['min_ms']:.1f}ms)"
                )

                print(f"\n[3/5] Benchmarking load_mmap() (PRIMARY TARGET)...")
                load_stats = benchmark_load_mmap(
                    projection_matrix, cache_path, size + 1_000, n_trials=5
                )
                bench_result["load_mmap"] = load_stats
                status = "✓ PASS" if load_stats["mean_ms"] < PRIMARY_TARGET_LOAD_MS else "✗ FAIL"
                print(
                    f"    load_mmap: mean={load_stats['mean_ms']:.1f}ms "
                    f"(min {load_stats['min_ms']:.1f}ms) {status}"
                )

                print(f"\n[4/5] Benchmarking first query after load_mmap()...")
                first_query_stats = benchmark_first_query(
                    projection_matrix,
                    cache_path,
                    size + 1_000,
                    sample_embeddings,
                    persistence_format="v2",
                )
                bench_result["first_query"] = first_query_stats
                print(
                    f"    First query: {first_query_stats['first_query_ms']:.3f}ms "
                    f"(hit={first_query_stats['hit']})"
                )
                print(
                    f"    Subsequent (mean): "
                    f"{first_query_stats['mean_subsequent_ms']:.3f}ms"
                )

            if include_legacy and size <= 100_000:
                print("\n[Legacy] Benchmarking save/load() for comparison...")
                save_legacy = benchmark_legacy_save(cache, legacy_npz_path, n_trials=2)
                load_legacy = benchmark_legacy_load(
                    projection_matrix, legacy_npz_path, size + 1_000, n_trials=2
                )
                bench_result["save_legacy"] = save_legacy
                bench_result["load_legacy"] = load_legacy
                speedup = load_legacy["mean_ms"] / (
                    bench_result.get("load_v3", {})
                    .get("full_load", {})
                    .get("mean_ms", load_legacy["mean_ms"])
                    if persistence_format == "v3"
                    else bench_result["load_mmap"]["mean_ms"]
                )
                bench_result["load_speedup"] = float(speedup)
                print(
                    f"    legacy load: {load_legacy['mean_ms']:.1f}ms "
                    f"(speedup {speedup:.1f}x)"
                )

            results["benchmarks"].append(bench_result)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if persistence_format == "v3":
        print(
            f"\n{'Size':>12} | {'Save_v3 (ms)':>15} | "
            f"{'LoadCodes (ms)':>15} | {'LoadFull (ms)':>15} | {'Checksum (ms)':>15}"
        )
        print("-" * 80)
        for bench in results["benchmarks"]:
            size = bench["n_entries"]
            save_stats = bench["save_v3"]
            load_stats = bench["load_v3"]["full_load"]
            codes_ms = bench["load_v3"]["codes_only_ms"]
            print(
                f"{size:>12,} | {save_stats['mean_ms']:>15.1f} | "
                f"{codes_ms:>15.1f} | {load_stats['mean_ms']:>15.1f} | "
                f"{save_stats['checksum_ms']:>15.1f}"
            )
    else:
        print(
            f"\n{'Size':>12} | {'load_mmap (ms)':>15} | "
            f"{'load_npz (ms)':>15} | {'Speedup':>10}"
        )
        print("-" * 60)
        for bench in results["benchmarks"]:
            size = bench["n_entries"]
            load_mmap = bench["load_mmap"]["mean_ms"]
            load_npz = (bench.get("load_legacy") or {}).get("mean_ms", "N/A")
            speedup = bench.get("load_speedup") or "N/A"
            npz_str = f"{load_npz:.1f}" if isinstance(load_npz, float) else load_npz
            speedup_str = f"{speedup:.1f}x" if isinstance(speedup, float) else speedup
            print(f"{size:>12,} | {load_mmap:>15.1f} | {npz_str:>15} | {speedup_str:>10}")

    print("-" * 80)

    primary_result = next(
        (b for b in results["benchmarks"] if b["n_entries"] == PRIMARY_TARGET_SIZE),
        None,
    )
    if primary_result:
        if persistence_format == "v3":
            load_time = primary_result["load_v3"]["full_load"]["mean_ms"]
        else:
            load_time = primary_result["load_mmap"]["mean_ms"]
        passed = load_time < PRIMARY_TARGET_LOAD_MS
        print(f"\nPRIMARY TARGET: Load {PRIMARY_TARGET_SIZE:,} entries")
        print(f"  Target: < {PRIMARY_TARGET_LOAD_MS:.0f}ms")
        print(f"  Result: {load_time:.1f}ms")
        print(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")
        results["primary_target_passed"] = passed
    else:
        results["primary_target_passed"] = None
        print(f"\nNote: Primary target ({PRIMARY_TARGET_SIZE:,} entries) was not benchmarked")

    output_path = _ROOT / "benchmarks" / "results" / "persistence_bench.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(results, handle, indent=2)

    print(f"\nResults saved to: {output_path}")
    return results


def main() -> int:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Zero-Copy Persistence Benchmark")
    parser.add_argument(
        "--entries",
        type=int,
        nargs="+",
        default=None,
        help="Cache sizes to benchmark (default: 100k, 1M)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark (100k only)",
    )
    parser.add_argument(
        "--format",
        choices=["v3", "v2"],
        default="v3",
        help="Persistence format to benchmark (default: v3)",
    )
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Also benchmark legacy npz save/load (advisory metric)",
    )
    
    args = parser.parse_args()
    
    if args.quick:
        sizes = [100_000]
    elif args.entries:
        sizes = args.entries
    else:
        sizes = SIZES
    
    try:
        results = run_benchmarks(
            sizes,
            persistence_format=args.format,
            include_legacy=args.include_legacy,
        )
        
        # Return code based on primary target
        if results.get("primary_target_passed") is False:
            print("\n⚠️ PRIMARY TARGET NOT MET")
            return 1
        return 0
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

