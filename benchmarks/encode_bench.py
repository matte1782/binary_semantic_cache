#!/usr/bin/env python3
"""
Encode Benchmark

Benchmarks BinaryEncoder performance for various scenarios:
- Single embedding encode
- Batch encode
- Different batch sizes

Target: mean < 1ms for single embedding
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

# Configuration
WARMUP_ITERATIONS = 10
MEASURE_ITERATIONS = 100
EMBEDDING_DIM = 384
CODE_BITS = 256
SEED = 42


def get_system_info() -> Dict[str, Any]:
    """Get system metadata for reproducibility."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
    }


def create_embeddings(n: int, dim: int = EMBEDDING_DIM, seed: int = SEED) -> np.ndarray:
    """Create normalized random embeddings."""
    rng = np.random.default_rng(seed)
    embeddings = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def benchmark_single_encode(
    encoder: BinaryEncoder, 
    n_samples: int,
    warmup: int = WARMUP_ITERATIONS,
    measure: int = MEASURE_ITERATIONS,
) -> Dict[str, float]:
    """Benchmark single embedding encode."""
    embeddings = create_embeddings(max(n_samples, measure))
    
    # Warmup
    for i in range(warmup):
        encoder.encode(embeddings[i % n_samples])
    
    # Measure
    times_ms: List[float] = []
    for i in range(measure):
        start = time.perf_counter()
        encoder.encode(embeddings[i % n_samples])
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)
    
    arr = np.array(times_ms)
    return {
        "n_samples": n_samples,
        "iterations": measure,
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def benchmark_batch_encode(
    encoder: BinaryEncoder,
    batch_size: int,
    warmup: int = WARMUP_ITERATIONS,
    measure: int = MEASURE_ITERATIONS,
) -> Dict[str, float]:
    """Benchmark batch embedding encode."""
    embeddings = create_embeddings(batch_size)
    
    # Warmup
    for _ in range(warmup):
        encoder.encode_batch(embeddings)
    
    # Measure
    times_ms: List[float] = []
    for _ in range(measure):
        start = time.perf_counter()
        encoder.encode_batch(embeddings)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)
    
    arr = np.array(times_ms)
    return {
        "batch_size": batch_size,
        "iterations": measure,
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "per_sample_ms": float(np.mean(arr) / batch_size),
    }


def run_benchmarks() -> Dict[str, Any]:
    """Run all encode benchmarks."""
    print("=" * 60)
    print("ENCODE BENCHMARK")
    print("=" * 60)
    
    encoder = BinaryEncoder(
        embedding_dim=EMBEDDING_DIM,
        code_bits=CODE_BITS,
        seed=SEED,
    )
    print(f"Encoder: {encoder}")
    
    results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "system_info": get_system_info(),
        "config": {
            "embedding_dim": EMBEDDING_DIM,
            "code_bits": CODE_BITS,
            "warmup_iterations": WARMUP_ITERATIONS,
            "measure_iterations": MEASURE_ITERATIONS,
        },
        "single_encode": [],
        "batch_encode": [],
    }
    
    # Single encode benchmarks
    print("\n[1/2] Single Encode Benchmarks")
    print("-" * 40)
    
    for n_samples in [1, 10, 100, 1000]:
        stats = benchmark_single_encode(encoder, n_samples)
        results["single_encode"].append(stats)
        
        status = "✓ PASS" if stats["mean_ms"] < 1.0 else "✗ FAIL"
        print(f"  n={n_samples:4d}: mean={stats['mean_ms']:.3f}ms, "
              f"p95={stats['p95_ms']:.3f}ms {status}")
    
    # Batch encode benchmarks
    print("\n[2/2] Batch Encode Benchmarks")
    print("-" * 40)
    
    for batch_size in [10, 100, 1000]:
        stats = benchmark_batch_encode(encoder, batch_size)
        results["batch_encode"].append(stats)
        
        print(f"  batch={batch_size:4d}: total={stats['mean_ms']:.3f}ms, "
              f"per_sample={stats['per_sample_ms']:.3f}ms")
    
    # Summary
    single_mean = results["single_encode"][0]["mean_ms"]
    target_met = single_mean < 1.0
    
    print("\n" + "=" * 60)
    print(f"SINGLE ENCODE: mean={single_mean:.3f}ms {'✓ PASS' if target_met else '✗ FAIL'}")
    print(f"TARGET: < 1.0 ms")
    print("=" * 60)
    
    results["summary"] = {
        "single_encode_mean_ms": single_mean,
        "target_ms": 1.0,
        "target_met": target_met,
    }
    
    # Save results
    output_path = _ROOT / "benchmarks" / "results" / "encode_bench.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


def main() -> int:
    """Run benchmarks."""
    try:
        results = run_benchmarks()
        return 0 if results["summary"]["target_met"] else 1
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

