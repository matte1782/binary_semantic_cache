"""
Stage 1: Latency Validation Benchmark
Purpose: Validate latency and memory assumptions for Binary Semantic Cache

KILL TRIGGERS:
- Encode latency > 5ms → Architecture at risk
- Lookup latency > 1ms for 100K entries → Architecture at risk  
- Memory > 10MB for 100K entries → Architecture at risk

TARGETS:
- Encode: <1ms per embedding
- Lookup: <100µs for 100K entries (brute force Hamming)
- Memory: <4MB for 100K entries (256-bit codes = 32 bytes each)
"""

import sys
import time
import json
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

# Add BinaryLLM to path
BINARYLLM_PATH = Path(__file__).parent.parent.parent.parent / "binary_llm"
sys.path.insert(0, str(BINARYLLM_PATH))

print(f"BinaryLLM path: {BINARYLLM_PATH}")

# Import BinaryLLM components
from src.quantization.binarization import RandomProjection, binarize_sign
from src.quantization.packing import pack_codes


class BinaryAdapter:
    """Adapter for encoding embeddings to binary codes using BinaryLLM."""
    
    def __init__(self, embedding_dim: int = 384, code_bits: int = 256, seed: int = 42):
        self._projection = RandomProjection(embedding_dim, code_bits, seed)
        self._code_bits = code_bits
        
    def encode(self, embedding: np.ndarray) -> np.ndarray:
        """Convert float embedding to packed uint64 binary code."""
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        projected = self._projection.project(embedding)
        codes_pm1 = binarize_sign(projected)
        codes_01 = ((codes_pm1 + 1.0) / 2.0).astype(np.int8)
        return pack_codes(codes_01)


def hamming_distance_batch(query: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Compute Hamming distances between query and all codes using popcount."""
    # XOR query with all codes, then count bits
    xored = query ^ codes  # Broadcasting: (1, W) ^ (N, W) = (N, W)
    # Popcount per word, sum across words
    distances = np.zeros(codes.shape[0], dtype=np.int32)
    for i in range(codes.shape[1]):
        # Use numpy's binary representation for popcount
        distances += np.vectorize(lambda x: bin(x).count('1'))(xored[:, i])
    return distances


def benchmark_encode(adapter: BinaryAdapter, n_samples: int = 1000, 
                     embedding_dim: int = 384) -> Dict[str, Any]:
    """Benchmark encoding latency."""
    embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    
    # Warm up
    _ = adapter.encode(embeddings[0])
    
    # Benchmark batch encoding
    start = time.perf_counter()
    for emb in embeddings:
        _ = adapter.encode(emb)
    elapsed = time.perf_counter() - start
    
    per_sample_ms = (elapsed / n_samples) * 1000
    
    return {
        "n_samples": n_samples,
        "total_seconds": elapsed,
        "per_sample_ms": per_sample_ms,
        "target_ms": 1.0,
        "pass": per_sample_ms < 5.0  # Kill trigger threshold
    }


def benchmark_lookup(code_bits: int = 256, n_entries: int = 100_000, 
                     n_queries: int = 100) -> Dict[str, Any]:
    """Benchmark lookup latency with brute-force Hamming search."""
    n_words = (code_bits + 63) // 64
    
    # Generate random packed codes
    codes = np.random.randint(0, 2**63, size=(n_entries, n_words), dtype=np.uint64)
    queries = np.random.randint(0, 2**63, size=(n_queries, n_words), dtype=np.uint64)
    
    # Warm up
    _ = hamming_distance_batch(queries[0:1], codes)
    
    # Benchmark
    start = time.perf_counter()
    for q in queries:
        _ = hamming_distance_batch(q.reshape(1, -1), codes)
    elapsed = time.perf_counter() - start
    
    per_query_us = (elapsed / n_queries) * 1_000_000
    
    return {
        "n_entries": n_entries,
        "n_queries": n_queries,
        "total_seconds": elapsed,
        "per_query_us": per_query_us,
        "target_us": 100.0,  # Target: <100µs
        "pass": per_query_us < 1000.0  # Kill trigger: <1ms
    }


def benchmark_memory(code_bits: int = 256, n_entries: int = 100_000) -> Dict[str, Any]:
    """Benchmark memory usage for code storage."""
    n_words = (code_bits + 63) // 64
    bytes_per_entry = n_words * 8  # uint64 = 8 bytes
    
    tracemalloc.start()
    
    # Allocate codes
    codes = np.zeros((n_entries, n_words), dtype=np.uint64)
    # Fill with random data to ensure allocation
    codes[:] = np.random.randint(0, 2**63, size=(n_entries, n_words), dtype=np.uint64)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    theoretical_mb = (n_entries * bytes_per_entry) / (1024 * 1024)
    actual_mb = current / (1024 * 1024)
    
    return {
        "n_entries": n_entries,
        "code_bits": code_bits,
        "bytes_per_entry": bytes_per_entry,
        "theoretical_mb": theoretical_mb,
        "actual_mb": actual_mb,
        "peak_mb": peak / (1024 * 1024),
        "target_mb": 4.0,  # Target: <4MB
        "pass": actual_mb < 10.0  # Kill trigger: <10MB
    }


def run_all_benchmarks() -> Dict[str, Any]:
    """Run all Stage 1 benchmarks."""
    results = {
        "stage": "S1",
        "name": "Latency Validation",
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "benchmarks": {},
        "kill_triggers": [],
        "overall_status": "PASS"
    }
    
    print("\n" + "="*60)
    print("STAGE 1: Latency Validation Benchmarks")
    print("="*60)
    
    # Benchmark 1: Encode latency
    print("\n[1/3] Benchmarking encode latency...")
    adapter = BinaryAdapter(embedding_dim=384, code_bits=256, seed=42)
    encode_results = benchmark_encode(adapter)
    results["benchmarks"]["encode"] = encode_results
    
    status = "✓ PASS" if encode_results["pass"] else "✗ FAIL"
    print(f"  Encode: {encode_results['per_sample_ms']:.3f} ms/sample (target: <1ms) [{status}]")
    
    if not encode_results["pass"]:
        results["kill_triggers"].append("encode_latency_exceeded")
        results["overall_status"] = "FAIL"
    
    # Benchmark 2: Lookup latency
    print("\n[2/3] Benchmarking lookup latency (100K entries)...")
    lookup_results = benchmark_lookup(n_entries=100_000, n_queries=100)
    results["benchmarks"]["lookup"] = lookup_results
    
    status = "✓ PASS" if lookup_results["pass"] else "✗ FAIL"
    print(f"  Lookup: {lookup_results['per_query_us']:.1f} µs/query (target: <100µs) [{status}]")
    
    if not lookup_results["pass"]:
        results["kill_triggers"].append("lookup_latency_exceeded")
        results["overall_status"] = "FAIL"
    
    # Benchmark 3: Memory usage
    print("\n[3/3] Benchmarking memory usage (100K entries)...")
    memory_results = benchmark_memory(n_entries=100_000)
    results["benchmarks"]["memory"] = memory_results
    
    status = "✓ PASS" if memory_results["pass"] else "✗ FAIL"
    print(f"  Memory: {memory_results['actual_mb']:.2f} MB (target: <4MB, theoretical: {memory_results['theoretical_mb']:.2f} MB) [{status}]")
    
    if not memory_results["pass"]:
        results["kill_triggers"].append("memory_exceeded")
        results["overall_status"] = "FAIL"
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"  Encode:  {encode_results['per_sample_ms']:.3f} ms/sample")
    print(f"  Lookup:  {lookup_results['per_query_us']:.1f} µs/query")
    print(f"  Memory:  {memory_results['actual_mb']:.2f} MB")
    print("="*60)
    print(f"OVERALL STATUS: {results['overall_status']}")
    if results["kill_triggers"]:
        print(f"KILL TRIGGERS ACTIVATED: {results['kill_triggers']}")
    print("="*60)
    
    return results


def main():
    """Main entry point."""
    results = run_all_benchmarks()
    
    # Save results
    results_path = Path(__file__).parent.parent / "results" / "s1_latency_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Exit with appropriate code
    sys.exit(0 if results["overall_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

