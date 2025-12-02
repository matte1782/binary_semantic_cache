#!/usr/bin/env python3
"""
Sprint 3 Benchmark: OpenAI Backend Performance

Measures:
- Mode A (Mock): Python-side overhead of batching logic and cost calculation
- Mode B (Real): Actual latency against OpenAI API (requires OPENAI_API_KEY)

Usage:
    # Mock mode only (default, no API key required)
    python benchmarks/openai_bench.py

    # Include real API tests (requires OPENAI_API_KEY)
    python benchmarks/openai_bench.py --real

    # Quick mode (fewer iterations)
    python benchmarks/openai_bench.py --quick

WARNING: Real mode incurs actual API costs!
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "openai_bench.json"

# Benchmark configuration
DEFAULT_BATCH_SIZES = [1, 10, 50, 100, 150, 200]
DEFAULT_TRIALS = 10
QUICK_TRIALS = 3

# Targets
# NOTE: 50ms target is realistic for Python-side processing of batches
# (list slicing, numpy array creation, mock response generation).
# Real network latency (200ms+) dominates in production.
# The mock generates 1536-dim float arrays which is expensive.
TARGET_OVERHEAD_PER_CALL_MS = 50.0  # Python overhead should be < 50ms per call
TARGET_RATE_LIMITER_OVERHEAD_MS = 0.1  # Rate limiter should add < 0.1ms


# =============================================================================
# MOCK HELPERS
# =============================================================================

def create_mock_embedding_response(
    n: int,
    dim: int = 1536,
    seed: int = 42,
) -> MagicMock:
    """Create a mock OpenAI embedding response."""
    rng = np.random.default_rng(seed)
    embeddings = rng.random((n, dim)).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(index=i, embedding=embeddings[i].tolist())
        for i in range(n)
    ]
    mock_response.usage = MagicMock(total_tokens=n * 10)  # ~10 tokens per text
    return mock_response


def setup_mock_openai():
    """Setup mock OpenAI module and client."""
    mock_openai = MagicMock()
    
    # Define exception classes
    mock_openai.RateLimitError = type("RateLimitError", (Exception,), {})
    mock_openai.APIConnectionError = type("APIConnectionError", (Exception,), {})
    mock_openai.InternalServerError = type("InternalServerError", (Exception,), {})
    mock_openai.APITimeoutError = type("APITimeoutError", (Exception,), {})
    mock_openai.AuthenticationError = type("AuthenticationError", (Exception,), {})
    mock_openai.BadRequestError = type("BadRequestError", (Exception,), {})
    
    # Create mock client
    mock_client = MagicMock()
    
    def create_embeddings(model: str, input: List[str]):
        return create_mock_embedding_response(len(input))
    
    mock_client.embeddings.create = MagicMock(side_effect=create_embeddings)
    mock_openai.OpenAI.return_value = mock_client
    
    return mock_openai, mock_client


def setup_mock_tenacity():
    """Setup mock tenacity module."""
    mock_tenacity = MagicMock()
    
    def mock_retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    mock_tenacity.retry = mock_retry
    mock_tenacity.stop_after_attempt = MagicMock()
    mock_tenacity.wait_exponential_jitter = MagicMock()
    mock_tenacity.retry_if_exception_type = MagicMock()
    
    return mock_tenacity


def setup_mock_tiktoken():
    """Setup mock tiktoken module."""
    mock_tiktoken = MagicMock()
    mock_encoding = MagicMock()
    mock_encoding.encode.return_value = list(range(10))  # 10 tokens per text
    mock_tiktoken.get_encoding.return_value = mock_encoding
    return mock_tiktoken


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def benchmark_batching_overhead(
    batch_sizes: List[int],
    trials: int,
) -> Dict[str, Any]:
    """
    Mode A: Measure Python-side overhead of batching logic.
    
    This measures the time spent in Python code (chunking, result aggregation)
    without any network latency.
    """
    print("\n" + "=" * 70)
    print("MODE A: BATCHING LOGIC OVERHEAD (Mock)")
    print("=" * 70)
    print(f"\nMeasuring Python-side overhead without network latency...")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Trials per size: {trials}")
    
    results = {}
    
    # Setup mocks
    mock_openai, mock_client = setup_mock_openai()
    mock_tenacity = setup_mock_tenacity()
    mock_tiktoken = setup_mock_tiktoken()
    
    with patch.dict("sys.modules", {
        "openai": mock_openai,
        "tenacity": mock_tenacity,
        "tiktoken": mock_tiktoken,
    }):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-mock-key"}):
            # Import after mocking
            from binary_semantic_cache.embeddings.openai_backend import OpenAIEmbeddingBackend
            
            backend = OpenAIEmbeddingBackend()
            
            for batch_size in batch_sizes:
                texts = [f"Sample text {i} for embedding" for i in range(batch_size)]
                times = []
                
                # Warmup
                backend.embed_texts(texts[:min(10, batch_size)])
                
                # Benchmark
                for _ in range(trials):
                    mock_client.embeddings.create.reset_mock()
                    start = time.perf_counter()
                    result = backend.embed_texts(texts)
                    elapsed = (time.perf_counter() - start) * 1000  # ms
                    times.append(elapsed)
                
                mean_ms = np.mean(times)
                std_ms = np.std(times)
                min_ms = np.min(times)
                max_ms = np.max(times)
                
                # Count API calls (should be ceil(batch_size / 100))
                expected_calls = (batch_size + 99) // 100
                actual_calls = mock_client.embeddings.create.call_count // trials
                
                results[batch_size] = {
                    "mean_ms": mean_ms,
                    "std_ms": std_ms,
                    "min_ms": min_ms,
                    "max_ms": max_ms,
                    "overhead_per_call_ms": mean_ms / expected_calls if expected_calls > 0 else 0,
                    "expected_api_calls": expected_calls,
                    "actual_api_calls_per_trial": actual_calls,
                }
                
                status = "✓ PASS" if mean_ms / expected_calls < TARGET_OVERHEAD_PER_CALL_MS else "✗ FAIL"
                print(f"\n  {batch_size:6d} texts: mean={mean_ms:.3f}ms (std={std_ms:.3f}), "
                      f"calls={expected_calls}, overhead/call={mean_ms/expected_calls:.3f}ms {status}")
    
    return results


def benchmark_rate_limiter_overhead(trials: int) -> Dict[str, Any]:
    """
    Measure rate limiter overhead (token bucket acquire time).
    """
    print("\n" + "=" * 70)
    print("MODE A: RATE LIMITER OVERHEAD (Mock)")
    print("=" * 70)
    print(f"\nMeasuring Token Bucket acquire() overhead...")
    print(f"Trials: {trials * 100}")
    
    # Import rate limiter directly
    mock_openai, _ = setup_mock_openai()
    mock_tenacity = setup_mock_tenacity()
    mock_tiktoken = setup_mock_tiktoken()
    
    with patch.dict("sys.modules", {
        "openai": mock_openai,
        "tenacity": mock_tenacity,
        "tiktoken": mock_tiktoken,
    }):
        from binary_semantic_cache.embeddings.openai_backend import RateLimiter
        
        # High RPM to avoid blocking
        limiter = RateLimiter(rpm_limit=100000)
        
        times = []
        for _ in range(trials * 100):
            limiter.reset()  # Ensure tokens available
            start = time.perf_counter()
            limiter.acquire(timeout=0.001)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)
        
        mean_ms = np.mean(times)
        std_ms = np.std(times)
        p99_ms = np.percentile(times, 99)
        
        status = "✓ PASS" if mean_ms < TARGET_RATE_LIMITER_OVERHEAD_MS else "✗ FAIL"
        print(f"\n  acquire() latency: mean={mean_ms:.4f}ms, p99={p99_ms:.4f}ms {status}")
        
        return {
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "p99_ms": p99_ms,
            "target_ms": TARGET_RATE_LIMITER_OVERHEAD_MS,
            "pass": mean_ms < TARGET_RATE_LIMITER_OVERHEAD_MS,
        }


def benchmark_cost_calculation_overhead(trials: int) -> Dict[str, Any]:
    """
    Measure cost tracking calculation overhead.
    """
    print("\n" + "=" * 70)
    print("MODE A: COST CALCULATION OVERHEAD (Mock)")
    print("=" * 70)
    print(f"\nMeasuring cost tracking overhead...")
    
    mock_openai, mock_client = setup_mock_openai()
    mock_tenacity = setup_mock_tenacity()
    mock_tiktoken = setup_mock_tiktoken()
    
    with patch.dict("sys.modules", {
        "openai": mock_openai,
        "tenacity": mock_tenacity,
        "tiktoken": mock_tiktoken,
    }):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-mock-key"}):
            from binary_semantic_cache.embeddings.openai_backend import OpenAIEmbeddingBackend
            
            backend = OpenAIEmbeddingBackend()
            
            # Warmup
            backend.embed_text("warmup")
            
            # Measure get_stats() overhead
            times = []
            for _ in range(trials * 100):
                start = time.perf_counter()
                stats = backend.get_stats()
                elapsed = (time.perf_counter() - start) * 1000  # ms
                times.append(elapsed)
            
            mean_ms = np.mean(times)
            p99_ms = np.percentile(times, 99)
            
            print(f"\n  get_stats() latency: mean={mean_ms:.4f}ms, p99={p99_ms:.4f}ms")
            print(f"  Stats: {stats}")
            
            return {
                "get_stats_mean_ms": mean_ms,
                "get_stats_p99_ms": p99_ms,
                "sample_stats": stats,
            }


def benchmark_real_api(
    batch_sizes: List[int],
    trials: int,
) -> Optional[Dict[str, Any]]:
    """
    Mode B: Measure actual latency against OpenAI API.
    
    WARNING: This incurs real API costs!
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n" + "=" * 70)
        print("MODE B: REAL API BENCHMARK (Skipped)")
        print("=" * 70)
        print("\n  ⚠ OPENAI_API_KEY not set. Skipping real API benchmark.")
        print("  To run: export OPENAI_API_KEY=sk-... && python benchmarks/openai_bench.py --real")
        return None
    
    if not api_key.startswith("sk-"):
        print("\n  ⚠ OPENAI_API_KEY appears invalid (should start with 'sk-'). Skipping.")
        return None
    
    print("\n" + "=" * 70)
    print("MODE B: REAL API BENCHMARK")
    print("=" * 70)
    print("\n  ⚠ WARNING: This will incur real API costs!")
    print(f"  API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Trials per size: {trials}")
    
    try:
        from binary_semantic_cache.embeddings.openai_backend import OpenAIEmbeddingBackend
        
        backend = OpenAIEmbeddingBackend()
        results = {}
        
        for batch_size in batch_sizes:
            texts = [f"Sample text {i} for embedding benchmark" for i in range(batch_size)]
            times = []
            
            print(f"\n  Benchmarking {batch_size} texts...")
            
            for trial in range(trials):
                start = time.perf_counter()
                result = backend.embed_texts(texts)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                times.append(elapsed)
                print(f"    Trial {trial + 1}: {elapsed:.1f}ms")
            
            mean_ms = np.mean(times)
            std_ms = np.std(times)
            
            results[batch_size] = {
                "mean_ms": mean_ms,
                "std_ms": std_ms,
                "min_ms": np.min(times),
                "max_ms": np.max(times),
                "latency_per_text_ms": mean_ms / batch_size,
            }
            
            print(f"    Mean: {mean_ms:.1f}ms ({mean_ms/batch_size:.2f}ms/text)")
        
        # Get final stats
        final_stats = backend.get_stats()
        results["final_stats"] = final_stats
        print(f"\n  Final stats: {final_stats}")
        print(f"  Estimated cost: ${final_stats.get('cost_usd', 0):.6f}")
        
        return results
        
    except Exception as e:
        print(f"\n  ✗ Error during real API benchmark: {e}")
        return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sprint 3 Benchmark: OpenAI Backend Performance"
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Include real API benchmark (requires OPENAI_API_KEY, incurs costs!)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with fewer trials"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Custom batch sizes to test"
    )
    args = parser.parse_args()
    
    batch_sizes = args.batch_sizes or DEFAULT_BATCH_SIZES
    trials = QUICK_TRIALS if args.quick else DEFAULT_TRIALS
    
    print("=" * 70)
    print("SPRINT 3 BENCHMARK: OpenAI Backend Performance")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Real API: {'Yes' if args.real else 'No (mock only)'}")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "batch_sizes": batch_sizes,
            "trials": trials,
            "real_api": args.real,
        },
        "mode_a": {},
        "mode_b": None,
    }
    
    # Mode A: Mock benchmarks
    results["mode_a"]["batching"] = benchmark_batching_overhead(batch_sizes, trials)
    results["mode_a"]["rate_limiter"] = benchmark_rate_limiter_overhead(trials)
    results["mode_a"]["cost_tracking"] = benchmark_cost_calculation_overhead(trials)
    
    # Mode B: Real API benchmark (optional)
    if args.real:
        # Use smaller batch sizes for real API to minimize costs
        real_batch_sizes = [1, 10, 50] if not args.batch_sizes else batch_sizes[:3]
        results["mode_b"] = benchmark_real_api(real_batch_sizes, min(trials, 3))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    batching = results["mode_a"]["batching"]
    rate_limiter = results["mode_a"]["rate_limiter"]
    
    print("\n  Mode A (Mock) Results:")
    print(f"    Batching overhead (100 texts): {batching.get(100, {}).get('mean_ms', 'N/A'):.3f}ms")
    print(f"    Rate limiter overhead: {rate_limiter['mean_ms']:.4f}ms")
    print(f"    Target: < {TARGET_OVERHEAD_PER_CALL_MS}ms per API call")
    
    # Check targets
    all_pass = True
    for size, data in batching.items():
        if isinstance(size, int):
            overhead_per_call = data.get("overhead_per_call_ms", 0)
            if overhead_per_call >= TARGET_OVERHEAD_PER_CALL_MS:
                all_pass = False
    
    if not rate_limiter["pass"]:
        all_pass = False
    
    status = "✓ ALL PASS" if all_pass else "✗ SOME FAILED"
    print(f"\n  Overall: {status}")
    
    if results["mode_b"]:
        print("\n  Mode B (Real API) Results:")
        for size, data in results["mode_b"].items():
            if isinstance(size, int):
                print(f"    {size} texts: {data['mean_ms']:.1f}ms ({data['latency_per_text_ms']:.2f}ms/text)")
        if "final_stats" in results["mode_b"]:
            print(f"    Total cost: ${results['mode_b']['final_stats'].get('cost_usd', 0):.6f}")
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {RESULTS_FILE}")
    
    print("\n" + "=" * 70)
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

