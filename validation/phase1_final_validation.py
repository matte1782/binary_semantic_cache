#!/usr/bin/env python3
"""
Phase 1 Final Scientific Validation

This script validates that all Phase 1 fixes:
1. Introduce no regressions
2. Maintain correctness guarantees
3. Preserve performance targets

Run this after all engineer fixes are applied.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from binary_semantic_cache.core.encoder import BinaryEncoder
from binary_semantic_cache.core.cache import BinarySemanticCache
from binary_semantic_cache.core.similarity import hamming_similarity

print("=" * 70)
print("PHASE 1 FINAL SCIENTIFIC VALIDATION")
print("=" * 70)
print(f"Timestamp: {datetime.now().isoformat()}")
print()

results: Dict[str, Any] = {
    "timestamp": datetime.now().isoformat(),
    "tests": {},
    "metrics": {},
    "verdict": None,
}

# =============================================================================
# TEST 1: Similarity Correlation (MUST be >= 0.95)
# =============================================================================
print("-" * 70)
print("TEST 1: Similarity Correlation")
print("-" * 70)

def create_similar_embedding(base: np.ndarray, target_cosine: float, seed: int) -> np.ndarray:
    """Create embedding with target cosine similarity to base."""
    rng = np.random.RandomState(seed)
    dim = len(base)
    
    base_norm = base / np.linalg.norm(base)
    random_vec = rng.randn(dim).astype(np.float32)
    random_vec = random_vec - np.dot(random_vec, base_norm) * base_norm
    random_vec = random_vec / np.linalg.norm(random_vec)
    
    theta = np.arccos(np.clip(target_cosine, -1, 1))
    result = np.cos(theta) * base_norm + np.sin(theta) * random_vec
    return result.astype(np.float32)

encoder = BinaryEncoder(embedding_dim=384, code_bits=256, seed=42)
base = np.random.randn(384).astype(np.float32)
base = base / np.linalg.norm(base)

# Test multiple similarity levels
test_cosines = [0.99, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]
cosine_hamming_pairs = []

print(f"  {'Target':>8} | {'Actual':>8} | {'Hamming':>8} | {'Error':>8}")
print(f"  {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")

for i, target_cos in enumerate(test_cosines):
    similar = create_similar_embedding(base, target_cos, seed=100+i)
    actual_cos = float(np.dot(base, similar))
    
    base_code = encoder.encode(base)
    similar_code = encoder.encode(similar)
    
    hamming_sim = hamming_similarity(base_code, similar_code.reshape(1, -1), 256)[0]
    
    error = abs(actual_cos - hamming_sim)
    cosine_hamming_pairs.append((actual_cos, hamming_sim))
    
    print(f"  {target_cos:>8.2f} | {actual_cos:>8.4f} | {hamming_sim:>8.4f} | {error:>8.4f}")

# Compute correlation
cosines = np.array([p[0] for p in cosine_hamming_pairs])
hammings = np.array([p[1] for p in cosine_hamming_pairs])
correlation = np.corrcoef(cosines, hammings)[0, 1]

print(f"\n  Correlation (Pearson r): {correlation:.4f}")
print(f"  Target: ≥ 0.95")
print(f"  Status: {'✓ PASS' if correlation >= 0.95 else '✗ FAIL'}")

results["tests"]["correlation"] = {
    "value": float(correlation),
    "target": 0.95,
    "passed": correlation >= 0.95,
}
results["metrics"]["correlation"] = float(correlation)

# =============================================================================
# TEST 2: Threshold Behavior at Multiple Values
# =============================================================================
print()
print("-" * 70)
print("TEST 2: Threshold Behavior (0.70, 0.80, 0.85)")
print("-" * 70)

# Create test cache with various thresholds
threshold_results = {}

for threshold in [0.70, 0.80, 0.85]:
    cache = BinarySemanticCache(
        encoder=encoder,
        max_entries=100,
        similarity_threshold=threshold,
    )
    
    # Insert base embedding
    base_emb = np.random.randn(384).astype(np.float32)
    base_emb = base_emb / np.linalg.norm(base_emb)
    cache.put(base_emb, {"type": "base"})
    
    # Test queries at various similarities
    test_cases = []
    for target_sim in [0.95, 0.85, 0.80, 0.75, 0.65]:
        query = create_similar_embedding(base_emb, target_sim, seed=int(target_sim * 100))
        result = cache.get(query)
        is_hit = result is not None
        
        expected_hit = target_sim >= threshold - 0.05  # Allow 5% quantization error
        test_cases.append({
            "target_sim": target_sim,
            "is_hit": is_hit,
            "expected": expected_hit,
            "correct": is_hit == expected_hit or (target_sim > threshold - 0.10 and target_sim < threshold + 0.05),
        })
    
    hits = sum(1 for t in test_cases if t["is_hit"])
    threshold_results[threshold] = {
        "test_cases": test_cases,
        "hits": hits,
        "total": len(test_cases),
    }
    
    print(f"\n  Threshold: {threshold}")
    for tc in test_cases:
        status = "HIT" if tc["is_hit"] else "MISS"
        print(f"    cosine={tc['target_sim']:.2f} → {status}")

results["tests"]["threshold_behavior"] = threshold_results

# =============================================================================
# TEST 3: Shape Handling (1D vs 2D)
# =============================================================================
print()
print("-" * 70)
print("TEST 3: Shape Handling")
print("-" * 70)

shape_tests = []

# Test 1: Single 1D embedding
try:
    single = np.random.randn(384).astype(np.float32)
    code = encoder.encode(single)
    assert code.ndim == 1, f"Expected 1D output, got {code.ndim}D"
    assert code.shape == (4,), f"Expected shape (4,), got {code.shape}"
    shape_tests.append(("1D input → 1D output", True, "OK"))
    print("  ✓ 1D input → 1D output (shape (4,))")
except Exception as e:
    shape_tests.append(("1D input → 1D output", False, str(e)))
    print(f"  ✗ 1D input → 1D output: {e}")

# Test 2: Batch 2D embeddings
try:
    batch = np.random.randn(10, 384).astype(np.float32)
    codes = encoder.encode(batch)
    assert codes.ndim == 2, f"Expected 2D output, got {codes.ndim}D"
    assert codes.shape == (10, 4), f"Expected shape (10, 4), got {codes.shape}"
    shape_tests.append(("2D input → 2D output", True, "OK"))
    print("  ✓ 2D input → 2D output (shape (10, 4))")
except Exception as e:
    shape_tests.append(("2D input → 2D output", False, str(e)))
    print(f"  ✗ 2D input → 2D output: {e}")

# Test 3: Single-item batch
try:
    single_batch = np.random.randn(1, 384).astype(np.float32)
    code = encoder.encode(single_batch)
    assert code.ndim == 2, f"Expected 2D output for (1, 384) input, got {code.ndim}D"
    assert code.shape == (1, 4), f"Expected shape (1, 4), got {code.shape}"
    shape_tests.append(("(1, D) input → (1, W) output", True, "OK"))
    print("  ✓ (1, D) input → (1, W) output (shape (1, 4))")
except Exception as e:
    shape_tests.append(("(1, D) input → (1, W) output", False, str(e)))
    print(f"  ✗ (1, D) input → (1, W) output: {e}")

# Test 4: Edge case - 2-entry batch
try:
    two_batch = np.random.randn(2, 384).astype(np.float32)
    codes = encoder.encode(two_batch)
    assert codes.shape == (2, 4)
    shape_tests.append(("2-entry batch", True, "OK"))
    print("  ✓ 2-entry batch → (2, 4)")
except Exception as e:
    shape_tests.append(("2-entry batch", False, str(e)))
    print(f"  ✗ 2-entry batch: {e}")

# Test 5: Invalid shape should raise
try:
    invalid = np.random.randn(2, 3, 384).astype(np.float32)
    encoder.encode(invalid)
    shape_tests.append(("3D input rejection", False, "No error raised"))
    print("  ✗ 3D input should raise error")
except ValueError:
    shape_tests.append(("3D input rejection", True, "OK"))
    print("  ✓ 3D input correctly rejected")

all_shape_passed = all(t[1] for t in shape_tests)
results["tests"]["shape_handling"] = {
    "tests": shape_tests,
    "passed": all_shape_passed,
}

# =============================================================================
# TEST 4: Lookup Latency (< 2.5ms)
# =============================================================================
print()
print("-" * 70)
print("TEST 4: Lookup Latency (target < 2.5ms @ 100K entries)")
print("-" * 70)

# Create large cache
large_cache = BinarySemanticCache(
    encoder=encoder,
    max_entries=100_000,
    similarity_threshold=0.80,
)

# Insert 100K entries
print("  Creating 100K entry cache...")
embeddings = np.random.randn(100_000, 384).astype(np.float32)
for i in range(100_000):
    large_cache.put(embeddings[i], {"id": i})
    if (i + 1) % 25000 == 0:
        print(f"    Inserted {i+1:,} entries...")

# Measure lookup time
print("  Measuring lookup latency (100 queries)...")
query_times = []
for _ in range(100):
    query = np.random.randn(384).astype(np.float32)
    start = time.perf_counter()
    large_cache.get(query)
    elapsed = (time.perf_counter() - start) * 1000
    query_times.append(elapsed)

mean_latency = np.mean(query_times)
p95_latency = np.percentile(query_times, 95)
p99_latency = np.percentile(query_times, 99)

print(f"\n  Mean: {mean_latency:.3f} ms")
print(f"  P95:  {p95_latency:.3f} ms")
print(f"  P99:  {p99_latency:.3f} ms")
print(f"  Target: < 2.5 ms (mean)")
print(f"  Status: {'✓ PASS' if mean_latency < 2.5 else '✗ FAIL'}")

results["tests"]["latency"] = {
    "mean_ms": float(mean_latency),
    "p95_ms": float(p95_latency),
    "p99_ms": float(p99_latency),
    "target_ms": 2.5,
    "passed": mean_latency < 2.5,
}
results["metrics"]["lookup_latency_ms"] = float(mean_latency)

# =============================================================================
# TEST 5: Cache Logic Correctness
# =============================================================================
print()
print("-" * 70)
print("TEST 5: Cache Logic Correctness")
print("-" * 70)

logic_tests = []

# Test exact match
cache = BinarySemanticCache(encoder=encoder, max_entries=100, similarity_threshold=0.80)
exact_emb = np.random.randn(384).astype(np.float32)
cache.put(exact_emb, {"type": "exact"})
result = cache.get(exact_emb)
passed = result is not None and result.response["type"] == "exact"
logic_tests.append(("Exact match", passed))
print(f"  {'✓' if passed else '✗'} Exact match returns correct entry")

# Test high similarity hit
similar_emb = create_similar_embedding(exact_emb, 0.95, seed=999)
result = cache.get(similar_emb)
passed = result is not None
logic_tests.append(("High similarity hit", passed))
print(f"  {'✓' if passed else '✗'} High similarity (0.95 cosine) returns hit")

# Test low similarity miss
different_emb = np.random.randn(384).astype(np.float32)
result = cache.get(different_emb)
passed = result is None
logic_tests.append(("Low similarity miss", passed))
print(f"  {'✓' if passed else '✗'} Random embedding returns miss")

# Test stats tracking
stats = cache.stats()
passed = stats.hits >= 2 and stats.misses >= 1
logic_tests.append(("Stats tracking", passed))
print(f"  {'✓' if passed else '✗'} Stats tracking (hits={stats.hits}, misses={stats.misses})")

all_logic_passed = all(t[1] for t in logic_tests)
results["tests"]["cache_logic"] = {
    "tests": logic_tests,
    "passed": all_logic_passed,
}

# =============================================================================
# FINAL VERDICT
# =============================================================================
print()
print("=" * 70)
print("FINAL VERDICT")
print("=" * 70)

all_passed = all([
    results["tests"]["correlation"]["passed"],
    all_shape_passed,
    results["tests"]["latency"]["passed"],
    all_logic_passed,
])

if all_passed:
    verdict = "PASS"
    print("✓ ALL TESTS PASSED")
else:
    if results["tests"]["correlation"]["passed"] and results["tests"]["latency"]["passed"]:
        verdict = "PASS-WITH-LIMITATIONS"
        print("⚠ PASS WITH LIMITATIONS")
    else:
        verdict = "FAIL"
        print("✗ FAIL")

results["verdict"] = verdict

print()
print("Summary:")
print(f"  - Correlation:    {results['metrics']['correlation']:.4f} ≥ 0.95 {'✓' if results['tests']['correlation']['passed'] else '✗'}")
print(f"  - Shape handling: {'✓' if all_shape_passed else '✗'}")
print(f"  - Lookup latency: {results['metrics']['lookup_latency_ms']:.3f}ms < 2.5ms {'✓' if results['tests']['latency']['passed'] else '✗'}")
print(f"  - Cache logic:    {'✓' if all_logic_passed else '✗'}")

# Save results
output_path = project_root / "validation" / "results" / "phase1_final_validation.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to: {output_path}")

sys.exit(0 if verdict == "PASS" else 1)

