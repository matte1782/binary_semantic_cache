# STAGE 3 PROMPT: Minimal Proof-of-Concept

**Agent Role:** Engineer  
**Duration:** 2-3 hours  
**Blocking:** YES — Core loop must work before investing 200+ hours

---

## CONTEXT

You have validated imports (S0) and measured latency (S1). Now you must prove the **core cache loop** works end-to-end:

```
Embedding → Binary Code → Insert → Lookup → Match → Return
```

This is NOT production code. It's a 100-line script that proves the concept.

---

## YOUR TASK

### Step 1: Implement Minimal PoC

Write `validation/poc/semantic_cache_poc.py` that demonstrates:

1. Convert embedding to binary code
2. Store code with response
3. Query with similar embedding
4. Find the match via Hamming distance
5. Return cached response

### Step 2: Validate Correctness

- Insert 100 entries
- Query with exact same embeddings → expect 100% hit rate
- Query with similar embeddings (small noise) → expect high hit rate
- Query with random embeddings → expect low hit rate

### Step 3: Validate Correlation

- Compute cosine similarity between embeddings
- Compute Hamming distance between codes
- Verify Spearman correlation > 0.85

---

## POC SCRIPT TEMPLATE

```python
# validation/poc/semantic_cache_poc.py
"""
Stage 3: Minimal Proof-of-Concept for Binary Semantic Cache

This script demonstrates the core loop:
1. Embedding → Binary code
2. Insert code + response
3. Query with embedding
4. Find nearest via Hamming
5. Return cached response

NOT PRODUCTION CODE. Just proof that the concept works.
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np
from scipy.stats import spearmanr

# Add BinaryLLM to path
BINARYLLM_PATH = Path(__file__).parent.parent.parent.parent / "binary_llm"
sys.path.insert(0, str(BINARYLLM_PATH))

from src.quantization.binarization import RandomProjection, binarize_sign
from src.quantization.packing import pack_codes, unpack_codes

# Configuration
EMBEDDING_DIM = 384
CODE_BITS = 256
SEED = 42
SIMILARITY_THRESHOLD = 0.85  # 1 - (hamming_distance / code_bits)

results = {
    "stage": "S3",
    "name": "Minimal Proof-of-Concept",
    "timestamp": datetime.now().isoformat(),
    "tests": [],
    "overall_status": "PENDING",
}

@dataclass
class MinimalCache:
    """Minimal cache for PoC - NOT production code."""
    
    embedding_dim: int = EMBEDDING_DIM
    code_bits: int = CODE_BITS
    seed: int = SEED
    threshold: float = SIMILARITY_THRESHOLD
    
    # Storage
    codes: np.ndarray = field(default=None, repr=False)
    responses: Dict[int, bytes] = field(default_factory=dict)
    n_entries: int = 0
    
    # Projection
    _projection: RandomProjection = field(default=None, repr=False)
    
    def __post_init__(self):
        self._projection = RandomProjection(
            input_dim=self.embedding_dim,
            output_bits=self.code_bits,
            seed=self.seed
        )
        self.codes = np.zeros((0, self.code_bits // 64), dtype=np.uint64)
    
    def _encode(self, embedding: np.ndarray) -> np.ndarray:
        """Convert embedding to packed binary code."""
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        projected = self._projection.project(embedding.astype(np.float32))
        codes_pm1 = binarize_sign(projected)
        codes_01 = ((codes_pm1 + 1.0) / 2.0).astype(np.int8)
        return pack_codes(codes_01)
    
    def _hamming_distance(self, query: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Compute Hamming distance from query to all codes."""
        xor = np.bitwise_xor(codes, query)
        # Vectorized popcount (slow but correct)
        distances = np.zeros(len(codes), dtype=np.int32)
        for i, row in enumerate(xor):
            distances[i] = sum(bin(w).count('1') for w in row)
        return distances
    
    def insert(self, embedding: np.ndarray, response: bytes) -> int:
        """Insert entry, return entry ID."""
        code = self._encode(embedding)
        entry_id = self.n_entries
        
        if self.n_entries == 0:
            self.codes = code
        else:
            self.codes = np.vstack([self.codes, code])
        
        self.responses[entry_id] = response
        self.n_entries += 1
        return entry_id
    
    def lookup(self, embedding: np.ndarray) -> Tuple[Optional[int], Optional[bytes], float]:
        """
        Lookup nearest entry above threshold.
        
        Returns: (entry_id, response, similarity) or (None, None, 0.0)
        """
        if self.n_entries == 0:
            return None, None, 0.0
        
        query = self._encode(embedding).squeeze()
        distances = self._hamming_distance(query, self.codes)
        
        # Convert to similarity
        similarities = 1.0 - (distances / self.code_bits)
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity >= self.threshold:
            return best_idx, self.responses[best_idx], float(best_similarity)
        else:
            return None, None, float(best_similarity)

def test_exact_match():
    """Test: Query with exact same embedding should always hit."""
    print("\n[Test 1] Exact match test...")
    
    cache = MinimalCache()
    embeddings = np.random.randn(100, EMBEDDING_DIM).astype(np.float32)
    
    # Insert all
    for i, emb in enumerate(embeddings):
        cache.insert(emb, f"response_{i}".encode())
    
    # Query with exact same embeddings
    hits = 0
    for i, emb in enumerate(embeddings):
        entry_id, response, similarity = cache.lookup(emb)
        if entry_id == i:  # Should match the exact entry
            hits += 1
    
    hit_rate = hits / len(embeddings)
    passed = hit_rate == 1.0
    
    results["tests"].append({
        "name": "exact_match",
        "hit_rate": hit_rate,
        "expected": 1.0,
        "passed": passed
    })
    
    print(f"  Hit rate: {hit_rate:.2%} (expected: 100%)")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed

def test_similar_match():
    """Test: Query with small noise should mostly hit."""
    print("\n[Test 2] Similar match test (with noise)...")
    
    cache = MinimalCache()
    embeddings = np.random.randn(100, EMBEDDING_DIM).astype(np.float32)
    
    # Insert all
    for i, emb in enumerate(embeddings):
        cache.insert(emb, f"response_{i}".encode())
    
    # Query with small noise added
    noise_level = 0.1
    hits = 0
    for i, emb in enumerate(embeddings):
        noisy_emb = emb + np.random.randn(EMBEDDING_DIM).astype(np.float32) * noise_level
        entry_id, response, similarity = cache.lookup(noisy_emb)
        if entry_id is not None:
            hits += 1
    
    hit_rate = hits / len(embeddings)
    passed = hit_rate >= 0.7  # Expect at least 70% hit rate with small noise
    
    results["tests"].append({
        "name": "similar_match",
        "noise_level": noise_level,
        "hit_rate": hit_rate,
        "threshold": 0.7,
        "passed": passed
    })
    
    print(f"  Noise level: {noise_level}")
    print(f"  Hit rate: {hit_rate:.2%} (threshold: 70%)")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed

def test_random_no_match():
    """Test: Query with random embeddings should mostly miss."""
    print("\n[Test 3] Random no-match test...")
    
    cache = MinimalCache()
    embeddings = np.random.randn(100, EMBEDDING_DIM).astype(np.float32)
    
    # Insert all
    for i, emb in enumerate(embeddings):
        cache.insert(emb, f"response_{i}".encode())
    
    # Query with completely random embeddings
    random_queries = np.random.randn(100, EMBEDDING_DIM).astype(np.float32)
    hits = 0
    for query in random_queries:
        entry_id, response, similarity = cache.lookup(query)
        if entry_id is not None:
            hits += 1
    
    hit_rate = hits / len(random_queries)
    passed = hit_rate <= 0.3  # Expect at most 30% false positives
    
    results["tests"].append({
        "name": "random_no_match",
        "hit_rate": hit_rate,
        "threshold": 0.3,
        "passed": passed
    })
    
    print(f"  Hit rate: {hit_rate:.2%} (threshold: ≤30%)")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed

def test_correlation():
    """Test: Hamming distance should correlate with cosine distance."""
    print("\n[Test 4] Cosine-Hamming correlation test...")
    
    cache = MinimalCache()
    n_samples = 100
    embeddings = np.random.randn(n_samples, EMBEDDING_DIM).astype(np.float32)
    
    # Normalize embeddings for cosine
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    
    # Compute cosine similarity matrix
    cosine_sim = normalized @ normalized.T
    
    # Encode all
    codes = cache._encode(embeddings)
    
    # Compute Hamming distances
    hamming_dist = np.zeros((n_samples, n_samples), dtype=np.float32)
    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                xor = np.bitwise_xor(codes[i], codes[j])
                hamming_dist[i, j] = sum(bin(w).count('1') for w in xor)
    
    # Get upper triangle (excluding diagonal)
    idx = np.triu_indices(n_samples, k=1)
    cosine_values = cosine_sim[idx]
    hamming_values = hamming_dist[idx]
    
    # Compute Spearman correlation (Hamming is distance, so should be negative with similarity)
    correlation, p_value = spearmanr(cosine_values, -hamming_values)
    
    passed = correlation >= 0.85
    
    results["tests"].append({
        "name": "correlation",
        "spearman_rho": float(correlation),
        "p_value": float(p_value),
        "threshold": 0.85,
        "passed": passed
    })
    
    print(f"  Spearman ρ: {correlation:.4f} (threshold: ≥0.85)")
    print(f"  p-value: {p_value:.2e}")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed

def test_no_memory_leak():
    """Test: Memory shouldn't grow unexpectedly."""
    print("\n[Test 5] Memory stability test...")
    
    import tracemalloc
    
    tracemalloc.start()
    cache = MinimalCache()
    
    # Insert 1000 entries
    for i in range(1000):
        emb = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        cache.insert(emb, f"response_{i}".encode())
    
    current1, peak1 = tracemalloc.get_traced_memory()
    
    # Insert 1000 more
    for i in range(1000, 2000):
        emb = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        cache.insert(emb, f"response_{i}".encode())
    
    current2, peak2 = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Memory should roughly double (linear growth)
    ratio = current2 / current1
    passed = 1.5 <= ratio <= 2.5  # Should be close to 2.0
    
    results["tests"].append({
        "name": "memory_stability",
        "memory_1000_kb": current1 / 1024,
        "memory_2000_kb": current2 / 1024,
        "ratio": ratio,
        "expected_ratio": "1.5-2.5",
        "passed": passed
    })
    
    print(f"  Memory at 1000 entries: {current1/1024:.1f} KB")
    print(f"  Memory at 2000 entries: {current2/1024:.1f} KB")
    print(f"  Ratio: {ratio:.2f} (expected: 1.5-2.5)")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed

def main():
    print("="*60)
    print("STAGE 3: Minimal Proof-of-Concept")
    print("="*60)
    
    tests = [
        test_exact_match,
        test_similar_match,
        test_random_no_match,
        test_correlation,
        test_no_memory_leak,
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    results["overall_status"] = "PASS" if all_passed else "FAIL"
    
    # Save results
    results_path = Path(__file__).parent.parent / "results" / "s3_poc_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed_count = sum(1 for t in results["tests"] if t["passed"])
    print(f"Tests passed: {passed_count}/{len(results['tests'])}")
    print(f"Overall: {results['overall_status']}")
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
```

---

## ACCEPTANCE CRITERIA

| Test | Criterion | Pass | Fail |
|------|-----------|------|------|
| Exact match | 100% hit rate | ✓ | ✗ |
| Similar match | ≥70% hit rate with 0.1 noise | ✓ | ✗ |
| Random no-match | ≤30% hit rate | ✓ | ✗ |
| Correlation | Spearman ρ ≥ 0.85 | ✓ | ✗ |
| Memory stability | Ratio 1.5-2.5 for 2× entries | ✓ | ✗ |

---

## KILL TRIGGERS

| Trigger | Action |
|---------|--------|
| Exact match < 100% | STOP: Binary encoding is broken |
| Correlation < 0.70 | STOP: Fundamental algorithm issue |
| Random hit rate > 50% | STOP: Threshold too low or bug |
| Memory ratio > 3.0 | INVESTIGATE: Memory leak |

---

## AFTER COMPLETION

If **PASS**: Core loop works. Proceed to Stage 4.  
If **FAIL**: Debug before investing more time.

---

*This is the moment of truth. If the PoC fails, stop here.*

