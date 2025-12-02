"""
Stage 3: Minimal Proof-of-Concept
Purpose: Demonstrate end-to-end cache flow works

This PoC validates:
1. Embedding → Binary encode
2. Binary → Cache store
3. Query → Cache lookup (hit/miss)
4. Threshold-based matching

NOT included (deferred to Phase 1):
- HTTP proxy
- Real LLM integration
- Persistence
- Eviction
"""

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json

import numpy as np

# Add BinaryLLM to path
BINARYLLM_PATH = Path(__file__).parent.parent.parent.parent / "binary_llm"
sys.path.insert(0, str(BINARYLLM_PATH))

from src.quantization.binarization import RandomProjection, binarize_sign
from src.quantization.packing import pack_codes


@dataclass
class CacheEntry:
    """A single cache entry."""
    binary_code: np.ndarray  # Packed uint64
    response: str
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    hits: int = 0


class BinarySemanticCache:
    """Minimal Binary Semantic Cache PoC."""
    
    def __init__(
        self,
        embedding_dim: int = 384,
        code_bits: int = 256,
        seed: int = 42,
        similarity_threshold: float = 0.85,
        max_entries: int = 10_000
    ):
        self._projection = RandomProjection(embedding_dim, code_bits, seed)
        self._code_bits = code_bits
        self._threshold = similarity_threshold
        self._max_entries = max_entries
        
        # Storage
        self._entries: Dict[int, CacheEntry] = {}  # ID → Entry
        self._codes: Optional[np.ndarray] = None   # (N, W) packed codes
        self._ids: list = []                        # Ordered list of IDs
        self._next_id = 0
        
    def _encode(self, embedding: np.ndarray) -> np.ndarray:
        """Convert embedding to packed binary code."""
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        projected = self._projection.project(embedding)
        codes_pm1 = binarize_sign(projected)
        codes_01 = ((codes_pm1 + 1.0) / 2.0).astype(np.int8)
        return pack_codes(codes_01)
    
    def _hamming_similarity(self, query_code: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Compute normalized Hamming similarity (1 = identical, 0 = opposite)."""
        # XOR then popcount
        xored = query_code ^ codes
        distances = np.zeros(codes.shape[0], dtype=np.int32)
        for w in range(codes.shape[1]):
            distances += np.vectorize(lambda x: bin(x).count('1'))(xored[:, w])
        # Normalize: similarity = 1 - (distance / code_bits)
        return 1.0 - (distances.astype(np.float32) / self._code_bits)
    
    def lookup(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Look up embedding in cache.
        
        Returns:
            (response, similarity) if hit, (None, 0.0) if miss
        """
        if self._codes is None or len(self._ids) == 0:
            return None, 0.0
        
        query_code = self._encode(embedding)
        similarities = self._hamming_similarity(query_code, self._codes)
        
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        
        if best_sim >= self._threshold:
            entry_id = self._ids[best_idx]
            entry = self._entries[entry_id]
            entry.hits += 1
            return entry.response, float(best_sim)
        
        return None, float(best_sim)
    
    def store(self, embedding: np.ndarray, response: str) -> int:
        """
        Store embedding-response pair in cache.
        
        Returns:
            Entry ID
        """
        code = self._encode(embedding)
        
        entry_id = self._next_id
        self._next_id += 1
        
        self._entries[entry_id] = CacheEntry(
            binary_code=code,
            response=response
        )
        
        if self._codes is None:
            self._codes = code
        else:
            self._codes = np.vstack([self._codes, code])
        
        self._ids.append(entry_id)
        
        # Simple eviction: remove oldest if full
        if len(self._ids) > self._max_entries:
            oldest_id = self._ids.pop(0)
            del self._entries[oldest_id]
            self._codes = self._codes[1:]
        
        return entry_id
    
    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total_hits = sum(e.hits for e in self._entries.values())
        return {
            "entries": len(self._entries),
            "max_entries": self._max_entries,
            "total_hits": total_hits,
            "code_bits": self._code_bits,
            "threshold": self._threshold
        }


def run_poc_demo():
    """Run minimal PoC demonstration."""
    print("\n" + "="*60)
    print("STAGE 3: Minimal PoC Demonstration")
    print("="*60)
    
    # Initialize cache
    cache = BinarySemanticCache(
        embedding_dim=384,
        code_bits=256,
        seed=42,
        similarity_threshold=0.85
    )
    
    print("\n[1/5] Cache initialized")
    print(f"  - Embedding dim: 384")
    print(f"  - Code bits: 256")
    print(f"  - Threshold: 0.85")
    
    # Create test embeddings
    np.random.seed(123)
    emb1 = np.random.randn(384).astype(np.float32)
    emb1 /= np.linalg.norm(emb1)  # Normalize
    
    # Similar embedding (add small noise)
    noise = np.random.randn(384).astype(np.float32) * 0.1
    emb2 = emb1 + noise
    emb2 /= np.linalg.norm(emb2)
    
    # Different embedding
    emb3 = np.random.randn(384).astype(np.float32)
    emb3 /= np.linalg.norm(emb3)
    
    print("\n[2/5] Test embeddings created")
    print(f"  - emb1: original")
    print(f"  - emb2: similar (small noise)")
    print(f"  - emb3: different")
    
    # Store first embedding
    entry_id = cache.store(emb1, "Response for embedding 1")
    print(f"\n[3/5] Stored emb1 with ID {entry_id}")
    
    # Lookup with similar embedding (should hit)
    response, similarity = cache.lookup(emb2)
    print(f"\n[4/5] Lookup with emb2 (similar):")
    print(f"  - Similarity: {similarity:.4f}")
    print(f"  - Hit: {response is not None}")
    if response:
        print(f"  - Response: {response[:50]}...")
    
    # Lookup with different embedding (should miss)
    response, similarity = cache.lookup(emb3)
    print(f"\n[5/5] Lookup with emb3 (different):")
    print(f"  - Similarity: {similarity:.4f}")
    print(f"  - Hit: {response is not None}")
    
    # Stats
    stats = cache.stats()
    print("\n" + "="*60)
    print("CACHE STATS")
    print("="*60)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Validate results
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)
    
    tests_passed = 0
    tests_total = 3
    
    # Test 1: Similar embedding should hit
    response, sim = cache.lookup(emb2)
    if response is not None and sim >= 0.85:
        print("  ✓ Similar embedding hits cache")
        tests_passed += 1
    else:
        print(f"  ✗ Similar embedding missed (sim={sim:.4f})")
    
    # Test 2: Different embedding should miss
    response, sim = cache.lookup(emb3)
    if response is None:
        print("  ✓ Different embedding misses cache")
        tests_passed += 1
    else:
        print(f"  ✗ Different embedding hit unexpectedly (sim={sim:.4f})")
    
    # Test 3: Exact same embedding should hit with high similarity
    response, sim = cache.lookup(emb1)
    if response is not None and sim >= 0.99:
        print("  ✓ Exact embedding hits with ~100% similarity")
        tests_passed += 1
    else:
        print(f"  ✗ Exact embedding not matching (sim={sim:.4f})")
    
    print("="*60)
    print(f"RESULT: {tests_passed}/{tests_total} tests passed")
    print("="*60)
    
    # Save results
    results = {
        "stage": "S3",
        "name": "Minimal PoC",
        "timestamp": datetime.now().isoformat(),
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "status": "PASS" if tests_passed == tests_total else "FAIL",
        "stats": stats
    }
    
    results_path = Path(__file__).parent.parent / "results" / "s3_poc_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return tests_passed == tests_total


if __name__ == "__main__":
    success = run_poc_demo()
    sys.exit(0 if success else 1)

