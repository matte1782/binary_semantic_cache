"""
Numba JIT-optimized Hamming distance
This achieves near-C performance in Python

Install: pip install numba
"""

import sys
import time
import numpy as np
from pathlib import Path

print("="*60)
print("NUMBA-OPTIMIZED HAMMING DISTANCE")
print("="*60)

try:
    from numba import jit, prange
    print("✓ Numba available")
    NUMBA_AVAILABLE = True
except ImportError:
    print("✗ Numba not installed. Run: pip install numba")
    print("  Falling back to pure NumPy")
    NUMBA_AVAILABLE = False

# Add BinaryLLM to path
BINARYLLM_PATH = Path(__file__).parent.parent.parent.parent / "binary_llm"
sys.path.insert(0, str(BINARYLLM_PATH))

# Popcount lookup table
POPCOUNT_8BIT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, fastmath=True)
    def hamming_distance_numba(query, codes):
        """Numba-optimized Hamming distance."""
        n_entries = codes.shape[0]
        n_words = codes.shape[1]
        distances = np.zeros(n_entries, dtype=np.int32)
        
        # Parallel loop over entries
        for i in prange(n_entries):
            dist = 0
            for w in range(n_words):
                xor_val = query[w] ^ codes[i, w]
                # Inline popcount using bit manipulation
                while xor_val:
                    dist += 1
                    xor_val &= xor_val - 1  # Brian Kernighan's algorithm
            distances[i] = dist
        
        return distances
    
    @jit(nopython=True, parallel=True)
    def hamming_similarity_numba(query, codes, code_bits=256):
        """Convert distances to similarities."""
        distances = hamming_distance_numba(query, codes)
        similarities = np.zeros(len(distances), dtype=np.float32)
        for i in prange(len(distances)):
            similarities[i] = 1.0 - (distances[i] / code_bits)
        return similarities
    
    # Alternative: Using lookup table (sometimes faster for large batches)
    @jit(nopython=True, parallel=True)
    def hamming_similarity_numba_lookup(query, codes, code_bits=256):
        """Numba with lookup table for popcount."""
        n_entries = codes.shape[0]
        n_words = codes.shape[1]
        similarities = np.zeros(n_entries, dtype=np.float32)
        
        for i in prange(n_entries):
            dist = 0
            for w in range(n_words):
                xor_val = query[w] ^ codes[i, w]
                # Extract bytes and use lookup
                for shift in range(0, 64, 8):
                    byte_val = (xor_val >> shift) & 0xFF
                    dist += POPCOUNT_8BIT[byte_val]
            similarities[i] = 1.0 - (dist / code_bits)
        
        return similarities

# Test setup
n_entries = 100_000
n_queries = 100
n_words = 4
code_bits = 256

print(f"\nTest setup:")
print(f"  Entries: {n_entries:,}")
print(f"  Queries: {n_queries}")
print()

# Generate test data
np.random.seed(42)
codes = np.random.randint(0, 2**63, size=(n_entries, n_words), dtype=np.uint64)
queries = np.random.randint(0, 2**63, size=(n_queries, n_words), dtype=np.uint64)

# Benchmark function
def benchmark_method(name, func, warm_up=True):
    if warm_up:
        # Warm up JIT compilation
        _ = func(queries[0], codes[:100])
    
    start = time.perf_counter()
    for q in queries:
        _ = func(q, codes)
    elapsed = time.perf_counter() - start
    
    per_query_us = (elapsed / n_queries) * 1_000_000
    print(f"{name:35s}: {per_query_us:8.1f} µs/query")
    return per_query_us

# Original NumPy method for comparison
def hamming_numpy_original(query, codes):
    xored = codes ^ query
    bytes_view = xored.view(np.uint8)
    distances = POPCOUNT_8BIT[bytes_view].reshape(codes.shape[0], -1).sum(axis=1)
    return 1.0 - (distances.astype(np.float32) / code_bits)

print("Running benchmarks...")
print("-" * 55)

# Benchmark NumPy baseline
numpy_time = benchmark_method("NumPy (lookup table)", hamming_numpy_original)

if NUMBA_AVAILABLE:
    # Benchmark Numba versions
    numba_time = benchmark_method("Numba (bit manipulation)", 
                                  lambda q, c: hamming_similarity_numba(q, c, code_bits))
    
    numba_lookup_time = benchmark_method("Numba (lookup table)", 
                                         lambda q, c: hamming_similarity_numba_lookup(q, c, code_bits))
    
    print("-" * 55)
    print(f"\nSpeedup vs NumPy:")
    print(f"  Numba (bit manipulation): {numpy_time/numba_time:.1f}x")
    print(f"  Numba (lookup table):     {numpy_time/numba_lookup_time:.1f}x")
    
    best_time = min(numba_time, numba_lookup_time)
    if best_time < 1000:
        print(f"\n✅ ACHIEVED SUB-MILLISECOND: {best_time:.1f} µs/query!")
    else:
        print(f"\n⚠️  Best time: {best_time/1000:.2f}ms (still above 1ms)")
    
    # Verify accuracy
    print("\nAccuracy check...")
    q_test = queries[0]
    baseline = hamming_numpy_original(q_test, codes)
    numba_result = hamming_similarity_numba(q_test, codes, code_bits)
    
    if np.allclose(baseline, numba_result, atol=1e-6):
        print("✓ Numba results match NumPy")
    else:
        print("✗ Results differ!")

print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)

if NUMBA_AVAILABLE and best_time < 1000:
    print("\n✅ SUCCESS: Numba achieves sub-millisecond lookup!")
    print("\nTo use in production:")
    print("1. Add numba to requirements.txt")
    print("2. Replace hamming_similarity_fast with numba version")
    print("3. Cache JIT compilation with cache=True")
else:
    print("\nOptions to achieve <1ms:")
    print("1. Install numba: pip install numba")
    print("2. Use C extension (e.g., Cython)")
    print("3. Reduce query batch size")
    print("4. Use approximate methods (LSH)")
    print("5. Accept current performance with explanation")

print("\nNOTE: The 100µs target is extremely aggressive for Python.")
print("Most production systems would be happy with <10ms.")
