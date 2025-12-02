"""
OPTIMIZED Hamming Distance Benchmarks
Goal: Get lookup latency below 1ms for 100K entries

Optimization techniques:
1. Batch all queries together (no Python loop)
2. Use numpy's packbits/unpackbits for efficient bit manipulation
3. Minimize memory allocations
4. Try multiple approaches and pick fastest
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add BinaryLLM to path
BINARYLLM_PATH = Path(__file__).parent.parent.parent.parent / "binary_llm"
sys.path.insert(0, str(BINARYLLM_PATH))

from src.quantization.packing import pack_codes

print("="*60)
print("HAMMING DISTANCE OPTIMIZATION BENCHMARK")
print("="*60)
print()

# Setup test data
n_entries = 100_000
n_queries = 100
n_words = 4  # 256 bits / 64
code_bits = 256

print(f"Test setup:")
print(f"  Entries: {n_entries:,}")
print(f"  Queries: {n_queries}")
print(f"  Bits per code: {code_bits}")
print(f"  Words per code: {n_words}")
print()

# Generate random codes
np.random.seed(42)
codes = np.random.randint(0, 2**63, size=(n_entries, n_words), dtype=np.uint64)
queries = np.random.randint(0, 2**63, size=(n_queries, n_words), dtype=np.uint64)

# Precompute popcount table
POPCOUNT_8BIT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

def benchmark_method(name, func, warm_up=True):
    """Benchmark a Hamming distance function."""
    if warm_up:
        # Warm up
        _ = func(queries[:5], codes[:1000])
    
    # Time the full benchmark
    start = time.perf_counter()
    for q in queries:
        _ = func(q.reshape(1, -1), codes)
    elapsed = time.perf_counter() - start
    
    per_query_us = (elapsed / n_queries) * 1_000_000
    print(f"{name:30s}: {per_query_us:8.1f} µs/query ({elapsed:.4f}s total)")
    return per_query_us

# Method 1: Original optimized (from v2)
def hamming_v2_original(queries, codes):
    """Original v2 implementation."""
    results = []
    for q in queries:
        xored = q ^ codes
        bytes_view = xored.view(np.uint8)
        distances = POPCOUNT_8BIT[bytes_view].reshape(codes.shape[0], -1).sum(axis=1)
        similarities = 1.0 - (distances.astype(np.float32) / code_bits)
        results.append(similarities)
    return np.array(results) if len(queries) > 1 else results[0]

# Method 2: Batch processing (no Python loop)
def hamming_batch(queries, codes):
    """Process all queries in one batch."""
    # Broadcast: (Q, 1, W) ^ (1, N, W) = (Q, N, W)
    xored = queries[:, np.newaxis, :] ^ codes[np.newaxis, :, :]
    
    # Convert to uint8 view for popcount
    bytes_view = xored.view(np.uint8)  # Shape: (Q, N, W*8)
    
    # Apply popcount lookup
    distances = POPCOUNT_8BIT[bytes_view].sum(axis=2)  # Shape: (Q, N)
    
    # Convert to similarities
    similarities = 1.0 - (distances.astype(np.float32) / code_bits)
    
    return similarities[0] if queries.shape[0] == 1 else similarities

# Method 3: Using unpackbits (built-in)
def hamming_unpackbits(queries, codes):
    """Use numpy's unpackbits for bit counting."""
    results = []
    for q in queries:
        xored = q ^ codes
        # View as uint8 and unpack to bits
        bytes_view = xored.view(np.uint8)
        bits = np.unpackbits(bytes_view, axis=1)
        distances = bits.sum(axis=1)
        similarities = 1.0 - (distances.astype(np.float32) / code_bits)
        results.append(similarities)
    return np.array(results) if len(queries) > 1 else results[0]

# Method 4: Optimized with contiguous memory
def hamming_optimized(queries, codes):
    """Optimized with contiguous memory access."""
    # Ensure contiguous memory
    codes_contig = np.ascontiguousarray(codes)
    
    results = []
    for q in queries:
        # XOR with broadcasting
        xored = codes_contig ^ q
        
        # Flatten to 1D for efficient popcount
        xored_flat = xored.ravel()
        bytes_flat = xored_flat.view(np.uint8)
        
        # Popcount on flat array (more cache-friendly)
        popcounts = POPCOUNT_8BIT[bytes_flat]
        
        # Reshape and sum
        distances = popcounts.reshape(n_entries, -1).sum(axis=1)
        similarities = 1.0 - (distances.astype(np.float32) / code_bits)
        results.append(similarities)
    
    return np.array(results) if len(queries) > 1 else results[0]

# Method 5: Pure NumPy bit manipulation
def hamming_numpy_bits(queries, codes):
    """Use NumPy's bit manipulation functions."""
    results = []
    for q in queries:
        xored = codes ^ q
        
        # Count bits using Brian Kernighan's algorithm in NumPy
        distances = np.zeros(codes.shape[0], dtype=np.int32)
        for word_idx in range(n_words):
            word = xored[:, word_idx]
            # Count set bits in each uint64
            # This is slow in pure Python but shows the concept
            for i in range(64):
                distances += (word >> i) & 1
        
        similarities = 1.0 - (distances.astype(np.float32) / code_bits)
        results.append(similarities)
    
    return np.array(results) if len(queries) > 1 else results[0]

# Method 6: Split by uint32 (sometimes faster)
def hamming_uint32_split(queries, codes):
    """Split uint64 into uint32 for potentially better performance."""
    # View as uint32 (twice as many elements)
    codes_32 = codes.view(np.uint32)
    
    results = []
    for q in queries:
        q_32 = q.view(np.uint32)
        xored = codes_32 ^ q_32
        
        # Popcount on uint32 might be more efficient
        bytes_view = xored.view(np.uint8)
        distances = POPCOUNT_8BIT[bytes_view].reshape(codes.shape[0], -1).sum(axis=1)
        similarities = 1.0 - (distances.astype(np.float32) / code_bits)
        results.append(similarities)
    
    return np.array(results) if len(queries) > 1 else results[0]

# Run benchmarks
print("Running benchmarks...")
print()

# Warm up NumPy
_ = codes @ queries.T

results = {}

print("Method performances:")
print("-" * 50)

# Skip the slow numpy_bits method
methods = [
    ("Original v2", hamming_v2_original),
    ("Batch processing", hamming_batch),
    ("Unpackbits", hamming_unpackbits),
    ("Optimized contiguous", hamming_optimized),
    ("uint32 split", hamming_uint32_split),
]

for name, func in methods:
    try:
        us_per_query = benchmark_method(name, func)
        results[name] = us_per_query
    except Exception as e:
        print(f"{name:30s}: ERROR - {e}")

print("-" * 50)

# Find the best method
if results:
    best_method = min(results.items(), key=lambda x: x[1])
    print(f"\nBest method: {best_method[0]} at {best_method[1]:.1f} µs/query")
    print(f"Speedup vs v2: {results.get('Original v2', 0) / best_method[1]:.1f}x")
    
    if best_method[1] < 1000:
        print("\n✅ ACHIEVED SUB-MILLISECOND LOOKUP!")
    else:
        print(f"\n⚠️  Still above 1ms target. Best: {best_method[1]/1000:.2f}ms")
        
# Test accuracy of best methods
print("\n" + "="*60)
print("ACCURACY CHECK")
print("="*60)

# Pick two random indices to verify
idx1, idx2 = 10, 50
q_test = queries[idx1:idx1+1]

print("\nVerifying all methods give same results...")
baseline = hamming_v2_original(q_test, codes)

for name, func in methods[1:]:  # Skip baseline
    try:
        result = func(q_test, codes)
        if np.allclose(baseline, result):
            print(f"  {name}: ✓ Matches")
        else:
            print(f"  {name}: ✗ DIFFERS!")
    except:
        print(f"  {name}: ✗ ERROR")

# Additional optimization suggestions
print("\n" + "="*60)
print("ADDITIONAL OPTIMIZATION OPTIONS")
print("="*60)
print()
print("If still above 1ms, consider:")
print("1. Use Cython or Numba JIT compilation")
print("2. Implement in C with Python bindings")
print("3. Use SIMD intrinsics (AVX2/AVX512)")
print("4. GPU acceleration for batch queries")
print("5. Approximate algorithms (LSH, learned indices)")
print()
print("For this PoC, Python performance may be acceptable with caveats.")
