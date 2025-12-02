"""
PROPER SIMILARITY TEST
Tests that Hamming similarity correlates with cosine similarity.

This test creates embedding pairs with KNOWN cosine similarities
and verifies the binary encoding preserves the similarity structure.
"""

import sys
from pathlib import Path
import numpy as np

# Setup paths
script_dir = Path(__file__).parent
binaryllm_path = script_dir.parent.parent.parent / "binary_llm"
sys.path.insert(0, str(binaryllm_path))

from src.quantization.binarization import RandomProjection, binarize_sign
from src.quantization.packing import pack_codes

print("="*60)
print("PROPER SIMILARITY CORRELATION TEST")
print("="*60)
print("Purpose: Verify Hamming similarity tracks cosine similarity")
print()

# Initialize projection
proj = RandomProjection(384, 256, seed=42)

def encode(emb):
    """Encode embedding to packed binary code."""
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    projected = proj.project(emb)
    codes_pm1 = binarize_sign(projected)
    codes_01 = ((codes_pm1 + 1.0) / 2.0).astype(np.int8)
    return pack_codes(codes_01)

def hamming_similarity(code1, code2):
    """Compute normalized Hamming similarity."""
    xored = code1 ^ code2
    dist = sum(bin(int(xored[0, w])).count('1') for w in range(code1.shape[1]))
    return 1.0 - dist / 256.0

def create_similar_embedding(base, target_cosine):
    """
    Create an embedding with approximately target cosine similarity to base.
    
    Uses the formula: emb2 = cos(θ) * base + sin(θ) * orthogonal
    where θ = arccos(target_cosine)
    """
    # Normalize base
    base = base / np.linalg.norm(base)
    
    # Create random orthogonal vector
    random_vec = np.random.randn(*base.shape).astype(np.float32)
    # Gram-Schmidt: make orthogonal to base
    orthogonal = random_vec - np.dot(random_vec.flatten(), base.flatten()) * base
    orthogonal = orthogonal / np.linalg.norm(orthogonal)
    
    # Compute angle
    theta = np.arccos(np.clip(target_cosine, -1, 1))
    
    # Create new embedding
    emb2 = np.cos(theta) * base + np.sin(theta) * orthogonal
    emb2 = emb2 / np.linalg.norm(emb2)
    
    return emb2.astype(np.float32)

# Test with various similarity levels
print("Testing similarity preservation across different cosine values:")
print("-" * 60)
print(f"{'Target Cosine':>15} | {'Actual Cosine':>15} | {'Hamming Sim':>15} | {'Error':>10}")
print("-" * 60)

np.random.seed(42)
results = []

for target_cos in [0.99, 0.95, 0.90, 0.85, 0.80, 0.70, 0.50, 0.30]:
    # Create base embedding
    base = np.random.randn(1, 384).astype(np.float32)
    base /= np.linalg.norm(base)
    
    # Create embedding with target similarity
    similar = create_similar_embedding(base, target_cos)
    
    # Verify actual cosine similarity
    actual_cos = float(np.dot(base.flatten(), similar.flatten()))
    
    # Encode both
    code_base = encode(base)
    code_similar = encode(similar)
    
    # Compute Hamming similarity
    ham_sim = hamming_similarity(code_base, code_similar)
    
    # Compute error
    error = abs(ham_sim - actual_cos)
    
    results.append({
        'target': target_cos,
        'actual': actual_cos,
        'hamming': ham_sim,
        'error': error
    })
    
    status = "✓" if error < 0.15 else "⚠"
    print(f"{target_cos:>15.2f} | {actual_cos:>15.4f} | {ham_sim:>15.4f} | {error:>10.4f} {status}")

print("-" * 60)

# Compute correlation
cosines = [r['actual'] for r in results]
hammings = [r['hamming'] for r in results]
correlation = np.corrcoef(cosines, hammings)[0, 1]

print(f"\nCorrelation (Pearson r): {correlation:.4f}")
print(f"Target: r > 0.85")
print(f"Status: {'✓ PASS' if correlation > 0.85 else '✗ FAIL'}")

# Critical test: Does 0.95 cosine give us a cache hit?
print("\n" + "="*60)
print("CRITICAL TEST: Would high-similarity embeddings hit cache?")
print("="*60)

# Find the result for 0.95 target
high_sim_result = next(r for r in results if r['target'] == 0.95)
print(f"Target cosine: 0.95")
print(f"Actual cosine: {high_sim_result['actual']:.4f}")
print(f"Hamming similarity: {high_sim_result['hamming']:.4f}")
print(f"Threshold: 0.85")
print(f"Would hit cache: {high_sim_result['hamming'] >= 0.85}")

if high_sim_result['hamming'] >= 0.85:
    print("\n✓ ARCHITECTURE VALIDATED: High-similarity queries hit cache")
else:
    print("\n⚠ WARNING: May need to adjust threshold or increase code bits")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
avg_error = np.mean([r['error'] for r in results])
max_error = max(r['error'] for r in results)
print(f"Average error: {avg_error:.4f}")
print(f"Max error: {max_error:.4f}")
print(f"Correlation: {correlation:.4f}")

all_pass = correlation > 0.85 and high_sim_result['hamming'] >= 0.80
print(f"\nOverall: {'✓ PASS' if all_pass else '⚠ NEEDS ATTENTION'}")

# Save results
import json
results_file = script_dir.parent / "results" / "similarity_correlation_test.json"
with open(results_file, "w") as f:
    json.dump({
        "correlation": correlation,
        "results": results,
        "pass": all_pass,
        "avg_error": avg_error,
        "max_error": max_error
    }, f, indent=2)
print(f"\nResults saved to: {results_file}")

