"""
CRITICAL TEST: Real Embeddings Validation

This test uses actual sentence embeddings to validate:
1. Hamming similarity correlates with cosine similarity
2. Similar semantic queries produce high Hamming similarity
3. The 0.85 threshold is appropriate

REQUIREMENT: pip install sentence-transformers
"""

import sys
from pathlib import Path
import numpy as np
from scipy import stats
import json
from datetime import datetime

# Add BinaryLLM
BINARYLLM_PATH = Path(__file__).parent.parent.parent.parent / "binary_llm"
sys.path.insert(0, str(BINARYLLM_PATH))

from src.quantization.binarization import RandomProjection, binarize_sign
from src.quantization.packing import pack_codes

print("="*60)
print("REAL EMBEDDINGS VALIDATION TEST")
print("="*60)

# Check for sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    print("✓ sentence-transformers available")
except ImportError:
    print("✗ sentence-transformers not installed")
    print("\nInstall with: pip install sentence-transformers")
    print("\nRunning fallback test with synthetic similar embeddings...")
    
    # Fallback: create embeddings that are ACTUALLY similar (small angle)
    print("\n--- Fallback: Controlled Similarity Test ---")
    
    def create_similar_embedding(base, target_cosine):
        """Create an embedding with specific cosine similarity to base."""
        # Generate orthogonal component
        ortho = np.random.randn(len(base)).astype(np.float32)
        ortho = ortho - np.dot(ortho, base) * base  # Make orthogonal
        ortho = ortho / np.linalg.norm(ortho)
        
        # Mix to achieve target cosine
        # cos(θ) = target_cosine, so θ = arccos(target_cosine)
        angle = np.arccos(target_cosine)
        result = np.cos(angle) * base + np.sin(angle) * ortho
        return result / np.linalg.norm(result)
    
    # Setup
    proj = RandomProjection(384, 256, 42)
    
    def encode_to_binary(emb):
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        projected = proj.project(emb)
        codes_pm1 = binarize_sign(projected)
        codes_01 = ((codes_pm1 + 1.0) / 2.0).astype(np.int8)
        return pack_codes(codes_01)
    
    def hamming_similarity(code1, code2):
        xored = code1 ^ code2
        dist = sum(bin(int(xored[0, w])).count('1') for w in range(code1.shape[1]))
        return 1.0 - dist / 256.0
    
    # Test at various cosine similarities
    print("\nTesting Hamming vs Cosine correlation:")
    print("-" * 50)
    
    np.random.seed(42)
    base = np.random.randn(384).astype(np.float32)
    base = base / np.linalg.norm(base)
    base_code = encode_to_binary(base)
    
    test_cosines = [0.99, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.30]
    results = []
    
    print(f"{'Cosine':<10} {'Hamming':<10} {'Would Hit (0.85)':<15}")
    print("-" * 50)
    
    for target_cos in test_cosines:
        similar = create_similar_embedding(base, target_cos)
        actual_cos = float(np.dot(base, similar))
        similar_code = encode_to_binary(similar)
        ham_sim = hamming_similarity(base_code, similar_code)
        hit = ham_sim >= 0.85
        
        results.append({
            "target_cosine": target_cos,
            "actual_cosine": actual_cos,
            "hamming": ham_sim,
            "would_hit": hit
        })
        
        print(f"{actual_cos:<10.4f} {ham_sim:<10.4f} {'✓ YES' if hit else '✗ NO':<15}")
    
    # Compute correlation
    cosines = [r["actual_cosine"] for r in results]
    hammings = [r["hamming"] for r in results]
    
    spearman_rho, spearman_p = stats.spearmanr(cosines, hammings)
    pearson_r, pearson_p = stats.pearsonr(cosines, hammings)
    
    print("\n" + "-" * 50)
    print("CORRELATION ANALYSIS")
    print("-" * 50)
    print(f"Spearman ρ: {spearman_rho:.4f} (p={spearman_p:.2e})")
    print(f"Pearson r:  {pearson_r:.4f} (p={pearson_p:.2e})")
    
    # Verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    
    if spearman_rho >= 0.85:
        print("✓ PASS: Strong correlation between Hamming and Cosine")
        print("  → Binary encoding preserves semantic similarity")
        verdict = "PASS"
    elif spearman_rho >= 0.70:
        print("⚠ CONDITIONAL: Moderate correlation")
        print("  → May need threshold adjustment")
        verdict = "CONDITIONAL"
    else:
        print("✗ FAIL: Weak correlation")
        print("  → Binary encoding loses too much information")
        verdict = "FAIL"
    
    # Threshold analysis
    print("\n--- Threshold Analysis ---")
    for threshold in [0.90, 0.85, 0.80, 0.75, 0.70]:
        hits_at_threshold = sum(1 for r in results if r["hamming"] >= threshold)
        min_cosine_hit = min((r["actual_cosine"] for r in results if r["hamming"] >= threshold), default=None)
        print(f"Threshold {threshold}: {hits_at_threshold}/9 would hit, min cosine = {min_cosine_hit or 'N/A'}")
    
    # Save results
    output = {
        "test_type": "fallback_controlled_similarity",
        "timestamp": datetime.now().isoformat(),
        "spearman_rho": spearman_rho,
        "pearson_r": pearson_r,
        "verdict": verdict,
        "results": results
    }
    
    results_file = Path(__file__).parent.parent / "results" / "real_embeddings_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    sys.exit(0 if verdict == "PASS" else 1)

# === FULL TEST WITH SENTENCE TRANSFORMERS ===

print("\nLoading embedding model (this may take a moment)...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim embeddings
print(f"✓ Model loaded: {model.get_sentence_embedding_dimension()}-dim embeddings")

# Setup binary encoding
proj = RandomProjection(384, 256, 42)

def encode_to_binary(emb):
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    emb = emb.astype(np.float32)
    projected = proj.project(emb)
    codes_pm1 = binarize_sign(projected)
    codes_01 = ((codes_pm1 + 1.0) / 2.0).astype(np.int8)
    return pack_codes(codes_01)

def hamming_similarity(code1, code2):
    xored = code1 ^ code2
    dist = sum(bin(int(xored[0, w])).count('1') for w in range(code1.shape[1]))
    return 1.0 - dist / 256.0

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Test pairs with expected similarity levels
test_pairs = [
    # Very similar (expected cosine > 0.9)
    ("What is the capital of France?", "Tell me the capital city of France"),
    ("How do I cook pasta?", "What's the best way to make pasta?"),
    ("What time is it?", "Can you tell me the current time?"),
    
    # Somewhat similar (expected cosine 0.6-0.8)
    ("What is the capital of France?", "Paris is a beautiful city"),
    ("How do I cook pasta?", "Italian cuisine is delicious"),
    
    # Different topics (expected cosine < 0.5)
    ("What is the capital of France?", "How do I fix a flat tire?"),
    ("How do I cook pasta?", "What is machine learning?"),
    ("What time is it?", "The stock market crashed yesterday"),
]

print("\n" + "-" * 70)
print("SEMANTIC SIMILARITY TEST")
print("-" * 70)
print(f"{'Query Pair':<50} {'Cosine':<10} {'Hamming':<10}")
print("-" * 70)

results = []

for q1, q2 in test_pairs:
    # Get embeddings
    emb1 = model.encode(q1)
    emb2 = model.encode(q2)
    
    # Compute similarities
    cos_sim = cosine_similarity(emb1, emb2)
    
    code1 = encode_to_binary(emb1)
    code2 = encode_to_binary(emb2)
    ham_sim = hamming_similarity(code1, code2)
    
    # Truncate query for display
    pair_display = f"{q1[:20]}... vs {q2[:20]}..."
    print(f"{pair_display:<50} {cos_sim:<10.4f} {ham_sim:<10.4f}")
    
    results.append({
        "query1": q1,
        "query2": q2,
        "cosine": cos_sim,
        "hamming": ham_sim
    })

# Correlation analysis
cosines = [r["cosine"] for r in results]
hammings = [r["hamming"] for r in results]

spearman_rho, spearman_p = stats.spearmanr(cosines, hammings)
pearson_r, pearson_p = stats.pearsonr(cosines, hammings)

print("\n" + "-" * 70)
print("CORRELATION ANALYSIS")
print("-" * 70)
print(f"Spearman ρ: {spearman_rho:.4f} (p={spearman_p:.4f})")
print(f"Pearson r:  {pearson_r:.4f} (p={pearson_p:.4f})")

# Cache hit analysis
print("\n" + "-" * 70)
print("CACHE HIT ANALYSIS (threshold = 0.85)")
print("-" * 70)

high_cosine_pairs = [r for r in results if r["cosine"] >= 0.85]
high_hamming_pairs = [r for r in results if r["hamming"] >= 0.85]

true_positives = sum(1 for r in results if r["cosine"] >= 0.85 and r["hamming"] >= 0.85)
false_negatives = sum(1 for r in results if r["cosine"] >= 0.85 and r["hamming"] < 0.85)
false_positives = sum(1 for r in results if r["cosine"] < 0.85 and r["hamming"] >= 0.85)
true_negatives = sum(1 for r in results if r["cosine"] < 0.85 and r["hamming"] < 0.85)

print(f"True Positives:  {true_positives}")
print(f"False Negatives: {false_negatives} (similar queries missed by cache)")
print(f"False Positives: {false_positives} (dissimilar queries incorrectly cached)")
print(f"True Negatives:  {true_negatives}")

if true_positives + false_negatives > 0:
    recall = true_positives / (true_positives + false_negatives)
    print(f"\nRecall: {recall:.2%}")
else:
    recall = None
    print("\nRecall: N/A (no high-cosine pairs)")

if true_positives + false_positives > 0:
    precision = true_positives / (true_positives + false_positives)
    print(f"Precision: {precision:.2%}")
else:
    precision = None
    print("Precision: N/A (no cache hits)")

# Verdict
print("\n" + "="*70)
print("VERDICT")
print("="*70)

verdict = "UNKNOWN"

if spearman_rho >= 0.85:
    print("✓ STRONG CORRELATION: Hamming similarity tracks cosine similarity well")
    verdict = "PASS"
elif spearman_rho >= 0.70:
    print("⚠ MODERATE CORRELATION: Binary encoding works but may miss some matches")
    verdict = "CONDITIONAL"
else:
    print("✗ WEAK CORRELATION: Binary encoding loses too much semantic information")
    verdict = "FAIL"

if false_negatives > 0:
    print(f"\n⚠ WARNING: {false_negatives} similar query pairs would NOT hit cache")
    print("  Consider lowering threshold or increasing code_bits")

if false_positives > 0:
    print(f"\n⚠ WARNING: {false_positives} dissimilar query pairs WOULD hit cache incorrectly")
    print("  Consider raising threshold")

# Threshold recommendation
print("\n--- Threshold Recommendation ---")
for threshold in [0.90, 0.85, 0.80, 0.75, 0.70, 0.65]:
    tp = sum(1 for r in results if r["cosine"] >= 0.85 and r["hamming"] >= threshold)
    fn = sum(1 for r in results if r["cosine"] >= 0.85 and r["hamming"] < threshold)
    fp = sum(1 for r in results if r["cosine"] < 0.85 and r["hamming"] >= threshold)
    
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"Threshold {threshold:.2f}: Recall={rec:.0%}, Precision={prec:.0%}")

print("\n" + "="*70)
print(f"FINAL VERDICT: {verdict}")
print("="*70)

# Save results
output = {
    "test_type": "sentence_transformers",
    "model": "all-MiniLM-L6-v2",
    "timestamp": datetime.now().isoformat(),
    "spearman_rho": spearman_rho,
    "pearson_r": pearson_r,
    "recall": recall,
    "precision": precision,
    "verdict": verdict,
    "results": results
}

results_file = Path(__file__).parent.parent / "results" / "real_embeddings_results.json"
with open(results_file, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to: {results_file}")

