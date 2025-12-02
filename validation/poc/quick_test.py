"""Quick validation test - outputs to console and file."""
import sys
from pathlib import Path

print("="*50)
print("QUICK VALIDATION TEST")
print("="*50)

# Setup paths
script_dir = Path(__file__).parent
binaryllm_path = script_dir.parent.parent.parent / "binary_llm"
print(f"Script dir: {script_dir}")
print(f"BinaryLLM path: {binaryllm_path}")
print(f"BinaryLLM exists: {binaryllm_path.exists()}")

sys.path.insert(0, str(binaryllm_path))

# Test imports
print("\n--- Import Tests ---")
try:
    from src.quantization.binarization import RandomProjection, binarize_sign
    print("✓ RandomProjection, binarize_sign imported")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

try:
    from src.quantization.packing import pack_codes
    print("✓ pack_codes imported")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("✓ numpy imported")
except Exception as e:
    print(f"✗ numpy failed: {e}")
    sys.exit(1)

# Quick encode test
print("\n--- Encode Test ---")
import time
proj = RandomProjection(384, 256, 42)
emb = np.random.randn(1, 384).astype(np.float32)

start = time.perf_counter()
for _ in range(100):
    projected = proj.project(emb)
    codes_pm1 = binarize_sign(projected)
    codes_01 = ((codes_pm1 + 1.0) / 2.0).astype(np.int8)
    packed = pack_codes(codes_01)
elapsed = (time.perf_counter() - start) * 10  # ms per sample
print(f"Encode latency: {elapsed:.3f}ms per sample")
print(f"Target: <1ms, Kill: >5ms")
print(f"Status: {'✓ PASS' if elapsed < 5 else '✗ FAIL'}")

# Memory test
print("\n--- Memory Test ---")
n_entries = 100_000
n_words = 4
codes = np.zeros((n_entries, n_words), dtype=np.uint64)
mem_mb = codes.nbytes / (1024*1024)
print(f"100K entries: {mem_mb:.2f}MB")
print(f"Target: <4MB, Kill: >10MB")
print(f"Status: {'✓ PASS' if mem_mb < 10 else '✗ FAIL'}")

# Cache test
print("\n--- Cache Test ---")
emb1 = np.random.randn(1, 384).astype(np.float32)
emb1 /= np.linalg.norm(emb1)

noise = np.random.randn(1, 384).astype(np.float32) * 0.1
emb2 = emb1 + noise
emb2 /= np.linalg.norm(emb2)

code1 = pack_codes(((binarize_sign(proj.project(emb1)) + 1) / 2).astype(np.int8))
code2 = pack_codes(((binarize_sign(proj.project(emb2)) + 1) / 2).astype(np.int8))

# Hamming distance
xored = code1 ^ code2
dist = sum(bin(int(xored[0, w])).count('1') for w in range(4))
sim = 1.0 - dist / 256.0
print(f"Similar embeddings similarity: {sim:.4f}")
print(f"Threshold: 0.85")
print(f"Would hit cache: {sim >= 0.85}")

print("\n" + "="*50)
print("QUICK TEST COMPLETE")
print("="*50)

# Write results
results_file = script_dir.parent / "results" / "quick_test_results.txt"
results_file.parent.mkdir(exist_ok=True)
with open(results_file, "w") as f:
    f.write(f"encode_ms={elapsed:.3f}\n")
    f.write(f"memory_mb={mem_mb:.2f}\n")
    f.write(f"similarity={sim:.4f}\n")
    f.write(f"all_pass={elapsed < 5 and mem_mb < 10}\n")
print(f"Results written to: {results_file}")

