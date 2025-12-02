"""
Stage 0: BinaryLLM Import Test
Purpose: Validate that BinaryLLM can be used as a standalone library

This is a BLOCKING gate - if imports fail, the Binary Semantic Cache cannot proceed.
"""

import sys
import json
from datetime import datetime
from pathlib import Path

# Add BinaryLLM to path
BINARYLLM_PATH = Path(__file__).parent.parent.parent.parent / "binary_llm"
sys.path.insert(0, str(BINARYLLM_PATH))

print(f"BinaryLLM path: {BINARYLLM_PATH}")
print(f"Path exists: {BINARYLLM_PATH.exists()}")

results = {
    "stage": "S0",
    "name": "BinaryLLM Import Test",
    "timestamp": datetime.now().isoformat(),
    "python_version": sys.version,
    "binaryllm_path": str(BINARYLLM_PATH),
    "tests": [],
    "overall_status": "PASS",
    "notes": ""
}

def test_import(name: str, import_fn):
    """Test a single import and record result."""
    try:
        result = import_fn()
        results["tests"].append({
            "name": name,
            "status": "PASS",
            "error": None
        })
        print(f"  ✓ {name}: PASS")
        return result
    except Exception as e:
        results["tests"].append({
            "name": name,
            "status": "FAIL",
            "error": str(e)
        })
        results["overall_status"] = "FAIL"
        print(f"  ✗ {name}: FAIL - {e}")
        return None

print("\n" + "="*60)
print("STAGE 0: BinaryLLM Import Test")
print("="*60)

# Test 1: Import RandomProjection
print("\n[1/8] Testing RandomProjection import...")
def import_random_projection():
    from src.quantization.binarization import RandomProjection
    return RandomProjection

RandomProjection = test_import("import_RandomProjection", import_random_projection)

# Test 2: Import binarize_sign
print("[2/8] Testing binarize_sign import...")
def import_binarize_sign():
    from src.quantization.binarization import binarize_sign
    return binarize_sign

binarize_sign = test_import("import_binarize_sign", import_binarize_sign)

# Test 3: Import pack_codes
print("[3/8] Testing pack_codes import...")
def import_pack_codes():
    from src.quantization.packing import pack_codes
    return pack_codes

pack_codes = test_import("import_pack_codes", import_pack_codes)

# Test 4: Import unpack_codes
print("[4/8] Testing unpack_codes import...")
def import_unpack_codes():
    from src.quantization.packing import unpack_codes
    return unpack_codes

unpack_codes = test_import("import_unpack_codes", import_unpack_codes)

# Test 5: Create RandomProjection instance
print("[5/8] Testing RandomProjection constructor...")
projection = None
if RandomProjection is not None:
    def create_projection():
        import numpy as np
        proj = RandomProjection(input_dim=384, output_bits=256, seed=42)
        assert hasattr(proj, '_weights'), "Projection matrix not created"
        assert proj._weights.shape == (384, 256), f"Wrong shape: {proj._weights.shape}"
        return proj

    projection = test_import("create_projection", create_projection)

# Test 6: Project embedding
print("[6/8] Testing projection operation...")
if projection is not None:
    def project_embedding():
        import numpy as np
        embedding = np.random.randn(1, 384).astype(np.float32)
        result = projection.project(embedding)
        assert result.shape == (1, 256), f"Expected (1, 256), got {result.shape}"
        return result

    test_import("project_embedding", project_embedding)

# Test 7: Binarize values
print("[7/8] Testing binarize_sign operation...")
if binarize_sign is not None:
    def test_binarize():
        import numpy as np
        result = binarize_sign(np.array([0.5, -0.5, 0.0]))
        expected = np.array([1.0, -1.0, 1.0])  # 0.0 -> +1 by convention
        assert np.allclose(result, expected), f"Expected {expected}, got {result}"
        return result

    test_import("binarize_sign_values", test_binarize)

# Test 8: Pack codes
print("[8/8] Testing pack_codes operation...")
if pack_codes is not None:
    def test_pack():
        import numpy as np
        # Create 64 bits for a single uint64 word
        codes_01 = np.array([[1, 0, 1, 0, 1, 0, 1, 0] * 8], dtype=np.int8)  # 64 bits
        packed = pack_codes(codes_01)
        assert packed.dtype == np.uint64, f"Expected uint64, got {packed.dtype}"
        assert packed.shape == (1, 1), f"Expected (1, 1), got {packed.shape}"
        return packed

    test_import("pack_codes_dtype", test_pack)

# Save results
results_path = Path(__file__).parent.parent / "results" / "s0_import_results.json"
results_path.parent.mkdir(parents=True, exist_ok=True)
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

# Print summary
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
passed = sum(1 for t in results["tests"] if t["status"] == "PASS")
failed = sum(1 for t in results["tests"] if t["status"] == "FAIL")
print(f"  Passed: {passed}/{len(results['tests'])}")
print(f"  Failed: {failed}/{len(results['tests'])}")
print("="*60)
print(f"OVERALL STATUS: {results['overall_status']}")
print("="*60)
print(f"\nResults saved to: {results_path}")

# Exit with appropriate code
sys.exit(0 if results["overall_status"] == "PASS" else 1)

