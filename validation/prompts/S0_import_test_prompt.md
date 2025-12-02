# STAGE 0 PROMPT: BinaryLLM Import Test

**Agent Role:** Engineer (Test-First)  
**Duration:** 1 hour max  
**Blocking:** YES — Cannot proceed without passing

---

## CONTEXT

You are validating that BinaryLLM Phase 1 can be used as a standalone library. This is a **blocking gate** — if imports fail, the entire Binary Semantic Cache project cannot proceed.

BinaryLLM is located at:
```
binary_llm/
├── src/
│   ├── quantization/
│   │   ├── binarization.py    # RandomProjection, binarize_sign
│   │   └── packing.py         # pack_codes, unpack_codes
│   └── ...
└── ...
```

---

## YOUR TASK

### Step 1: Create Import Test Script

Write `validation/poc/test_imports.py` that:

1. Attempts to import each required module
2. Tests basic functionality of each import
3. Reports success/failure with clear messages
4. Exits with code 0 (success) or 1 (failure)

### Step 2: Run the Test

Execute the script and record results.

### Step 3: Document Results

Create `validation/results/s0_import_results.json` with:
```json
{
  "stage": "S0",
  "name": "BinaryLLM Import Test",
  "timestamp": "ISO-8601",
  "python_version": "3.x.x",
  "binaryllm_path": "/path/to/binary_llm",
  "tests": [
    {
      "name": "import_RandomProjection",
      "status": "PASS|FAIL",
      "error": null or "error message"
    },
    ...
  ],
  "overall_status": "PASS|FAIL",
  "notes": "any additional observations"
}
```

---

## ACCEPTANCE CRITERIA

| Test | Expected | Status |
|------|----------|--------|
| `from src.quantization.binarization import RandomProjection` | No error | |
| `from src.quantization.binarization import binarize_sign` | No error | |
| `from src.quantization.packing import pack_codes` | No error | |
| `from src.quantization.packing import unpack_codes` | No error | |
| `RandomProjection(384, 256, 42)` creates object | No error | |
| `projection.project(np.random.randn(1, 384))` returns shape (1, 256) | Correct shape | |
| `binarize_sign(np.array([0.5, -0.5]))` returns `[1, -1]` | Correct values | |
| `pack_codes(np.array([[1, 0, 1, 0]]))` returns uint64 array | Correct dtype | |

---

## KILL TRIGGERS

| Trigger | Action |
|---------|--------|
| Any import fails | STOP: Fix PYTHONPATH or install BinaryLLM |
| RandomProjection constructor fails | STOP: Check BinaryLLM version |
| Shape mismatch in projection | STOP: Debug BinaryLLM |

---

## COMMON ISSUES AND FIXES

### Issue 1: ModuleNotFoundError

```
Fix: Add BinaryLLM to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/binary_llm"
```

### Issue 2: Relative Import Error

```
Fix: Run from binary_llm root or install as package
cd binary_llm && pip install -e .
```

### Issue 3: Missing numpy

```
Fix: Install dependencies
pip install numpy scipy
```

---

## OUTPUT TEMPLATE

```python
# validation/poc/test_imports.py
"""
Stage 0: BinaryLLM Import Test
Purpose: Validate that BinaryLLM can be used as a library
"""

import sys
import json
from datetime import datetime
from pathlib import Path

# Add BinaryLLM to path
BINARYLLM_PATH = Path(__file__).parent.parent.parent.parent / "binary_llm"
sys.path.insert(0, str(BINARYLLM_PATH))

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
        return result
    except Exception as e:
        results["tests"].append({
            "name": name,
            "status": "FAIL",
            "error": str(e)
        })
        results["overall_status"] = "FAIL"
        return None

# Test 1: Import RandomProjection
def import_random_projection():
    from src.quantization.binarization import RandomProjection
    return RandomProjection

RandomProjection = test_import("import_RandomProjection", import_random_projection)

# Test 2: Import binarize_sign
def import_binarize_sign():
    from src.quantization.binarization import binarize_sign
    return binarize_sign

binarize_sign = test_import("import_binarize_sign", import_binarize_sign)

# Test 3: Import pack_codes
def import_pack_codes():
    from src.quantization.packing import pack_codes
    return pack_codes

pack_codes = test_import("import_pack_codes", import_pack_codes)

# Test 4: Import unpack_codes
def import_unpack_codes():
    from src.quantization.packing import unpack_codes
    return unpack_codes

unpack_codes = test_import("import_unpack_codes", import_unpack_codes)

# Test 5: Create RandomProjection instance
def create_projection():
    import numpy as np
    proj = RandomProjection(input_dim=384, output_bits=256, seed=42)
    assert hasattr(proj, '_weights'), "Projection matrix not created"
    return proj

projection = test_import("create_projection", create_projection)

# Test 6: Project embedding
def project_embedding():
    import numpy as np
    embedding = np.random.randn(1, 384).astype(np.float32)
    result = projection.project(embedding)
    assert result.shape == (1, 256), f"Expected (1, 256), got {result.shape}"
    return result

test_import("project_embedding", project_embedding)

# Test 7: Binarize values
def test_binarize():
    import numpy as np
    result = binarize_sign(np.array([0.5, -0.5, 0.0]))
    expected = np.array([1.0, -1.0, 1.0])  # 0.0 -> +1 by convention
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    return result

test_import("binarize_sign_values", test_binarize)

# Test 8: Pack codes
def test_pack():
    import numpy as np
    codes_01 = np.array([[1, 0, 1, 0, 1, 0, 1, 0] * 8], dtype=np.int8)  # 64 bits
    packed = pack_codes(codes_01)
    assert packed.dtype == np.uint64, f"Expected uint64, got {packed.dtype}"
    return packed

test_import("pack_codes_dtype", test_pack)

# Save results
results_path = Path(__file__).parent.parent / "results" / "s0_import_results.json"
results_path.parent.mkdir(parents=True, exist_ok=True)
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

# Print summary
print("\n" + "="*60)
print("STAGE 0: BinaryLLM Import Test Results")
print("="*60)
for test in results["tests"]:
    status = "✓" if test["status"] == "PASS" else "✗"
    print(f"  {status} {test['name']}: {test['status']}")
    if test["error"]:
        print(f"      Error: {test['error']}")
print("="*60)
print(f"OVERALL: {results['overall_status']}")
print("="*60)

# Exit with appropriate code
sys.exit(0 if results["overall_status"] == "PASS" else 1)
```

---

## AFTER COMPLETION

If **PASS**: Proceed to Stage 1 (Latency Validation)  
If **FAIL**: Fix the issue before proceeding

---

*This stage validates the foundation. Do not skip or shortcut it.*

