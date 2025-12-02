# 🚀 MANUAL VALIDATION - RUN THESE COMMANDS

The agent's terminal output isn't capturing properly. Please run this in your own terminal:

---

## Step 1: Open a NEW Terminal

**Option A:** Open PowerShell from Start Menu  
**Option B:** In Cursor, press `Ctrl+Shift+`\` to open a new terminal

---

## Step 2: Navigate and Run

Copy-paste this entire block:

```powershell
cd "C:\Users\matte\Desktop\Desktop OLD\AI\Università AI\courses\personal_project\fortress_problem_driven\binary_semantic_cache\validation\poc"
python quick_test.py
```

---

## Step 3: Check Results

After running, you should see output like:

```
==================================================
QUICK VALIDATION TEST
==================================================
Script dir: ...
BinaryLLM path: ...
BinaryLLM exists: True

--- Import Tests ---
✓ RandomProjection, binarize_sign imported
✓ pack_codes imported
✓ numpy imported

--- Encode Test ---
Encode latency: 0.XXX ms per sample
Target: <1ms, Kill: >5ms
Status: ✓ PASS

--- Memory Test ---
100K entries: 3.05MB
Target: <4MB, Kill: >10MB
Status: ✓ PASS

--- Cache Test ---
Similar embeddings similarity: 0.9XXX
Threshold: 0.85
Would hit cache: True

==================================================
QUICK TEST COMPLETE
==================================================
```

---

## Step 4: Report Back

Copy the output and paste it in chat, or just confirm:
- ✅ "All tests passed" 
- ❌ "Test failed: [which one]"

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `python not found` | Activate conda/venv first |
| `ModuleNotFoundError: numpy` | `pip install numpy` |
| `cannot find BinaryLLM` | Check path exists |

---

## Alternative: Run Full Validation

For the complete benchmark suite:

```powershell
cd "C:\Users\matte\Desktop\Desktop OLD\AI\Università AI\courses\personal_project\fortress_problem_driven\binary_semantic_cache\validation\poc"
python run_all_validation.py
```

This runs S0, S1, and S3 together with detailed logging.

