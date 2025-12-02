# MANUAL VALIDATION INSTRUCTIONS

The terminal output capture is not working. Please run these commands manually in a new terminal.

---

## Quick Start

Open a **new PowerShell or CMD terminal** and run:

```powershell
cd "C:\Users\matte\Desktop\Desktop OLD\AI\Università AI\courses\personal_project\fortress_problem_driven\binary_semantic_cache\validation\poc"
python run_all_validation.py
```

---

## Expected Output

You should see something like:

```
============================================================
BINARY SEMANTIC CACHE - VALIDATION RUN
============================================================
Timestamp: 2025-11-28T...
BinaryLLM path: ...
BinaryLLM exists: True

============================================================
STAGE 0: Import Test
============================================================
  ✓ RandomProjection imported
  ✓ binarize_sign imported
  ✓ pack_codes imported
  ✓ unpack_codes imported
  ✓ numpy imported

STAGE 0: PASS

============================================================
STAGE 1: Latency Validation
============================================================

[1/3] Encode Latency Test
  Samples: 1000
  Total time: ~0.5s
  Per sample: ~0.5ms
  Target: <1ms, Kill: >5ms
  Status: ✓ PASS

[2/3] Lookup Latency Test
  Entries: 100000
  Queries: 100
  Per query: ~500-1000µs (Python is slow, C would be faster)
  Target: <100µs, Kill: >1000µs
  Status: ✓ PASS or ⚠ CONDITIONAL

[3/3] Memory Usage Test
  Theoretical: 3.05MB
  Actual: ~3-4MB
  Target: <4MB, Kill: >10MB
  Status: ✓ PASS

STAGE 1: PASS (or CONDITIONAL PASS)

============================================================
STAGE 3: Minimal PoC
============================================================

[1/3] Test: Similar embedding should hit
  Status: ✓ PASS (similarity ~0.9+)

[2/3] Test: Different embedding should miss
  Status: ✓ PASS

[3/3] Test: Exact embedding should hit with high similarity
  Status: ✓ PASS (similarity = 1.0)

STAGE 3: PASS

============================================================
VALIDATION SUMMARY
============================================================
  S0 Import Test: ✓ PASS
  S1 Latency:     ✓ PASS (or ⚠ CONDITIONAL)
  S2 Decision Log: ✓ COMPLETE
  S3 PoC:         ✓ PASS
  S4 Market:      ⏳ PENDING
  S5 Final Review: ⏳ PENDING
============================================================
TECHNICAL VALIDATION: ✓ PASS
============================================================
```

---

## Results Files

After running, check:
- `results/s1_latency_results.json` — Latency benchmark results
- `results/s3_poc_results.json` — PoC test results
- `results/validation_run_*.log` — Full execution log

---

## Kill Triggers

If you see any of these, STOP immediately:

| Issue | Action |
|-------|--------|
| Import errors | Fix Python/BinaryLLM path |
| Encode > 5ms | Architecture broken |
| Memory > 10MB | Model broken |
| PoC tests fail | Debug before proceeding |

---

## After Successful Run

1. Paste the output in chat
2. I'll update the execution log
3. We'll proceed to Stage 4 (Market Signals)

