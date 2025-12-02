# Stage 1: Latency Validation — Expected Results

**Date:** 2025-11-28  
**Status:** 🔄 PENDING EXECUTION

---

## Expected Benchmarks

### 1. Encode Latency

| Metric | Target | Kill Trigger | Expected |
|--------|--------|--------------|----------|
| Per-sample encode | <1ms | >5ms | ~0.1-0.5ms |

**Rationale:**
- RandomProjection: single matrix multiply (384 × 256 floats)
- Binarization: element-wise sign comparison
- Packing: bit manipulation loop (could optimize with vectorization)

### 2. Lookup Latency (100K entries)

| Metric | Target | Kill Trigger | Expected |
|--------|--------|--------------|----------|
| Per-query lookup | <100µs | >1ms | ~200-800µs |

**Rationale:**
- Brute-force Hamming: XOR + popcount over 4 uint64 words × 100K entries
- Python loop is slow; NumPy vectorization helps
- With C/SIMD (future): could hit <10µs

**Note:** Initial Python implementation may exceed target but should pass kill trigger.

### 3. Memory Usage (100K entries)

| Metric | Target | Kill Trigger | Expected |
|--------|--------|--------------|----------|
| Total memory | <4MB | >10MB | ~3.2MB |

**Calculation:**
- 256 bits = 32 bytes per entry
- 100K entries × 32 bytes = 3.2 MB (theoretical)
- Actual: 3.2-4MB (NumPy overhead)

---

## How to Run

From PowerShell:
```powershell
cd "C:\Users\matte\Desktop\Desktop OLD\AI\Università AI\courses\personal_project\fortress_problem_driven\binary_semantic_cache\validation\poc"
python benchmark_latency.py
```

Or double-click: `run_benchmarks.bat`

---

## Kill Trigger Logic

```
IF encode_latency > 5ms:
    STOP — "Encoding too slow for real-time proxy use"
    
IF lookup_latency > 1ms:
    STOP — "Need indexing structure (not brute force)"
    
IF memory > 10MB:
    STOP — "Memory model broken; reconsider code_bits"
```

---

## Pass Criteria

✅ All benchmarks under kill trigger thresholds
✅ Results saved to `s1_latency_results.json`
✅ Proceed to Stage 2 (Decision Log)

