# STAGE 1 PROMPT: Core Latency Validation

**Agent Role:** Engineer (Benchmark-First)  
**Duration:** 2-3 hours  
**Blocking:** YES — Performance assumptions must be validated

---

## CONTEXT

You are measuring the actual performance of core operations that the Binary Semantic Cache depends on. The architecture report claims:

- Projection latency: ~0.16ms (single embedding)
- Hamming scan at 100K: ~5ms
- Memory footprint: 384KB for projection

**These are estimates. You will measure the REAL values.**

---

## YOUR TASK

### Step 1: Create Benchmark Script

Write `validation/poc/bench_latency.py` that measures:

1. **Projection Latency**
   - Single embedding (384-dim → 256-bit)
   - Batch of 1000 embeddings
   - Report: mean, p50, p95, p99

2. **Hamming Distance Scan**
   - Query vs 1K entries
   - Query vs 10K entries
   - Query vs 100K entries
   - Report: mean, p50, p95, p99

3. **Memory Footprint**
   - Projection matrix size
   - Packed codes at 1K, 10K, 100K entries

### Step 2: Run Benchmarks

Execute with multiple warm-up runs and measurement runs.

### Step 3: Document Results

Create `validation/results/s1_latency_results.json` with all measurements.

---

## ACCEPTANCE CRITERIA

| Metric | Target | Acceptable | Fail |
|--------|--------|------------|------|
| Projection (single) | < 0.2ms | < 0.5ms | > 1ms |
| Projection (batch 1000) | < 5ms | < 10ms | > 20ms |
| Hamming scan 1K | < 0.5ms | < 1ms | > 5ms |
| Hamming scan 10K | < 2ms | < 5ms | > 10ms |
| Hamming scan 100K | < 10ms | < 20ms | > 50ms |
| Memory (projection) | < 400KB | < 600KB | > 1MB |
| Memory (100K codes) | < 4MB | < 6MB | > 10MB |

---

## BENCHMARK SCRIPT TEMPLATE

```python
# validation/poc/bench_latency.py
"""
Stage 1: Core Latency Validation
Purpose: Measure actual performance of projection and Hamming operations
"""

import sys
import json
import time
import gc
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Add BinaryLLM to path
BINARYLLM_PATH = Path(__file__).parent.parent.parent.parent / "binary_llm"
sys.path.insert(0, str(BINARYLLM_PATH))

from src.quantization.binarization import RandomProjection, binarize_sign
from src.quantization.packing import pack_codes

# Configuration
EMBEDDING_DIM = 384
CODE_BITS = 256
SEED = 42
N_WARMUP = 10
N_RUNS = 100
N_ENTRIES_LIST = [1_000, 10_000, 100_000]

results: Dict[str, Any] = {
    "stage": "S1",
    "name": "Core Latency Validation",
    "timestamp": datetime.now().isoformat(),
    "config": {
        "embedding_dim": EMBEDDING_DIM,
        "code_bits": CODE_BITS,
        "seed": SEED,
        "n_warmup": N_WARMUP,
        "n_runs": N_RUNS,
    },
    "benchmarks": {},
    "memory": {},
    "overall_status": "PENDING",
    "notes": ""
}

def percentile(data: List[float], p: int) -> float:
    """Calculate percentile."""
    return float(np.percentile(data, p))

def benchmark_projection_single():
    """Benchmark single embedding projection."""
    print("\n[1/6] Benchmarking single projection...")
    
    proj = RandomProjection(input_dim=EMBEDDING_DIM, output_bits=CODE_BITS, seed=SEED)
    embedding = np.random.randn(1, EMBEDDING_DIM).astype(np.float32)
    
    # Warm-up
    for _ in range(N_WARMUP):
        _ = proj.project(embedding)
    
    # Measure
    times = []
    for _ in range(N_RUNS):
        gc.disable()
        start = time.perf_counter()
        _ = proj.project(embedding)
        end = time.perf_counter()
        gc.enable()
        times.append((end - start) * 1000)  # Convert to ms
    
    return {
        "mean_ms": float(np.mean(times)),
        "p50_ms": percentile(times, 50),
        "p95_ms": percentile(times, 95),
        "p99_ms": percentile(times, 99),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }

def benchmark_projection_batch():
    """Benchmark batch embedding projection."""
    print("[2/6] Benchmarking batch projection (1000 embeddings)...")
    
    proj = RandomProjection(input_dim=EMBEDDING_DIM, output_bits=CODE_BITS, seed=SEED)
    embeddings = np.random.randn(1000, EMBEDDING_DIM).astype(np.float32)
    
    # Warm-up
    for _ in range(N_WARMUP):
        _ = proj.project(embeddings)
    
    # Measure
    times = []
    for _ in range(N_RUNS):
        gc.disable()
        start = time.perf_counter()
        _ = proj.project(embeddings)
        end = time.perf_counter()
        gc.enable()
        times.append((end - start) * 1000)
    
    return {
        "batch_size": 1000,
        "mean_ms": float(np.mean(times)),
        "p50_ms": percentile(times, 50),
        "p95_ms": percentile(times, 95),
        "p99_ms": percentile(times, 99),
        "per_embedding_us": float(np.mean(times) * 1000 / 1000),  # μs per embedding
    }

def benchmark_hamming_scan(n_entries: int):
    """Benchmark Hamming distance scan."""
    print(f"[3-5/6] Benchmarking Hamming scan ({n_entries:,} entries)...")
    
    # Generate packed codes
    proj = RandomProjection(input_dim=EMBEDDING_DIM, output_bits=CODE_BITS, seed=SEED)
    embeddings = np.random.randn(n_entries, EMBEDDING_DIM).astype(np.float32)
    projected = proj.project(embeddings)
    codes_pm1 = binarize_sign(projected)
    codes_01 = ((codes_pm1 + 1.0) / 2.0).astype(np.int8)
    packed = pack_codes(codes_01)
    
    # Query
    query = packed[0]
    
    # Warm-up
    for _ in range(min(N_WARMUP, 5)):
        xor = np.bitwise_xor(packed, query)
        # Popcount - this is the critical path
        distances = np.zeros(len(packed), dtype=np.int32)
        for i, row in enumerate(xor):
            distances[i] = sum(bin(w).count('1') for w in row)
    
    # Measure
    times = []
    n_measure = min(N_RUNS, 20) if n_entries >= 100_000 else N_RUNS
    for _ in range(n_measure):
        gc.disable()
        start = time.perf_counter()
        xor = np.bitwise_xor(packed, query)
        distances = np.zeros(len(packed), dtype=np.int32)
        for i, row in enumerate(xor):
            distances[i] = sum(bin(w).count('1') for w in row)
        end = time.perf_counter()
        gc.enable()
        times.append((end - start) * 1000)
    
    return {
        "n_entries": n_entries,
        "mean_ms": float(np.mean(times)),
        "p50_ms": percentile(times, 50),
        "p95_ms": percentile(times, 95),
        "p99_ms": percentile(times, 99),
        "throughput_entries_per_sec": int(n_entries / (np.mean(times) / 1000)),
    }

def measure_memory():
    """Measure memory footprint."""
    print("[6/6] Measuring memory footprint...")
    
    memory = {}
    
    # Projection matrix
    tracemalloc.start()
    proj = RandomProjection(input_dim=EMBEDDING_DIM, output_bits=CODE_BITS, seed=SEED)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory["projection_matrix_bytes"] = proj._weights.nbytes
    memory["projection_matrix_kb"] = proj._weights.nbytes / 1024
    
    # Packed codes at different scales
    for n_entries in N_ENTRIES_LIST:
        proj = RandomProjection(input_dim=EMBEDDING_DIM, output_bits=CODE_BITS, seed=SEED)
        embeddings = np.random.randn(n_entries, EMBEDDING_DIM).astype(np.float32)
        projected = proj.project(embeddings)
        codes_pm1 = binarize_sign(projected)
        codes_01 = ((codes_pm1 + 1.0) / 2.0).astype(np.int8)
        packed = pack_codes(codes_01)
        
        memory[f"packed_codes_{n_entries}_bytes"] = packed.nbytes
        memory[f"packed_codes_{n_entries}_kb"] = packed.nbytes / 1024
        memory[f"bytes_per_entry_{n_entries}"] = packed.nbytes / n_entries
    
    return memory

def evaluate_results() -> str:
    """Determine overall status based on acceptance criteria."""
    status = "PASS"
    notes = []
    
    # Check projection single
    proj_single = results["benchmarks"]["projection_single"]["p99_ms"]
    if proj_single > 1.0:
        status = "FAIL"
        notes.append(f"Projection single p99 ({proj_single:.2f}ms) > 1ms FAIL threshold")
    elif proj_single > 0.5:
        if status == "PASS":
            status = "ACCEPTABLE"
        notes.append(f"Projection single p99 ({proj_single:.2f}ms) > 0.5ms target")
    
    # Check Hamming 100K
    if "hamming_scan_100000" in results["benchmarks"]:
        hamming_100k = results["benchmarks"]["hamming_scan_100000"]["p99_ms"]
        if hamming_100k > 50:
            status = "FAIL"
            notes.append(f"Hamming 100K p99 ({hamming_100k:.2f}ms) > 50ms FAIL threshold")
        elif hamming_100k > 20:
            if status == "PASS":
                status = "ACCEPTABLE"
            notes.append(f"Hamming 100K p99 ({hamming_100k:.2f}ms) > 20ms target")
    
    # Check memory
    proj_mem = results["memory"]["projection_matrix_kb"]
    if proj_mem > 1024:
        status = "FAIL"
        notes.append(f"Projection memory ({proj_mem:.0f}KB) > 1MB FAIL threshold")
    elif proj_mem > 600:
        if status == "PASS":
            status = "ACCEPTABLE"
        notes.append(f"Projection memory ({proj_mem:.0f}KB) > 600KB target")
    
    results["notes"] = "; ".join(notes) if notes else "All metrics within target"
    return status

def main():
    print("="*60)
    print("STAGE 1: Core Latency Validation")
    print("="*60)
    
    # Run benchmarks
    results["benchmarks"]["projection_single"] = benchmark_projection_single()
    results["benchmarks"]["projection_batch"] = benchmark_projection_batch()
    
    for n_entries in N_ENTRIES_LIST:
        results["benchmarks"][f"hamming_scan_{n_entries}"] = benchmark_hamming_scan(n_entries)
    
    results["memory"] = measure_memory()
    
    # Evaluate
    results["overall_status"] = evaluate_results()
    
    # Save results
    results_path = Path(__file__).parent.parent / "results" / "s1_latency_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print("\nProjection Latency:")
    ps = results["benchmarks"]["projection_single"]
    print(f"  Single:  mean={ps['mean_ms']:.3f}ms, p50={ps['p50_ms']:.3f}ms, p99={ps['p99_ms']:.3f}ms")
    pb = results["benchmarks"]["projection_batch"]
    print(f"  Batch:   mean={pb['mean_ms']:.3f}ms for {pb['batch_size']} ({pb['per_embedding_us']:.2f}μs/embedding)")
    
    print("\nHamming Scan Latency:")
    for n in N_ENTRIES_LIST:
        hs = results["benchmarks"][f"hamming_scan_{n}"]
        print(f"  {n:>7,} entries: mean={hs['mean_ms']:.2f}ms, p99={hs['p99_ms']:.2f}ms, {hs['throughput_entries_per_sec']:,}/sec")
    
    print("\nMemory Footprint:")
    print(f"  Projection matrix: {results['memory']['projection_matrix_kb']:.1f} KB")
    for n in N_ENTRIES_LIST:
        print(f"  {n:>7,} codes: {results['memory'][f'packed_codes_{n}_kb']:.1f} KB ({results['memory'][f'bytes_per_entry_{n}']:.1f} bytes/entry)")
    
    print("\n" + "="*60)
    print(f"OVERALL STATUS: {results['overall_status']}")
    if results["notes"]:
        print(f"NOTES: {results['notes']}")
    print("="*60)
    
    return 0 if results["overall_status"] in ["PASS", "ACCEPTABLE"] else 1

if __name__ == "__main__":
    sys.exit(main())
```

---

## KILL TRIGGERS

| Trigger | Threshold | Action |
|---------|-----------|--------|
| Projection single p99 > 1ms | Fail | STOP: Investigate RandomProjection |
| Hamming 100K p99 > 50ms | Fail | REDESIGN: Need SIMD optimization |
| Projection memory > 1MB | Fail | INVESTIGATE: Unexpected allocation |

---

## CRITICAL WARNING: Hamming Popcount

The Python `bin(w).count('1')` method is **extremely slow**. Real production code will need:

1. **NumPy vectorized popcount** (if available)
2. **Lookup table** (256 entries for byte-wise popcount)
3. **SIMD/C extension** (for maximum performance)

If the benchmark shows Hamming scan > 20ms at 100K, you'll need optimization in Phase 1.

---

## AFTER COMPLETION

If **PASS**: Proceed to Stage 2 (Decision Log)  
If **ACCEPTABLE**: Proceed with documented concerns  
If **FAIL**: Investigate root cause before proceeding

---

*This stage validates the performance assumptions that your entire architecture depends on.*

