# Validation Plots

**Date:** 2025-11-28  
**Status:** Generated from Validation Results  

This folder contains visualizations of the key metrics collected during the pre-production validation phase.

---

## 1. Similarity Correlation
**File:** `correlation_float_vs_binary.png`

- **Description:** Scatter plot comparing the ground-truth cosine similarity (float embeddings) against the measured Hamming similarity (binary codes).
- **Source:** `similarity_correlation_test.json`
- **Key Takeaway:** 
  - Pearson correlation **r = 0.9378**
  - High similarity region (Cosine > 0.85) maps consistently to high Hamming similarity.
  - The relationship is monotonic and strong in the critical caching region.
- **Support for GO:** Confirms that 256-bit binary codes preserve semantic information well enough for caching decisions.

## 2. Latency Metrics
**File:** `latency_comparison.png`

- **Description:** Comparison of measured latency vs target thresholds for Encoding and Lookup operations.
- **Source:** `s1_latency_results_v3.json`
- **Key Takeaway:**
  - **Encode:** 0.72 ms (Passes < 1.0 ms target)
  - **Lookup:** 13.48 ms (Exceeds 1.0 ms target, but within 20 ms Python PoC limit)
- **Support for GO:** 
  - Encoding is performant.
  - Lookup latency is the main bottleneck identified, but acceptable for MVP (orders of magnitude faster than LLM calls).
  - Clear optimization path (Numba) exists for Phase 1.

## 3. Memory Usage
**File:** `memory_usage.png`

- **Description:** Actual memory usage for 100,000 cache entries.
- **Source:** `s1_latency_results_v3.json`
- **Key Takeaway:**
  - **Usage:** 3.05 MB
  - **Target:** < 4.0 MB
- **Support for GO:** Validates the high efficiency of binary storage (32x compression compared to float32).

---

## Summary
These plots visually confirm the quantitative findings in the `FINAL_VALIDATION_REPORT.md`. The architecture is fundamentally sound (correlation, memory), with a known optimization task for lookup latency in Phase 1.

