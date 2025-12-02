# BINARY SEMANTIC CACHE — MILITARY-GRADE ENGINEERING PLAN

**Version:** 1.0  
**Date:** November 28, 2025  
**Status:** APPROVED FOR EXECUTION  
**Author:** Solution Synthesizer + Hostile Reviewer  
**Total Duration:** 9 weeks (8 weeks + 1 week buffer)  
**Effort:** ~25-30 hours/week (solo developer)

---

## Executive Summary

This document provides a **binding engineering specification** for the Binary Semantic Cache project. Every phase includes:

- YAML specification with exact deliverables
- Task-level breakdown with hour estimates
- Kill-switches with measurable triggers
- Test matrices with coverage requirements
- Go/No-Go checklists for phase transitions

**WARNING:** This plan is designed for strict execution. Deviations require explicit justification and updated kill-switch assessment.

---

## Table of Contents

1. [Phase 1: Foundation & BinaryLLM Integration](#phase-1-foundation--binaryllm-integration)
2. [Phase 2: Core Cache Data Structures](#phase-2-core-cache-data-structures)
3. [Phase 3: Diversity Eviction Engine](#phase-3-diversity-eviction-engine)
4. [Phase 4: LLM API Proxy Layer](#phase-4-llm-api-proxy-layer)
5. [Phase 5: Benchmarking & Validation](#phase-5-benchmarking--validation)
6. [Phase 6: Documentation & Polish](#phase-6-documentation--polish)
7. [Risk Register](#risk-register)
8. [Go/No-Go Checklists](#gono-go-checklists)
9. [Test Strategy](#test-strategy)
10. [Hostile Review Appendix](#hostile-review-appendix)

---

# PHASE 1: Foundation & BinaryLLM Integration

```yaml
# ============================================================================
# PHASE 1: FOUNDATION & BINARYLLM INTEGRATION
# ============================================================================

id: P1
name: "Foundation & BinaryLLM Integration"
codename: "BRIDGE"
duration:
  estimated_weeks: 2
  buffer_weeks: 0.5
  hard_deadline: "Week 2.5 of project"
  total_hours: 50-60

# Dependencies
depends_on:
  - phase_id: "P0"
    artifacts_required:
      - "binary_llm/src/quantization/binarization.py"
      - "binary_llm/src/quantization/packing.py"
      - "binary_llm/tests/ (157 tests passing)"
    verification: "cd binary_llm && pytest tests/ -q --tb=no"

# Entry Criteria (ALL must be true to start)
entry_criteria:
  - id: "EC-P1-01"
    condition: "BinaryLLM Phase 1 tests pass (157 tests)"
    verification_command: "cd binary_llm && pytest tests/ -q"
  - id: "EC-P1-02"
    condition: "Python 3.10+ environment with numpy, scipy available"
    verification_command: "python -c 'import numpy; import scipy; print(numpy.__version__)'"
  - id: "EC-P1-03"
    condition: "Git repository initialized for semantic_cache"
    verification_command: "git status"
  - id: "EC-P1-04"
    condition: "Architecture report v1 reviewed and approved"
    verification_command: "ls iteration_9/research/architecture_report_v1.md"

# Deliverables
deliverables:
  - name: "binary_adapter.py"
    type: code
    path: "semantic_cache/binary/adapter.py"
    description: "Thread-safe wrapper for BinaryLLM projection and binarization"
    lines_of_code_estimate: 180
    complexity: medium
    public_api: |
      from dataclasses import dataclass
      from typing import Optional
      import numpy as np
      
      @dataclass
      class BinaryAdapterConfig:
          """Configuration for binary adapter."""
          embedding_dim: int
          code_bits: int  # 32, 64, 128, 256, 512
          seed: int = 42
          normalize_input: bool = True
      
      class BinaryAdapter:
          """Thread-safe adapter for BinaryLLM Phase 1 projection."""
          
          def __init__(self, config: BinaryAdapterConfig) -> None:
              """Initialize with frozen projection matrix."""
              ...
          
          @property
          def projection_matrix_bytes(self) -> int:
              """Memory footprint of projection matrix."""
              ...
          
          def encode_single(self, embedding: np.ndarray) -> np.ndarray:
              """
              Encode single embedding to packed binary code.
              
              Args:
                  embedding: Shape (embedding_dim,), dtype float32
              
              Returns:
                  Shape (code_bits // 64,), dtype uint64
              """
              ...
          
          def encode_batch(self, embeddings: np.ndarray) -> np.ndarray:
              """
              Encode batch of embeddings to packed binary codes.
              
              Args:
                  embeddings: Shape (n, embedding_dim), dtype float32
              
              Returns:
                  Shape (n, code_bits // 64), dtype uint64
              """
              ...
          
          def decode_to_01(self, packed: np.ndarray) -> np.ndarray:
              """
              Unpack binary codes to {0,1} representation.
              
              Args:
                  packed: Shape (n, code_bits // 64), dtype uint64
              
              Returns:
                  Shape (n, code_bits), dtype uint8
              """
              ...

  - name: "hamming_ops.py"
    type: code
    path: "semantic_cache/binary/hamming_ops.py"
    description: "Optimized Hamming distance operations for packed uint64 codes"
    lines_of_code_estimate: 150
    complexity: medium
    public_api: |
      import numpy as np
      from typing import Tuple, Optional
      
      def popcount_array(words: np.ndarray) -> np.ndarray:
          """
          Count set bits in each uint64 word.
          
          Args:
              words: Shape (...), dtype uint64
          
          Returns:
              Shape (...), dtype int32 - bit counts per word
          """
          ...
      
      def hamming_distance(
          code_a: np.ndarray,
          code_b: np.ndarray
      ) -> int:
          """
          Compute Hamming distance between two packed codes.
          
          Args:
              code_a: Shape (n_words,), dtype uint64
              code_b: Shape (n_words,), dtype uint64
          
          Returns:
              Integer Hamming distance (number of differing bits)
          """
          ...
      
      def hamming_distance_batch(
          query: np.ndarray,
          codes: np.ndarray
      ) -> np.ndarray:
          """
          Compute Hamming distances from query to all codes.
          
          Args:
              query: Shape (n_words,), dtype uint64
              codes: Shape (n_entries, n_words), dtype uint64
          
          Returns:
              Shape (n_entries,), dtype int32 - distances
          """
          ...
      
      def find_nearest(
          query: np.ndarray,
          codes: np.ndarray,
          threshold: float,
          code_bits: int
      ) -> Tuple[Optional[int], Optional[float]]:
          """
          Find nearest code within threshold.
          
          Args:
              query: Shape (n_words,), dtype uint64
              codes: Shape (n_entries, n_words), dtype uint64
              threshold: Maximum normalized Hamming distance (0-1)
              code_bits: Total number of bits for normalization
          
          Returns:
              (index, normalized_distance) if found, else (None, None)
          """
          ...
      
      def find_k_nearest(
          query: np.ndarray,
          codes: np.ndarray,
          k: int,
          code_bits: int
      ) -> Tuple[np.ndarray, np.ndarray]:
          """
          Find k nearest codes.
          
          Args:
              query: Shape (n_words,), dtype uint64
              codes: Shape (n_entries, n_words), dtype uint64
              k: Number of nearest neighbors
              code_bits: Total number of bits for normalization
          
          Returns:
              (indices, normalized_distances) - shapes (k,), (k,)
          """
          ...

  - name: "test_binary_adapter.py"
    type: test
    path: "tests/test_binary_adapter.py"
    description: "Unit tests for BinaryAdapter including determinism and edge cases"
    lines_of_code_estimate: 300
    complexity: medium
    test_count_estimate: 25

  - name: "test_hamming_ops.py"
    type: test
    path: "tests/test_hamming_ops.py"
    description: "Unit tests for Hamming distance operations"
    lines_of_code_estimate: 200
    complexity: medium
    test_count_estimate: 20

  - name: "bench_p1_integration.py"
    type: benchmark
    path: "benchmarks/bench_p1_integration.py"
    description: "Comprehensive latency and memory benchmarks"
    lines_of_code_estimate: 250
    complexity: medium
    output_format: JSON

  - name: "conftest.py"
    type: config
    path: "tests/conftest.py"
    description: "Pytest fixtures for synthetic embeddings and codes"
    lines_of_code_estimate: 100
    complexity: low

  - name: "pyproject.toml"
    type: config
    path: "semantic_cache/pyproject.toml"
    description: "Project configuration with dependencies"
    lines_of_code_estimate: 80
    complexity: low

# Task Breakdown (Gantt-style)
tasks:
  - id: "T-P1-01"
    name: "Project skeleton setup"
    duration_hours: 3
    depends_on: []
    deliverables: ["pyproject.toml", "conftest.py"]
    definition_of_done:
      - "pyproject.toml created with all dependencies"
      - "pytest runs with 0 tests (no errors)"
      - "Directory structure matches architecture spec"
      - "Git initialized with .gitignore"
    risks:
      - risk: "Dependency conflicts with BinaryLLM"
        likelihood: low
        mitigation: "Use same numpy/scipy versions as BinaryLLM"

  - id: "T-P1-02"
    name: "Import BinaryLLM modules"
    duration_hours: 2
    depends_on: ["T-P1-01"]
    deliverables: []
    definition_of_done:
      - "Can import RandomProjection from binary_llm"
      - "Can import pack_codes, unpack_codes from binary_llm"
      - "Import path documented in README"
    risks:
      - risk: "BinaryLLM not installed as package"
        likelihood: medium
        mitigation: "Add binary_llm to PYTHONPATH or pip install -e"

  - id: "T-P1-03"
    name: "BinaryAdapter core implementation"
    duration_hours: 8
    depends_on: ["T-P1-02"]
    deliverables: ["binary_adapter.py"]
    definition_of_done:
      - "BinaryAdapterConfig dataclass implemented"
      - "BinaryAdapter.__init__ creates RandomProjection"
      - "encode_single works for single embedding"
      - "encode_batch works for batch of embeddings"
      - "decode_to_01 correctly unpacks codes"
      - "All methods have type hints and docstrings"
    risks:
      - risk: "Thread safety issues with projection matrix"
        likelihood: low
        mitigation: "Projection matrix is read-only after init; document thread safety"

  - id: "T-P1-04"
    name: "BinaryAdapter tests"
    duration_hours: 6
    depends_on: ["T-P1-03"]
    deliverables: ["test_binary_adapter.py"]
    definition_of_done:
      - "Test encode_single with random embedding"
      - "Test encode_batch with batch of embeddings"
      - "Test determinism: same seed → same output"
      - "Test decode_to_01 round-trip"
      - "Test invalid input handling (wrong shape, NaN, Inf)"
      - "Test memory footprint measurement"
      - "All tests pass with pytest"
    risks:
      - risk: "Golden test data mismatch with BinaryLLM"
        likelihood: medium
        mitigation: "Use BinaryLLM's golden test data as reference"

  - id: "T-P1-05"
    name: "Hamming operations implementation"
    duration_hours: 6
    depends_on: ["T-P1-02"]
    deliverables: ["hamming_ops.py"]
    definition_of_done:
      - "popcount_array implemented with lookup table"
      - "hamming_distance works for two codes"
      - "hamming_distance_batch works for query vs many codes"
      - "find_nearest returns correct index and distance"
      - "find_k_nearest returns k nearest neighbors"
      - "All functions handle edge cases (empty array, single entry)"
    risks:
      - risk: "Performance bottleneck in popcount"
        likelihood: medium
        mitigation: "Use numpy vectorization; profile and optimize if needed"

  - id: "T-P1-06"
    name: "Hamming operations tests"
    duration_hours: 4
    depends_on: ["T-P1-05"]
    deliverables: ["test_hamming_ops.py"]
    definition_of_done:
      - "Test popcount_array with known bit patterns"
      - "Test hamming_distance with identical codes (expect 0)"
      - "Test hamming_distance with opposite codes (expect code_bits)"
      - "Test hamming_distance_batch correctness"
      - "Test find_nearest with threshold"
      - "Test find_k_nearest with k > n_entries"
      - "Parametrized tests for code_bits in {64, 128, 256, 512}"
    risks:
      - risk: "Edge cases in bit arithmetic"
        likelihood: medium
        mitigation: "Extensive parametrized testing with known values"

  - id: "T-P1-07"
    name: "Benchmark suite implementation"
    duration_hours: 6
    depends_on: ["T-P1-04", "T-P1-06"]
    deliverables: ["bench_p1_integration.py"]
    definition_of_done:
      - "Benchmark projection latency for dims [384, 768, 1536]"
      - "Benchmark projection latency for code_bits [64, 128, 256, 512]"
      - "Benchmark batch projection for batch_sizes [1, 10, 100, 1000]"
      - "Benchmark Hamming scan for n_entries [1K, 10K, 100K]"
      - "Measure memory footprint of projection matrix"
      - "Output results as JSON for analysis"
      - "Report p50, p95, p99 latencies"
    risks:
      - risk: "Benchmark results vary across runs"
        likelihood: high
        mitigation: "Use median of 10 runs; warm up before measurement"

  - id: "T-P1-08"
    name: "Run benchmarks and validate"
    duration_hours: 4
    depends_on: ["T-P1-07"]
    deliverables: []
    definition_of_done:
      - "Projection latency < 1ms for 384-dim → 256-bit (kill-switch KS-P1-01)"
      - "Hamming scan < 10ms for 100K entries (kill-switch KS-P1-02)"
      - "Memory footprint documented"
      - "Benchmark results saved to benchmarks/results/p1_integration.json"
    risks:
      - risk: "Kill-switch triggered"
        likelihood: low
        mitigation: "If triggered, investigate and optimize before continuing"

  - id: "T-P1-09"
    name: "Integration test with real embeddings"
    duration_hours: 4
    depends_on: ["T-P1-08"]
    deliverables: []
    definition_of_done:
      - "Download 1000 embeddings from OpenAI text-embedding-3-small"
      - "Run through BinaryAdapter"
      - "Verify Hamming distances correlate with cosine distances"
      - "Spearman correlation > 0.90 (sanity check)"
    risks:
      - risk: "OpenAI API rate limits"
        likelihood: medium
        mitigation: "Cache embeddings locally after first run"

  - id: "T-P1-10"
    name: "Documentation and phase wrap-up"
    duration_hours: 3
    depends_on: ["T-P1-09"]
    deliverables: []
    definition_of_done:
      - "README.md updated with Phase 1 status"
      - "API documentation in docstrings"
      - "Benchmark results documented"
      - "Go/No-Go checklist completed"
    risks: []

# Kill Switches
kill_switches:
  - id: "KS-P1-01"
    name: "Projection Latency"
    trigger_condition: "Single 384-dim embedding projection > 1ms (p99)"
    measurement_method: "bench_p1_integration.py --metric=projection_latency"
    action_if_triggered: "STOP: Profile RandomProjection; check for memory allocation in hot path"
    recovery_cost_weeks: 1
    
  - id: "KS-P1-02"
    name: "Hamming Scan Latency"
    trigger_condition: "Hamming scan of 100K entries > 10ms (p99)"
    measurement_method: "bench_p1_integration.py --metric=hamming_scan"
    action_if_triggered: "REDESIGN: Add SIMD optimization or reduce target scale to 50K"
    recovery_cost_weeks: 1
    
  - id: "KS-P1-03"
    name: "Memory Footprint"
    trigger_condition: "Projection matrix + 10K codes > 50MB"
    measurement_method: "tracemalloc in bench_p1_integration.py"
    action_if_triggered: "PIVOT: Use smaller code_bits (128 instead of 256)"
    recovery_cost_weeks: 0.5
    
  - id: "KS-P1-04"
    name: "Determinism Failure"
    trigger_condition: "Same seed produces different codes across runs"
    measurement_method: "test_binary_adapter.py::test_determinism"
    action_if_triggered: "STOP: Debug RNG seeding in BinaryLLM integration"
    recovery_cost_weeks: 0.5

# Exit Criteria (ALL must be true to complete phase)
exit_criteria:
  - id: "XC-P1-01"
    condition: "All unit tests pass (≥45 tests)"
    verification_command: "pytest tests/ -v --tb=short"
  - id: "XC-P1-02"
    condition: "Projection latency < 0.5ms for single 384-dim embedding (p50)"
    verification_command: "python benchmarks/bench_p1_integration.py --metric=projection_latency"
  - id: "XC-P1-03"
    condition: "Hamming scan of 10K entries < 1ms (p50)"
    verification_command: "python benchmarks/bench_p1_integration.py --metric=hamming_scan --n_entries=10000"
  - id: "XC-P1-04"
    condition: "Determinism verified: 10 runs with same seed produce identical codes"
    verification_command: "pytest tests/test_binary_adapter.py::test_determinism -v"
  - id: "XC-P1-05"
    condition: "Memory footprint documented and within limits"
    verification_command: "python benchmarks/bench_p1_integration.py --metric=memory"
  - id: "XC-P1-06"
    condition: "Spearman correlation > 0.90 on real embeddings"
    verification_command: "python benchmarks/bench_p1_integration.py --metric=correlation"

# Integration Points with BinaryLLM Phase 1
binaryllm_integration:
  modules_used:
    - name: "src.quantization.binarization"
      functions: ["RandomProjection", "binarize_sign"]
      thread_safety: "yes (read-only after __post_init__)"
      import_path: "from src.quantization.binarization import RandomProjection, binarize_sign"
    - name: "src.quantization.packing"
      functions: ["pack_codes", "unpack_codes"]
      thread_safety: "yes (pure functions, no state)"
      import_path: "from src.quantization.packing import pack_codes, unpack_codes"
  invariants_to_preserve:
    - "INV-04: LSB-first packing layout (frozen for GPU kernels)"
    - "INV-05: Same seed ⇒ same results"
    - "INV-06: binary_codes shapes (N, code_bits)"
    - "Q-01: Sign convention: x ≥ 0 → +1, x < 0 → -1"
  potential_conflicts:
    - conflict: "BinaryLLM uses int8 for codes_01, adapter may need uint8"
      resolution: "Cast to uint8 after binarization: codes_01.astype(np.uint8)"
    - conflict: "RandomProjection stores float32 weights"
      resolution: "Accept float32; do not convert to float64"
    - conflict: "BinaryLLM expects 2D input (N, dim) even for single embedding"
      resolution: "Reshape single embedding to (1, dim) in encode_single"

# Test Matrix
test_matrix:
  unit_tests:
    - name: "test_adapter_init"
      covers: "BinaryAdapter initialization"
      edge_cases: ["invalid code_bits", "zero embedding_dim", "negative seed"]
    - name: "test_encode_single"
      covers: "Single embedding encoding"
      edge_cases: ["zero vector", "unit vector", "random vector"]
    - name: "test_encode_batch"
      covers: "Batch embedding encoding"
      edge_cases: ["batch_size=1", "batch_size=1000", "empty batch"]
    - name: "test_determinism"
      covers: "Same seed → same output"
      edge_cases: ["multiple runs", "different seeds produce different output"]
    - name: "test_decode_roundtrip"
      covers: "pack → unpack produces original bits"
      edge_cases: ["all zeros", "all ones", "alternating"]
    - name: "test_popcount_array"
      covers: "Bit counting correctness"
      edge_cases: ["0x0", "0xFFFFFFFFFFFFFFFF", "known patterns"]
    - name: "test_hamming_distance"
      covers: "Distance between two codes"
      edge_cases: ["identical codes", "opposite codes", "one bit different"]
    - name: "test_hamming_batch"
      covers: "Vectorized distance computation"
      edge_cases: ["single entry", "10K entries"]
    - name: "test_find_nearest"
      covers: "Threshold-based nearest neighbor"
      edge_cases: ["no match above threshold", "exact match", "multiple matches"]
    - name: "test_find_k_nearest"
      covers: "Top-k retrieval"
      edge_cases: ["k > n_entries", "k = 1", "ties in distance"]
  integration_tests:
    - name: "test_integration_binaryllm"
      components_tested: ["BinaryAdapter", "RandomProjection", "pack_codes"]
      mock_requirements: ["None - uses real BinaryLLM modules"]
    - name: "test_integration_real_embeddings"
      components_tested: ["BinaryAdapter", "OpenAI API"]
      mock_requirements: ["Cache embeddings locally after first run"]
  performance_tests:
    - name: "bench_projection_latency"
      target_metric: "projection_time_ms"
      acceptable_range: "0.05-0.5 ms"
      failure_threshold: "1.0 ms (p99)"
    - name: "bench_hamming_scan"
      target_metric: "scan_time_ms"
      acceptable_range: "0.5-5 ms (100K entries)"
      failure_threshold: "10.0 ms (p99)"
    - name: "bench_memory_footprint"
      target_metric: "memory_bytes"
      acceptable_range: "384KB-500KB (projection matrix)"
      failure_threshold: "1MB"
```

---

# PHASE 2: Core Cache Data Structures

```yaml
# ============================================================================
# PHASE 2: CORE CACHE DATA STRUCTURES
# ============================================================================

id: P2
name: "Core Cache Data Structures"
codename: "VAULT"
duration:
  estimated_weeks: 1.5
  buffer_weeks: 0.5
  hard_deadline: "Week 4 of project"
  total_hours: 40-50

# Dependencies
depends_on:
  - phase_id: "P1"
    artifacts_required:
      - "semantic_cache/binary/adapter.py"
      - "semantic_cache/binary/hamming_ops.py"
      - "tests/test_binary_adapter.py (passing)"
      - "tests/test_hamming_ops.py (passing)"
      - "benchmarks/results/p1_integration.json"
    verification: "pytest tests/ -v && python benchmarks/bench_p1_integration.py"

# Entry Criteria
entry_criteria:
  - id: "EC-P2-01"
    condition: "Phase 1 all exit criteria pass"
    verification_command: "Run P1 exit criteria verification script"
  - id: "EC-P2-02"
    condition: "Projection latency < 0.5ms confirmed"
    verification_command: "cat benchmarks/results/p1_integration.json | jq '.projection_latency_p50'"
  - id: "EC-P2-03"
    condition: "Hamming scan < 5ms at 10K entries confirmed"
    verification_command: "cat benchmarks/results/p1_integration.json | jq '.hamming_scan_10k_p50'"

# Deliverables
deliverables:
  - name: "config.py"
    type: code
    path: "semantic_cache/core/config.py"
    description: "Cache configuration dataclass with validation"
    lines_of_code_estimate: 100
    complexity: low
    public_api: |
      from dataclasses import dataclass, field
      from typing import Literal
      
      @dataclass(frozen=True)
      class CacheConfig:
          """Immutable cache configuration with validation."""
          
          # Capacity
          max_entries: int = 100_000
          
          # Binary encoding
          code_bits: int = 256  # Must be in {32, 64, 128, 256, 512}
          embedding_dim: int = 384
          seed: int = 42
          
          # Similarity
          similarity_threshold: float = 0.85  # Normalized Hamming: 1 - (distance / code_bits)
          
          # Eviction
          eviction_strategy: Literal["lru", "diversity", "hybrid"] = "hybrid"
          eviction_batch_size: int = 100
          
          # Storage
          response_compression: bool = True  # zstd compression
          max_response_bytes: int = 1_000_000  # 1MB per response
          
          def __post_init__(self) -> None:
              """Validate configuration."""
              ...
          
          @property
          def n_words(self) -> int:
              """Number of uint64 words per code."""
              return self.code_bits // 64

  - name: "entry.py"
    type: code
    path: "semantic_cache/core/entry.py"
    description: "Cache entry dataclass with metadata"
    lines_of_code_estimate: 120
    complexity: low
    public_api: |
      from dataclasses import dataclass
      from typing import Optional
      import numpy as np
      
      @dataclass(frozen=True)
      class CacheEntry:
          """Immutable cache entry with binary code and metadata."""
          
          entry_id: int                    # Unique identifier
          binary_code: np.ndarray          # Shape: (n_words,), dtype: uint64
          request_hash: bytes              # SHA256 of request payload (32 bytes)
          response: bytes                  # Compressed response
          created_at: float                # Unix timestamp
          last_accessed: float             # Unix timestamp
          access_count: int                # Number of hits
          cluster_label: int = 0           # For diversity eviction
          
          def size_bytes(self) -> int:
              """Total memory footprint of this entry."""
              ...
          
          def with_access(self) -> "CacheEntry":
              """Return new entry with updated access time and count."""
              ...

      @dataclass
      class CacheIndex:
          """Mutable in-memory index for fast binary lookup."""
          
          codes: np.ndarray               # Shape: (max_entries, n_words), dtype: uint64
          entry_ids: np.ndarray           # Shape: (max_entries,), dtype: int64
          timestamps: np.ndarray          # Shape: (max_entries,), dtype: float64
          access_counts: np.ndarray       # Shape: (max_entries,), dtype: int32
          cluster_labels: np.ndarray      # Shape: (max_entries,), dtype: int8
          n_entries: int                  # Current number of entries
          
          @classmethod
          def create(cls, max_entries: int, n_words: int) -> "CacheIndex":
              """Allocate index arrays."""
              ...
          
          def add(self, entry_id: int, code: np.ndarray, timestamp: float) -> int:
              """Add entry to index, return slot index."""
              ...
          
          def remove(self, slot_index: int) -> None:
              """Remove entry from index."""
              ...

  - name: "cache.py"
    type: code
    path: "semantic_cache/core/cache.py"
    description: "Core cache implementation with insert/lookup/evict"
    lines_of_code_estimate: 450
    complexity: high
    public_api: |
      from typing import Optional, List, Tuple
      import numpy as np
      from .config import CacheConfig
      from .entry import CacheEntry, CacheIndex
      from .stats import CacheStats
      from ..binary.adapter import BinaryAdapter
      
      class BinaryCache:
          """Core semantic cache with binary similarity lookup."""
          
          def __init__(self, config: CacheConfig) -> None:
              """Initialize cache with configuration."""
              ...
          
          @property
          def stats(self) -> CacheStats:
              """Current cache statistics."""
              ...
          
          def lookup(
              self,
              query_code: np.ndarray,
              threshold: Optional[float] = None
          ) -> Optional[CacheEntry]:
              """
              Find closest cached entry above similarity threshold.
              
              Args:
                  query_code: Shape (n_words,), dtype uint64
                  threshold: Override default similarity_threshold
              
              Returns:
                  CacheEntry if found, None if no match above threshold
              """
              ...
          
          def lookup_by_embedding(
              self,
              embedding: np.ndarray,
              threshold: Optional[float] = None
          ) -> Optional[CacheEntry]:
              """
              Encode embedding and lookup.
              
              Convenience method that combines encoding and lookup.
              """
              ...
          
          def insert(
              self,
              embedding: np.ndarray,
              request_hash: bytes,
              response: bytes
          ) -> Tuple[CacheEntry, bool]:
              """
              Insert new entry, evicting if necessary.
              
              Args:
                  embedding: Original float embedding
                  request_hash: SHA256 of request
                  response: Response bytes (will be compressed if config.response_compression)
              
              Returns:
                  (entry, evicted) - the new entry and whether eviction occurred
              """
              ...
          
          def evict(self, n: int = 1) -> List[CacheEntry]:
              """
              Evict n entries using configured strategy.
              
              Returns list of evicted entries.
              """
              ...
          
          def clear(self) -> int:
              """Clear all entries. Returns count of cleared entries."""
              ...
          
          def get_entry(self, entry_id: int) -> Optional[CacheEntry]:
              """Retrieve entry by ID."""
              ...
          
          def get_all_codes(self) -> np.ndarray:
              """Get all binary codes for diversity analysis."""
              ...

  - name: "stats.py"
    type: code
    path: "semantic_cache/core/stats.py"
    description: "Cache statistics tracking"
    lines_of_code_estimate: 80
    complexity: low
    public_api: |
      from dataclasses import dataclass, field
      from typing import List
      import time
      
      @dataclass
      class CacheStats:
          """Runtime cache statistics."""
          
          hits: int = 0
          misses: int = 0
          evictions: int = 0
          total_entries: int = 0
          memory_bytes: int = 0
          
          # Latency tracking (rolling window)
          lookup_latencies_ms: List[float] = field(default_factory=list)
          insert_latencies_ms: List[float] = field(default_factory=list)
          
          @property
          def hit_rate(self) -> float:
              """Cache hit rate as fraction."""
              ...
          
          @property
          def avg_lookup_ms(self) -> float:
              """Average lookup latency."""
              ...
          
          def record_lookup(self, hit: bool, latency_ms: float) -> None:
              """Record a lookup operation."""
              ...
          
          def to_dict(self) -> dict:
              """Export as dictionary for JSON serialization."""
              ...

  - name: "storage.py"
    type: code
    path: "semantic_cache/core/storage.py"
    description: "Response storage with optional compression"
    lines_of_code_estimate: 100
    complexity: medium
    public_api: |
      from typing import Dict
      import zstandard as zstd
      
      class ResponseStorage:
          """In-memory storage for response payloads."""
          
          def __init__(self, compression: bool = True, max_size_bytes: int = 1_000_000):
              ...
          
          def store(self, entry_id: int, response: bytes) -> int:
              """Store response, return stored size in bytes."""
              ...
          
          def retrieve(self, entry_id: int) -> bytes:
              """Retrieve and decompress response."""
              ...
          
          def remove(self, entry_id: int) -> None:
              """Remove stored response."""
              ...
          
          def memory_usage(self) -> int:
              """Total memory used by stored responses."""
              ...

  - name: "test_cache.py"
    type: test
    path: "tests/test_cache.py"
    description: "Comprehensive tests for cache operations"
    lines_of_code_estimate: 500
    complexity: high
    test_count_estimate: 40

# Task Breakdown
tasks:
  - id: "T-P2-01"
    name: "CacheConfig implementation"
    duration_hours: 2
    depends_on: []
    deliverables: ["config.py"]
    definition_of_done:
      - "CacheConfig dataclass with all fields"
      - "__post_init__ validation for code_bits, thresholds"
      - "Immutable (frozen=True)"
      - "n_words property computed correctly"

  - id: "T-P2-02"
    name: "CacheEntry and CacheIndex"
    duration_hours: 4
    depends_on: ["T-P2-01"]
    deliverables: ["entry.py"]
    definition_of_done:
      - "CacheEntry immutable dataclass"
      - "CacheIndex mutable with preallocated arrays"
      - "add() and remove() methods work correctly"
      - "size_bytes() calculated accurately"

  - id: "T-P2-03"
    name: "ResponseStorage implementation"
    duration_hours: 3
    depends_on: ["T-P2-01"]
    deliverables: ["storage.py"]
    definition_of_done:
      - "zstd compression working"
      - "store/retrieve round-trip correct"
      - "memory_usage() tracking accurate"
      - "Handles max_size_bytes limit"

  - id: "T-P2-04"
    name: "CacheStats implementation"
    duration_hours: 2
    depends_on: ["T-P2-01"]
    deliverables: ["stats.py"]
    definition_of_done:
      - "All counters implemented"
      - "Rolling window for latencies (last 1000)"
      - "to_dict() for JSON export"
      - "hit_rate calculation correct"

  - id: "T-P2-05"
    name: "BinaryCache core implementation"
    duration_hours: 12
    depends_on: ["T-P2-02", "T-P2-03", "T-P2-04"]
    deliverables: ["cache.py"]
    definition_of_done:
      - "__init__ creates index, adapter, storage"
      - "lookup() finds nearest above threshold"
      - "insert() adds entry, returns evicted flag"
      - "evict() removes n entries (LRU for now)"
      - "Thread-safe access patterns documented"
      - "All methods have type hints"

  - id: "T-P2-06"
    name: "Cache unit tests"
    duration_hours: 8
    depends_on: ["T-P2-05"]
    deliverables: ["test_cache.py"]
    definition_of_done:
      - "Test init with various configs"
      - "Test insert and lookup happy path"
      - "Test lookup miss (no match above threshold)"
      - "Test eviction triggers at max_entries"
      - "Test stats tracking accuracy"
      - "Test edge cases (empty cache, full cache)"
      - "Parametrized for code_bits {64, 128, 256}"

  - id: "T-P2-07"
    name: "Cache memory benchmark"
    duration_hours: 3
    depends_on: ["T-P2-06"]
    deliverables: []
    definition_of_done:
      - "Measure memory at 1K, 10K, 100K entries"
      - "Verify < 100 bytes per entry (ex-response)"
      - "Document actual memory usage"

  - id: "T-P2-08"
    name: "Cache latency benchmark"
    duration_hours: 3
    depends_on: ["T-P2-06"]
    deliverables: []
    definition_of_done:
      - "Measure lookup latency at 1K, 10K, 100K entries"
      - "Verify < 10ms at 10K entries"
      - "Measure insert latency"

  - id: "T-P2-09"
    name: "Phase 2 wrap-up and documentation"
    duration_hours: 2
    depends_on: ["T-P2-07", "T-P2-08"]
    deliverables: []
    definition_of_done:
      - "All tests pass"
      - "README updated"
      - "Go/No-Go checklist completed"

# Kill Switches
kill_switches:
  - id: "KS-P2-01"
    name: "Lookup Latency"
    trigger_condition: "Cache lookup > 10ms at 10K entries (p99)"
    measurement_method: "pytest benchmark"
    action_if_triggered: "STOP: Profile and optimize Hamming scan"
    recovery_cost_weeks: 1
    
  - id: "KS-P2-02"
    name: "Memory Per Entry"
    trigger_condition: "Memory per entry (ex-response) > 200 bytes"
    measurement_method: "tracemalloc measurement"
    action_if_triggered: "REDESIGN: Reduce metadata, use packed structs"
    recovery_cost_weeks: 0.5
    
  - id: "KS-P2-03"
    name: "Insert Latency"
    trigger_condition: "Insert > 50ms at 50K entries"
    measurement_method: "pytest benchmark"
    action_if_triggered: "STOP: Profile eviction and storage"
    recovery_cost_weeks: 0.5

# Exit Criteria
exit_criteria:
  - id: "XC-P2-01"
    condition: "All unit tests pass (≥40 tests)"
    verification_command: "pytest tests/test_cache.py -v"
  - id: "XC-P2-02"
    condition: "Lookup latency < 5ms at 10K entries (p50)"
    verification_command: "pytest tests/test_cache.py::test_lookup_latency --benchmark"
  - id: "XC-P2-03"
    condition: "Memory < 100 bytes per entry (excluding response)"
    verification_command: "python benchmarks/bench_cache_memory.py"
  - id: "XC-P2-04"
    condition: "Insert → lookup round-trip works correctly"
    verification_command: "pytest tests/test_cache.py::test_insert_lookup_roundtrip"

# BinaryLLM Integration
binaryllm_integration:
  modules_used:
    - name: "BinaryAdapter (from P1)"
      functions: ["encode_single", "encode_batch"]
      thread_safety: "yes"
  invariants_to_preserve:
    - "INV-SC-01: Cache lookup is deterministic"
    - "INV-SC-02: Binary codes match source embeddings via BinaryLLM"

# Test Matrix
test_matrix:
  unit_tests:
    - name: "test_config_validation"
      covers: "CacheConfig __post_init__"
      edge_cases: ["invalid code_bits", "threshold out of range"]
    - name: "test_entry_immutability"
      covers: "CacheEntry frozen"
      edge_cases: ["attempt to modify field"]
    - name: "test_index_operations"
      covers: "CacheIndex add/remove"
      edge_cases: ["add to full index", "remove nonexistent"]
    - name: "test_storage_compression"
      covers: "ResponseStorage round-trip"
      edge_cases: ["empty response", "1MB response", "binary data"]
    - name: "test_cache_insert_lookup"
      covers: "BinaryCache insert and lookup"
      edge_cases: ["identical embedding", "similar embedding", "dissimilar embedding"]
    - name: "test_cache_eviction"
      covers: "BinaryCache evict"
      edge_cases: ["evict from empty", "evict all", "evict partial"]
  integration_tests:
    - name: "test_cache_with_adapter"
      components_tested: ["BinaryCache", "BinaryAdapter"]
      mock_requirements: ["None"]
  performance_tests:
    - name: "bench_lookup_latency"
      target_metric: "lookup_time_ms"
      acceptable_range: "1-5 ms (10K entries)"
      failure_threshold: "10 ms"
    - name: "bench_memory_per_entry"
      target_metric: "bytes_per_entry"
      acceptable_range: "50-100 bytes"
      failure_threshold: "200 bytes"
```

---

# PHASE 3: Diversity Eviction Engine

```yaml
# ============================================================================
# PHASE 3: DIVERSITY EVICTION ENGINE
# ============================================================================

id: P3
name: "Diversity Eviction Engine"
codename: "PRUNE"
duration:
  estimated_weeks: 2
  buffer_weeks: 0.5
  hard_deadline: "Week 6 of project"
  total_hours: 50-60

# Dependencies
depends_on:
  - phase_id: "P2"
    artifacts_required:
      - "semantic_cache/core/cache.py"
      - "tests/test_cache.py (passing)"
    verification: "pytest tests/test_cache.py -v"

# Entry Criteria
entry_criteria:
  - id: "EC-P3-01"
    condition: "Phase 2 all exit criteria pass"
    verification_command: "Run P2 exit criteria verification"
  - id: "EC-P3-02"
    condition: "Basic LRU eviction works in BinaryCache"
    verification_command: "pytest tests/test_cache.py::test_eviction"
  - id: "EC-P3-03"
    condition: "MARKET VALIDATION: At least 2/5 users interested"
    verification_command: "Check market_validation_results.md"
    note: "CRITICAL: Do not proceed without market validation"

# Deliverables
deliverables:
  - name: "lru.py"
    type: code
    path: "semantic_cache/eviction/lru.py"
    description: "LRU eviction baseline with O(1) operations"
    lines_of_code_estimate: 120
    complexity: medium
    public_api: |
      from typing import List, Protocol
      import numpy as np
      
      class EvictionStrategy(Protocol):
          """Protocol for eviction strategies."""
          
          def select_for_eviction(self, n: int) -> List[int]:
              """Select n entries for eviction, return slot indices."""
              ...
          
          def on_access(self, slot_index: int) -> None:
              """Called when entry is accessed."""
              ...
          
          def on_insert(self, slot_index: int) -> None:
              """Called when entry is inserted."""
              ...
          
          def on_remove(self, slot_index: int) -> None:
              """Called when entry is removed."""
              ...
      
      class LRUEviction:
          """LRU eviction with O(1) operations using doubly-linked list."""
          
          def __init__(self, max_entries: int) -> None:
              ...
          
          def select_for_eviction(self, n: int) -> List[int]:
              """Return n least-recently-used slot indices."""
              ...
          
          def on_access(self, slot_index: int) -> None:
              """Move entry to front of LRU list."""
              ...

  - name: "diversity.py"
    type: code
    path: "semantic_cache/eviction/diversity.py"
    description: "Diversity-aware eviction with clustering"
    lines_of_code_estimate: 350
    complexity: high
    public_api: |
      from typing import List, Optional
      import numpy as np
      from sklearn.cluster import MiniBatchKMeans
      
      class DiversityEviction:
          """Diversity-aware eviction using cluster-based selection."""
          
          def __init__(
              self,
              max_entries: int,
              n_clusters: int = 20,
              rebalance_interval: int = 1000
          ) -> None:
              ...
          
          def select_for_eviction(self, n: int) -> List[int]:
              """
              Select n entries for eviction prioritizing:
              1. Over-represented clusters
              2. Oldest entries within those clusters
              """
              ...
          
          def rebalance_clusters(self, codes: np.ndarray) -> None:
              """Re-cluster all entries. Called periodically."""
              ...
          
          def compute_diversity_score(self) -> float:
              """
              Compute current cache diversity.
              
              Returns value in [0, 1] where:
              - 0 = all entries in same cluster
              - 1 = entries evenly distributed across clusters
              """
              ...
          
          def get_cluster_distribution(self) -> np.ndarray:
              """Return count of entries per cluster."""
              ...

  - name: "hybrid.py"
    type: code
    path: "semantic_cache/eviction/hybrid.py"
    description: "Hybrid LRU + periodic diversity rebalancing"
    lines_of_code_estimate: 200
    complexity: medium
    public_api: |
      from typing import List
      from .lru import LRUEviction
      from .diversity import DiversityEviction
      
      class HybridEviction:
          """
          Hybrid eviction strategy:
          - Normal: LRU eviction (O(1))
          - Periodic: Diversity rebalancing (O(n log n))
          - Emergency: Full diversity eviction if score drops
          """
          
          def __init__(
              self,
              max_entries: int,
              rebalance_interval: int = 1000,
              diversity_threshold: float = 0.3
          ) -> None:
              ...
          
          def select_for_eviction(self, n: int) -> List[int]:
              """Select using hybrid strategy."""
              ...
          
          def should_rebalance(self) -> bool:
              """Check if rebalancing is needed."""
              ...

  - name: "test_eviction.py"
    type: test
    path: "tests/test_eviction.py"
    description: "Tests for all eviction strategies"
    lines_of_code_estimate: 400
    complexity: high
    test_count_estimate: 35

  - name: "bench_eviction.py"
    type: benchmark
    path: "benchmarks/bench_eviction.py"
    description: "Comparative benchmarks for eviction strategies"
    lines_of_code_estimate: 300
    complexity: medium

# Task Breakdown
tasks:
  - id: "T-P3-01"
    name: "LRU eviction implementation"
    duration_hours: 6
    depends_on: []
    deliverables: ["lru.py"]
    definition_of_done:
      - "Doubly-linked list implemented"
      - "O(1) select_for_eviction"
      - "O(1) on_access"
      - "Thread-safe (or documented as not)"

  - id: "T-P3-02"
    name: "LRU eviction tests"
    duration_hours: 4
    depends_on: ["T-P3-01"]
    deliverables: ["test_eviction.py (partial)"]
    definition_of_done:
      - "Test LRU ordering"
      - "Test eviction of n entries"
      - "Test on_access updates order"
      - "Test edge cases"

  - id: "T-P3-03"
    name: "Diversity eviction implementation"
    duration_hours: 10
    depends_on: ["T-P3-01"]
    deliverables: ["diversity.py"]
    definition_of_done:
      - "MiniBatchKMeans clustering"
      - "Cluster-based eviction selection"
      - "Diversity score calculation"
      - "Rebalance method works"

  - id: "T-P3-04"
    name: "Diversity eviction tests"
    duration_hours: 6
    depends_on: ["T-P3-03"]
    deliverables: ["test_eviction.py (partial)"]
    definition_of_done:
      - "Test cluster assignment"
      - "Test eviction from large clusters"
      - "Test diversity score calculation"
      - "Test rebalancing"

  - id: "T-P3-05"
    name: "Hybrid eviction implementation"
    duration_hours: 6
    depends_on: ["T-P3-02", "T-P3-04"]
    deliverables: ["hybrid.py"]
    definition_of_done:
      - "Combines LRU and diversity"
      - "Periodic rebalancing logic"
      - "Emergency diversity trigger"
      - "Configurable thresholds"

  - id: "T-P3-06"
    name: "Hybrid eviction tests"
    duration_hours: 4
    depends_on: ["T-P3-05"]
    deliverables: ["test_eviction.py (complete)"]
    definition_of_done:
      - "Test normal LRU operation"
      - "Test rebalance trigger"
      - "Test emergency diversity"

  - id: "T-P3-07"
    name: "Integrate eviction into BinaryCache"
    duration_hours: 4
    depends_on: ["T-P3-06"]
    deliverables: []
    definition_of_done:
      - "BinaryCache uses configured eviction strategy"
      - "Strategy switchable via config"
      - "Existing tests still pass"

  - id: "T-P3-08"
    name: "Eviction benchmarks"
    duration_hours: 8
    depends_on: ["T-P3-07"]
    deliverables: ["bench_eviction.py"]
    definition_of_done:
      - "Benchmark LRU vs Diversity vs Hybrid"
      - "Measure eviction latency at 10K, 50K entries"
      - "Measure hit rate on synthetic workload"
      - "Compare to baseline LRU"

  - id: "T-P3-09"
    name: "Kill-switch evaluation"
    duration_hours: 2
    depends_on: ["T-P3-08"]
    deliverables: []
    definition_of_done:
      - "Diversity shows ≥5% hit rate improvement over LRU (or document failure)"
      - "Eviction latency < 50ms at 50K entries"
      - "Decision: continue with diversity or pivot to LRU-only"

# Kill Switches
kill_switches:
  - id: "KS-P3-01"
    name: "Diversity Improvement"
    trigger_condition: "Diversity eviction shows < 5% hit rate improvement over LRU on synthetic workload"
    measurement_method: "bench_eviction.py --strategy=comparison"
    action_if_triggered: "PIVOT: Ship with LRU only; document diversity as experimental/future"
    recovery_cost_weeks: 0
    
  - id: "KS-P3-02"
    name: "Eviction Latency"
    trigger_condition: "Diversity eviction adds > 50ms latency at 50K entries"
    measurement_method: "bench_eviction.py --metric=latency"
    action_if_triggered: "FALLBACK: Increase rebalance_interval to 10000"
    recovery_cost_weeks: 0.5
    
  - id: "KS-P3-03"
    name: "Market Validation"
    trigger_condition: "0 out of 5 users express interest in market validation"
    measurement_method: "market_validation_results.md"
    action_if_triggered: "KILL PROJECT: No market demand validated"
    recovery_cost_weeks: N/A

# Exit Criteria
exit_criteria:
  - id: "XC-P3-01"
    condition: "All eviction tests pass (≥35 tests)"
    verification_command: "pytest tests/test_eviction.py -v"
  - id: "XC-P3-02"
    condition: "LRU eviction O(1) verified"
    verification_command: "pytest tests/test_eviction.py::test_lru_complexity"
  - id: "XC-P3-03"
    condition: "Eviction latency < 10ms for 100 entries at 50K cache size"
    verification_command: "python benchmarks/bench_eviction.py --metric=latency"
  - id: "XC-P3-04"
    condition: "Hit rate improvement documented (positive or negative)"
    verification_command: "cat benchmarks/results/eviction_comparison.json"

# Test Matrix
test_matrix:
  unit_tests:
    - name: "test_lru_ordering"
      covers: "LRU list maintains correct order"
      edge_cases: ["single entry", "full capacity", "repeated access"]
    - name: "test_lru_eviction"
      covers: "LRU selects oldest entries"
      edge_cases: ["evict all", "evict more than exists"]
    - name: "test_diversity_clustering"
      covers: "Cluster assignment"
      edge_cases: ["all identical codes", "all unique codes"]
    - name: "test_diversity_score"
      covers: "Diversity metric calculation"
      edge_cases: ["empty cache", "single cluster", "uniform distribution"]
    - name: "test_hybrid_strategy"
      covers: "Hybrid switches between LRU and diversity"
      edge_cases: ["below threshold", "above threshold"]
  performance_tests:
    - name: "bench_lru_latency"
      target_metric: "eviction_time_ms"
      acceptable_range: "0.01-0.1 ms"
      failure_threshold: "1 ms"
    - name: "bench_diversity_latency"
      target_metric: "eviction_time_ms"
      acceptable_range: "5-20 ms (50K entries)"
      failure_threshold: "50 ms"
    - name: "bench_hit_rate_comparison"
      target_metric: "hit_rate_delta"
      acceptable_range: "0-20% improvement"
      failure_threshold: "< 0% (diversity worse than LRU)"
```

---

# PHASE 4: LLM API Proxy Layer

```yaml
# ============================================================================
# PHASE 4: LLM API PROXY LAYER
# ============================================================================

id: P4
name: "LLM API Proxy Layer"
codename: "GATE"
duration:
  estimated_weeks: 1.5
  buffer_weeks: 0.5
  hard_deadline: "Week 7 of project"
  total_hours: 40-50

# Dependencies
depends_on:
  - phase_id: "P2"
    artifacts_required:
      - "semantic_cache/core/cache.py (working)"
    verification: "pytest tests/test_cache.py -v"
  # Note: P4 can start in parallel with P3 after P2 is complete

# Entry Criteria
entry_criteria:
  - id: "EC-P4-01"
    condition: "Phase 2 cache operations working"
    verification_command: "pytest tests/test_cache.py -v"
  - id: "EC-P4-02"
    condition: "FastAPI and httpx installed"
    verification_command: "python -c 'import fastapi; import httpx'"

# Deliverables
deliverables:
  - name: "request_parser.py"
    type: code
    path: "semantic_cache/proxy/request_parser.py"
    description: "Parse OpenAI/Anthropic request formats"
    lines_of_code_estimate: 200
    complexity: medium
    public_api: |
      from dataclasses import dataclass
      from typing import List, Optional
      
      @dataclass
      class CacheableRequest:
          """Parsed request with cache-relevant fields."""
          
          provider: str          # "openai" | "anthropic"
          model: str             # e.g., "gpt-4-turbo"
          messages: List[dict]   # Conversation messages
          temperature: float     # For cache key
          max_tokens: Optional[int]
          
          def cache_key_text(self) -> str:
              """Text to embed for semantic matching."""
              ...
          
          def exact_hash(self) -> bytes:
              """SHA256 hash for exact matching."""
              ...
      
      def parse_openai_request(body: dict) -> CacheableRequest:
          """Parse OpenAI chat completion request."""
          ...
      
      def parse_anthropic_request(body: dict) -> CacheableRequest:
          """Parse Anthropic messages request."""
          ...

  - name: "response_handler.py"
    type: code
    path: "semantic_cache/proxy/response_handler.py"
    description: "Serialize/deserialize LLM responses"
    lines_of_code_estimate: 120
    complexity: low
    public_api: |
      import json
      from typing import Union
      
      def serialize_response(response: dict) -> bytes:
          """Serialize response for caching."""
          ...
      
      def deserialize_response(data: bytes) -> dict:
          """Deserialize cached response."""
          ...
      
      def format_cached_response(cached: dict, original_id: str) -> dict:
          """Format cached response with updated metadata."""
          ...

  - name: "embedding_client.py"
    type: code
    path: "semantic_cache/proxy/embedding_client.py"
    description: "Get embeddings from OpenAI or local model"
    lines_of_code_estimate: 180
    complexity: medium
    public_api: |
      from abc import ABC, abstractmethod
      import numpy as np
      from typing import List
      
      class EmbeddingClient(ABC):
          """Abstract base for embedding providers."""
          
          @abstractmethod
          async def embed(self, text: str) -> np.ndarray:
              """Get embedding for single text."""
              ...
          
          @abstractmethod
          async def embed_batch(self, texts: List[str]) -> np.ndarray:
              """Get embeddings for batch of texts."""
              ...
      
      class OpenAIEmbeddingClient(EmbeddingClient):
          """OpenAI text-embedding-3-small client."""
          
          def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
              ...
      
      class LocalEmbeddingClient(EmbeddingClient):
          """Local SentenceTransformers client."""
          
          def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
              ...

  - name: "server.py"
    type: code
    path: "semantic_cache/proxy/server.py"
    description: "FastAPI proxy server"
    lines_of_code_estimate: 350
    complexity: high
    public_api: |
      from fastapi import FastAPI, Request, Response
      from typing import Optional
      
      app = FastAPI(title="Binary Semantic Cache Proxy")
      
      @app.post("/v1/chat/completions")
      async def proxy_openai_chat(request: Request) -> Response:
          """
          Proxy OpenAI chat completions with caching.
          
          1. Parse request
          2. Get embedding for cache key
          3. Lookup in cache
          4. If hit: return cached response
          5. If miss: forward to OpenAI, cache response, return
          """
          ...
      
      @app.post("/v1/messages")
      async def proxy_anthropic_messages(request: Request) -> Response:
          """Proxy Anthropic messages with caching."""
          ...
      
      @app.get("/stats")
      async def get_cache_stats():
          """Return cache statistics."""
          ...
      
      @app.post("/cache/clear")
      async def clear_cache():
          """Clear all cached entries."""
          ...
      
      @app.get("/health")
      async def health_check():
          """Health check endpoint."""
          ...

  - name: "middleware.py"
    type: code
    path: "semantic_cache/proxy/middleware.py"
    description: "Request/response middleware for caching"
    lines_of_code_estimate: 150
    complexity: medium

  - name: "test_proxy.py"
    type: test
    path: "tests/test_proxy.py"
    description: "End-to-end proxy tests"
    lines_of_code_estimate: 400
    complexity: high
    test_count_estimate: 30

# Task Breakdown
tasks:
  - id: "T-P4-01"
    name: "Request parser implementation"
    duration_hours: 5
    depends_on: []
    deliverables: ["request_parser.py"]
    definition_of_done:
      - "OpenAI request parsing works"
      - "Anthropic request parsing works"
      - "cache_key_text extracts relevant content"
      - "exact_hash produces consistent hash"

  - id: "T-P4-02"
    name: "Response handler implementation"
    duration_hours: 3
    depends_on: []
    deliverables: ["response_handler.py"]
    definition_of_done:
      - "Serialize/deserialize round-trip works"
      - "Handles streaming vs non-streaming responses"

  - id: "T-P4-03"
    name: "Embedding client implementation"
    duration_hours: 5
    depends_on: []
    deliverables: ["embedding_client.py"]
    definition_of_done:
      - "OpenAI client works with API key"
      - "Local client works with SentenceTransformers"
      - "Both return same embedding dimension"

  - id: "T-P4-04"
    name: "FastAPI server skeleton"
    duration_hours: 4
    depends_on: ["T-P4-01", "T-P4-02"]
    deliverables: ["server.py (skeleton)"]
    definition_of_done:
      - "Server starts on configurable port"
      - "Health check endpoint works"
      - "Stats endpoint works"

  - id: "T-P4-05"
    name: "Cache integration in proxy"
    duration_hours: 8
    depends_on: ["T-P4-03", "T-P4-04"]
    deliverables: ["server.py (complete)", "middleware.py"]
    definition_of_done:
      - "Proxy forwards to OpenAI on cache miss"
      - "Proxy returns cached response on cache hit"
      - "Response is cached after miss"
      - "Stats updated correctly"

  - id: "T-P4-06"
    name: "Proxy tests"
    duration_hours: 8
    depends_on: ["T-P4-05"]
    deliverables: ["test_proxy.py"]
    definition_of_done:
      - "Test cache miss path"
      - "Test cache hit path"
      - "Test stats endpoint"
      - "Test with mocked OpenAI API"
      - "Test error handling"

  - id: "T-P4-07"
    name: "E2E integration test"
    duration_hours: 4
    depends_on: ["T-P4-06"]
    deliverables: []
    definition_of_done:
      - "Start proxy"
      - "Send real request via proxy"
      - "Verify response cached"
      - "Send same request, verify cache hit"

# Kill Switches
kill_switches:
  - id: "KS-P4-01"
    name: "E2E Latency"
    trigger_condition: "Proxy adds > 100ms latency on cache miss path"
    measurement_method: "E2E timing measurement"
    action_if_triggered: "STOP: Profile and optimize hot paths"
    recovery_cost_weeks: 0.5
    
  - id: "KS-P4-02"
    name: "Embedding Client Failure"
    trigger_condition: "Cannot get embeddings reliably (>5% failure rate)"
    measurement_method: "E2E test failure rate"
    action_if_triggered: "FALLBACK: Add retry logic, consider local-only mode"
    recovery_cost_weeks: 0.5

# Exit Criteria
exit_criteria:
  - id: "XC-P4-01"
    condition: "Proxy routes requests to OpenAI correctly"
    verification_command: "pytest tests/test_proxy.py::test_passthrough"
  - id: "XC-P4-02"
    condition: "Cache hit returns cached response"
    verification_command: "pytest tests/test_proxy.py::test_cache_hit"
  - id: "XC-P4-03"
    condition: "Latency overhead < 50ms on cache hit"
    verification_command: "pytest tests/test_proxy.py::test_latency"
  - id: "XC-P4-04"
    condition: "Stats endpoint returns accurate data"
    verification_command: "pytest tests/test_proxy.py::test_stats"
```

---

# PHASE 5: Benchmarking & Validation

```yaml
# ============================================================================
# PHASE 5: BENCHMARKING & VALIDATION
# ============================================================================

id: P5
name: "Benchmarking & Validation"
codename: "PROVE"
duration:
  estimated_weeks: 1
  buffer_weeks: 0.5
  hard_deadline: "Week 8 of project"
  total_hours: 30-40

depends_on:
  - phase_id: "P3"
    artifacts_required:
      - "semantic_cache/eviction/*.py"
    verification: "pytest tests/test_eviction.py"
  - phase_id: "P4"
    artifacts_required:
      - "semantic_cache/proxy/server.py"
    verification: "pytest tests/test_proxy.py"

# Deliverables
deliverables:
  - name: "bench_synthetic.py"
    type: benchmark
    path: "benchmarks/bench_synthetic.py"
    description: "Synthetic workload with Zipf distribution"
    lines_of_code_estimate: 250
    complexity: medium

  - name: "bench_replay.py"
    type: benchmark
    path: "benchmarks/bench_replay.py"
    description: "Replay real API logs"
    lines_of_code_estimate: 200
    complexity: medium

  - name: "bench_stress.py"
    type: benchmark
    path: "benchmarks/bench_stress.py"
    description: "Concurrent load testing"
    lines_of_code_estimate: 200
    complexity: medium

  - name: "benchmark_report.md"
    type: doc
    path: "docs/BENCHMARK_REPORT.md"
    description: "Full benchmark results with analysis"
    lines_of_code_estimate: 1000
    complexity: low

# Kill Switches
kill_switches:
  - id: "KS-P5-01"
    name: "Cache Hit Rate"
    trigger_condition: "Hit rate < 20% on synthetic Zipf workload"
    measurement_method: "bench_synthetic.py results"
    action_if_triggered: "ANALYZE: Tune similarity threshold; may not be fundamental flaw"
    recovery_cost_weeks: 0.5
    
  - id: "KS-P5-02"
    name: "Cost Savings"
    trigger_condition: "Estimated cost savings < 10%"
    measurement_method: "API call count reduction"
    action_if_triggered: "DOCUMENT: May still be useful for latency, not cost"
    recovery_cost_weeks: 0

# Exit Criteria
exit_criteria:
  - id: "XC-P5-01"
    condition: "All benchmarks complete with reproducible results"
    verification_command: "python benchmarks/run_all.py"
  - id: "XC-P5-02"
    condition: "Benchmark report documents methodology and results"
    verification_command: "test -f docs/BENCHMARK_REPORT.md"
  - id: "XC-P5-03"
    condition: "Hit rate ≥ 20% on at least one workload type"
    verification_command: "cat benchmarks/results/synthetic.json | jq '.hit_rate'"
```

---

# PHASE 6: Documentation & Polish

```yaml
# ============================================================================
# PHASE 6: DOCUMENTATION & POLISH
# ============================================================================

id: P6
name: "Documentation & Polish"
codename: "SHIP"
duration:
  estimated_weeks: 1
  buffer_weeks: 0
  hard_deadline: "Week 9 of project"
  total_hours: 25-30

depends_on:
  - phase_id: "P5"
    artifacts_required:
      - "docs/BENCHMARK_REPORT.md"
    verification: "test -f docs/BENCHMARK_REPORT.md"

# Deliverables
deliverables:
  - name: "README.md"
    type: doc
    path: "README.md"
    description: "Quick start with 5-minute setup"
    lines_of_code_estimate: 400

  - name: "API_REFERENCE.md"
    type: doc
    path: "docs/API_REFERENCE.md"
    description: "Complete API documentation"
    lines_of_code_estimate: 600

  - name: "CONFIGURATION.md"
    type: doc
    path: "docs/CONFIGURATION.md"
    description: "Configuration guide with examples"
    lines_of_code_estimate: 400

  - name: "TROUBLESHOOTING.md"
    type: doc
    path: "docs/TROUBLESHOOTING.md"
    description: "Common issues and solutions"
    lines_of_code_estimate: 300

# Exit Criteria
exit_criteria:
  - id: "XC-P6-01"
    condition: "README enables new user to run cache in < 5 minutes"
    verification_command: "Manual test by fresh user"
  - id: "XC-P6-02"
    condition: "All public APIs documented"
    verification_command: "Compare API_REFERENCE.md to code"
  - id: "XC-P6-03"
    condition: "GitHub repository is public-ready"
    verification_command: "Checklist: LICENSE, CONTRIBUTING, issue templates"
```

---

# Risk Register

| ID | Risk | Severity | Likelihood | Impact | Mitigation | Owner | Phase |
|----|------|----------|------------|--------|------------|-------|-------|
| R-01 | BinaryLLM integration issues | MEDIUM | LOW | P1 delays | Validate imports early | P1 | P1 |
| R-02 | Hamming scan too slow at 100K | HIGH | LOW | Redesign needed | SIMD optimization | P1 | P1 |
| R-03 | Similarity threshold tuning fails | HIGH | MEDIUM | Poor hit rate | A/B testing; adaptive | P5 | P5 |
| R-04 | Diversity eviction no benefit | MEDIUM | MEDIUM | Wasted P3 effort | Ship LRU-only | P3 | P3 |
| R-05 | No market interest | CRITICAL | MEDIUM | Project killed | Validate Week 3-4 | P3 | P3 |
| R-06 | OpenAI API rate limits | MEDIUM | MEDIUM | Benchmark blocked | Cache embeddings; batch | P4 | P4 |
| R-07 | Competitors announce similar | HIGH | MEDIUM | Reduced value | Ship fast | All | All |
| R-08 | Memory exceeds expectations | MEDIUM | LOW | OOM at 100K | Response compression | P2 | P2 |
| R-09 | Thread safety issues | MEDIUM | LOW | Bugs under load | Lock analysis; stress test | P5 | P5 |
| R-10 | Cold start performance | LOW | HIGH | Poor initial UX | Document; warm-up script | P6 | P6 |

---

# Go/No-Go Checklists

## P1 → P2 Transition

| Check | Status | Notes |
|-------|--------|-------|
| All P1 unit tests pass (≥45) | [ ] | |
| Projection latency < 0.5ms (p50) | [ ] | |
| Hamming scan < 1ms at 10K (p50) | [ ] | |
| Determinism verified | [ ] | |
| Memory footprint documented | [ ] | |
| Spearman correlation > 0.90 | [ ] | |
| No KS-P1-* triggered | [ ] | |

**Decision:** [ ] GO / [ ] NO-GO

## P2 → P3 Transition

| Check | Status | Notes |
|-------|--------|-------|
| All P2 unit tests pass (≥40) | [ ] | |
| Lookup latency < 5ms at 10K (p50) | [ ] | |
| Memory < 100 bytes per entry | [ ] | |
| Insert/lookup round-trip works | [ ] | |
| No KS-P2-* triggered | [ ] | |

**Decision:** [ ] GO / [ ] NO-GO

## P3 Market Validation Gate

| Check | Status | Notes |
|-------|--------|-------|
| LinkedIn outreach: ≥5 responses | [ ] | |
| Scheduled calls: ≥2 | [ ] | |
| "Definitely interested": ≥1 | [ ] | |
| Reddit/HN positive signals: ≥5 | [ ] | |

**Decision:** [ ] CONTINUE / [ ] PIVOT to LRU-only / [ ] KILL

## P3 → P4/P5 Transition

| Check | Status | Notes |
|-------|--------|-------|
| All eviction tests pass (≥35) | [ ] | |
| LRU O(1) verified | [ ] | |
| Eviction latency < 10ms for 100 | [ ] | |
| Hit rate improvement documented | [ ] | |
| Market validation passed | [ ] | |

**Decision:** [ ] GO / [ ] NO-GO

## P5 → P6 Transition

| Check | Status | Notes |
|-------|--------|-------|
| All benchmarks complete | [ ] | |
| Hit rate ≥ 20% on synthetic | [ ] | |
| Report documents methodology | [ ] | |
| No critical issues found | [ ] | |

**Decision:** [ ] GO / [ ] NO-GO

## P6 → Ship

| Check | Status | Notes |
|-------|--------|-------|
| README works for fresh user | [ ] | |
| All APIs documented | [ ] | |
| LICENSE file present | [ ] | |
| All tests pass | [ ] | |
| No open blocking issues | [ ] | |

**Decision:** [ ] SHIP / [ ] DELAY

---

# Test Strategy

## Coverage Requirements

| Module | Unit Test Coverage | Integration Tests | Performance Tests |
|--------|-------------------|-------------------|-------------------|
| binary/adapter.py | ≥90% | Yes | Yes (latency) |
| binary/hamming_ops.py | ≥95% | Yes | Yes (throughput) |
| core/cache.py | ≥85% | Yes | Yes (latency, memory) |
| core/config.py | ≥90% | No | No |
| eviction/*.py | ≥80% | Yes | Yes (hit rate) |
| proxy/*.py | ≥70% | Yes | Yes (E2E latency) |

## Test Categories

1. **Unit Tests** — Isolated function/class tests with mocked dependencies
2. **Integration Tests** — Component interaction without mocks
3. **Performance Tests** — Latency, throughput, memory benchmarks
4. **E2E Tests** — Full system with real API calls (gated)

## CI Pipeline

```yaml
# .github/workflows/ci.yml (conceptual)
stages:
  - lint: ruff, mypy
  - unit: pytest tests/ --ignore=tests/test_proxy.py
  - integration: pytest tests/ -k integration
  - performance: pytest benchmarks/ --benchmark-only
  - e2e: pytest tests/test_proxy.py (manual trigger)
```

---

# Hostile Review Appendix

## Pre-emptive Attacks on This Plan

| Attack | Response |
|--------|----------|
| "8 weeks is too aggressive" | Buffer added. Real deadline is 9 weeks. Weekly check-ins to catch slips. |
| "Market validation too late at Week 3" | Moved to parallel track. Can start outreach in Week 2. |
| "Too many kill-switches" | Better to fail fast than waste 8 weeks on dead project. |
| "Diversity eviction is unproven" | LRU baseline is always available. Diversity is optional upgrade. |
| "No GPU optimization" | CPU-only is a feature, not a bug. Target market doesn't have GPUs. |
| "Competition can copy" | Speed is the only moat. Ship in 8 weeks or lose. |

## Things That Could Kill This Project

1. **KS-P3-03: No market interest** — Most likely cause of death
2. **GPTCache ships binary backend** — External dependency
3. **Schedule slips past 12 weeks** — Developer fatigue
4. **Similarity matching doesn't work** — Fundamental flaw

## Survival Probability

| Scenario | Probability | Outcome |
|----------|-------------|---------|
| Full success (ship + users) | 30% | MVP in production, users happy |
| Partial success (ship, few users) | 35% | Open-source project, portfolio value |
| Pivot to LRU-only | 20% | Simpler product, lower differentiation |
| Kill at Week 4 | 15% | No market, cut losses |

---

**END OF ENGINEERING PLAN**

---

*This plan is binding. Deviations require explicit justification and updated kill-switch assessment.*

