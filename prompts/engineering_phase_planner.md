# ENGINEERING PHASE PLANNER — Binary Semantic Cache

**Version:** 1.0  
**Purpose:** Detailed week-by-week engineering plan with dependency graphs  
**Use After:** `deep_research_semantic_cache_architecture.md` has been analyzed

---

## CONTEXT

You are creating a **military-grade engineering plan** for implementing the Binary Semantic Cache. This plan must be:

1. **Executable by a solo developer** working part-time (~25-30 hrs/week)
2. **Dependency-aware**: No task starts before its prerequisites are complete
3. **Kill-switch protected**: Every phase has explicit failure criteria
4. **Integration-first**: BinaryLLM Phase 1 integration is validated before building on top

---

## PHASE PLANNING TEMPLATE

For each phase, produce this exact structure:

```yaml
# ============================================================================
# PHASE {N}: {PHASE_NAME}
# ============================================================================

id: P{N}
name: "{Phase Name}"
codename: "{Single-word codename}"
duration:
  estimated_weeks: {N}
  buffer_weeks: {N}
  hard_deadline: "Week {N} of project"

# Dependencies
depends_on:
  - phase_id: "P{N-1}"
    artifacts_required:
      - "{artifact_name}"
      - "{artifact_name}"
    verification: "{how to verify dependency is met}"

# Entry Criteria (ALL must be true to start)
entry_criteria:
  - id: "EC-P{N}-01"
    condition: "{exact condition}"
    verification_command: "{command or test to verify}"
  - id: "EC-P{N}-02"
    condition: "{exact condition}"
    verification_command: "{command or test to verify}"

# Deliverables
deliverables:
  - name: "{file_or_module_name}"
    type: "{code|test|doc|config}"
    path: "src/{path}"
    description: "{1-sentence description}"
    lines_of_code_estimate: {N}
    complexity: "{low|medium|high}"
    public_api: |
      def function_name(arg: Type) -> ReturnType:
          \"\"\"Docstring\"\"\"
          
  - name: "{file_or_module_name}"
    # ... repeat for each deliverable

# Task Breakdown (Gantt-style)
tasks:
  - id: "T-P{N}-01"
    name: "{Task name}"
    duration_hours: {N}
    depends_on: []  # or ["T-P{N}-XX"]
    deliverables: ["{deliverable_name}"]
    definition_of_done:
      - "{specific condition}"
      - "{specific condition}"
    risks:
      - risk: "{what could go wrong}"
        likelihood: "{low|medium|high}"
        mitigation: "{how to handle}"

# Kill Switches
kill_switches:
  - id: "KS-P{N}-01"
    name: "{Short name}"
    trigger_condition: "{exact measurable condition}"
    measurement_method: "{how to measure}"
    action_if_triggered: "{STOP|PIVOT|REDESIGN}: {specific action}"
    recovery_cost_weeks: {N}
  
# Exit Criteria (ALL must be true to complete phase)
exit_criteria:
  - id: "XC-P{N}-01"
    condition: "{exact condition}"
    verification_command: "{command or test to verify}"
    
# Integration Points with BinaryLLM Phase 1
binaryllm_integration:
  modules_used:
    - name: "quantization.binarization"
      functions: ["binarize_sign", "RandomProjection"]
      thread_safety: "{yes|no|unknown}"
    - name: "quantization.packing"
      functions: ["pack_codes", "unpack_codes"]
      thread_safety: "{yes|no|unknown}"
  invariants_to_preserve:
    - "INV-{ID}: {description}"
  potential_conflicts:
    - conflict: "{description}"
      resolution: "{how to resolve}"

# Test Matrix
test_matrix:
  unit_tests:
    - name: "test_{function_name}"
      covers: "{what it tests}"
      edge_cases: ["{case1}", "{case2}"]
  integration_tests:
    - name: "test_integration_{feature}"
      components_tested: ["{comp1}", "{comp2}"]
      mock_requirements: ["{what needs mocking}"]
  performance_tests:
    - name: "bench_{operation}"
      target_metric: "{metric_name}"
      acceptable_range: "{min}-{max} {unit}"
      failure_threshold: "{value} {unit}"
```

---

## PHASES TO PLAN

### PHASE 1: Foundation & BinaryLLM Integration (Weeks 1-2)

**Goal:** Prove that BinaryLLM Phase 1 can be used as a library for semantic matching.

**Key Questions to Answer:**
1. Can we instantiate `RandomProjection` once and reuse it?
2. What is the latency for projecting a single 384-dim embedding to 256-bit code?
3. Can we do batch projection efficiently?
4. What is the memory footprint of the projection matrix?

**Required Experiments:**
```python
# Experiment 1: Projection latency
for dim in [384, 768, 1536]:
    for code_bits in [64, 128, 256, 512]:
        # Measure time to project 1 embedding
        # Measure time to project 1000 embeddings
        # Report mean, p50, p99 latency

# Experiment 2: Memory footprint
# Measure RAM usage with projection matrix loaded
# Compare to baseline without matrix

# Experiment 3: Hamming distance at scale
for n_entries in [1_000, 10_000, 100_000]:
    # Measure time to compute Hamming distance to all entries
    # Report throughput (entries/sec)
```

---

### PHASE 2: Core Cache Data Structures (Weeks 2-3)

**Goal:** Build the in-memory cache with fast binary similarity lookup.

**Key Decisions:**
1. **Storage format for binary codes**: Packed uint64 array vs. sparse representation
2. **Index structure**: Flat array scan vs. LSH vs. custom structure
3. **Metadata storage**: Inline with codes vs. separate array
4. **LRU tracking**: Linked list vs. heap vs. hash map

**Data Structure Prototype:**
```python
import numpy as np
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class CacheConfig:
    max_entries: int = 100_000
    code_bits: int = 256
    embedding_dim: int = 384
    similarity_threshold: float = 0.85  # Normalized Hamming
    eviction_batch_size: int = 100

@dataclass(frozen=True)
class CacheEntry:
    """Immutable cache entry."""
    entry_id: int
    binary_code: np.ndarray  # Shape: (code_bits // 64,), dtype: uint64
    embedding_hash: bytes    # SHA256 of original embedding
    request_hash: bytes      # SHA256 of request payload
    response: bytes          # Serialized response
    created_at: float        # Unix timestamp
    last_accessed: float     # Unix timestamp
    access_count: int

class BinaryCache:
    """Core cache with binary similarity lookup."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._codes: np.ndarray = None  # Packed binary codes
        self._metadata: list = []
        self._lru_heap: list = []
        # ... initialization
    
    def lookup(
        self, 
        query_code: np.ndarray,
        threshold: Optional[float] = None
    ) -> Optional[CacheEntry]:
        """Find closest match above threshold."""
        ...
    
    def insert(
        self,
        entry: CacheEntry
    ) -> bool:
        """Insert entry, evicting if necessary."""
        ...
    
    def evict(self, n: int) -> list[CacheEntry]:
        """Evict n entries using diversity-aware strategy."""
        ...
```

---

### PHASE 3: Diversity Eviction Engine (Weeks 3-5)

**Goal:** Implement and benchmark diversity-aware eviction.

**Algorithm Options:**

#### Option A: MMR-Based Eviction
```python
def mmr_evict(cache: BinaryCache, n: int, lambda_: float = 0.5) -> list[int]:
    """
    Evict entries that minimize Maximal Marginal Relevance.
    
    Score(e) = lambda * recency(e) - (1 - lambda) * max_similarity(e, remaining)
    
    Evict entries with lowest score.
    """
    evict_indices = []
    remaining = set(range(len(cache)))
    
    for _ in range(n):
        scores = []
        for i in remaining:
            recency = cache.recency_score(i)
            diversity = cache.diversity_score(i, remaining - {i})
            score = lambda_ * recency - (1 - lambda_) * diversity
            scores.append((score, i))
        
        _, worst = min(scores)
        evict_indices.append(worst)
        remaining.remove(worst)
    
    return evict_indices
```

#### Option B: Cluster-Based Eviction
```python
def cluster_evict(cache: BinaryCache, n: int, n_clusters: int = 10) -> list[int]:
    """
    Evict from over-represented clusters.
    
    1. Cluster all entries by binary code similarity
    2. Identify largest clusters
    3. Evict oldest entries from largest clusters
    """
    from sklearn.cluster import MiniBatchKMeans
    
    # Cluster binary codes (treat as continuous for clustering)
    codes = cache.get_all_codes_unpacked()
    kmeans = MiniBatchKMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(codes)
    
    # Count per cluster
    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    
    # Evict from largest clusters
    evict_indices = []
    for _ in range(n):
        largest_cluster = np.argmax(cluster_sizes)
        cluster_members = np.where(labels == largest_cluster)[0]
        # Evict oldest in this cluster
        oldest = min(cluster_members, key=lambda i: cache.last_accessed(i))
        evict_indices.append(oldest)
        cluster_sizes[largest_cluster] -= 1
    
    return evict_indices
```

**Benchmarking Requirements:**
| Metric | LRU Baseline | MMR | Cluster | Target |
|--------|--------------|-----|---------|--------|
| Eviction latency (10K entries) | <1ms | ??? | ??? | <10ms |
| Cache hit rate (synthetic) | X% | X+?% | X+?% | +10% over LRU |
| Memory overhead | 0 | ??? | ??? | <20% |

---

### PHASE 4: LLM API Proxy Layer (Weeks 5-6)

**Goal:** HTTP proxy that intercepts OpenAI/Anthropic API calls and caches responses.

**Architecture:**
```
Client → Proxy (localhost:8080) → OpenAI API
                ↓
         [Semantic Cache]
                ↓
         [BinaryLLM Engine]
```

**Key Components:**
1. **Request Parser**: Extract embedding-relevant content from request
2. **Response Serializer**: Store and reconstruct responses
3. **Cache Middleware**: Integrate with core cache
4. **Metrics Collector**: Track hit rate, latency, cost savings

**Proxy Implementation Options:**
| Option | Library | Async | HTTPS | Complexity |
|--------|---------|-------|-------|------------|
| asyncio + aiohttp | aiohttp | Yes | Yes | Medium |
| FastAPI | uvicorn | Yes | Yes | Low |
| mitmproxy | mitmproxy | Yes | Yes | High |
| Flask + gunicorn | Flask | No | Manual | Low |

**Recommended:** FastAPI for simplicity and async support.

---

### PHASE 5: Benchmarking & Validation (Weeks 6-7)

**Goal:** Prove the cache works on real workloads.

**Benchmark Suite:**

```python
# Benchmark 1: Synthetic workload
def bench_synthetic():
    """
    Generate embeddings from a distribution.
    Measure: hit rate, latency, memory.
    """
    pass

# Benchmark 2: Replay real API logs
def bench_replay(log_file: str):
    """
    Replay actual API requests.
    Compare: with cache vs without cache.
    """
    pass

# Benchmark 3: Stress test
def bench_stress():
    """
    High concurrency test.
    Measure: latency under load, error rate.
    """
    pass
```

**Success Criteria:**
| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| Cache hit rate | 20% | 35% | 50% |
| Latency overhead | <50ms | <20ms | <10ms |
| Memory per 100K entries | <500MB | <200MB | <100MB |
| Cost savings (estimated) | 20% | 35% | 50% |

---

### PHASE 6: Documentation & Polish (Weeks 7-8)

**Deliverables:**
1. README with quick start
2. API documentation
3. Benchmark results
4. Configuration guide
5. Troubleshooting guide

---

## DEPENDENCY GRAPH

```
                    ┌─────────────────┐
                    │  BinaryLLM P1   │
                    │   (EXISTING)    │
                    └────────┬────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│  PHASE 1: Foundation & BinaryLLM Integration               │
│  - Projection matrix management                            │
│  - Batch binarization                                      │
│  - Hamming distance benchmarks                             │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│  PHASE 2: Core Cache Data Structures                       │
│  - CacheEntry, CacheConfig                                 │
│  - In-memory storage                                       │
│  - Basic lookup/insert                                     │
└────────────────────────────┬───────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────────┐
│  PHASE 3: Eviction      │   │  PHASE 4: Proxy Layer       │
│  - LRU baseline         │   │  - HTTP server              │
│  - Diversity algorithms │   │  - Request parsing          │
│  - Benchmarks           │   │  - Response serialization   │
└────────────┬────────────┘   └──────────────┬──────────────┘
             │                               │
             └───────────────┬───────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│  PHASE 5: Benchmarking & Validation                        │
│  - Synthetic benchmarks                                    │
│  - Real workload replay                                    │
│  - Stress testing                                          │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│  PHASE 6: Documentation & Polish                           │
│  - README                                                  │
│  - API docs                                                │
│  - Benchmark report                                        │
└────────────────────────────────────────────────────────────┘
```

---

## WEEKLY SCHEDULE

| Week | Phase | Focus | Deliverables | Kill-Switch Check |
|------|-------|-------|--------------|-------------------|
| 1 | P1 | BinaryLLM integration experiments | Benchmark data | Latency < 1ms? |
| 2 | P1→P2 | Projection wrapper + cache skeleton | `binary/adapter.py` | Memory OK? |
| 3 | P2 | Core cache implementation | `core/cache.py` | Lookup < 5ms? |
| 4 | P3 | LRU baseline + diversity v1 | `eviction/*.py` | Eviction works? |
| 5 | P3→P4 | Diversity benchmarks + proxy start | Benchmark data | +10% hit rate? |
| 6 | P4 | Proxy complete | `proxy/*.py` | E2E works? |
| 7 | P5 | Benchmarking | Results report | Targets met? |
| 8 | P6 | Documentation | README, docs | Ready to ship? |

---

## OUTPUT FORMAT

When using this prompt, ask Claude to produce:

1. **Full YAML for each phase** (using template above)
2. **Risk register** with mitigation plans
3. **Test plan** with coverage requirements
4. **Go/No-Go checklist** for each phase transition

---

## FOLLOW-UP PROMPTS

After receiving the phase plan:

1. **Code Skeleton**: "Generate the Python file structure with stub implementations for Phase 2. Include all type hints and docstrings."

2. **Test-First**: "Write the test file for `core/cache.py` before the implementation. Use pytest with parametrized tests for edge cases."

3. **Benchmark Script**: "Create a standalone benchmark script that measures projection latency, Hamming distance throughput, and memory usage. Output results as JSON."

4. **Failure Analysis**: "For Phase 3, what happens if diversity eviction adds >100ms latency? What's the fallback plan?"

---

*This prompt produces a detailed, actionable engineering plan that can be directly executed.*

