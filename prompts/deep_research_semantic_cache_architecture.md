# DEEP RESEARCH PROMPT: Binary Semantic Cache Architecture & Phase Planning

**Version:** 1.0  
**Date:** November 28, 2025  
**Purpose:** Master-level architectural research for integrating BinaryLLM Phase 1 with Semantic Cache  
**Target Model:** Claude Opus 4.5 (Maximum reasoning depth)

---

## CONTEXT INJECTION (READ FIRST)

You are performing deep architectural research for a solo developer building a **Binary Semantic Cache with Diversity Eviction**. This system must:

1. **Reuse BinaryLLM Phase 1** (frozen, deterministic binary embedding engine)
2. **Operate without GPU** (CPU-only, no renting compute)
3. **Target high-volume LLM API users** who want to reduce costs
4. **Avoid architectural bottlenecks** that would require major rewrites
5. **Be buildable in ~8 weeks** by a single skilled developer

### Existing Infrastructure Summary

**BinaryLLM Phase 1 (COMPLETE)**:
- Gaussian random projection + sign binarization
- LSB-first packing into uint64 words
- Code lengths: 32, 64, 128, 256 bits
- Cosine-Hamming Spearman correlation: ~0.94
- Deterministic: same seed → identical results
- 157 tests, frozen at "Stability Seal"
- Key invariant: `INV-07: Packed layout LSB-first (frozen for GPU kernels)`

**Key API**:
```python
@dataclass
class BinaryEmbeddingEngine:
    encoder_spec: EncoderSpec
    dataset_spec: DatasetSpec
    code_bits: int            # 32, 64, 128, 256
    projection_type: str      # "gaussian" only
    seed: int
    normalize: bool = True

def run(
    embeddings: ArrayLike,
    *,
    metrics: Iterable[str] = ("similarity", "retrieval"),
    retrieval_k: int = 3,
) -> Dict[str, object]
```

**Merkle-RAG (CONDITIONAL GO)**: Available for reuse if needed:
- RFC 6962 Merkle tree implementation
- Append-only log + blob store
- Qdrant integration
- 14-week roadmap already planned

---

## RESEARCH DIRECTIVES

### DIRECTIVE 1: ARCHITECTURAL DEEP DIVE

Analyze and answer the following architectural questions with maximum depth:

#### 1.1 Cache Architecture Patterns

Research and compare these semantic cache architectures:

| Pattern | Description | Pros | Cons | Fit for Binary? |
|---------|-------------|------|------|-----------------|
| **Hash-based lookup** | Direct key→value after embedding hash | O(1) lookup | Collisions, no semantic matching | ? |
| **HNSW/IVFPQ hybrid** | Approximate nearest neighbor + exact ranking | Fast, proven | Complex, memory overhead | ? |
| **Flat brute-force** | Scan all cached embeddings | Simple, exact | O(n) per query | ? |
| **Locality-Sensitive Hashing** | Hash-based approximate similarity | Fast, simple | Lower recall | ? |
| **Binary Hamming scan** | Direct Hamming distance on packed codes | Very fast, CPU-only | Requires binary codes | ? |

**For each pattern, answer:**
1. How does it integrate with BinaryLLM's packed uint64 format?
2. What is the memory overhead per cached entry?
3. What is the latency profile at 10K, 100K, 1M cached entries?
4. Where are the hidden complexity traps?

#### 1.2 Integration Points with BinaryLLM Phase 1

**Critical Question:** How do we call BinaryLLM Phase 1 from the cache?

Consider these integration patterns:

```python
# Pattern A: Direct engine instantiation per query
engine = BinaryEmbeddingEngine(...)
result = engine.run(query_embedding)
binary_code = result["binary_codes"]["packed"]

# Pattern B: Pre-initialized engine with reused projection matrix
class SemanticCache:
    def __init__(self, code_bits: int, seed: int):
        self._projection = RandomProjection(input_dim, code_bits, seed)
    
    def get_binary_code(self, embedding: np.ndarray) -> np.ndarray:
        projected = self._projection.project(embedding)
        return pack_codes(binarize_sign(projected))

# Pattern C: Batch processing with amortized overhead
class BatchCache:
    def process_batch(self, embeddings: List[np.ndarray]):
        # Batch all embeddings, run once
        ...
```

**For each pattern, analyze:**
1. Memory overhead (projection matrix size for 384-dim → 256-bit)
2. Latency per query (projection + binarization + packing)
3. Thread safety considerations
4. Determinism guarantees under concurrent access
5. Integration complexity with existing Phase 1 tests

#### 1.3 Eviction Policy Architecture

The "diversity eviction" is the novel component. Research these eviction strategies:

| Strategy | Description | Compute Cost | Implementation Complexity |
|----------|-------------|--------------|---------------------------|
| **LRU** | Least Recently Used | O(1) | Low |
| **LFU** | Least Frequently Used | O(1) | Low |
| **MMR-based** | Maximal Marginal Relevance eviction | O(n²) per eviction | Medium |
| **Cluster-based** | Evict from over-represented clusters | O(n log n) per eviction | Medium |
| **Hybrid LRU+Diversity** | LRU with diversity penalty | O(n) per eviction | Medium |

**For each strategy, answer:**
1. How do we compute "diversity" using Hamming distance on binary codes?
2. What data structures are needed (heap, tree, hash)?
3. What is the memory overhead per cached entry?
4. How do we handle cache warmup (cold start)?
5. What are the failure modes under high load?

---

### DIRECTIVE 2: PHASE DECOMPOSITION

Decompose the project into phases with **explicit dependencies** and **kill-switches**.

#### Required Phase Structure

For each phase, provide:

```yaml
phase_id: P1
name: "Core Cache Infrastructure"
duration_weeks: 2
depends_on: []
deliverables:
  - name: "cache_core.py"
    description: "..."
    loc_estimate: 300
  - name: "test_cache_core.py"
    description: "..."
    loc_estimate: 200
kill_switches:
  - id: "KS-P1-1"
    trigger: "Cache lookup latency > 10ms at 10K entries"
    action: "STOP: Redesign storage layer"
  - id: "KS-P1-2"
    trigger: "Memory per entry > 1KB"
    action: "STOP: Optimize data structures"
integration_risks:
  - risk: "BinaryLLM projection matrix not thread-safe"
    mitigation: "..."
    effort_if_triggered: "1 week"
entry_criteria:
  - "BinaryLLM Phase 1 tests pass"
  - "Seed determinism verified"
exit_criteria:
  - "10K entries cached without OOM"
  - "Lookup latency < 5ms at 10K entries"
  - "100% test coverage on core operations"
```

#### Phases to Define

1. **P1: Core Cache Infrastructure** (Weeks 1-2)
2. **P2: BinaryLLM Integration** (Weeks 2-3)
3. **P3: Diversity Eviction Engine** (Weeks 3-5)
4. **P4: LLM API Proxy Layer** (Weeks 5-6)
5. **P5: Benchmarking & Validation** (Weeks 6-7)
6. **P6: Documentation & Polish** (Weeks 7-8)

---

### DIRECTIVE 3: BOTTLENECK ANALYSIS

Identify and analyze potential bottlenecks with maximum paranoia.

#### 3.1 Compute Bottlenecks

| Operation | Estimated Cost | At 10K entries | At 100K entries | At 1M entries |
|-----------|----------------|----------------|-----------------|---------------|
| Embed query (external API) | 100-500ms | ??? | ??? | ??? |
| Project to binary (CPU) | ??? | ??? | ??? | ??? |
| Hamming distance scan | ??? | ??? | ??? | ??? |
| Diversity calculation | ??? | ??? | ??? | ??? |
| Eviction decision | ??? | ??? | ??? | ??? |

**Fill in the ???s with realistic estimates and justify.**

#### 3.2 Memory Bottlenecks

| Component | Size per entry | At 10K | At 100K | At 1M |
|-----------|---------------|--------|---------|-------|
| Binary code (256-bit) | 32 bytes | ??? | ??? | ??? |
| Metadata (timestamp, etc.) | ??? | ??? | ??? | ??? |
| LRU pointers | ??? | ??? | ??? | ??? |
| Diversity index | ??? | ??? | ??? | ??? |
| Response payload | ??? | ??? | ??? | ??? |

**Calculate total memory and identify the limiting factor.**

#### 3.3 Architectural Bottlenecks

For each, explain why it could become a bottleneck:

1. **Single-threaded projection**: 
2. **Lock contention on cache access**:
3. **Disk I/O for response persistence**:
4. **Embedding API rate limits**:
5. **Garbage collection pressure**:

---

### DIRECTIVE 4: HOSTILE REVIEW (SELF-ATTACK)

Attack the proposed architecture with maximum hostility.

#### 4.1 Why This Will Fail

List 10 reasons why this project will fail, ordered by severity:

| # | Reason | Severity | Evidence | Mitigation Cost |
|---|--------|----------|----------|-----------------|
| 1 | ??? | CRITICAL | ??? | ??? |
| 2 | ??? | HIGH | ??? | ??? |
| ... | ... | ... | ... | ... |

#### 4.2 Competition Threat Analysis

| Competitor | What They Have | When They Could Ship | Why They Might Not |
|------------|----------------|---------------------|-------------------|
| GPTCache (Zilliz) | ??? | ??? | ??? |
| LangChain caching | ??? | ??? | ??? |
| Redis semantic cache | ??? | ??? | ??? |
| OpenAI native caching | ??? | ??? | ??? |

#### 4.3 Why Binary Might Be Wrong

Challenge the core assumption that binary embeddings are valuable for semantic caching:

1. **Precision loss**: At what similarity threshold do binary codes produce false positives/negatives?
2. **Domain sensitivity**: Does Cosine-Hamming correlation hold for all query types?
3. **Embedding model dependency**: What if the user's embedding model differs from training?
4. **Cache hit rate**: What empirical cache hit rate is needed for ROI?

---

### DIRECTIVE 5: ENGINEERING SPEC OUTPUT

Produce a concrete engineering specification with:

#### 5.1 Module Structure

```
semantic_cache/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── cache.py              # Core cache operations
│   ├── storage.py            # Persistence layer
│   └── config.py             # Configuration management
├── binary/
│   ├── __init__.py
│   ├── adapter.py            # BinaryLLM Phase 1 integration
│   └── similarity.py         # Hamming distance operations
├── eviction/
│   ├── __init__.py
│   ├── lru.py                # LRU baseline
│   ├── diversity.py          # Diversity-aware eviction
│   └── hybrid.py             # Combined strategy
├── proxy/
│   ├── __init__.py
│   ├── server.py             # HTTP proxy server
│   └── middleware.py         # Request/response interception
├── benchmarks/
│   └── ...
└── tests/
    └── ...
```

**For each module, specify:**
- Purpose (1 sentence)
- Public API (function signatures)
- Dependencies (internal and external)
- Test requirements

#### 5.2 Data Structures

Define the core data structures with byte-level precision:

```python
@dataclass(frozen=True)
class CacheEntry:
    """Single cached LLM response."""
    # Define all fields with types and sizes
    ...

@dataclass
class CacheIndex:
    """In-memory index for fast lookup."""
    # Define structure for binary code storage
    ...

@dataclass
class EvictionState:
    """State for diversity-aware eviction."""
    # Define what state is needed
    ...
```

#### 5.3 Critical Invariants

Define the invariants that MUST hold:

| ID | Invariant | Enforcement | Test |
|----|-----------|-------------|------|
| INV-SC-01 | Cache lookup is deterministic | ??? | ??? |
| INV-SC-02 | Binary codes match source embeddings | ??? | ??? |
| INV-SC-03 | Eviction never removes recently used entries | ??? | ??? |
| ... | ... | ... | ... |

---

### DIRECTIVE 6: DECISION MATRIX

For key architectural decisions, provide a structured comparison:

#### Decision 1: Storage Backend

| Option | Pros | Cons | Complexity | Recommendation |
|--------|------|------|------------|----------------|
| In-memory only | Fast | No persistence | Low | ??? |
| SQLite | Durable, SQL | Slower | Medium | ??? |
| RocksDB | Fast, durable | C++ dependency | High | ??? |
| Memory-mapped file | Fast, durable | Complex | High | ??? |

**Final Decision:** ???  
**Justification:** ???

#### Decision 2: Similarity Search Method

(Similar table format)

#### Decision 3: Eviction Granularity

(Similar table format)

---

## OUTPUT FORMAT

Produce your response in the following structure:

```markdown
# BINARY SEMANTIC CACHE — ARCHITECTURAL RESEARCH REPORT

## Executive Summary
(3 paragraphs max)

## 1. Architectural Analysis
### 1.1 Cache Pattern Recommendation
### 1.2 BinaryLLM Integration Design
### 1.3 Eviction Engine Architecture

## 2. Phase Plan
### Phase 1: ...
### Phase 2: ...
...

## 3. Bottleneck Analysis
### 3.1 Compute Bottlenecks
### 3.2 Memory Bottlenecks
### 3.3 Architectural Bottlenecks

## 4. Hostile Review
### 4.1 Failure Modes
### 4.2 Competition Threats
### 4.3 Core Assumption Challenges

## 5. Engineering Specification
### 5.1 Module Structure
### 5.2 Data Structures
### 5.3 Invariants

## 6. Decision Log
### Decision 1: ...
### Decision 2: ...
...

## 7. Risk Register
(Table format with severity, likelihood, mitigation)

## 8. Recommended Next Steps
(Numbered list of immediate actions)
```

---

## REASONING GUIDELINES

When analyzing, apply these thinking patterns:

1. **First Principles**: Don't assume existing patterns apply. Question everything.
2. **Adversarial Thinking**: What would an attacker or hostile competitor do?
3. **Constraint Satisfaction**: Always check against: no GPU, 8 weeks, solo developer.
4. **Failure Mode Analysis**: For every design choice, ask "how does this fail?"
5. **Integration Paranoia**: Assume interfaces between components will break.
6. **Scaling Analysis**: What happens at 10x, 100x, 1000x the expected load?

---

## APPENDIX: BinaryLLM Phase 1 Technical Reference

(Attach the full TECHNICAL_REFERENCE.md content here for complete context)

---

**END OF PROMPT**

---

# HOW TO USE THIS PROMPT

1. **Copy this entire document** into a new Claude conversation
2. **Attach BinaryLLM TECHNICAL_REFERENCE.md** as context
3. **Optionally attach** Merkle-RAG architecture docs if considering integration
4. **Ask Claude to produce the full architectural research report**
5. **Follow up with specific questions** on any section that needs deeper analysis

## Suggested Follow-up Prompts

After receiving the initial response, consider these follow-ups:

1. **Deeper Phase Analysis**: "Expand Phase 2 (BinaryLLM Integration) with code-level details. Show me the exact adapter interface and how projection matrix state is managed."

2. **Bottleneck Stress Test**: "Assume 1M cache entries and 100 QPS. Walk through a single request lifecycle and identify every microsecond of latency."

3. **Eviction Deep Dive**: "Compare diversity eviction algorithms in detail. Show me the mathematical formulation and pseudocode for the top 3 candidates."

4. **Failure Scenario**: "Assume the cache has been running for 1 week and suddenly starts returning stale responses. Trace through all possible causes and how to detect each."

5. **Competition Preemption**: "GPTCache just announced semantic similarity caching. How do we differentiate? What can we build that they can't easily copy?"

---

*This prompt is designed to extract maximum architectural insight from Claude Opus 4.5. Use it as a starting point and iterate based on the responses.*

