# Stage 2: Decision Log

## Objective
Consolidate all architectural decisions made during research into a single, versioned document.

## Deliverable
`docs/DECISION_LOG_v1.md` containing:
1. All major architectural choices with rationale
2. Trade-offs considered and rejected alternatives
3. Assumptions that must hold for the architecture to work
4. Kill triggers derived from assumptions

## Why This Matters
- Prevents knowledge loss as the project evolves
- Makes future hostile reviews faster
- Documents what was rejected and why (avoids re-litigating)
- Creates accountability for assumptions

## Input Sources
1. `iteration_9/research/architecture_report_v1.md` (if available)
2. `iteration_9/engineering/engineering_plan_v1.md` (if available)
3. BinaryLLM Phase 1 specifications
4. All hostile review feedback

## Decision Categories

### D1. Binary Encoding
- Why 256 bits? (trade-off: precision vs speed)
- Why Gaussian projection? (vs learned, PQ, etc.)
- Why sign binarization? (vs thresholding)

### D2. Cache Architecture
- Why in-memory first? (vs disk-backed)
- Why single-node? (vs distributed)
- Why flat search? (vs tree/graph index)

### D3. Similarity Threshold
- Default threshold (e.g., 0.85)
- Why Hamming normalized? (vs raw distance)
- Calibration strategy

### D4. Eviction Strategy
- Why LRU first? (vs diversity, hybrid)
- Deferred: diversity eviction as Phase 2

### D5. API Design
- OpenAI-compatible proxy
- Why FastAPI? (vs other frameworks)
- Streaming support strategy

### D6. Integration Pattern
- BinaryLLM as vendored dependency
- Path insertion vs package install
- Version pinning strategy

## Template Per Decision

```markdown
## Decision: [SHORT_NAME]

**Date:** YYYY-MM-DD  
**Status:** ACCEPTED | DEFERRED | REJECTED

### Context
Why was this decision needed?

### Options Considered
1. Option A: [description]
2. Option B: [description]
3. Option N: [description]

### Decision
We chose Option X because...

### Consequences
- Positive: ...
- Negative: ...
- Risks: ...

### Assumptions
- [List assumptions this depends on]

### Kill Trigger
If [condition], reconsider this decision.
```

## Time Budget
1-2 hours total

## Kill Trigger
If consolidation reveals fundamental contradictions in architecture → STOP and resolve before proceeding.

