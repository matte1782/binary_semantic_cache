# PROMPT MASTERY GUIDE — Maximizing Claude Opus 4.5 for Architectural Research

**Version:** 1.0  
**Purpose:** Meta-guide for using the research prompts effectively  
**Philosophy:** Experience and mastery through deliberate practice

---

## CORE PRINCIPLES FOR OPUS 4.5 MASTERY

### Principle 1: Structured Complexity Injection

Claude Opus 4.5 excels at complex, multi-layered reasoning. Feed it structured complexity:

```
❌ WEAK: "Design a cache system"

✅ STRONG: "Design a cache system with these constraints:
   1. Must integrate with existing BinaryLLM Phase 1 (API surface attached)
   2. Must operate under 100MB RAM for 10K entries
   3. Must achieve <10ms lookup latency at 100K entries
   4. Must support diversity-aware eviction with configurable λ parameter
   5. Must be testable with deterministic seeds
   
   For each constraint, explain:
   - How it affects the architecture
   - What trade-offs it forces
   - What failure modes it introduces"
```

### Principle 2: Adversarial Framing

Opus 4.5 produces better analysis when asked to attack its own ideas:

```
❌ WEAK: "What are the risks?"

✅ STRONG: "You are a hostile reviewer who believes this project will fail.
   Attack it on these dimensions:
   1. Technical feasibility
   2. Market timing
   3. Competition
   4. Hidden complexity
   5. Maintenance burden
   
   For each attack, rate severity (1-10) and explain the evidence.
   Then, switch to defender role and propose mitigations.
   Finally, as an impartial judge, rate whether each mitigation is convincing."
```

### Principle 3: Explicit Output Scaffolding

Opus 4.5 follows detailed output formats precisely:

```
❌ WEAK: "Give me a plan"

✅ STRONG: "Produce output in this exact YAML structure:
   
   phase:
     id: P{N}
     name: string
     duration_weeks: int
     depends_on: list[phase_id]
     deliverables:
       - name: string
         path: string
         loc_estimate: int
         public_api: |
           def function(arg: Type) -> Return:
               '''Docstring'''
     kill_switches:
       - trigger: string
         action: 'STOP|PIVOT|REDESIGN'
         recovery_weeks: int
     exit_criteria:
       - condition: string
         test_command: string
   
   Fill in all fields. Use realistic estimates based on:
   - Solo developer working 25-30 hrs/week
   - Python with pytest
   - No external dependencies beyond numpy, aiohttp"
```

### Principle 4: Chain-of-Thought Elicitation

For complex decisions, ask for explicit reasoning chains:

```
❌ WEAK: "Which storage backend should I use?"

✅ STRONG: "Evaluate storage backends for the semantic cache:
   
   Candidates: {in-memory, SQLite, RocksDB, memory-mapped file}
   Criteria: {latency, durability, complexity, dependency cost}
   
   For each candidate:
   1. List 3 strengths with evidence
   2. List 3 weaknesses with evidence
   3. Estimate implementation effort (hours)
   4. Identify the single biggest risk
   
   After evaluating all candidates:
   1. Create a weighted decision matrix
   2. Identify any disqualifying factors
   3. Make a recommendation with confidence level (low/medium/high)
   4. Describe what would change your recommendation"
```

### Principle 5: Context Maximization

Opus 4.5 uses context deeply. Provide maximum relevant context:

```
❌ WEAK: "How should I integrate with the binary embedding engine?"

✅ STRONG: "Here is the complete BinaryLLM Phase 1 technical reference:

   [PASTE FULL TECHNICAL_REFERENCE.md]
   
   Here is the complete test suite structure:

   [PASTE test directory listing]
   
   Here is the invariants table:

   [PASTE invariants]
   
   Given this context, design an integration that:
   1. Preserves all 13 invariants (list which ones are at risk)
   2. Adds no new dependencies to Phase 1
   3. Can be tested independently of Phase 1's test suite
   4. Handles the projection matrix as a reusable resource"
```

---

## PROMPT SEQUENCE FOR SEMANTIC CACHE PROJECT

Use these prompts in order for maximum learning and mastery:

### Session 1: Architectural Deep Dive (2-3 hours)

**Goal:** Understand the full problem space before coding.

1. **Load Context**
   ```
   Attach these files to the conversation:
   - binary_llm/docs/TECHNICAL_REFERENCE.md
   - merkle_rag/iteration_2/architecture_v0.1_core.md
   - iteration_9/prompts/deep_research_semantic_cache_architecture.md
   ```

2. **Run Main Prompt**
   ```
   [Paste full content of deep_research_semantic_cache_architecture.md]
   ```

3. **Follow-Up: Bottleneck Drill**
   ```
   You identified {N} bottlenecks. For the top 3 by severity:
   
   1. Show me the exact code path where the bottleneck occurs
   2. Calculate the worst-case latency with realistic numbers
   3. Propose 3 mitigation strategies, ranked by effort
   4. For the recommended mitigation, show pseudocode
   ```

4. **Follow-Up: Competition War-Game**
   ```
   Assume GPTCache releases a new version in 2 weeks with:
   - Binary embedding support
   - Diversity-aware eviction
   - 10x faster lookup
   
   How do we respond? Options:
   A) Abandon project
   B) Pivot to different differentiator
   C) Race to ship first
   D) Focus on integration moat
   
   Analyze each option with 3 pros, 3 cons, and a recommendation.
   ```

### Session 2: Engineering Planning (2-3 hours)

**Goal:** Create an executable week-by-week plan.

1. **Load Context**
   ```
   Attach:
   - Results from Session 1
   - iteration_9/prompts/engineering_phase_planner.md
   ```

2. **Run Planning Prompt**
   ```
   [Paste full content of engineering_phase_planner.md]
   
   Additional constraints:
   - Available time: 25 hours/week
   - Must maintain BinaryLLM Phase 1 tests passing at all times
   - No new GPU dependencies
   - Target: working v0.1 in 6 weeks, v0.2 with diversity in 8 weeks
   ```

3. **Follow-Up: Test-First Specification**
   ```
   For Phase 2 (Core Cache Data Structures), write the complete
   test file BEFORE the implementation.
   
   Include:
   - Unit tests for CacheEntry creation
   - Unit tests for CacheConfig validation
   - Integration tests for lookup/insert
   - Performance tests with parametrized entry counts
   - Edge case tests (empty cache, full cache, duplicate entries)
   
   Use pytest with fixtures. Include docstrings explaining each test.
   ```

4. **Follow-Up: Failure Scenarios**
   ```
   For each phase, describe:
   
   1. The most likely failure mode
   2. How you would detect it (what test would fail?)
   3. The recovery action
   4. Time cost of recovery
   
   Format as a table with columns:
   Phase | Failure Mode | Detection | Recovery | Cost
   ```

### Session 3: Implementation Scaffolding (1-2 hours)

**Goal:** Generate code skeletons to start implementation.

1. **Module Structure Generation**
   ```
   Based on the engineering plan from Session 2, generate:
   
   1. Complete directory structure with __init__.py files
   2. Stub implementations for all public APIs
   3. Type hints for all parameters and returns
   4. Docstrings with usage examples
   5. TODO comments marking implementation points
   
   Output as a series of file contents that can be directly written.
   ```

2. **Configuration System**
   ```
   Design a configuration system for the semantic cache that:
   
   1. Supports YAML config files
   2. Has sensible defaults for all values
   3. Validates constraints (e.g., code_bits must be 64, 128, or 256)
   4. Supports environment variable overrides
   5. Is fully typed with dataclasses
   
   Include:
   - config.py implementation
   - default_config.yaml
   - test_config.py with validation tests
   ```

3. **BinaryLLM Adapter**
   ```
   Create the adapter module that wraps BinaryLLM Phase 1:
   
   Requirements:
   1. Lazy initialization of projection matrix
   2. Thread-safe access to shared projection
   3. Batch processing API
   4. Consistent seeding with Phase 1
   
   Show:
   - binary/adapter.py (full implementation)
   - test_adapter.py (unit tests)
   - Integration test that verifies determinism
   ```

### Session 4: Benchmarking Framework (1 hour)

**Goal:** Create measurement infrastructure.

1. **Benchmark Suite**
   ```
   Create a comprehensive benchmark suite that measures:
   
   1. Projection latency (single embedding, batch of 100, batch of 1000)
   2. Hamming distance computation (10K, 100K, 1M entries)
   3. Cache lookup latency (varying cache sizes)
   4. Memory usage (per-entry overhead)
   5. Eviction latency (LRU vs diversity)
   
   Output format: JSON with structured results
   Visualization: Generate matplotlib charts
   
   Include:
   - benchmarks/run_benchmarks.py
   - benchmarks/visualize.py
   - benchmarks/README.md
   ```

---

## ADVANCED TECHNIQUES

### Technique 1: Iterative Refinement

Don't accept the first response. Iterate:

```
Response 1 → "This is good, but I notice you didn't address X. 
              Expand that section with specific numbers."

Response 2 → "The numbers look reasonable but assume Y. 
              What if Y is false? Recalculate."

Response 3 → "Now compare this approach to Z alternative. 
              Which is better under what conditions?"
```

### Technique 2: Role Switching

Use multiple perspectives:

```
"Analyze this architecture as:

1. A systems engineer focused on latency
2. A product manager focused on user experience
3. A security researcher focused on attack surface
4. A maintainability expert focused on code quality
5. A skeptical investor focused on competitive advantage

For each role, provide the top 3 concerns and how they'd be addressed."
```

### Technique 3: Concrete Anchoring

Ground abstract concepts in specifics:

```
❌ WEAK: "The cache should be fast"

✅ STRONG: "The cache must achieve:
   - p50 lookup latency: <5ms
   - p99 lookup latency: <20ms
   - Insert latency: <10ms
   - Eviction batch (100 entries): <50ms
   
   These targets assume:
   - Cache size: 100K entries
   - Entry size: 32 bytes binary code + 1KB average response
   - Hardware: Single core, 4GB RAM
   
   If targets cannot be met, explain why and propose revised targets."
```

### Technique 4: Failure Pre-Mortem

Assume failure and work backward:

```
"It is 3 months from now. The semantic cache project has failed.

The post-mortem revealed these causes:
1. [You fill in the most likely cause]
2. [You fill in the second most likely cause]
3. [You fill in the third most likely cause]

For each cause:
- When could we have detected it?
- What early warning signs did we miss?
- What decision would have prevented it?

Now, design the monitoring and kill-switches to catch these failures early."
```

---

## PROMPT CHAINING STRATEGY

For maximum learning, chain prompts across sessions:

```
Session 1: Architecture Research
    ↓ (save output as architecture_v1.md)
    
Session 2: Hostile Review of architecture_v1.md
    ↓ (save output as hostile_review_v1.md)
    
Session 3: Revise architecture based on hostile review
    ↓ (save output as architecture_v2.md)
    
Session 4: Engineering Plan for architecture_v2.md
    ↓ (save output as engineering_plan_v1.md)
    
Session 5: Test Specification from engineering_plan_v1.md
    ↓ (save output as test_spec_v1.md)
    
Session 6: Implementation Scaffolding
    ↓ (generate actual code files)
    
Session 7: Benchmark Results Analysis
    ↓ (iterate on implementation)
```

---

## MASTERY CHECKLIST

Track your progress through these milestones:

- [ ] **Architecture Understanding**: Can explain the system to someone else in 5 minutes
- [ ] **Risk Awareness**: Can list top 5 risks without looking at notes
- [ ] **Code Mapping**: Know which file implements which component
- [ ] **Test Coverage**: Every public function has at least 3 tests planned
- [ ] **Benchmark Baseline**: Have concrete numbers for current Phase 1 performance
- [ ] **Kill-Switch Ready**: Know exactly what triggers project termination
- [ ] **Competition Prepared**: Can articulate differentiation in one sentence
- [ ] **Documentation First**: README written before full implementation
- [ ] **Integration Verified**: BinaryLLM Phase 1 tests still pass with wrapper
- [ ] **First User**: Have run the cache on real API traffic (even synthetic)

---

## LEARNING ACCELERATION TIPS

1. **Export All Responses**: Save every Claude response as markdown for future reference
2. **Highlight Surprises**: Mark anything unexpected for deeper investigation
3. **Question Everything**: If Claude says "typically X", ask "show me the edge case where X fails"
4. **Build Mental Models**: Draw diagrams of architectures discussed
5. **Time Your Sessions**: Track how long each type of analysis takes
6. **Reuse Patterns**: Notice which prompt patterns work best and template them
7. **Fail Fast**: If a direction isn't working, ask Claude for alternatives

---

## FINAL NOTE

The goal is **mastery through deliberate practice**, not just getting answers. 

Each prompt session should leave you with:
1. A specific artifact (document, code, or analysis)
2. At least one surprise or insight
3. A question for the next session
4. Increased confidence in the architecture

The prompts in this directory are designed to be **reusable tools** that you can adapt for future projects. The patterns transfer to any complex engineering endeavor.

---

*"The master has failed more times than the beginner has tried."*

---

*End of Prompt Mastery Guide*

