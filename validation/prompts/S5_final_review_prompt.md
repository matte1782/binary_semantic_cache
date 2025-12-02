# STAGE 5 PROMPT: Final Hostile Review

**Agent Role:** Hostile Reviewer (NVIDIA-Grade)  
**Duration:** 1-2 hours  
**Blocking:** YES — Final gate before Phase 1

---

## CONTEXT

All validation stages are complete. You must now conduct a **comprehensive hostile review** of everything before committing to 200+ hours of production work.

This is your last chance to kill the project cheaply.

---

## YOUR TASK

### Step 1: Review All Stage Results

Load and review:
- `validation/results/s0_import_results.json`
- `validation/results/s1_latency_results.json`
- `validation/stages/S2_DECISION_LOG.md`
- `validation/results/s3_poc_results.json`
- `validation/results/s4_market_signals.json`

### Step 2: Attack Everything

For each area, find weaknesses:
- Technical assumptions
- Market validation
- Competitive threats
- Hidden complexity
- Timeline risks

### Step 3: Issue Final Verdict

Create `validation/GO_NO_GO_DECISION.md` with:
- Summary of all stage results
- Identified risks and mitigations
- Final recommendation
- Conditions for proceeding

---

## ATTACK VECTORS

### 1. Technical Attacks

| Attack | Question | Evidence Required |
|--------|----------|-------------------|
| Import fragility | Can imports break in production? | S0 results |
| Latency variance | Is p99 acceptable, not just mean? | S1 results |
| Memory growth | Does memory grow linearly? | S1, S3 results |
| Correlation stability | Is 0.85 Spearman consistent? | S3 results |
| Threshold sensitivity | What happens at 0.80 or 0.90? | Not tested |
| Concurrency | Tested under concurrent access? | NOT TESTED |

### 2. Market Attacks

| Attack | Question | Evidence Required |
|--------|----------|-------------------|
| Signal quality | Are signals from real potential users? | S4 results |
| Competition response | What if GPTCache adds binary? | Risk assessment |
| Willingness to pay | Did anyone mention paying? | S4 responses |
| Alternative solutions | Did they mention existing tools? | S4 responses |

### 3. Execution Attacks

| Attack | Question | Evidence Required |
|--------|----------|-------------------|
| Timeline realism | Is 9 weeks achievable solo? | Engineering plan |
| Scope creep risk | What could expand? | Decision log |
| Kill-switch enforcement | Will you actually kill if triggered? | Commitment |
| Distraction risk | What else competes for time? | Personal assessment |

### 4. Architecture Attacks

| Attack | Question | Evidence Required |
|--------|----------|-------------------|
| Decision contradictions | Any conflicts in decision log? | S2 review |
| Missing decisions | What's not decided yet? | S2 gaps |
| Assumption validity | All assumptions tested? | Validation results |
| Fallback plans | What if primary approach fails? | Engineering plan |

---

## CHECKLIST

### Technical Validation

| Check | Status | Notes |
|-------|--------|-------|
| S0: All imports successful | [ ] | |
| S1: Projection latency acceptable | [ ] | |
| S1: Hamming scan latency acceptable | [ ] | |
| S1: Memory footprint acceptable | [ ] | |
| S3: Exact match 100% | [ ] | |
| S3: Similar match ≥70% | [ ] | |
| S3: Random hit rate ≤30% | [ ] | |
| S3: Correlation ≥0.85 | [ ] | |
| S3: Memory stable | [ ] | |

### Market Validation

| Check | Status | Notes |
|-------|--------|-------|
| S4: At least 1 positive signal | [ ] | |
| S4: Negative signals <50% | [ ] | |
| S4: No competitor announcement | [ ] | |

### Documentation Validation

| Check | Status | Notes |
|-------|--------|-------|
| S2: Decision log complete | [ ] | |
| S2: No contradictions | [ ] | |
| All results documented | [ ] | |

### Execution Readiness

| Check | Status | Notes |
|-------|--------|-------|
| Engineering plan finalized | [ ] | |
| Kill-switches defined | [ ] | |
| Phase 1 tasks clear | [ ] | |
| Time commitment confirmed | [ ] | |

---

## VERDICT OPTIONS

### STRONG GO ✅

All checks pass. Proceed with full confidence.

**Requirements:**
- All S0-S3 tests pass
- S4 weighted score ≥10
- No contradictions in S2
- No critical gaps found

### GO ⚠️

Most checks pass with minor concerns. Proceed with caution.

**Requirements:**
- All S0-S3 tests pass (some acceptable)
- S4 weighted score ≥5
- Minor issues documented with mitigations
- Timeline may need adjustment

### CONDITIONAL GO 🟡

Significant concerns exist. Proceed only if addressed.

**Requirements:**
- S0-S3 pass but with concerns
- S4 signals weak
- Must define specific conditions before continuing

### NO-GO 🔴

Critical blockers exist. Do not proceed until resolved.

**Examples:**
- S0 import fails
- S1 latency in "Fail" range
- S3 core tests fail
- S4 overwhelmingly negative

### KILL ⛔

Fundamental flaw discovered. Terminate project.

**Examples:**
- Binary encoding doesn't preserve similarity
- No market demand whatsoever
- Competitor already shipped equivalent
- Infeasible with available resources

---

## OUTPUT TEMPLATE

```markdown
# GO/NO-GO DECISION: Binary Semantic Cache

**Date:** [DATE]
**Reviewer:** [NAME]
**Status:** [STRONG GO | GO | CONDITIONAL GO | NO-GO | KILL]

---

## Executive Summary

[2-3 sentence summary of decision and key factors]

---

## Stage Results Summary

| Stage | Status | Key Finding |
|-------|--------|-------------|
| S0: Import Test | PASS/FAIL | [1-sentence summary] |
| S1: Latency Validation | PASS/ACCEPTABLE/FAIL | [1-sentence summary] |
| S2: Decision Log | COMPLETE/INCOMPLETE | [1-sentence summary] |
| S3: PoC | PASS/FAIL | [1-sentence summary] |
| S4: Market Signals | [score] | [1-sentence summary] |

---

## Risks Identified

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| ... | HIGH/MED/LOW | HIGH/MED/LOW | ... |

---

## Conditions for Proceeding

1. [Condition 1]
2. [Condition 2]
3. ...

---

## Recommendation

[Detailed recommendation with reasoning]

---

## Next Steps

If GO:
1. Create project skeleton
2. Begin Phase 1
3. [Additional steps]

If NO-GO:
1. Address [specific blockers]
2. Re-validate
3. Return to this stage

If KILL:
1. Archive all materials
2. Document lessons learned
3. Redirect effort to [alternatives]

---

## Sign-Off

- [ ] Reviewer confirms all stages reviewed
- [ ] Reviewer confirms no critical gaps
- [ ] Reviewer commits to kill-switch enforcement

Signed: _________________ Date: _________
```

---

## HOSTILE QUESTIONS TO ANSWER

Before signing off, answer these honestly:

1. **Would you bet $10,000 that this works as described?**
2. **If you're wrong about latency, what's the fallback?**
3. **If 0 users care after Week 8, was this worth it?**
4. **What's the ONE thing most likely to kill this project?**
5. **Are you proceeding because evidence supports it, or because you want to?**

---

*This is your last cheap exit. Use it wisely.*

