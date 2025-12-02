# STAGE 4 PROMPT: Market Signal Collection

**Agent Role:** Market Miner  
**Duration:** 2-3 hours (+ 48h wait for responses)  
**Blocking:** CONDITIONAL — At least 1 signal required

---

## CONTEXT

You have validated the technical foundation. Now you must validate that **anyone actually wants this**.

The architecture is sound. The code will work. But if 0 people care, you're building a museum piece.

---

## YOUR TASK

### Step 1: Outreach Campaign

Execute these parallel activities:

1. **LinkedIn (5 messages)**
   - Target: AI startup CTOs, ML Engineers
   - Goal: Schedule 2 calls

2. **Reddit Post**
   - Target: r/MachineLearning or r/LocalLLaMA
   - Goal: 10+ upvotes, 3+ substantive comments

3. **HackerNews** (optional)
   - Target: "Show HN" or comment on relevant thread
   - Goal: 5+ upvotes

4. **GPTCache Issue Mining**
   - Target: GitHub issues
   - Goal: Find 10+ issues indicating unmet needs

### Step 2: Document Signals

Create `validation/results/s4_market_signals.json` with all signals.

### Step 3: Evaluate

Determine if there's enough interest to proceed.

---

## OUTREACH TEMPLATES

### LinkedIn Message

```
Subject: Quick question about LLM API costs

Hi [Name],

I saw [Company] is working with LLM APIs. I'm researching a specific problem:

Many teams spend 30-50% more on OpenAI/Anthropic than necessary because 
similar queries aren't cached semantically.

I'm building a lightweight, CPU-only semantic cache that uses binary 
embeddings (48× smaller than float vectors) with <10ms lookup.

Quick question: Is this a real pain point for your team?

Not selling anything — just validating if this problem is worth solving.
Would you have 15 minutes for a quick call?

Best,
[Your Name]
```

### Reddit Post

```
Title: Building a CPU-only semantic cache for LLM APIs — would you use this?

I'm working on a semantic caching layer for OpenAI/Anthropic API calls. 
The key differentiator: binary embeddings (32 bytes per entry vs 1.5KB 
for float vectors), CPU-only operation, <10ms lookup.

Before investing more time, I want to validate interest:

1. How much do you spend on LLM APIs per month?
2. Would 30% cost savings be meaningful?
3. Would you prefer open-source (self-hosted) or paid SaaS?
4. What's the dealbreaker feature that would make this a must-have?

Current approach: Binary codes from Gaussian random projection, 
Hamming distance for similarity, diversity-aware eviction to maintain 
cache coverage.

Any feedback appreciated. Will share results when ready.
```

### GPTCache Issue Queries

Search for:
- `memory` (memory usage complaints)
- `performance` (speed issues)
- `CPU` (no GPU requests)
- `scale` (scaling problems)
- `cost` (cost reduction desires)

---

## SIGNAL CLASSIFICATION

### Positive Signals

| Type | Example | Weight |
|------|---------|--------|
| Scheduled call | "Let's chat Thursday" | 5 |
| Expressed interest | "This would be useful" | 3 |
| Upvote/Like | Reddit upvote | 1 |
| Substantive comment | Asks technical questions | 2 |
| GitHub issue alignment | "Need lighter cache" | 2 |

### Negative Signals

| Type | Example | Weight |
|------|---------|--------|
| "Not needed" | "We use GPTCache fine" | -3 |
| No response | Silence after 48h | -1 |
| "Build it yourself" | "Easy to DIY" | -2 |
| Downvote | Reddit downvote | -1 |

### Neutral Signals

| Type | Example | Weight |
|------|---------|--------|
| "Interesting" | Generic positive | 0 |
| Question only | "How does it work?" | 1 |

---

## RESULTS TEMPLATE

```json
{
  "stage": "S4",
  "name": "Market Signal Collection",
  "timestamp": "ISO-8601",
  "collection_period": "48 hours",
  "channels": {
    "linkedin": {
      "messages_sent": 5,
      "responses": 0,
      "calls_scheduled": 0,
      "positive_signals": 0,
      "negative_signals": 0
    },
    "reddit": {
      "post_url": "...",
      "upvotes": 0,
      "comments": 0,
      "positive_comments": 0,
      "negative_comments": 0
    },
    "hackernews": {
      "post_url": "...",
      "upvotes": 0,
      "comments": 0
    },
    "gptcache_issues": {
      "issues_reviewed": 0,
      "relevant_issues": [],
      "unmet_needs_found": 0
    }
  },
  "signal_summary": {
    "total_positive": 0,
    "total_negative": 0,
    "total_neutral": 0,
    "weighted_score": 0
  },
  "key_insights": [],
  "overall_status": "PENDING"
}
```

---

## ACCEPTANCE CRITERIA

| Criterion | Target | Minimum | Fail |
|-----------|--------|---------|------|
| Positive signals (weighted) | ≥10 | ≥5 | <2 |
| Scheduled calls | ≥2 | ≥1 | 0 |
| "Interested" responses | ≥3 | ≥1 | 0 |
| Negative signals | <30% | <50% | >70% |

---

## DECISION MATRIX

| Weighted Score | Calls | Decision |
|----------------|-------|----------|
| ≥10 | ≥2 | **STRONG GO** — Clear demand |
| 5-9 | 1+ | **GO** — Enough interest |
| 3-4 | 0-1 | **CONDITIONAL** — Proceed with reduced scope |
| 0-2 | 0 | **PAUSE** — Intensify outreach before Phase 1 |
| <0 | 0 | **RECONSIDER** — May not be worth building |

---

## TIMELINE

| Time | Activity |
|------|----------|
| Hour 0-1 | Draft messages, find targets |
| Hour 1-2 | Send LinkedIn messages |
| Hour 2 | Post on Reddit |
| Hour 2-3 | Review GPTCache issues |
| Hour 24 | Check initial responses |
| Hour 48 | Final evaluation |

---

## KILL TRIGGERS

| Trigger | Action |
|---------|--------|
| 0 responses after 48h | PAUSE: Revise messaging or targets |
| >70% negative signals | RECONSIDER: Market may not exist |
| "GPTCache does this" ×5 | INVESTIGATE: Is differentiation real? |

---

## IMPORTANT NOTES

1. **Be honest** — Don't oversell. You're validating, not selling.
2. **Be specific** — "Binary embeddings, 48× smaller" is better than "AI cache"
3. **Ask questions** — "Would you use this?" not "Here's what I built"
4. **Listen to objections** — They reveal real concerns
5. **Document everything** — Every response is data

---

## AFTER COMPLETION

If **STRONG GO** or **GO**: Proceed to Stage 5 with confidence  
If **CONDITIONAL**: Note concerns, proceed with caution  
If **PAUSE**: Address before investing more  
If **RECONSIDER**: Serious discussion about project viability

---

*The market doesn't care about your code quality. Validate demand before building.*

