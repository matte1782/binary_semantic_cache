# Threshold Tuning Guide

**Date:** 2025-11-29  
**Version:** 1.0

---

## Understanding Semantic vs Topical Similarity

### The Core Challenge

Embedding models distinguish between:
1. **Semantic similarity:** Same meaning, possibly different words
2. **Topical similarity:** Same topic, different meaning

**Example:**
- Query 1: "What's the weather forecast?"
- Query 2: "What's the weather like today?"

These are **topically similar** (both about weather) but **semantically different** (forecast vs current).

Most embedding models (including `nomic-embed-text`) will score these around 0.70-0.75 similarity, NOT 0.90+.

---

## Why This Matters for Caching

A semantic cache must decide:
- **High threshold (0.85+):** Only cache near-identical queries (safer but more cache misses)
- **Low threshold (0.70-0.75):** Cache topically related queries (more hits but risk wrong answers)

**Golden Rule:** A cache returning the wrong answer is worse than a cache miss.

---

## How to Find Your Optimal Threshold

### Step 1: Create a Test Corpus

```python
TEST_QUERIES = [
    # Semantic duplicates (should definitely cache)
    ("How do I sort a list in Python?", "Python list sorting", True),
    ("What's the Python syntax to sort a list?", "Python list sorting", True),
    
    # Topical but different (maybe cache?)
    ("How do I sort in ascending order?", "sort direction", False),
    ("How do I sort in descending order?", "sort direction", False),
    
    # Completely different (should never cache)
    ("What is quantum computing?", "unrelated", False),
]
```

### Step 2: Run Threshold Sweep

```python
from binary_semantic_cache.validation.threshold_sweep_experiment import run_threshold_sweep

# Run sweep from 0.6 to 0.9
thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
results = []

for threshold in thresholds:
    cache = BinarySemanticCache(
        encoder=encoder,
        similarity_threshold=threshold
    )
    # Test your queries...
    accuracy = evaluate_cache_accuracy(cache, TEST_QUERIES)
    results.append((threshold, accuracy))

# Find optimal threshold
optimal = max(results, key=lambda x: x[1])
print(f"Best threshold: {optimal[0]} (accuracy: {optimal[1]})")
```

### Step 3: Validate on Real Data

Test with your actual use case:
1. Log real user queries for a week
2. Manually label which should cache-hit
3. Run threshold sweep on this real dataset
4. Pick threshold with best precision/recall tradeoff

---

## Recommended Defaults by Embedder

Based on our testing with common embedding models:

| Embedding Model | Recommended Threshold | Notes |
|-----------------|----------------------|-------|
| **nomic-embed-text** | 0.80 | Good balance, tested extensively |
| **OpenAI text-embedding-3** | 0.85 | Higher quality embeddings, can be stricter |
| **sentence-transformers** | 0.82 | Varies by specific model |
| **Cohere embed-v3** | 0.83 | Good semantic discrimination |
| **Local models (BERT-based)** | 0.78 | Often noisier, need lower threshold |

---

## When to Adjust the Threshold

### Raise Threshold (Stricter Matching)
- **Symptoms:** Cache returning incorrect/unexpected answers
- **When:** Factual Q&A, medical/legal queries, high-stakes applications
- **Try:** 0.85 → 0.88 → 0.90

### Lower Threshold (More Permissive)
- **Symptoms:** Very low cache hit rate despite similar queries
- **When:** Conversational AI, creative writing, general chatbots
- **Try:** 0.80 → 0.77 → 0.75

### Domain-Specific Tuning
Different domains need different thresholds:
- **Technical documentation:** 0.85+ (precision critical)
- **Customer support:** 0.78-0.82 (topic matching OK)
- **Creative writing:** 0.75-0.80 (more flexibility)

---

## Advanced: Per-Query Threshold

For maximum control, implement query-specific thresholds:

```python
class AdaptiveBinarySemanticCache(BinarySemanticCache):
    def get(self, embedding, query_type=None):
        # Adjust threshold based on query type
        if query_type == "factual":
            threshold = 0.88
        elif query_type == "conversational":
            threshold = 0.75
        else:
            threshold = self._threshold
            
        # Use custom threshold for this query
        return self._get_with_threshold(embedding, threshold)
```

---

## Debugging Threshold Issues

### Check Your Similarity Distribution

```python
# Analyze similarity scores in your cache
similarities = []
for query in test_queries:
    result = cache.get(query_embedding)
    if result:
        similarities.append(result.similarity)

import matplotlib.pyplot as plt
plt.hist(similarities, bins=50)
plt.xlabel('Similarity Score')
plt.ylabel('Count')
plt.title('Cache Hit Similarity Distribution')
plt.axvline(x=cache._threshold, color='r', linestyle='--', label='Threshold')
plt.show()
```

If most hits cluster just above threshold, you may need to adjust.

### Common Pitfalls

1. **"Why isn't 'weather forecast' matching 'weather today'?"**
   - These are different questions. Working as intended.
   - Lower threshold if you want topic matching.

2. **"My cache hit rate is only 20%"**
   - Check if queries are actually semantic duplicates
   - Users rarely ask the exact same question twice
   - Consider implementing query normalization first

3. **"Cache returns wrong answers sometimes"**
   - Threshold too low
   - Increase by 0.05 and re-test

---

## Production Best Practices

1. **Start Conservative:** Begin with threshold=0.85, lower if needed
2. **Log Everything:** Track similarity scores for all cache hits/misses
3. **A/B Test:** Run different thresholds for different user segments
4. **Monitor Metrics:**
   - Cache hit rate
   - User satisfaction (for cached responses)
   - False positive rate

5. **Implement Safeguards:**
```python
# Never cache below a minimum threshold
MIN_SAFE_THRESHOLD = 0.70

if similarity < MIN_SAFE_THRESHOLD:
    return None  # Always miss, too risky
```

---

## Conclusion

The optimal threshold depends on:
- Your embedding model
- Your use case (factual vs conversational)
- Your tolerance for incorrect cache hits

Start with 0.80, measure extensively, and adjust based on real-world performance.

Remember: **It's better to miss the cache than return the wrong answer.**
