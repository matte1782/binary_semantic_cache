# Using Ollama for Local Testing

This guide explains how to use Ollama for real embedding generation and end-to-end testing of the Binary Semantic Cache.

## ⚠️ IMPORTANT: Embedding Models vs Chat Models

**Not all Ollama models support embeddings!**

| Model Type | Supports Embeddings | Examples |
|------------|---------------------|----------|
| **Embedding Models** | ✅ YES | `nomic-embed-text`, `mxbai-embed-large` |
| **Chat/LLM Models** | ❌ NO | `kimi`, `llama3`, `qwen`, `gemma`, `glm` |

**Cloud models like `kimi-k2:1t-cloud` are chat models and will NOT work for embeddings.**

---

## Quick Start

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download
```

### 2. Start Ollama Server

```bash
ollama serve
```

### 3. Pull an EMBEDDING Model

```bash
# Pull a dedicated embedding model (REQUIRED)
ollama pull nomic-embed-text
```

### 4. Run the End-to-End Test

```bash
cd binary_semantic_cache
python validation/ollama_end_to_end_test.py --model nomic-embed-text
```

---

## Choosing the Right Model (Embedding vs Chat)

### ✅ Embedding Models (USE THESE)

These models support the `/api/embeddings` endpoint:

| Model | Dimensions | Quality | Speed |
|-------|------------|---------|-------|
| `nomic-embed-text` | 768 | ⭐⭐⭐ Good | ⚡ Fast (~50ms) |
| `mxbai-embed-large` | 1024 | ⭐⭐⭐⭐ Better | ⚡ Fast (~80ms) |
| `snowflake-arctic-embed` | 1024 | ⭐⭐⭐⭐ Better | ⚡ Fast (~80ms) |
| `all-minilm` | 384 | ⭐⭐ OK | ⚡⚡ Very Fast |
| `bge-large` | 1024 | ⭐⭐⭐⭐ Better | ⚡ Fast |

**Recommended:** Start with `nomic-embed-text` - good balance of quality and speed.

### ❌ Chat Models (DO NOT USE)

These models are for chat/completion only and **do not support embeddings**:

- `kimi-k2:1t-cloud` ❌
- `llama3`, `llama3.2` ❌
- `qwen2.5:7b`, `qwen3-coder:480b-cloud` ❌
- `gemma3:1b` ❌
- `deepseek-r1:1.5b` ❌
- `glm-4.6:cloud` ❌
- `gpt-oss:*-cloud` ❌
- Any model with `-cloud` suffix (typically) ❌

**Why cloud models don't work:** Cloud models in Ollama are primarily hosted for chat/completion. They don't implement the embedding endpoint that we need.

---

## Installation Commands

### Install Recommended Embedding Model

```bash
# Best choice for testing
ollama pull nomic-embed-text

# Alternative: Higher quality
ollama pull mxbai-embed-large

# Alternative: Smallest/fastest
ollama pull all-minilm
```

### Run the Test

```bash
# With default model (nomic-embed-text)
python validation/ollama_end_to_end_test.py

# With specific embedding model
python validation/ollama_end_to_end_test.py --model mxbai-embed-large
```

---

## Troubleshooting

### Error: "Model does NOT support embeddings"

**Cause:** You're using a chat model instead of an embedding model.

**Fix:**
```bash
ollama pull nomic-embed-text
python validation/ollama_end_to_end_test.py --model nomic-embed-text
```

### Error: "Model not installed"

**Cause:** The embedding model isn't pulled yet.

**Fix:**
```bash
ollama pull nomic-embed-text
```

### Error: "Could not connect to Ollama"

**Cause:** Ollama server isn't running.

**Fix:**
```bash
ollama serve
```

### Error: "Single embedding must be 1D, got shape (N, dim)"

**Cause:** Batch embeddings were passed to a single-embedding function.

**Details:** The encoder now auto-detects batch vs single based on dimensions:
- 1D array `(dim,)` → treated as single embedding
- 2D array `(N, dim)` → treated as batch of N embeddings

The encoder.encode() method now handles both automatically.

### Error: "expected 1D result, got shape (N, dim)"

**Cause:** The embedding model returned an unexpected shape.

**Fix:** This indicates a problem with the Ollama model. Use a dedicated embedding model:
```bash
ollama pull nomic-embed-text
```

### Slow Embeddings (>500ms per embedding)

**Cause:** You're using a large local LLM for embeddings (if it even works).

**Fix:** Use a dedicated embedding model:
```bash
ollama pull nomic-embed-text
```

Dedicated embedding models are ~10-50x faster than LLMs.

---

## Shape Handling

### Embedding Shapes

The `OllamaEmbedder` guarantees consistent shapes:

| Method | Input | Output Shape |
|--------|-------|--------------|
| `embed_text("hello")` | single string | `(768,)` - 1D |
| `embed_texts(["a", "b"])` | list of strings | `(2, 768)` - 2D |

### Encoder Shapes

The `BinaryEncoder` auto-detects batch vs single:

| Input Shape | Treatment | Output Shape |
|-------------|-----------|--------------|
| `(768,)` | Single | `(4,)` - 1D |
| `(N, 768)` | Batch | `(N, 4)` - 2D |

### Example

```python
from binary_semantic_cache.embeddings import OllamaEmbedder
from binary_semantic_cache.core.encoder import BinaryEncoder

embedder = OllamaEmbedder(model_name="nomic-embed-text")
encoder = BinaryEncoder(embedding_dim=768, code_bits=256)

# Single embedding
single = embedder.embed_text("hello")
assert single.shape == (768,), f"Expected (768,), got {single.shape}"

single_code = encoder.encode(single)
assert single_code.shape == (4,), f"Expected (4,), got {single_code.shape}"

# Batch embeddings
batch = embedder.embed_texts(["hello", "world"])
assert batch.shape == (2, 768), f"Expected (2, 768), got {batch.shape}"

batch_codes = encoder.encode(batch)
assert batch_codes.shape == (2, 4), f"Expected (2, 4), got {batch_codes.shape}"
```

---

## Python Usage

### Basic Usage

```python
from binary_semantic_cache.embeddings import OllamaEmbedder

# Use a dedicated embedding model
embedder = OllamaEmbedder(model_name="nomic-embed-text")

# Check if it works
if embedder.is_available():
    embeddings = embedder.embed_texts(["Hello world", "How are you?"])
    print(f"Shape: {embeddings.shape}")  # (2, 768)
else:
    print("Model not available or doesn't support embeddings")
```

### Test Embedding Support

```python
from binary_semantic_cache.embeddings import OllamaEmbedder

embedder = OllamaEmbedder(model_name="nomic-embed-text")

# Explicit test
supports, message = embedder.test_embedding_support()
if supports:
    print(f"✓ {message}")
else:
    print(f"✗ {message}")
```

### List Available Embedding Models

```python
from binary_semantic_cache.embeddings import OllamaEmbedder

embedder = OllamaEmbedder()

# All models in Ollama
all_models = embedder.list_models()
print(f"All models: {all_models}")

# Only embedding-capable models
embedding_models = embedder.list_embedding_models()
print(f"Embedding models: {embedding_models}")
```

### With the Cache

```python
from binary_semantic_cache.embeddings import OllamaEmbedder
from binary_semantic_cache.core.encoder import BinaryEncoder
from binary_semantic_cache.core.cache import BinarySemanticCache

# Use embedding model
embedder = OllamaEmbedder(model_name="nomic-embed-text")

# Get embedding dimension
test_emb = embedder.embed_text("test")
embedding_dim = len(test_emb)

# Initialize encoder and cache
encoder = BinaryEncoder(embedding_dim=embedding_dim, code_bits=256, seed=42)
cache = BinarySemanticCache(encoder=encoder, max_entries=10000)

# Store an embedding
query = "How do I reverse a string in Python?"
embedding = embedder.embed_text(query)
cache.put(embedding, {"answer": "Use slicing: s[::-1]"})

# Query with similar text
similar_query = "Python string reversal"
similar_embedding = embedder.embed_text(similar_query)
result = cache.get(similar_embedding)

if result:
    print(f"Cache HIT! Similarity: {result.similarity:.3f}")
```

---

## Expected Test Output

When running with a proper embedding model:

```
Checking Ollama availability...
Ollama available at http://localhost:11434
Installed models: nomic-embed-text, llama3:latest...
Embedding models found: nomic-embed-text
✓ Model 'nomic-embed-text' supports embeddings (dim=768)
✓ Ollama is ready with embedding support

============================================================
OLLAMA END-TO-END TEST
============================================================
Embedder: OllamaEmbedder(model='nomic-embed-text', dim=768)
Embedding dimension: 768

Step 1: Embedding corpus
  [ 1]   45.2ms | How do I reverse a string in Python?
  [ 2]   38.1ms | What's the best way to reverse...
  ...

Query accuracy: 5/5 (100%)

============================================================
VERDICT
============================================================
✓ END-TO-END TEST PASSED
  - Query accuracy: 100% (≥60% required)
  - Similarity correlation: 0.87 (≥0.50 required)
```

---

## Summary

| Step | Command |
|------|---------|
| 1. Start Ollama | `ollama serve` |
| 2. Pull embedding model | `ollama pull nomic-embed-text` |
| 3. Run test | `python validation/ollama_end_to_end_test.py` |

**Remember:** Only use **embedding models** (`nomic-embed-text`, `mxbai-embed-large`, etc.), not chat models (`kimi`, `llama3`, `qwen`, etc.).
