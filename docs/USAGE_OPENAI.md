# OpenAI Backend Usage Guide

**Status:** ✅ Production Ready (Sprint 3)

This guide explains how to use `BinarySemanticCache` with the OpenAI embedding backend. This configuration allows you to cache OpenAI API calls locally, significantly reducing costs and latency for repeated semantic queries.

---

## 1. Installation

The OpenAI backend requires additional dependencies (`openai`, `tiktoken`, `tenacity`).

```bash
# Install with OpenAI extras
pip install "binary-semantic-cache[openai]"
```

---

## 2. Environment Setup (Exhaustive)

This library is designed to run in a Python virtual environment. Here is a bullet-proof guide to setting one up on any platform.

### Option A: Python `venv` (Standard)

**Windows (PowerShell)**
```powershell
# 1. Create virtual environment named 'venv'
python -m venv venv

# 2. Activate it
.\venv\Scripts\Activate.ps1

# Note: If you see "execution of scripts is disabled", run this first:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows (Command Prompt)**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Option B: Conda (Anaconda/Miniconda)

If you prefer Conda, use these commands:

```bash
# 1. Create environment
conda create -n bsc_env python=3.10

# 2. Activate
conda activate bsc_env

# 3. Install package
pip install "binary-semantic-cache[openai]"
```

---

## 3. Configuration

### API Key Setup

You must set the `OPENAI_API_KEY` environment variable. **Never hardcode your API key in your scripts.**

> **🔒 Security Note:** The `BinarySemanticCache` library **never** stores your API keys on disk or transmits them to any third-party server other than OpenAI itself.

**Windows (PowerShell)**
```powershell
$env:OPENAI_API_KEY = "sk-..."
```

**Linux / macOS**
```bash
export OPENAI_API_KEY="sk-..."
```

**Python (Runtime - Not Recommended for Production)**
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

### Rate Limiting

The backend implements a **Token Bucket** rate limiter to respect OpenAI's RPM (Requests Per Minute) limits.

- **Default:** 3,000 RPM (Tier 1 default)
- **Configurable:** Pass `rpm_limit` to the backend constructor.

---

## 4. Quick Start

```python
import os
from binary_semantic_cache import BinarySemanticCache, BinaryEncoder
from binary_semantic_cache.embeddings.openai_backend import OpenAIEmbeddingBackend

# 1. Initialize Backend (handles API calls, batching, and rate limiting)
backend = OpenAIEmbeddingBackend(
    model="text-embedding-3-small",  # Default
    rpm_limit=3000                   # Adjust based on your OpenAI Tier
)

# 2. Initialize Cache
# Note: text-embedding-3-small is 1536 dimensions
encoder = BinaryEncoder(embedding_dim=1536, code_bits=256)
cache = BinarySemanticCache(encoder=encoder, max_entries=1000)

# 3. Define Helper Function
def get_response(query: str):
    # Step A: Check Cache
    embedding = backend.embed_text(query)
    cached = cache.get(embedding)
    
    if cached:
        print(f"✅ CACHE HIT (Sim: {cached.similarity:.4f})")
        return cached.response
    
    # Step B: Call LLM (simulated here)
    print("❌ CACHE MISS - Calling LLM API...")
    response = f"Computed answer for: {query}"
    
    # Step C: Cache Result
    cache.put(embedding, response)
    return response

# 4. Run
print(get_response("What is the capital of France?")) # Miss
print(get_response("France capital city"))            # Hit!
```

---

## 5. Cost Management & Stats

The backend automatically tracks token usage and estimated costs.

```python
stats = backend.get_stats()
print(f"Total Requests: {stats['requests']}")
print(f"Total Tokens:   {stats['tokens']}")
print(f"Estimated Cost: ${stats['cost_usd']:.6f}")
```

**Supported Models:**
- `text-embedding-3-small` (Default): Cheapest, highly efficient.
- `text-embedding-3-large`: Higher precision, more expensive.
- `text-embedding-ada-002`: Legacy.

---

## 6. Troubleshooting

### "Rate Limit Exceeded" Error
- **Cause:** You are sending requests faster than your OpenAI tier allows.
- **Fix:** Increase the retry backoff or reduce `rpm_limit` in `OpenAIEmbeddingBackend`.
- **Note:** The backend uses `tenacity` to automatically retry on rate limits with exponential backoff.

### "API Key Missing" Error
- **Error:** `openai.AuthenticationError` or "OPENAI_API_KEY not found".
- **Fix:** Ensure you ran `export OPENAI_API_KEY=...` (or `$env:`) in the **same terminal** where you are running the script.

### "UnicodeEncodeError" on Windows
- **Cause:** Printing special characters (like the cost symbol or progress bars) in a legacy Command Prompt.
- **Fix:** Use PowerShell or set `PYTHONIOENCODING=utf-8` before running your script.

### Cache Misses on Similar Queries
- **Cause:** The default threshold (0.95) might be too strict for 256-bit binary codes.
- **Fix:** Lower `similarity_threshold` to `0.90` or `0.85` in `BinarySemanticCache`.

---

## 7. Architecture Notes

- **Batching:** The backend automatically chunks large lists of text into batches of 100 (OpenAI's limit) to minimize network overhead.
- **Normalization:** OpenAI embeddings are normalized to length 1. The binary encoder expects this for optimal quantization.
- **Parity:** The Rust encoder used internally produces bit-exact results matching the Python reference implementation.

