"""
FastAPI Proxy Server for Binary Semantic Cache.

Provides an OpenAI-compatible API that transparently caches LLM responses.

Usage:
    uvicorn binary_semantic_cache.proxy.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
from pathlib import Path

# Add src to path for imports
_SRC_PATH = Path(__file__).resolve().parents[2]
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from binary_semantic_cache.config import CacheConfig, ProxyConfig
from binary_semantic_cache.core.cache import BinarySemanticCache
from binary_semantic_cache.core.encoder import BinaryEncoder

logger = logging.getLogger(__name__)

# Global cache instance (initialized in lifespan)
_cache: Optional[BinarySemanticCache] = None
_config: Optional[ProxyConfig] = None
_http_client: Optional[httpx.AsyncClient] = None


# --- Pydantic Models (OpenAI-compatible) ---


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(..., description="Model to use")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: bool = Field(default=False)
    # Additional fields can be added as needed


class ChatCompletionChoice(BaseModel):
    """A single choice in the response."""

    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage = Field(default_factory=ChatCompletionUsage)
    cached: bool = Field(default=False, description="Whether response was from cache")


class CacheStatsResponse(BaseModel):
    """Cache statistics response."""

    size: int
    max_size: int
    hits: int
    misses: int
    hit_rate: float
    evictions: int
    memory_mb: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    cache_size: int
    uptime_seconds: float


# --- Utility Functions ---

_start_time: float = 0.0


def _hash_messages(messages: List[ChatMessage]) -> str:
    """Create a deterministic hash of messages for cache key."""
    content = json.dumps(
        [{"role": m.role, "content": m.content} for m in messages],
        sort_keys=True,
    )
    return hashlib.sha256(content.encode()).hexdigest()


def _messages_to_embedding(messages: List[ChatMessage]) -> np.ndarray:
    """
    Convert messages to an embedding vector.

    For MVP, we use a simple hash-based pseudo-embedding.
    In production, this would use a real embedding model.
    """
    # Concatenate all message contents
    full_text = " ".join(m.content for m in messages)

    # Create deterministic pseudo-embedding from hash
    # This is a placeholder - real implementation uses embedding model
    hash_bytes = hashlib.sha256(full_text.encode()).digest()

    # Expand hash to embedding dimension (384)
    rng = np.random.default_rng(int.from_bytes(hash_bytes[:8], "little"))
    embedding = rng.standard_normal(384).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)

    return embedding


# --- Lifespan ---


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: initialize and cleanup resources."""
    global _cache, _config, _http_client, _start_time

    _start_time = time.time()

    # Load configuration
    _config = ProxyConfig()
    logger.info("Configuration loaded: %s", _config)

    # Initialize encoder and cache
    encoder = BinaryEncoder(
        embedding_dim=_config.cache.embedding_dim,
        code_bits=_config.cache.code_bits,
        seed=42,
    )
    _cache = BinarySemanticCache(
        encoder=encoder,
        max_entries=_config.cache.max_entries,
        similarity_threshold=_config.cache.similarity_threshold,
    )
    logger.info("Cache initialized: max_entries=%d", _config.cache.max_entries)

    # Initialize HTTP client
    _http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(_config.request_timeout),
    )

    yield

    # Cleanup
    if _http_client:
        await _http_client.aclose()
    logger.info("Proxy server shutdown")


# --- FastAPI App ---

app = FastAPI(
    title="Binary Semantic Cache Proxy",
    description="OpenAI-compatible API proxy with semantic caching",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Routes ---


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        cache_size=len(_cache) if _cache else 0,
        uptime_seconds=time.time() - _start_time,
    )


@app.get("/metrics", response_model=CacheStatsResponse)
async def get_metrics() -> CacheStatsResponse:
    """Get cache statistics."""
    if _cache is None:
        raise HTTPException(status_code=503, detail="Cache not initialized")

    stats = _cache.stats()
    return CacheStatsResponse(
        size=stats.size,
        max_size=stats.max_size,
        hits=stats.hits,
        misses=stats.misses,
        hit_rate=stats.hit_rate,
        evictions=stats.evictions,
        memory_mb=stats.memory_mb,
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    OpenAI-compatible chat completions endpoint with caching.

    Flow:
    1. Convert messages to embedding
    2. Check cache for similar query
    3. If hit: return cached response
    4. If miss: forward to upstream, cache response, return
    """
    if _cache is None or _config is None or _http_client is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    # Streaming not supported in MVP
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Streaming not supported in cached mode. Set stream=false.",
        )

    # Convert messages to embedding
    embedding = _messages_to_embedding(request.messages)

    # Check cache
    cache_entry = _cache.get(embedding)

    if cache_entry is not None:
        # Cache hit
        logger.info("Cache hit")
        cached_response = cache_entry.response
        cached_response["cached"] = True
        return ChatCompletionResponse(**cached_response)

    # Cache miss - forward to upstream
    logger.info("Cache miss, forwarding to upstream")

    if not _config.openai_api_key:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured. Set BSC_PROXY_OPENAI_API_KEY.",
        )

    try:
        upstream_response = await _http_client.post(
            f"{_config.upstream_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {_config.openai_api_key}",
                "Content-Type": "application/json",
            },
            json=request.model_dump(exclude={"stream"}),
        )
        upstream_response.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.error("Upstream error: %s", e)
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Upstream error: {e.response.text}",
        )
    except httpx.RequestError as e:
        logger.error("Request error: %s", e)
        raise HTTPException(
            status_code=502,
            detail=f"Upstream request failed: {str(e)}",
        )

    # Parse response
    response_data = upstream_response.json()

    # Cache the response
    _cache.put(embedding, response_data)

    # Return with cached=False
    response_data["cached"] = False
    return ChatCompletionResponse(**response_data)


@app.post("/cache/clear")
async def clear_cache() -> Dict[str, str]:
    """Clear all cache entries."""
    if _cache is None:
        raise HTTPException(status_code=503, detail="Cache not initialized")

    _cache.clear()
    return {"status": "cleared"}


# --- Main ---


def main() -> None:
    """Run the server (for script entry point)."""
    import uvicorn

    config = ProxyConfig()
    uvicorn.run(
        "binary_semantic_cache.proxy.server:app",
        host=config.host,
        port=config.port,
        reload=False,
    )


if __name__ == "__main__":
    main()

