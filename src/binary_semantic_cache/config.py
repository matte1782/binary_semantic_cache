"""
Configuration module for Binary Semantic Cache.

Uses Pydantic BaseSettings for environment variable support.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class CacheConfig(BaseSettings):
    """
    Cache configuration with environment variable support.

    All settings can be configured via environment variables
    with the BSC_ prefix (e.g., BSC_MAX_ENTRIES=50000).
    """

    # Core settings
    max_entries: int = Field(
        default=100_000,
        ge=1,
        description="Maximum number of cache entries",
    )
    similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum similarity for cache hit",
    )
    code_bits: int = Field(
        default=256,
        ge=64,
        description="Number of bits in binary code",
    )
    embedding_dim: int = Field(
        default=384,
        ge=1,
        description="Dimension of embeddings",
    )

    # Embedding provider
    embedding_provider: Literal["openai", "local"] = Field(
        default="local",
        description="Embedding provider to use",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Model name for embeddings",
    )

    # Performance tuning
    numba_enabled: bool = Field(
        default=True,
        description="Enable Numba JIT acceleration",
    )

    model_config = {
        "env_prefix": "BSC_",
        "case_sensitive": False,
    }


class ProxyConfig(BaseSettings):
    """
    Proxy server configuration.
    """

    # Server settings
    host: str = Field(
        default="0.0.0.0",
        description="Host to bind to",
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Port to bind to",
    )

    # Upstream settings
    upstream_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for upstream LLM API",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for upstream calls",
    )
    request_timeout: float = Field(
        default=30.0,
        ge=1.0,
        description="Timeout for upstream requests (seconds)",
    )

    # Cache settings (embedded)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    model_config = {
        "env_prefix": "BSC_PROXY_",
        "case_sensitive": False,
    }

