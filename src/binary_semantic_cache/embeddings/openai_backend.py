"""
OpenAI embedding backend.

Provides embedding generation using OpenAI's embedding API with:
- Rate limiting (Token Bucket)
- Retry logic with exponential backoff (via tenacity)
- Cost tracking
- Batching (max 100 items per request)

Usage:
    from binary_semantic_cache.embeddings import OpenAIEmbeddingBackend
    
    # Initialize with API key (or use OPENAI_API_KEY env var)
    embedder = OpenAIEmbeddingBackend(api_key="sk-...", model="text-embedding-3-small")
    
    # Generate embeddings
    embeddings = embedder.embed_texts(["Hello world", "Goodbye world"])
    
    # Check usage statistics
    stats = embedder.get_stats()
    print(f"Cost: ${stats['cost_usd']:.6f}")

Environment Variables:
    OPENAI_API_KEY: OpenAI API key (fallback if not provided explicitly)
    OPENAI_ORG_ID: OpenAI organization ID (optional)
    OPENAI_BASE_URL: Custom base URL for API (optional)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from .base import BaseEmbedder

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_RPM_LIMIT = 3000
DEFAULT_TIMEOUT = 60.0
MAX_BATCH_SIZE = 100  # Conservative limit (OpenAI allows 2048)
MAX_RETRIES = 5

# Model dimensions
MODEL_DIMENSIONS: Dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Pricing per 1M tokens (USD)
MODEL_PRICING: Dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


# =============================================================================
# EXCEPTIONS
# =============================================================================

class OpenAIBackendError(Exception):
    """Base exception for OpenAI backend errors."""
    pass


class OpenAINotInstalledError(OpenAIBackendError):
    """Raised when openai package is not installed."""
    pass


class OpenAIAuthenticationError(OpenAIBackendError):
    """Raised when authentication fails (invalid API key)."""
    pass


class OpenAIRateLimitError(OpenAIBackendError):
    """Raised when rate limit is exceeded after all retries."""
    pass


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """
    Token Bucket rate limiter for API request pacing.
    
    Attributes:
        rpm_limit: Maximum requests per minute
        tokens: Current number of available tokens
        last_refill: Timestamp of last token refill
    """
    
    __slots__ = ("rpm_limit", "tokens", "last_refill", "_lock")
    
    def __init__(self, rpm_limit: int = DEFAULT_RPM_LIMIT) -> None:
        """
        Initialize rate limiter.
        
        Args:
            rpm_limit: Maximum requests per minute (bucket capacity)
        """
        self.rpm_limit = rpm_limit
        self.tokens = float(rpm_limit)
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        # Refill rate: rpm_limit / 60 tokens per second
        refill_amount = elapsed * (self.rpm_limit / 60.0)
        self.tokens = min(self.rpm_limit, self.tokens + refill_amount)
        self.last_refill = now
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a token, blocking if necessary.
        
        Args:
            timeout: Maximum time to wait for a token (None = wait forever)
            
        Returns:
            True if token acquired, False if timeout
        """
        start_time = time.monotonic()
        
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True
                
                # Calculate wait time for next token
                tokens_needed = 1.0 - self.tokens
                wait_time = tokens_needed / (self.rpm_limit / 60.0)
            
            # Check timeout
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed + wait_time > timeout:
                    return False
            
            # Sleep until we can get a token
            time.sleep(min(wait_time, 0.1))  # Cap sleep to 100ms for responsiveness
    
    def reset(self) -> None:
        """Reset the rate limiter to full capacity."""
        with self._lock:
            self.tokens = float(self.rpm_limit)
            self.last_refill = time.monotonic()


# =============================================================================
# MAIN CLASS
# =============================================================================

class OpenAIEmbeddingBackend(BaseEmbedder):
    """
    Embedding backend using OpenAI's embedding API.
    
    Features:
    - Rate limiting with Token Bucket algorithm
    - Automatic retry with exponential backoff for transient errors
    - Cost tracking based on token usage
    - Batching for efficient API usage
    
    Attributes:
        model_name: Name of the OpenAI embedding model
        embedding_dim: Dimensionality of embeddings
        rpm_limit: Requests per minute limit
    """
    
    __slots__ = (
        "_api_key",
        "_model",
        "_organization",
        "_base_url",
        "_timeout",
        "_client",
        "_rate_limiter",
        "_stats",
        "_stats_lock",
        "_encoding",
    )
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        rpm_limit: int = DEFAULT_RPM_LIMIT,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """
        Initialize OpenAI embedding backend.
        
        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            model: Model name (default: text-embedding-3-small)
            rpm_limit: Requests per minute limit for rate limiting
            organization: OpenAI organization ID (optional)
            base_url: Custom base URL for API (optional)
            timeout: Request timeout in seconds
            
        Raises:
            OpenAINotInstalledError: If openai package is not installed
            OpenAIAuthenticationError: If no API key is provided
        """
        # Lazy import openai
        try:
            import openai
        except ImportError as e:
            raise OpenAINotInstalledError(
                "OpenAI package is not installed. "
                "Install with: pip install binary-semantic-cache[openai]"
            ) from e
        
        # Get API key
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise OpenAIAuthenticationError(
                "No API key provided. Pass api_key parameter or set OPENAI_API_KEY env var."
            )
        
        self._model = model
        self._organization = organization or os.environ.get("OPENAI_ORG_ID")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._timeout = timeout
        
        # Initialize OpenAI client
        self._client = openai.OpenAI(
            api_key=self._api_key,
            organization=self._organization,
            base_url=self._base_url,
            timeout=self._timeout,
        )
        
        # Initialize rate limiter
        self._rate_limiter = RateLimiter(rpm_limit=rpm_limit)
        
        # Initialize stats
        self._stats: Dict[str, Any] = {
            "requests": 0,
            "tokens": 0,
            "cost_usd": 0.0,
            "errors": 0,
        }
        self._stats_lock = threading.Lock()
        
        # Lazy initialize tiktoken encoding
        self._encoding = None
        
        logger.info(
            f"OpenAIEmbeddingBackend initialized: model={model}, rpm_limit={rpm_limit}"
        )
    
    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings produced by this model."""
        return MODEL_DIMENSIONS.get(self._model, 1536)
    
    @property
    def model_name(self) -> str:
        """Return the name of the model being used."""
        return self._model
    
    def _get_encoding(self):
        """Lazy-load tiktoken encoding for the model."""
        if self._encoding is None:
            try:
                import tiktoken
                # Use cl100k_base for embedding models
                self._encoding = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                logger.warning("tiktoken not installed; token counting disabled")
                self._encoding = False  # Sentinel to avoid repeated import attempts
        return self._encoding if self._encoding else None
    
    def _count_tokens(self, texts: List[str]) -> int:
        """Count tokens in texts using tiktoken."""
        encoding = self._get_encoding()
        if encoding is None:
            # Rough estimate: 1 token per 4 characters
            return sum(len(t) // 4 for t in texts)
        return sum(len(encoding.encode(t)) for t in texts)
    
    def _update_stats(
        self,
        requests: int = 0,
        tokens: int = 0,
        errors: int = 0,
    ) -> None:
        """Thread-safe stats update."""
        with self._stats_lock:
            self._stats["requests"] += requests
            self._stats["tokens"] += tokens
            self._stats["errors"] += errors
            # Calculate cost
            price_per_million = MODEL_PRICING.get(self._model, 0.02)
            self._stats["cost_usd"] = self._stats["tokens"] * price_per_million / 1_000_000
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dict with keys: requests, tokens, cost_usd, errors
        """
        with self._stats_lock:
            return dict(self._stats)
    
    def reset_stats(self) -> None:
        """Reset usage statistics to zero."""
        with self._stats_lock:
            self._stats = {
                "requests": 0,
                "tokens": 0,
                "cost_usd": 0.0,
                "errors": 0,
            }
    
    def _call_api_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        Call OpenAI API with retry logic.
        
        Args:
            texts: List of texts to embed (must be <= MAX_BATCH_SIZE)
            
        Returns:
            List of embedding vectors
            
        Raises:
            OpenAIAuthenticationError: On auth failure (no retry)
            OpenAIRateLimitError: After max retries on rate limit
            OpenAIBackendError: On other API errors
        """
        import openai
        
        # Import tenacity for retry logic
        try:
            from tenacity import (
                retry,
                stop_after_attempt,
                wait_exponential_jitter,
                retry_if_exception_type,
            )
        except ImportError:
            # Fallback: no retry if tenacity not installed
            logger.warning("tenacity not installed; retry logic disabled")
            return self._call_api_once(texts)
        
        # Define retry decorator
        @retry(
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_exponential_jitter(initial=1, max=60, jitter=5),
            retry=retry_if_exception_type((
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.InternalServerError,
                openai.APITimeoutError,
            )),
            reraise=True,
        )
        def _call_with_retry():
            return self._call_api_once(texts)
        
        try:
            return _call_with_retry()
        except openai.AuthenticationError as e:
            # Fail fast on auth errors
            self._update_stats(errors=1)
            raise OpenAIAuthenticationError(
                f"Authentication failed: {e}. Check your API key."
            ) from e
        except openai.BadRequestError as e:
            # Fail fast on bad requests
            self._update_stats(errors=1)
            raise OpenAIBackendError(f"Bad request: {e}") from e
        except openai.RateLimitError as e:
            # Max retries exceeded
            self._update_stats(errors=1)
            raise OpenAIRateLimitError(
                f"Rate limit exceeded after {MAX_RETRIES} retries: {e}"
            ) from e
        except Exception as e:
            self._update_stats(errors=1)
            raise OpenAIBackendError(f"API error: {e}") from e
    
    def _call_api_once(self, texts: List[str]) -> List[List[float]]:
        """
        Make a single API call (no retry).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Acquire rate limit token
        if not self._rate_limiter.acquire(timeout=self._timeout):
            raise OpenAIRateLimitError("Rate limiter timeout")
        
        # Make API call
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        
        # Update stats from response
        tokens_used = response.usage.total_tokens if response.usage else self._count_tokens(texts)
        self._update_stats(requests=1, tokens=tokens_used)
        
        # Extract embeddings (sorted by index to ensure order)
        embeddings = sorted(response.data, key=lambda x: x.index)
        return [e.embedding for e in embeddings]
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        
        Args:
            text: The text to embed
            
        Returns:
            A 1D numpy array of shape (embedding_dim,) with dtype float32
        """
        result = self.embed_texts([text])
        return result[0]
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple text strings.
        
        Automatically batches requests to respect API limits.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            A 2D numpy array of shape (len(texts), embedding_dim) with dtype float32
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.embedding_dim)
        
        all_embeddings: List[List[float]] = []
        
        # Process in batches
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[i : i + MAX_BATCH_SIZE]
            batch_embeddings = self._call_api_with_retry(batch)
            all_embeddings.extend(batch_embeddings)
        
        # Convert to numpy and normalize
        result = np.array(all_embeddings, dtype=np.float32)
        return self.normalize(result)
    
    def is_available(self) -> bool:
        """
        Check if the OpenAI backend is available and working.
        
        Returns:
            True if a test embedding can be generated, False otherwise
        """
        try:
            self.embed_text("test")
            return True
        except Exception:
            return False
    
    def __repr__(self) -> str:
        return (
            f"OpenAIEmbeddingBackend(model={self._model!r}, "
            f"dim={self.embedding_dim}, rpm_limit={self._rate_limiter.rpm_limit})"
        )

