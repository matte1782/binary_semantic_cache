"""
Embedding backends for Binary Semantic Cache.

Provides pluggable embedding generation for different backends:
- Ollama (local embedding models like nomic-embed-text)
- OpenAI (planned)
- Sentence Transformers (planned)

IMPORTANT: For Ollama, use EMBEDDING models only, not chat models.
✅ nomic-embed-text, mxbai-embed-large, snowflake-arctic-embed
❌ kimi, llama3, qwen, gemma (chat-only, won't work)

Usage:
    from binary_semantic_cache.embeddings import OllamaEmbedder
    
    # Use a dedicated embedding model
    embedder = OllamaEmbedder(model_name="nomic-embed-text")
    embeddings = embedder.embed_texts(["Hello world", "Goodbye world"])
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseEmbedder
    from .ollama_backend import OllamaEmbedder


class EmbeddingBackend(str, Enum):
    """Supported embedding backends."""
    
    OLLAMA = "ollama"
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"  # Planned


def get_embedder(backend: EmbeddingBackend, **kwargs) -> "BaseEmbedder":
    """
    Factory function to get an embedder for the specified backend.
    
    Args:
        backend: The embedding backend to use
        **kwargs: Backend-specific configuration
        
    Returns:
        An embedder instance
        
    Raises:
        NotImplementedError: If backend is not yet implemented
        ImportError: If required dependencies are missing
    """
    if backend == EmbeddingBackend.OLLAMA:
        from .ollama_backend import OllamaEmbedder
        return OllamaEmbedder(**kwargs)
    elif backend == EmbeddingBackend.OPENAI:
        from .openai_backend import OpenAIEmbeddingBackend
        return OpenAIEmbeddingBackend(**kwargs)
    elif backend == EmbeddingBackend.SENTENCE_TRANSFORMERS:
        raise NotImplementedError("Sentence Transformers backend not yet implemented")
    else:
        raise ValueError(f"Unknown backend: {backend}")


# Lazy imports to avoid dependency issues
def __getattr__(name: str):
    if name == "OllamaEmbedder":
        from .ollama_backend import OllamaEmbedder
        return OllamaEmbedder
    elif name == "BaseEmbedder":
        from .base import BaseEmbedder
        return BaseEmbedder
    elif name == "OllamaConnectionError":
        from .ollama_backend import OllamaConnectionError
        return OllamaConnectionError
    elif name == "OllamaModelNotFoundError":
        from .ollama_backend import OllamaModelNotFoundError
        return OllamaModelNotFoundError
    elif name == "OllamaNotEmbeddingModelError":
        from .ollama_backend import OllamaNotEmbeddingModelError
        return OllamaNotEmbeddingModelError
    elif name == "OllamaModelError":
        from .ollama_backend import OllamaModelError
        return OllamaModelError
    elif name == "OpenAIEmbeddingBackend":
        from .openai_backend import OpenAIEmbeddingBackend
        return OpenAIEmbeddingBackend
    elif name == "OpenAIBackendError":
        from .openai_backend import OpenAIBackendError
        return OpenAIBackendError
    elif name == "OpenAINotInstalledError":
        from .openai_backend import OpenAINotInstalledError
        return OpenAINotInstalledError
    elif name == "OpenAIAuthenticationError":
        from .openai_backend import OpenAIAuthenticationError
        return OpenAIAuthenticationError
    elif name == "OpenAIRateLimitError":
        from .openai_backend import OpenAIRateLimitError
        return OpenAIRateLimitError
    elif name == "RateLimiter":
        from .openai_backend import RateLimiter
        return RateLimiter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EmbeddingBackend",
    "get_embedder",
    "OllamaEmbedder",
    "BaseEmbedder",
    "OllamaConnectionError",
    "OllamaModelNotFoundError",
    "OllamaNotEmbeddingModelError",
    "OllamaModelError",
    "OpenAIEmbeddingBackend",
    "OpenAIBackendError",
    "OpenAINotInstalledError",
    "OpenAIAuthenticationError",
    "OpenAIRateLimitError",
    "RateLimiter",
]

