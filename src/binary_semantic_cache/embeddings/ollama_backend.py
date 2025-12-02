"""
Ollama embedding backend.

Provides embedding generation using Ollama models that support the embeddings API.

IMPORTANT: Not all Ollama models support embeddings!
- Chat/LLM models (kimi, llama3, qwen, etc.) are typically chat-only
- Dedicated embedding models (nomic-embed-text, mxbai-embed-large) support embeddings

Usage:
    from binary_semantic_cache.embeddings import OllamaEmbedder
    
    # Use a dedicated embedding model (recommended)
    embedder = OllamaEmbedder(model_name="nomic-embed-text")
    
    # Generate embeddings
    embeddings = embedder.embed_texts(["Hello world", "Goodbye world"])
    
Environment Variables:
    OLLAMA_HOST: Ollama server URL (default: http://localhost:11434)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseEmbedder

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "nomic-embed-text"  # Recommended embedding model

# =============================================================================
# EMBEDDING MODEL ALLOWLIST
# =============================================================================
# Only these models are known to reliably support the Ollama /api/embeddings endpoint.
# Chat models (llama3, kimi, qwen, etc.) typically do NOT support embeddings.

EMBEDDING_MODELS: Dict[str, int] = {
    # Dedicated embedding models (RECOMMENDED)
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
    "bge-m3": 1024,
    "bge-large": 1024,
    "gte-large": 1024,
    "e5-large": 1024,
    "e5-small": 384,
    "paraphrase-multilingual": 768,
}

# Alias for backward compatibility with tests
KNOWN_DIMENSIONS = EMBEDDING_MODELS

# Models known to NOT support embeddings (chat-only)
# These will show in ollama list but fail on /api/embeddings
CHAT_ONLY_MODELS: List[str] = [
    "kimi",
    "llama",
    "qwen",
    "gemma",
    "deepseek",
    "glm",
    "minimax",
    "gpt-oss",
    "smollm",
    "granite",
    "mistral",
    "phi",
    "vicuna",
    "codellama",
]


# =============================================================================
# EXCEPTIONS
# =============================================================================

class OllamaConnectionError(Exception):
    """Raised when Ollama server is not reachable."""
    pass


class OllamaModelNotFoundError(Exception):
    """Raised when model is not installed in Ollama."""
    pass


class OllamaNotEmbeddingModelError(Exception):
    """Raised when model exists but doesn't support embeddings."""
    pass


class OllamaModelError(Exception):
    """Generic model error (for backwards compatibility)."""
    pass


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_known_embedding_model(model_name: str) -> bool:
    """
    Check if a model is in the known embedding models allowlist.
    
    Args:
        model_name: The model name (e.g., "nomic-embed-text")
        
    Returns:
        True if model is known to support embeddings
    """
    model_lower = model_name.lower()
    return any(embed_model in model_lower for embed_model in EMBEDDING_MODELS.keys())


def is_likely_chat_only_model(model_name: str) -> bool:
    """
    Check if a model is likely a chat-only model that doesn't support embeddings.
    
    Args:
        model_name: The model name
        
    Returns:
        True if model is likely chat-only
    """
    model_lower = model_name.lower()
    return any(chat_model in model_lower for chat_model in CHAT_ONLY_MODELS)


def get_recommended_embedding_models() -> List[str]:
    """Get list of recommended embedding models."""
    return list(EMBEDDING_MODELS.keys())[:5]


# =============================================================================
# MAIN CLASS
# =============================================================================

class OllamaEmbedder(BaseEmbedder):
    """
    Embedding backend using Ollama.
    
    IMPORTANT: Only use models that support the Ollama embeddings API.
    Recommended models: nomic-embed-text, mxbai-embed-large, snowflake-arctic-embed
    
    Chat models (llama3, kimi, qwen, etc.) do NOT support embeddings.
    
    Attributes:
        host: Ollama server URL
        model_name: Name of the model to use
        embedding_dim: Dimensionality of embeddings (detected at runtime)
    """
    
    __slots__ = ("_host", "_model_name", "_embedding_dim", "_session", "_timeout", "_models_cache")
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        host: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize Ollama embedder.
        
        Args:
            model_name: Ollama model to use for embeddings.
                        MUST be an embedding-capable model (e.g., nomic-embed-text).
            host: Ollama server URL (defaults to OLLAMA_HOST env or localhost:11434)
            timeout: Request timeout in seconds
        """
        self._host = host or os.environ.get("OLLAMA_HOST", DEFAULT_HOST)
        self._model_name = model_name
        self._embedding_dim: Optional[int] = None
        self._timeout = timeout
        self._models_cache: Optional[List[str]] = None
        
        # Lazy import httpx to avoid hard dependency
        try:
            import httpx
            self._session = httpx.Client(timeout=httpx.Timeout(timeout))
        except ImportError:
            try:
                import requests
                self._session = None  # Will use requests directly
            except ImportError:
                raise ImportError(
                    "Either httpx or requests is required for OllamaEmbedder. "
                    "Install with: pip install httpx"
                )
        
        logger.info(f"OllamaEmbedder initialized: host={self._host}, model={model_name}")
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimensionality (detected at runtime if not known)."""
        if self._embedding_dim is None:
            # Check known dimensions first
            for embed_model, dim in EMBEDDING_MODELS.items():
                if embed_model in self._model_name.lower():
                    self._embedding_dim = dim
                    break
            
            # If still unknown, detect by making a test call
            if self._embedding_dim is None:
                try:
                    test_embedding = self._get_embedding("test")
                    self._embedding_dim = len(test_embedding)
                except Exception as e:
                    logger.warning(f"Could not detect embedding dim: {e}")
                    self._embedding_dim = 768  # Fallback
        
        return self._embedding_dim
    
    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name
    
    @property
    def host(self) -> str:
        """Return the Ollama host URL."""
        return self._host
    
    def _fetch_available_models(self) -> List[str]:
        """Fetch list of models from Ollama server."""
        if self._models_cache is not None:
            return self._models_cache
            
        try:
            url = f"{self._host}/api/tags"
            if self._session is not None:
                response = self._session.get(url)
            else:
                import requests
                response = requests.get(url, timeout=5.0)
            
            if response.status_code == 200:
                data = response.json()
                self._models_cache = [m.get("name", "") for m in data.get("models", [])]
                return self._models_cache
        except Exception:
            pass
        return []
    
    def _is_model_in_ollama(self, model_name: str) -> bool:
        """Check if a model is available in Ollama (regardless of embedding support)."""
        models = self._fetch_available_models()
        model_base = model_name.split(":")[0].lower()
        return any(model_base in m.lower() for m in models)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text via Ollama API.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding as numpy array
            
        Raises:
            OllamaConnectionError: If server is not reachable
            OllamaModelNotFoundError: If model is not installed
            OllamaNotEmbeddingModelError: If model doesn't support embeddings
        """
        url = f"{self._host}/api/embeddings"
        payload = {
            "model": self._model_name,
            "prompt": text,
        }
        
        try:
            if self._session is not None:
                response = self._session.post(url, json=payload)
            else:
                import requests
                response = requests.post(url, json=payload, timeout=self._timeout)
            
            # Check for success
            if response.status_code == 200:
                data = response.json()
                raw_embedding = data.get("embedding")
                
                if raw_embedding is None:
                    raise OllamaNotEmbeddingModelError(
                        f"Model '{self._model_name}' returned null embedding. "
                        "This model may not support embeddings."
                    )
                
                # Convert to numpy array and validate shape
                embedding = np.array(raw_embedding, dtype=np.float32)
                logger.debug(f"Ollama raw embedding shape: {embedding.shape}")
                
                # SHAPE GUARANTEE: Must be 1D for single text
                if embedding.ndim == 0:
                    raise OllamaNotEmbeddingModelError(
                        f"Model '{self._model_name}' returned scalar, not embedding vector. "
                        "This model does not support embeddings."
                    )
                
                if embedding.ndim == 2:
                    # Some models might wrap in extra dimension - unwrap if single
                    if embedding.shape[0] == 1:
                        embedding = embedding.squeeze(0)
                        logger.debug(f"Squeezed embedding to shape: {embedding.shape}")
                    else:
                        raise OllamaNotEmbeddingModelError(
                            f"Model '{self._model_name}' returned unexpected shape {embedding.shape}. "
                            "Expected 1D vector for single text embedding."
                        )
                
                if embedding.ndim != 1:
                    raise OllamaNotEmbeddingModelError(
                        f"Model '{self._model_name}' returned {embedding.ndim}D array with shape {embedding.shape}. "
                        "Expected 1D embedding vector. This model may not be embedding-capable."
                    )
                
                if embedding.shape[0] == 0:
                    raise OllamaNotEmbeddingModelError(
                        f"Model '{self._model_name}' returned empty embedding (0 dimensions). "
                        "This model does not support embeddings."
                    )
                
                return embedding
            
            # Handle errors
            error_text = response.text.lower()
            
            # Check if it's a "model not found" error
            if response.status_code == 404 or "not found" in error_text:
                # Is the model in Ollama's list?
                if self._is_model_in_ollama(self._model_name):
                    # Model exists but doesn't support embeddings
                    recommended = get_recommended_embedding_models()
                    raise OllamaNotEmbeddingModelError(
                        f"Model '{self._model_name}' is installed but does NOT support embeddings.\n"
                        f"This is a chat-only model, not an embedding model.\n\n"
                        f"Please use a dedicated embedding model instead:\n"
                        f"  ollama pull nomic-embed-text\n"
                        f"  python validation/ollama_end_to_end_test.py --model nomic-embed-text\n\n"
                        f"Recommended embedding models: {', '.join(recommended)}"
                    )
                else:
                    # Model genuinely not installed
                    raise OllamaModelNotFoundError(
                        f"Model '{self._model_name}' is not installed.\n"
                        f"Pull it with: ollama pull {self._model_name}"
                    )
            
            # Other errors
            response.raise_for_status()
            
        except OllamaConnectionError:
            raise
        except OllamaModelNotFoundError:
            raise
        except OllamaNotEmbeddingModelError:
            raise
        except Exception as e:
            error_str = str(e).lower()
            
            # Connection errors
            if ("connect" in error_str or "refused" in error_str or 
                "timeout" in error_str or "getaddrinfo" in error_str):
                raise OllamaConnectionError(
                    f"Could not connect to Ollama at {self._host}.\n"
                    "Make sure Ollama is running: ollama serve"
                ) from e
            
            # Re-raise with context
            raise OllamaModelError(f"Embedding failed: {e}") from e
        
        # Should not reach here
        raise OllamaModelError(f"Unexpected response from Ollama")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        
        SHAPE GUARANTEE: Always returns 1D array of shape (embedding_dim,)
        
        Args:
            text: The text to embed
            
        Returns:
            A 1D numpy array of shape (embedding_dim,) with dtype float32
            
        Raises:
            OllamaNotEmbeddingModelError: If model returns invalid shape
        """
        embedding = self._get_embedding(text)
        normalized = self.normalize(embedding)
        
        # SHAPE GUARANTEE: Must be 1D
        if normalized.ndim != 1:
            raise OllamaNotEmbeddingModelError(
                f"embed_text() internal error: expected 1D result, got shape {normalized.shape}"
            )
        
        logger.debug(f"embed_text() returning shape: {normalized.shape}")
        return normalized
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple text strings.
        
        SHAPE GUARANTEE: Always returns 2D array of shape (len(texts), embedding_dim)
        
        Note: Ollama doesn't have a native batch API, so this makes
        sequential calls.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            A 2D numpy array of shape (len(texts), embedding_dim) with dtype float32
            
        Raises:
            OllamaNotEmbeddingModelError: If model returns invalid shape
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.embedding_dim)
        
        embeddings = []
        for i, text in enumerate(texts):
            embedding = self._get_embedding(text)
            
            # Validate each embedding is 1D
            if embedding.ndim != 1:
                raise OllamaNotEmbeddingModelError(
                    f"embed_texts() error at text #{i}: expected 1D, got shape {embedding.shape}"
                )
            
            embeddings.append(embedding)
        
        result = np.stack(embeddings, axis=0)
        normalized = self.normalize(result)
        
        # SHAPE GUARANTEE: Must be 2D
        if normalized.ndim != 2:
            raise OllamaNotEmbeddingModelError(
                f"embed_texts() internal error: expected 2D result, got shape {normalized.shape}"
            )
        
        logger.debug(f"embed_texts() returning shape: {normalized.shape}")
        return normalized
    
    def test_embedding_support(self) -> Tuple[bool, str]:
        """
        Test if the current model supports embeddings.
        
        Returns:
            Tuple of (supports_embeddings, message)
        """
        try:
            embedding = self._get_embedding("test")
            return True, f"Model '{self._model_name}' supports embeddings (dim={len(embedding)})"
        except OllamaNotEmbeddingModelError as e:
            return False, str(e)
        except OllamaModelNotFoundError as e:
            return False, str(e)
        except OllamaConnectionError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected error: {e}"
    
    def is_available(self) -> bool:
        """
        Check if Ollama is reachable and the model supports embeddings.
        
        Returns:
            True if embeddings can be generated, False otherwise
        """
        supports, _ = self.test_embedding_support()
        return supports
    
    def list_models(self) -> List[str]:
        """
        List available models on the Ollama server.
        
        Returns:
            List of model names
        """
        return self._fetch_available_models()
    
    def list_embedding_models(self) -> List[str]:
        """
        List models from Ollama that are likely to support embeddings.
        
        Returns:
            List of model names that may support embeddings
        """
        all_models = self._fetch_available_models()
        embedding_capable = []
        
        for model in all_models:
            model_lower = model.lower()
            # Check if it's a known embedding model
            if any(embed_model in model_lower for embed_model in EMBEDDING_MODELS.keys()):
                embedding_capable.append(model)
        
        return embedding_capable
    
    def close(self) -> None:
        """Close the HTTP session."""
        if self._session is not None:
            self._session.close()
    
    def __enter__(self) -> "OllamaEmbedder":
        return self
    
    def __exit__(self, *args) -> None:
        self.close()
    
    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
