"""Unit tests for the embeddings module.

Tests:
- BaseEmbedder interface
- OllamaEmbedder (mocked)
- Factory function
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add src to path
_SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))


class TestEmbeddingBackendEnum:
    """Test EmbeddingBackend enum."""

    def test_backend_values(self) -> None:
        """All expected backends should be defined."""
        from binary_semantic_cache.embeddings import EmbeddingBackend

        assert EmbeddingBackend.OLLAMA == "ollama"
        assert EmbeddingBackend.OPENAI == "openai"
        assert EmbeddingBackend.SENTENCE_TRANSFORMERS == "sentence_transformers"


class TestGetEmbedder:
    """Test get_embedder factory function."""

    def test_unknown_backend_raises(self) -> None:
        """Unknown backend should raise ValueError."""
        from binary_semantic_cache.embeddings import get_embedder

        with pytest.raises(ValueError, match="Unknown backend"):
            get_embedder("invalid_backend")  # type: ignore

    def test_openai_implemented(self) -> None:
        """OpenAI backend should be implemented (returns correct class type)."""
        # Mock openai module to avoid ImportError
        mock_openai = MagicMock()
        mock_openai.OpenAI = MagicMock()
        
        with patch.dict("sys.modules", {"openai": mock_openai}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
                from binary_semantic_cache.embeddings import EmbeddingBackend, get_embedder
                from binary_semantic_cache.embeddings.openai_backend import OpenAIEmbeddingBackend

                embedder = get_embedder(EmbeddingBackend.OPENAI)
                assert isinstance(embedder, OpenAIEmbeddingBackend)

    def test_sentence_transformers_not_implemented(self) -> None:
        """Sentence Transformers backend should raise NotImplementedError."""
        from binary_semantic_cache.embeddings import EmbeddingBackend, get_embedder

        with pytest.raises(NotImplementedError, match="Sentence Transformers"):
            get_embedder(EmbeddingBackend.SENTENCE_TRANSFORMERS)


class TestBaseEmbedder:
    """Test BaseEmbedder abstract class."""

    def test_normalize_1d(self) -> None:
        """1D normalization should work."""
        from binary_semantic_cache.embeddings.base import BaseEmbedder

        # Create a concrete implementation for testing
        class MockEmbedder(BaseEmbedder):
            @property
            def embedding_dim(self) -> int:
                return 3

            @property
            def model_name(self) -> str:
                return "mock"

            def embed_text(self, text: str) -> np.ndarray:
                return np.array([1.0, 0.0, 0.0], dtype=np.float32)

            def embed_texts(self, texts: List[str]) -> np.ndarray:
                return np.stack([self.embed_text(t) for t in texts])

        embedder = MockEmbedder()
        vec = np.array([3.0, 4.0, 0.0])
        normalized = embedder.normalize(vec)

        assert np.isclose(np.linalg.norm(normalized), 1.0)
        assert np.allclose(normalized, [0.6, 0.8, 0.0])

    def test_normalize_2d(self) -> None:
        """2D normalization should work."""
        from binary_semantic_cache.embeddings.base import BaseEmbedder

        class MockEmbedder(BaseEmbedder):
            @property
            def embedding_dim(self) -> int:
                return 3

            @property
            def model_name(self) -> str:
                return "mock"

            def embed_text(self, text: str) -> np.ndarray:
                return np.array([1.0, 0.0, 0.0], dtype=np.float32)

            def embed_texts(self, texts: List[str]) -> np.ndarray:
                return np.stack([self.embed_text(t) for t in texts])

        embedder = MockEmbedder()
        vecs = np.array([[3.0, 4.0, 0.0], [0.0, 5.0, 0.0]])
        normalized = embedder.normalize(vecs)

        # Check all rows are unit vectors
        norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(norms, 1.0)

    def test_repr(self) -> None:
        """Repr should include class name, model, and dim."""
        from binary_semantic_cache.embeddings.base import BaseEmbedder

        class MockEmbedder(BaseEmbedder):
            @property
            def embedding_dim(self) -> int:
                return 768

            @property
            def model_name(self) -> str:
                return "test-model"

            def embed_text(self, text: str) -> np.ndarray:
                return np.zeros(768, dtype=np.float32)

            def embed_texts(self, texts: List[str]) -> np.ndarray:
                return np.zeros((len(texts), 768), dtype=np.float32)

        embedder = MockEmbedder()
        assert "MockEmbedder" in repr(embedder)
        assert "test-model" in repr(embedder)
        assert "768" in repr(embedder)


class TestOllamaEmbedder:
    """Test OllamaEmbedder class."""

    def test_default_init(self) -> None:
        """Default initialization should work."""
        # We must preserve PATH for ctypes/dll loading on Windows, otherwise trio/httpx fails
        safe_env = {k: v for k, v in os.environ.items() if k.upper() in ["PATH", "SYSTEMROOT", "WINDIR"]}
        
        with patch.dict("os.environ", safe_env, clear=True):
            from binary_semantic_cache.embeddings.ollama_backend import (
                OllamaEmbedder,
                DEFAULT_HOST,
                DEFAULT_MODEL,
            )

            embedder = OllamaEmbedder()
            assert embedder.host == DEFAULT_HOST
            assert embedder.model_name == DEFAULT_MODEL

    def test_custom_host_from_env(self) -> None:
        """Host should be read from OLLAMA_HOST env var."""
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://custom:1234"}):
            from binary_semantic_cache.embeddings.ollama_backend import OllamaEmbedder

            embedder = OllamaEmbedder()
            assert embedder.host == "http://custom:1234"

    def test_custom_host_override(self) -> None:
        """Explicit host should override env var."""
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://custom:1234"}):
            from binary_semantic_cache.embeddings.ollama_backend import OllamaEmbedder

            embedder = OllamaEmbedder(host="http://explicit:5678")
            assert embedder.host == "http://explicit:5678"

    def test_custom_model(self) -> None:
        """Custom model name should be used."""
        from binary_semantic_cache.embeddings.ollama_backend import OllamaEmbedder

        embedder = OllamaEmbedder(model_name="qwen2.5:7b")
        assert embedder.model_name == "qwen2.5:7b"

    @patch("binary_semantic_cache.embeddings.ollama_backend.OllamaEmbedder._get_embedding")
    def test_embed_text_normalizes(self, mock_get: MagicMock) -> None:
        """embed_text should normalize the result."""
        from binary_semantic_cache.embeddings.ollama_backend import OllamaEmbedder

        mock_get.return_value = np.array([3.0, 4.0, 0.0], dtype=np.float32)

        embedder = OllamaEmbedder()
        result = embedder.embed_text("test")

        assert np.isclose(np.linalg.norm(result), 1.0)

    @patch("binary_semantic_cache.embeddings.ollama_backend.OllamaEmbedder._get_embedding")
    def test_embed_texts_normalizes(self, mock_get: MagicMock) -> None:
        """embed_texts should normalize all results."""
        from binary_semantic_cache.embeddings.ollama_backend import OllamaEmbedder

        mock_get.side_effect = [
            np.array([3.0, 4.0, 0.0], dtype=np.float32),
            np.array([0.0, 5.0, 0.0], dtype=np.float32),
        ]

        embedder = OllamaEmbedder()
        embedder._embedding_dim = 3  # Set directly to avoid detection call
        result = embedder.embed_texts(["a", "b"])

        assert result.shape == (2, 3)
        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, 1.0)

    @patch("binary_semantic_cache.embeddings.ollama_backend.OllamaEmbedder._get_embedding")
    def test_embed_texts_empty(self, mock_get: MagicMock) -> None:
        """embed_texts with empty list should return empty array."""
        from binary_semantic_cache.embeddings.ollama_backend import OllamaEmbedder

        embedder = OllamaEmbedder()
        embedder._embedding_dim = 768

        result = embedder.embed_texts([])
        assert result.shape == (0, 768)

    def test_known_dimensions(self) -> None:
        """Known models should have correct dimensions."""
        from binary_semantic_cache.embeddings.ollama_backend import KNOWN_DIMENSIONS

        assert "nomic-embed-text" in KNOWN_DIMENSIONS
        assert KNOWN_DIMENSIONS["nomic-embed-text"] == 768

    @patch("binary_semantic_cache.embeddings.ollama_backend.OllamaEmbedder._get_embedding")
    def test_embedding_dim_detection(self, mock_get: MagicMock) -> None:
        """Unknown model should detect dimension from test call."""
        from binary_semantic_cache.embeddings.ollama_backend import OllamaEmbedder

        mock_get.return_value = np.zeros(512, dtype=np.float32)

        embedder = OllamaEmbedder(model_name="unknown-model")
        dim = embedder.embedding_dim

        assert dim == 512
        mock_get.assert_called_once_with("test")


class TestOllamaErrors:
    """Test OllamaEmbedder error handling."""

    def test_connection_error(self) -> None:
        """Connection errors should be wrapped in OllamaConnectionError."""
        from binary_semantic_cache.embeddings.ollama_backend import (
            OllamaEmbedder,
            OllamaConnectionError,
        )

        embedder = OllamaEmbedder(host="http://nonexistent:99999")

        # This should fail with connection error
        with pytest.raises(OllamaConnectionError):
            embedder.embed_text("test")

    def test_is_available_false_when_offline(self) -> None:
        """is_available should return False when server is offline."""
        from binary_semantic_cache.embeddings.ollama_backend import OllamaEmbedder

        embedder = OllamaEmbedder(host="http://nonexistent:99999")
        assert embedder.is_available() is False

    def test_test_embedding_support(self) -> None:
        """test_embedding_support should return tuple."""
        from binary_semantic_cache.embeddings.ollama_backend import OllamaEmbedder

        embedder = OllamaEmbedder(host="http://nonexistent:99999")
        supports, message = embedder.test_embedding_support()
        assert supports is False
        assert isinstance(message, str)
        assert len(message) > 0


class TestEmbeddingModelHelpers:
    """Test helper functions for embedding model detection."""

    def test_is_known_embedding_model_true(self) -> None:
        """Known embedding models should return True."""
        from binary_semantic_cache.embeddings.ollama_backend import is_known_embedding_model

        assert is_known_embedding_model("nomic-embed-text") is True
        assert is_known_embedding_model("mxbai-embed-large") is True
        assert is_known_embedding_model("snowflake-arctic-embed") is True

    def test_is_known_embedding_model_false(self) -> None:
        """Chat models should return False."""
        from binary_semantic_cache.embeddings.ollama_backend import is_known_embedding_model

        assert is_known_embedding_model("llama3") is False
        assert is_known_embedding_model("kimi-k2:1t-cloud") is False
        assert is_known_embedding_model("qwen2.5:7b") is False

    def test_is_likely_chat_only_model_true(self) -> None:
        """Chat models should be detected."""
        from binary_semantic_cache.embeddings.ollama_backend import is_likely_chat_only_model

        assert is_likely_chat_only_model("llama3") is True
        assert is_likely_chat_only_model("kimi-k2:1t-cloud") is True
        assert is_likely_chat_only_model("qwen2.5:7b") is True
        assert is_likely_chat_only_model("gemma3:1b") is True

    def test_is_likely_chat_only_model_false(self) -> None:
        """Embedding models should not be detected as chat-only."""
        from binary_semantic_cache.embeddings.ollama_backend import is_likely_chat_only_model

        assert is_likely_chat_only_model("nomic-embed-text") is False
        assert is_likely_chat_only_model("mxbai-embed-large") is False

    def test_get_recommended_embedding_models(self) -> None:
        """Should return a list of recommended models."""
        from binary_semantic_cache.embeddings.ollama_backend import get_recommended_embedding_models

        models = get_recommended_embedding_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "nomic-embed-text" in models


class TestOllamaContextManager:
    """Test OllamaEmbedder context manager."""

    def test_context_manager(self) -> None:
        """Context manager should work."""
        from binary_semantic_cache.embeddings.ollama_backend import OllamaEmbedder

        with OllamaEmbedder() as embedder:
            assert embedder is not None
        # Session should be closed after exiting


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

