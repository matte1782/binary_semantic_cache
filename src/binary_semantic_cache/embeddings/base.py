"""
Abstract base class for embedding backends.

All embedding backends should implement this interface for consistency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class BaseEmbedder(ABC):
    """
    Abstract base class for embedding backends.
    
    Provides a consistent interface for embedding text using different backends.
    """
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings produced by this backend."""
        ...
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model being used."""
        ...
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        
        Args:
            text: The text to embed
            
        Returns:
            A 1D numpy array of shape (embedding_dim,) with dtype float32
        """
        ...
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple text strings.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            A 2D numpy array of shape (len(texts), embedding_dim) with dtype float32
        """
        ...
    
    def is_available(self) -> bool:
        """
        Check if this embedding backend is available and working.
        
        Returns:
            True if the backend can generate embeddings, False otherwise
        """
        try:
            # Try to embed a simple test string
            result = self.embed_text("test")
            return result is not None and len(result) > 0
        except Exception:
            return False
    
    def normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        L2-normalize embeddings.
        
        Args:
            embeddings: Embeddings to normalize (1D or 2D array)
            
        Returns:
            Normalized embeddings
        """
        if embeddings.ndim == 1:
            norm = np.linalg.norm(embeddings)
            if norm > 0:
                return embeddings / norm
            return embeddings
        else:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)  # Avoid division by zero
            return embeddings / norms
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r}, dim={self.embedding_dim})"

