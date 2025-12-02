"""
Binary Encoder Module

Converts float embeddings to compact binary codes using:
1. Gaussian Random Projection (dimensionality reduction)
2. Sign Binarization (float → {0, 1})
3. Bit Packing (bits → uint64 words)

FROZEN FORMULA - DO NOT MODIFY WITHOUT RE-VALIDATION
See: docs/DECISION_LOG_v1.md (D1, D2)

Shape Handling:
- Single embedding: (embedding_dim,) → (n_words,)
- Batch embeddings: (N, embedding_dim) → (N, n_words)
- Auto-detection: encode() infers batch from ndim
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Final, Union

import numpy as np

# Add BinaryLLM to path for imports
_BINARY_LLM_PATH = Path(__file__).resolve().parents[5] / "binary_llm"
if str(_BINARY_LLM_PATH) not in sys.path:
    sys.path.insert(0, str(_BINARY_LLM_PATH))

from src.quantization.binarization import RandomProjection, binarize_sign
from src.quantization.packing import pack_codes

logger = logging.getLogger(__name__)

# Frozen constants from validation
DEFAULT_EMBEDDING_DIM: Final[int] = 384
DEFAULT_CODE_BITS: Final[int] = 256
DEFAULT_SEED: Final[int] = 42


class BinaryEncoder:
    """
    Encode float embeddings to binary codes.

    Uses BinaryLLM's RandomProjection + binarize_sign + pack_codes pipeline.
    The encoding formula is FROZEN and validated for correlation r >= 0.93.

    Shape Handling:
        - encode(1D array) → single code (n_words,)
        - encode(2D array) → batch codes (N, n_words)
        - encode_batch() → explicitly batch mode

    Attributes:
        embedding_dim: Input embedding dimension (default: 384)
        code_bits: Output binary code length (default: 256)
        seed: Random seed for reproducible projection (default: 42)

    Example:
        >>> encoder = BinaryEncoder(embedding_dim=384, code_bits=256)
        >>> # Single embedding
        >>> embedding = np.random.randn(384).astype(np.float32)
        >>> code = encoder.encode(embedding)
        >>> code.shape
        (4,)  # 256 bits = 4 x uint64
        >>> # Batch of embeddings
        >>> embeddings = np.random.randn(10, 384).astype(np.float32)
        >>> codes = encoder.encode(embeddings)
        >>> codes.shape
        (10, 4)
    """

    __slots__ = ("_embedding_dim", "_code_bits", "_seed", "_projection", "_n_words")

    def __init__(
        self,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        code_bits: int = DEFAULT_CODE_BITS,
        seed: int = DEFAULT_SEED,
    ) -> None:
        """
        Initialize the binary encoder.

        Args:
            embedding_dim: Dimension of input float embeddings.
            code_bits: Number of bits in output binary code.
            seed: Random seed for projection matrix (ensures determinism).

        Raises:
            ValueError: If embedding_dim <= 0 or code_bits <= 0.
        """
        if embedding_dim <= 0:
            logger.error("embedding_dim must be positive, got %d", embedding_dim)
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if code_bits <= 0:
            logger.error("code_bits must be positive, got %d", code_bits)
            raise ValueError(f"code_bits must be positive, got {code_bits}")
        if code_bits % 64 != 0:
            logger.warning(
                "code_bits=%d is not a multiple of 64, will be padded", code_bits
            )

        self._embedding_dim = embedding_dim
        self._code_bits = code_bits
        self._seed = seed
        self._n_words = (code_bits + 63) // 64

        # Initialize projection matrix (deterministic)
        self._projection = RandomProjection(
            input_dim=embedding_dim,
            output_bits=code_bits,
            seed=seed,
        )
        logger.debug(
            "BinaryEncoder initialized: dim=%d, bits=%d, seed=%d",
            embedding_dim,
            code_bits,
            seed,
        )

    @property
    def embedding_dim(self) -> int:
        """Input embedding dimension."""
        return self._embedding_dim

    @property
    def code_bits(self) -> int:
        """Output binary code length in bits."""
        return self._code_bits

    @property
    def n_words(self) -> int:
        """Number of uint64 words in packed output."""
        return self._n_words

    @property
    def seed(self) -> int:
        """Random seed used for projection."""
        return self._seed

    def _validate_single(self, embedding: np.ndarray) -> np.ndarray:
        """
        Validate a single 1D embedding.
        
        Args:
            embedding: Must be 1D array of shape (embedding_dim,)
            
        Returns:
            Validated float32 embedding
            
        Raises:
            TypeError: If not numpy array
            ValueError: If wrong shape or contains non-finite values
        """
        if not isinstance(embedding, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(embedding).__name__}")
        
        if embedding.ndim != 1:
            raise ValueError(
                f"Single embedding must be 1D, got shape {embedding.shape}. "
                f"For batch encoding, pass a 2D array or use encode_batch()."
            )
        
        if embedding.shape[0] != self._embedding_dim:
            raise ValueError(
                f"Expected embedding dim {self._embedding_dim}, "
                f"got {embedding.shape[0]}"
            )
        
        # Ensure float32
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        
        # Check for non-finite values
        if not np.isfinite(embedding).all():
            raise ValueError("Embedding contains non-finite values (NaN or Inf)")
        
        return embedding

    def _validate_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Validate a batch of 2D embeddings.
        
        Args:
            embeddings: Must be 2D array of shape (N, embedding_dim)
            
        Returns:
            Validated float32 embeddings
            
        Raises:
            TypeError: If not numpy array
            ValueError: If wrong shape or contains non-finite values
        """
        if not isinstance(embeddings, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(embeddings).__name__}")
        
        if embeddings.ndim != 2:
            raise ValueError(
                f"Batch embeddings must be 2D, got shape {embeddings.shape}. "
                f"For single embedding, pass a 1D array."
            )
        
        if embeddings.shape[1] != self._embedding_dim:
            raise ValueError(
                f"Expected embedding dim {self._embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )
        
        # Ensure float32
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Check for non-finite values
        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain non-finite values (NaN or Inf)")
        
        return embeddings

    def _validate_embedding(self, embedding: np.ndarray, batch: bool = False) -> np.ndarray:
        """
        Validate embedding input (legacy method for compatibility).
        
        Args:
            embedding: Input array
            batch: If True, expect 2D. If False, expect 1D.
            
        Returns:
            Validated embedding(s)
        """
        if batch:
            # Allow 1D to be promoted to (1, dim) for backwards compatibility
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            return self._validate_batch(embedding)
        else:
            # Allow (1, dim) to be squeezed for backwards compatibility
            if embedding.ndim == 2 and embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)
            return self._validate_single(embedding)

    def encode(self, embedding: np.ndarray) -> np.ndarray:
        """
        Encode embedding(s) to binary code(s).
        
        Automatically detects single vs batch based on input dimensions:
        - 1D input (embedding_dim,) → returns 1D output (n_words,)
        - 2D input (N, embedding_dim) → returns 2D output (N, n_words)

        Args:
            embedding: Float array of shape (embedding_dim,) for single,
                       or (N, embedding_dim) for batch.

        Returns:
            Binary code(s) as uint64 array.
            - Single: shape (n_words,)
            - Batch: shape (N, n_words)

        Raises:
            TypeError: If embedding is not a numpy array.
            ValueError: If embedding has wrong shape, dtype, or contains non-finite values.
        """
        if not isinstance(embedding, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(embedding).__name__}")
        
        # Auto-detect batch vs single based on ndim
        if embedding.ndim == 1:
            # Single embedding path
            return self._encode_single(embedding)
        elif embedding.ndim == 2:
            # Batch embedding path
            return self._encode_batch(embedding)
        else:
            raise ValueError(
                f"Embedding must be 1D (single) or 2D (batch), got {embedding.ndim}D "
                f"with shape {embedding.shape}"
            )

    def _encode_single(self, embedding: np.ndarray) -> np.ndarray:
        """
        Encode a single 1D embedding.
        
        Args:
            embedding: 1D array of shape (embedding_dim,)
            
        Returns:
            Binary code of shape (n_words,)
        """
        embedding = self._validate_single(embedding)
        
        # Reshape for batch processing
        embedding_2d = embedding.reshape(1, -1)
        
        # FROZEN FORMULA: project → binarize → pack
        projected = self._projection.project(embedding_2d)
        binarized = binarize_sign(projected)  # Returns +1/-1
        
        # Convert from +1/-1 to 0/1 for packing
        codes_01 = (binarized >= 0).astype(np.uint8)
        
        # Pack to uint64
        packed = pack_codes(codes_01)
        
        return packed.squeeze(0)

    def _encode_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Encode a batch of 2D embeddings.
        
        Args:
            embeddings: 2D array of shape (N, embedding_dim)
            
        Returns:
            Binary codes of shape (N, n_words)
        """
        embeddings = self._validate_batch(embeddings)
        
        # FROZEN FORMULA: project → binarize → pack
        projected = self._projection.project(embeddings)
        binarized = binarize_sign(projected)  # Returns +1/-1
        
        # Convert from +1/-1 to 0/1 for packing
        codes_01 = (binarized >= 0).astype(np.uint8)
        
        # Pack to uint64
        packed = pack_codes(codes_01)
        
        return packed

    def encode_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Encode multiple embeddings to binary codes.
        
        Explicit batch mode (for callers who want to be explicit).

        Args:
            embeddings: Float array of shape (n, embedding_dim).

        Returns:
            Binary codes as uint64 array of shape (n, n_words).

        Raises:
            TypeError: If embeddings is not a numpy array.
            ValueError: If embeddings have wrong shape, dtype, or contain non-finite values.
        """
        if not isinstance(embeddings, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(embeddings).__name__}")
        
        # Allow 1D to be treated as single-item batch
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        return self._encode_batch(embeddings)

    def __repr__(self) -> str:
        return (
            f"BinaryEncoder(embedding_dim={self._embedding_dim}, "
            f"code_bits={self._code_bits}, seed={self._seed})"
        )
