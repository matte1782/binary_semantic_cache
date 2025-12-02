"""Unit tests for BinaryEncoder.

Tests:
- Deterministic encoding (same seed = same output)
- Output shape correctness
- Batch vs single consistency
- Input validation
"""

import numpy as np
import pytest

import sys
from pathlib import Path

# Add src to path
_SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from binary_semantic_cache.core.encoder import BinaryEncoder


class TestBinaryEncoderInit:
    """Test encoder initialization."""

    def test_default_init(self) -> None:
        """Test default initialization parameters."""
        encoder = BinaryEncoder()
        assert encoder.embedding_dim == 384
        assert encoder.code_bits == 256
        assert encoder.seed == 42
        assert encoder.n_words == 4  # 256 / 64

    def test_custom_init(self) -> None:
        """Test custom initialization parameters."""
        encoder = BinaryEncoder(embedding_dim=768, code_bits=512, seed=123)
        assert encoder.embedding_dim == 768
        assert encoder.code_bits == 512
        assert encoder.seed == 123
        assert encoder.n_words == 8  # 512 / 64

    def test_invalid_embedding_dim(self) -> None:
        """Test that invalid embedding_dim raises ValueError."""
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            BinaryEncoder(embedding_dim=0)
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            BinaryEncoder(embedding_dim=-1)

    def test_invalid_code_bits(self) -> None:
        """Test that invalid code_bits raises ValueError."""
        with pytest.raises(ValueError, match="code_bits must be positive"):
            BinaryEncoder(code_bits=0)
        with pytest.raises(ValueError, match="code_bits must be positive"):
            BinaryEncoder(code_bits=-1)


class TestDeterministicEncoding:
    """Test that encoding is deterministic with same seed."""

    def test_same_seed_same_output(self) -> None:
        """Same seed should produce identical encodings."""
        encoder1 = BinaryEncoder(seed=42)
        encoder2 = BinaryEncoder(seed=42)

        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        code1 = encoder1.encode(embedding)
        code2 = encoder2.encode(embedding)

        np.testing.assert_array_equal(code1, code2)

    def test_different_seed_different_output(self) -> None:
        """Different seeds should produce different encodings."""
        encoder1 = BinaryEncoder(seed=42)
        encoder2 = BinaryEncoder(seed=43)

        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        code1 = encoder1.encode(embedding)
        code2 = encoder2.encode(embedding)

        # They should be different (with very high probability)
        assert not np.array_equal(code1, code2)

    def test_multiple_calls_same_result(self) -> None:
        """Multiple encode calls with same input should give same result."""
        encoder = BinaryEncoder(seed=42)

        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        code1 = encoder.encode(embedding)
        code2 = encoder.encode(embedding)
        code3 = encoder.encode(embedding)

        np.testing.assert_array_equal(code1, code2)
        np.testing.assert_array_equal(code2, code3)


class TestEncodeShape:
    """Test output shape correctness."""

    def test_single_encode_shape(self) -> None:
        """Single encode should return (n_words,) shape."""
        encoder = BinaryEncoder(code_bits=256)
        embedding = np.random.randn(384).astype(np.float32)

        code = encoder.encode(embedding)

        assert code.shape == (4,)  # 256 / 64 = 4
        assert code.dtype == np.uint64

    def test_batch_encode_shape(self) -> None:
        """Batch encode should return (n, n_words) shape."""
        encoder = BinaryEncoder(code_bits=256)
        embeddings = np.random.randn(10, 384).astype(np.float32)

        codes = encoder.encode_batch(embeddings)

        assert codes.shape == (10, 4)
        assert codes.dtype == np.uint64

    def test_single_batch_encode_shape(self) -> None:
        """Batch encode with n=1 should return (1, n_words) shape."""
        encoder = BinaryEncoder(code_bits=256)
        embeddings = np.random.randn(1, 384).astype(np.float32)

        codes = encoder.encode_batch(embeddings)

        assert codes.shape == (1, 4)

    def test_custom_code_bits_shape(self) -> None:
        """Test shape with different code_bits values."""
        for code_bits, expected_words in [(64, 1), (128, 2), (256, 4), (512, 8)]:
            encoder = BinaryEncoder(embedding_dim=384, code_bits=code_bits)
            embedding = np.random.randn(384).astype(np.float32)

            code = encoder.encode(embedding)

            assert code.shape == (expected_words,), f"Failed for code_bits={code_bits}"


class TestBatchConsistency:
    """Test that batch encoding matches individual encoding."""

    def test_batch_equals_individual(self) -> None:
        """Batch encode should equal individual encodes stacked."""
        encoder = BinaryEncoder(seed=42)

        # Create 5 embeddings
        embeddings = np.random.randn(5, 384).astype(np.float32)

        # Batch encode
        batch_codes = encoder.encode_batch(embeddings)

        # Individual encode
        individual_codes = np.array([
            encoder.encode(embeddings[i]) for i in range(5)
        ])

        np.testing.assert_array_equal(batch_codes, individual_codes)

    def test_1d_input_to_batch(self) -> None:
        """1D input to encode_batch should work (reshaped to 2D)."""
        encoder = BinaryEncoder(seed=42)
        embedding = np.random.randn(384).astype(np.float32)

        # encode_batch with 1D input (should reshape)
        batch_code = encoder.encode_batch(embedding)

        # Single encode
        single_code = encoder.encode(embedding)

        np.testing.assert_array_equal(batch_code.squeeze(0), single_code)


class TestInputValidation:
    """Test input validation."""

    def test_wrong_embedding_dim(self) -> None:
        """Wrong embedding dimension should raise ValueError."""
        encoder = BinaryEncoder(embedding_dim=384)

        wrong_dim = np.random.randn(256).astype(np.float32)
        with pytest.raises(ValueError, match="Expected embedding dim 384"):
            encoder.encode(wrong_dim)

    def test_wrong_batch_dim(self) -> None:
        """Wrong batch embedding dimension should raise ValueError."""
        encoder = BinaryEncoder(embedding_dim=384)

        wrong_dim = np.random.randn(10, 256).astype(np.float32)
        with pytest.raises(ValueError, match="Expected embedding dim 384"):
            encoder.encode_batch(wrong_dim)

    def test_non_array_input(self) -> None:
        """Non-array input should raise TypeError."""
        encoder = BinaryEncoder()

        with pytest.raises(TypeError, match="Expected np.ndarray"):
            encoder.encode([1.0, 2.0, 3.0])  # type: ignore

    def test_nan_input(self) -> None:
        """NaN in input should raise ValueError."""
        encoder = BinaryEncoder()

        embedding = np.random.randn(384).astype(np.float32)
        embedding[0] = np.nan

        with pytest.raises(ValueError, match="non-finite"):
            encoder.encode(embedding)

    def test_inf_input(self) -> None:
        """Inf in input should raise ValueError."""
        encoder = BinaryEncoder()

        embedding = np.random.randn(384).astype(np.float32)
        embedding[0] = np.inf

        with pytest.raises(ValueError, match="non-finite"):
            encoder.encode(embedding)

    def test_dtype_conversion(self) -> None:
        """Float64 input should be converted to float32."""
        encoder = BinaryEncoder(seed=42)

        embedding_f64 = np.random.randn(384).astype(np.float64)
        embedding_f32 = embedding_f64.astype(np.float32)

        code_f64 = encoder.encode(embedding_f64)
        code_f32 = encoder.encode(embedding_f32)

        np.testing.assert_array_equal(code_f64, code_f32)


class TestRepr:
    """Test string representation."""

    def test_repr(self) -> None:
        """Test __repr__ output."""
        encoder = BinaryEncoder(embedding_dim=384, code_bits=256, seed=42)
        expected = "BinaryEncoder(embedding_dim=384, code_bits=256, seed=42)"
        assert repr(encoder) == expected


class TestEncodeAutoDetection:
    """Test that encode() auto-detects batch vs single based on ndim."""

    def test_1d_input_returns_1d(self) -> None:
        """1D input should return 1D output."""
        encoder = BinaryEncoder(embedding_dim=384, code_bits=256)
        embedding = np.random.randn(384).astype(np.float32)
        
        code = encoder.encode(embedding)
        
        assert code.ndim == 1, f"Expected 1D, got {code.ndim}D with shape {code.shape}"
        assert code.shape == (4,)

    def test_2d_input_returns_2d(self) -> None:
        """2D input should return 2D output (batch mode)."""
        encoder = BinaryEncoder(embedding_dim=384, code_bits=256)
        embeddings = np.random.randn(6, 384).astype(np.float32)
        
        codes = encoder.encode(embeddings)
        
        assert codes.ndim == 2, f"Expected 2D, got {codes.ndim}D with shape {codes.shape}"
        assert codes.shape == (6, 4)

    def test_2d_single_returns_2d(self) -> None:
        """2D input with n=1 should return 2D output with shape (1, n_words)."""
        encoder = BinaryEncoder(embedding_dim=384, code_bits=256)
        embeddings = np.random.randn(1, 384).astype(np.float32)
        
        codes = encoder.encode(embeddings)
        
        assert codes.ndim == 2, f"Expected 2D, got {codes.ndim}D with shape {codes.shape}"
        assert codes.shape == (1, 4)

    def test_3d_input_raises_error(self) -> None:
        """3D input should raise ValueError."""
        encoder = BinaryEncoder(embedding_dim=384, code_bits=256)
        embeddings = np.random.randn(2, 3, 384).astype(np.float32)
        
        with pytest.raises(ValueError, match="1D.*2D.*got 3D"):
            encoder.encode(embeddings)

    def test_encode_batch_mode_equals_explicit_batch(self) -> None:
        """encode() with 2D should equal encode_batch()."""
        encoder = BinaryEncoder(seed=42)
        embeddings = np.random.randn(5, 384).astype(np.float32)
        
        # Using auto-detection
        codes_auto = encoder.encode(embeddings)
        
        # Using explicit batch
        codes_explicit = encoder.encode_batch(embeddings)
        
        np.testing.assert_array_equal(codes_auto, codes_explicit)

    def test_encode_consistency_1d_vs_2d(self) -> None:
        """Single embedding encoded as 1D should match same embedding in 2D batch."""
        encoder = BinaryEncoder(seed=42)
        
        embedding_1d = np.random.randn(384).astype(np.float32)
        embedding_2d = embedding_1d.reshape(1, -1)
        
        code_1d = encoder.encode(embedding_1d)
        codes_2d = encoder.encode(embedding_2d)
        
        np.testing.assert_array_equal(code_1d, codes_2d.squeeze(0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

