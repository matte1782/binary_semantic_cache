"""Cross-validation tests for Rust BinaryEncoder.

These tests verify bit-exact compatibility between the Python
and Rust encoder implementations.

Prerequisites:
    Run `maturin develop` in the project root before running these tests.

Tests:
- Bit-exact output matching Python encoder
- Deterministic encoding with same projection matrix
- Shape correctness
- Input validation
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
_SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))


# Try to import the Rust extension - skip tests if not built
try:
    from binary_semantic_cache.binary_semantic_cache_rs import RustBinaryEncoder
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    RustBinaryEncoder = None


# Helper to generate projection matrix matching Python's RandomProjection
def generate_projection_matrix(
    code_bits: int, embedding_dim: int, seed: int = 42
) -> np.ndarray:
    """
    Generate a Gaussian random projection matrix.
    
    This replicates numpy's default_rng behavior for determinism.
    
    Args:
        code_bits: Number of output bits
        embedding_dim: Input embedding dimension
        seed: Random seed for reproducibility
        
    Returns:
        Projection matrix of shape (code_bits, embedding_dim), dtype float32
    """
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((code_bits, embedding_dim)).astype(np.float32)
    return matrix


# Reference implementation matching Python encoder logic
def python_encode_reference(
    embedding: np.ndarray,
    projection_matrix: np.ndarray,
    code_bits: int,
) -> np.ndarray:
    """
    Reference Python implementation of the encoding pipeline.
    
    FROZEN FORMULA: project → binarize → pack
    
    Args:
        embedding: Float32 array of shape (embedding_dim,) or (N, embedding_dim)
        projection_matrix: Float32 matrix of shape (code_bits, embedding_dim)
        code_bits: Number of bits in output
        
    Returns:
        Packed uint64 codes
    """
    # Handle single vs batch
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    
    # Project: (N, embedding_dim) @ (embedding_dim, code_bits).T -> (N, code_bits)
    projected = embedding @ projection_matrix.T
    
    # Binarize: 1 if >= 0, else 0
    binary = (projected >= 0).astype(np.uint8)
    
    # Pack into uint64 (LSB-first layout)
    n_words = (code_bits + 63) // 64
    n_samples = binary.shape[0]
    packed = np.zeros((n_samples, n_words), dtype=np.uint64)
    
    for sample_idx in range(n_samples):
        for bit_idx in range(code_bits):
            if binary[sample_idx, bit_idx]:
                word_idx = bit_idx // 64
                bit_pos = bit_idx % 64
                packed[sample_idx, word_idx] |= np.uint64(1) << np.uint64(bit_pos)
    
    if squeeze:
        return packed.squeeze(0)
    return packed


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")
class TestRustEncoderInit:
    """Test Rust encoder initialization."""

    def test_create_encoder(self) -> None:
        """Test basic encoder creation."""
        projection = generate_projection_matrix(256, 384)
        encoder = RustBinaryEncoder(384, 256, projection)
        
        assert encoder.embedding_dim == 384
        assert encoder.code_bits == 256
        assert encoder.n_words == 4

    def test_invalid_matrix_shape(self) -> None:
        """Test that wrong matrix shape raises error."""
        projection = generate_projection_matrix(128, 384)  # Wrong code_bits
        
        with pytest.raises(ValueError, match="shape"):
            RustBinaryEncoder(384, 256, projection)

    def test_non_finite_matrix_rejected(self) -> None:
        """Test that NaN in matrix raises error."""
        projection = generate_projection_matrix(256, 384)
        projection[0, 0] = np.nan
        
        with pytest.raises(ValueError, match="non-finite"):
            RustBinaryEncoder(384, 256, projection)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")
class TestRustEncoderBitExact:
    """Test bit-exact compatibility with Python implementation."""

    def test_single_embedding_matches_python(self) -> None:
        """Single embedding should match Python reference."""
        np.random.seed(12345)  # Fixed seed for reproducible test
        
        projection = generate_projection_matrix(256, 384, seed=42)
        embedding = np.random.randn(384).astype(np.float32)
        
        # Python reference
        python_code = python_encode_reference(embedding, projection, 256)
        
        # Rust encoder
        rust_encoder = RustBinaryEncoder(384, 256, projection)
        rust_code = rust_encoder.encode(embedding)
        
        np.testing.assert_array_equal(
            rust_code, python_code,
            err_msg="Rust output does not match Python reference"
        )
    
    def test_rust_python_encoder_parity_with_seed(self) -> None:
        """OAI-08 (BLOCKING): Verify RustBinaryEncoder(seed=N) == BinaryEncoder(seed=N) bit-exact.
        
        This is a regression guard for the Phase 1 Encoder Determinism contract.
        If this test fails, Rust and Python encoders cannot be swapped without re-encoding.
        """
        from binary_semantic_cache.core.encoder import BinaryEncoder
        
        # Initialize both encoders with same seed
        seed = 42
        rust_encoder = RustBinaryEncoder(384, 256, seed=seed)
        python_encoder = BinaryEncoder(embedding_dim=384, code_bits=256, seed=seed)
        
        # Generate test embeddings
        rng = np.random.default_rng(123)
        embeddings = rng.standard_normal((100, 384)).astype(np.float32)
        
        # Encode with both
        rust_codes = rust_encoder.encode(embeddings)
        python_codes = python_encoder.encode(embeddings)
        
        # Verify bit-exact match
        np.testing.assert_array_equal(
            rust_codes, python_codes,
            err_msg="CRITICAL: Rust and Python encoders produce different outputs with same seed"
        )
        
        # Also verify single-vector encoding
        single_emb = rng.standard_normal(384).astype(np.float32)
        rust_single = rust_encoder.encode(single_emb)
        python_single = python_encoder.encode(single_emb)
        
        np.testing.assert_array_equal(
            rust_single, python_single,
            err_msg="CRITICAL: Single-vector encoding mismatch"
        )

    def test_batch_embedding_matches_python(self) -> None:
        """Batch embeddings should match Python reference."""
        np.random.seed(12345)
        
        projection = generate_projection_matrix(256, 384, seed=42)
        embeddings = np.random.randn(10, 384).astype(np.float32)
        
        # Python reference
        python_codes = python_encode_reference(embeddings, projection, 256)
        
        # Rust encoder
        rust_encoder = RustBinaryEncoder(384, 256, projection)
        rust_codes = rust_encoder.encode(embeddings)
        
        np.testing.assert_array_equal(
            rust_codes, python_codes,
            err_msg="Rust batch output does not match Python reference"
        )

    def test_normalized_embedding_matches_python(self) -> None:
        """Normalized embedding should match Python reference."""
        np.random.seed(67890)
        
        projection = generate_projection_matrix(256, 384, seed=42)
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        
        python_code = python_encode_reference(embedding, projection, 256)
        
        rust_encoder = RustBinaryEncoder(384, 256, projection)
        rust_code = rust_encoder.encode(embedding)
        
        np.testing.assert_array_equal(rust_code, python_code)

    def test_different_code_bits_matches_python(self) -> None:
        """Different code_bits values should match Python."""
        np.random.seed(11111)
        
        for code_bits in [64, 128, 256, 512]:
            projection = generate_projection_matrix(code_bits, 384, seed=42)
            embedding = np.random.randn(384).astype(np.float32)
            
            python_code = python_encode_reference(embedding, projection, code_bits)
            
            rust_encoder = RustBinaryEncoder(384, code_bits, projection)
            rust_code = rust_encoder.encode(embedding)
            
            np.testing.assert_array_equal(
                rust_code, python_code,
                err_msg=f"Mismatch for code_bits={code_bits}"
            )


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")
class TestRustEncoderDeterminism:
    """Test deterministic behavior of Rust encoder."""

    def test_same_input_same_output(self) -> None:
        """Same input should produce same output."""
        projection = generate_projection_matrix(256, 384, seed=42)
        rust_encoder = RustBinaryEncoder(384, 256, projection)
        
        embedding = np.random.randn(384).astype(np.float32)
        
        code1 = rust_encoder.encode(embedding)
        code2 = rust_encoder.encode(embedding)
        code3 = rust_encoder.encode(embedding)
        
        np.testing.assert_array_equal(code1, code2)
        np.testing.assert_array_equal(code2, code3)

    def test_multiple_encoders_same_output(self) -> None:
        """Multiple encoders with same matrix should produce same output."""
        projection = generate_projection_matrix(256, 384, seed=42)
        
        encoder1 = RustBinaryEncoder(384, 256, projection)
        encoder2 = RustBinaryEncoder(384, 256, projection)
        
        embedding = np.random.randn(384).astype(np.float32)
        
        code1 = encoder1.encode(embedding)
        code2 = encoder2.encode(embedding)
        
        np.testing.assert_array_equal(code1, code2)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")
class TestRustEncoderShape:
    """Test output shape correctness."""

    def test_single_encode_shape(self) -> None:
        """Single encode should return 1D array."""
        projection = generate_projection_matrix(256, 384)
        encoder = RustBinaryEncoder(384, 256, projection)
        
        embedding = np.random.randn(384).astype(np.float32)
        code = encoder.encode(embedding)
        
        assert code.shape == (4,)
        assert code.dtype == np.uint64

    def test_batch_encode_shape(self) -> None:
        """Batch encode should return 2D array."""
        projection = generate_projection_matrix(256, 384)
        encoder = RustBinaryEncoder(384, 256, projection)
        
        embeddings = np.random.randn(10, 384).astype(np.float32)
        codes = encoder.encode(embeddings)
        
        assert codes.shape == (10, 4)
        assert codes.dtype == np.uint64

    def test_encode_batch_method(self) -> None:
        """encode_batch method should work correctly."""
        projection = generate_projection_matrix(256, 384)
        encoder = RustBinaryEncoder(384, 256, projection)
        
        embeddings = np.random.randn(5, 384).astype(np.float32)
        codes = encoder.encode_batch(embeddings)
        
        assert codes.shape == (5, 4)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")
class TestRustEncoderValidation:
    """Test input validation."""

    def test_wrong_embedding_dim(self) -> None:
        """Wrong embedding dimension should raise error."""
        projection = generate_projection_matrix(256, 384)
        encoder = RustBinaryEncoder(384, 256, projection)
        
        wrong_embedding = np.random.randn(256).astype(np.float32)
        
        with pytest.raises(ValueError, match="embedding dim"):
            encoder.encode(wrong_embedding)

    def test_nan_input_rejected(self) -> None:
        """NaN input should raise error."""
        projection = generate_projection_matrix(256, 384)
        encoder = RustBinaryEncoder(384, 256, projection)
        
        embedding = np.random.randn(384).astype(np.float32)
        embedding[0] = np.nan
        
        with pytest.raises(ValueError, match="non-finite"):
            encoder.encode(embedding)

    def test_inf_input_rejected(self) -> None:
        """Inf input should raise error."""
        projection = generate_projection_matrix(256, 384)
        encoder = RustBinaryEncoder(384, 256, projection)
        
        embedding = np.random.randn(384).astype(np.float32)
        embedding[0] = np.inf
        
        with pytest.raises(ValueError, match="non-finite"):
            encoder.encode(embedding)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")
class TestRustEncoderRepr:
    """Test string representation."""

    def test_repr(self) -> None:
        """Test __repr__ output."""
        projection = generate_projection_matrix(256, 384)
        encoder = RustBinaryEncoder(384, 256, projection)
        
        repr_str = repr(encoder)
        assert "RustBinaryEncoder" in repr_str
        assert "384" in repr_str
        assert "256" in repr_str


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")
class TestRustEncoderEdgeCases:
    """Test edge cases for Rust encoder (T1-T8 from PHASE2_TEST_MATRIX.md)."""

    def test_empty_batch_returns_empty(self) -> None:
        """T1 (Blocking): Verify encode with shape (0, 384) returns shape (0, n_words)."""
        projection = generate_projection_matrix(256, 384)
        encoder = RustBinaryEncoder(384, 256, projection)
        
        # Create empty batch with correct embedding dimension
        empty_batch = np.zeros((0, 384), dtype=np.float32)
        result = encoder.encode(empty_batch)
        
        assert result.shape == (0, 4)  # 256 bits = 4 words
        assert result.dtype == np.uint64

    def test_float64_input_rejected(self) -> None:
        """T2 (Blocking): Verify passing a float64 array raises TypeError (Rust expects float32)."""
        projection = generate_projection_matrix(256, 384)
        encoder = RustBinaryEncoder(384, 256, projection)
        
        # Create float64 embedding
        embedding = np.random.randn(384).astype(np.float64)
        
        with pytest.raises(TypeError, match="float32|f32|float"):
            encoder.encode(embedding)

    def test_all_zero_embedding(self) -> None:
        """T3 (Blocking): Verify zeros(384) produces a deterministic, non-empty code."""
        projection = generate_projection_matrix(256, 384, seed=42)
        encoder = RustBinaryEncoder(384, 256, projection)
        
        # All zeros embedding
        zeros = np.zeros(384, dtype=np.float32)
        
        # Encode multiple times to verify determinism
        code1 = encoder.encode(zeros)
        code2 = encoder.encode(zeros)
        
        # Verify deterministic
        np.testing.assert_array_equal(code1, code2)
        
        # Verify non-empty (at least one bit is set)
        assert code1.shape == (4,)
        assert code1.any(), "All-zero embedding should produce non-empty code"

    def test_negative_zero_handled(self) -> None:
        """T4 (Blocking): Verify -0.0 is treated as 0.0 (bit=1) in projection."""
        projection = generate_projection_matrix(256, 384)
        encoder = RustBinaryEncoder(384, 256, projection)
        
        # Create embeddings with positive and negative zeros
        embedding_pos = np.zeros(384, dtype=np.float32)
        embedding_neg = np.zeros(384, dtype=np.float32)
        embedding_neg[:] = -0.0
        
        code_pos = encoder.encode(embedding_pos)
        code_neg = encoder.encode(embedding_neg)
        
        # IEEE 754: -0.0 >= 0 should be True, so both should produce identical codes
        np.testing.assert_array_equal(
            code_pos, code_neg,
            err_msg="Negative zero should be treated same as positive zero"
        )

    def test_inf_in_matrix_rejected(self) -> None:
        """T5 (Blocking): Verify RustBinaryEncoder constructor raises ValueError if projection contains Inf."""
        projection = generate_projection_matrix(256, 384)
        # Add infinity to the projection matrix
        projection[0, 0] = np.inf
        
        with pytest.raises(ValueError, match="non-finite|Inf|infinity"):
            RustBinaryEncoder(384, 256, projection)

    def test_3d_input_rejected(self) -> None:
        """T6: Verify encoding a 3D array raises TypeError/ValueError."""
        projection = generate_projection_matrix(256, 384)
        encoder = RustBinaryEncoder(384, 256, projection)
        
        # Create 3D array
        array_3d = np.random.randn(2, 5, 384).astype(np.float32)
        
        with pytest.raises((TypeError, ValueError), match="2D|dimension|shape"):
            encoder.encode(array_3d)

    def test_code_bits_65_boundary(self) -> None:
        """T7: Initialize with code_bits=65, verify n_words=2, check bit packing across word boundary."""
        # Generate projection matrix for 65 bits
        projection = generate_projection_matrix(65, 384)
        encoder = RustBinaryEncoder(384, 65, projection)
        
        # Verify n_words is 2 (65 bits needs 2 uint64 words)
        assert encoder.n_words == 2
        
        # Encode a random embedding
        embedding = np.random.randn(384).astype(np.float32)
        code = encoder.encode(embedding)
        
        # Verify output shape
        assert code.shape == (2,)
        assert code.dtype == np.uint64
        
        # Encode batch to verify batch handling
        batch = np.random.randn(3, 384).astype(np.float32)
        codes = encoder.encode(batch)
        assert codes.shape == (3, 2)

    def test_large_batch_stress(self) -> None:
        """T8: Successfully encode a batch of 1000+ embeddings."""
        projection = generate_projection_matrix(256, 384)
        encoder = RustBinaryEncoder(384, 256, projection)
        
        # Create large batch
        large_batch = np.random.randn(1000, 384).astype(np.float32)
        
        # Should not raise any errors
        codes = encoder.encode(large_batch)
        
        # Verify output
        assert codes.shape == (1000, 4)
        assert codes.dtype == np.uint64
        
        # Verify no NaN or invalid values in output
        assert not np.isnan(codes).any()


if __name__ == "__main__":
    if not RUST_AVAILABLE:
        print("Rust extension not available. Run 'maturin develop' first.")
        print("Skipping tests.")
        sys.exit(0)
    
    pytest.main([__file__, "-v"])

