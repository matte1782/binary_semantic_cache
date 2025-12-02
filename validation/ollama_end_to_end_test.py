#!/usr/bin/env python3
"""
Ollama End-to-End Test

This script validates the binary semantic cache with REAL embeddings from Ollama.

IMPORTANT: You must use an EMBEDDING model, not a chat model.
- ✅ nomic-embed-text, mxbai-embed-large, snowflake-arctic-embed
- ❌ kimi, llama3, qwen, gemma (these are chat-only, won't work)

Usage:
    # Pull an embedding model first
    ollama pull nomic-embed-text
    
    # Run the test
    python validation/ollama_end_to_end_test.py
    
    # With specific embedding model
    python validation/ollama_end_to_end_test.py --model mxbai-embed-large
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add src to path
_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _ROOT / "src"
sys.path.insert(0, str(_SRC_PATH))

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

# Recommended embedding models (NOT chat models)
RECOMMENDED_EMBEDDING_MODELS = [
    "nomic-embed-text",      # 768 dim, fast, good quality
    "mxbai-embed-large",     # 1024 dim, higher quality
    "snowflake-arctic-embed", # 1024 dim, production-grade
    "all-minilm",            # 384 dim, smallest
    "bge-large",             # 1024 dim
]

# Test corpus - diverse texts to test semantic similarity
TEST_CORPUS = [
    # Programming questions (similar semantics)
    "How do I reverse a string in Python?",
    "What's the best way to reverse a string using Python?",
    "Python string reversal method",
    
    # Different topic - weather
    "What's the weather like today?",
    "How is the weather outside?",
    "Tell me the current weather conditions",
    
    # Another topic - cooking
    "How do I make pasta from scratch?",
    "What's the recipe for homemade pasta?",
    "Steps to prepare fresh pasta at home",
    
    # Technical - databases
    "How do I create an index in PostgreSQL?",
    "PostgreSQL index creation syntax",
    
    # Random unrelated
    "The quick brown fox jumps over the lazy dog",
    "Lorem ipsum dolor sit amet",
    
    # Edge cases
    "Hello",  # Very short
    "This is a longer sentence with more context about artificial intelligence and machine learning applications in modern software development.",  # Long
]

# Queries to test against the corpus
TEST_QUERIES = [
    # Should hit (TRUE PARAPHRASES of corpus items)
    ("How can I reverse a string in Python?", "string reverse", True),
    ("How is the weather right now?", "weather", True),  # Paraphrase of "How is the weather outside?"
    ("How to make homemade pasta?", "pasta", True),  # Paraphrase of "What's the recipe for homemade pasta?"
    
    # Should miss (very different)
    ("What is quantum computing?", "quantum", False),
    ("How tall is Mount Everest?", "mountain", False),
]


def print_embedding_model_help() -> None:
    """Print helpful information about embedding models."""
    logger.error("\n" + "=" * 60)
    logger.error("EMBEDDING MODEL REQUIRED")
    logger.error("=" * 60)
    logger.error("")
    logger.error("This test requires an EMBEDDING model, not a chat model.")
    logger.error("")
    logger.error("Chat models like kimi, llama3, qwen, gemma do NOT support embeddings.")
    logger.error("")
    logger.error("To fix, install a proper embedding model:")
    logger.error("")
    logger.error("    ollama pull nomic-embed-text")
    logger.error("")
    logger.error("Then run:")
    logger.error("")
    logger.error("    python validation/ollama_end_to_end_test.py --model nomic-embed-text")
    logger.error("")
    logger.error("Recommended embedding models:")
    for model in RECOMMENDED_EMBEDDING_MODELS:
        logger.error(f"    - {model}")
    logger.error("")


def check_ollama_available(host: Optional[str], model: str) -> Tuple[bool, str]:
    """
    Check if Ollama is available and the model supports embeddings.
    
    Returns:
        Tuple of (is_available, message)
    """
    try:
        from binary_semantic_cache.embeddings.ollama_backend import (
            OllamaEmbedder,
            OllamaConnectionError,
            OllamaModelNotFoundError,
            OllamaNotEmbeddingModelError,
        )
        
        embedder = OllamaEmbedder(model_name=model, host=host if host else None)
        
        # Check server connectivity
        models = embedder.list_models()
        if not models:
            return False, f"Ollama server at {embedder.host} is not responding or has no models"
        
        logger.info(f"Ollama available at {embedder.host}")
        logger.info(f"Installed models: {', '.join(models[:5])}{'...' if len(models) > 5 else ''}")
        
        # Check for known embedding models
        embedding_models = embedder.list_embedding_models()
        if embedding_models:
            logger.info(f"Embedding models found: {', '.join(embedding_models)}")
        
        # Test if the specified model supports embeddings
        supports, message = embedder.test_embedding_support()
        
        if not supports:
            embedder.close()
            return False, message
        
        logger.info(f"✓ {message}")
        embedder.close()
        return True, "Ollama is ready with embedding support"
        
    except ImportError as e:
        return False, f"Missing dependency: {e}. Install with: pip install httpx"
    except Exception as e:
        return False, f"Ollama check failed: {e}"


def run_end_to_end_test(
    model: str,
    host: Optional[str] = None,
    threshold: float = 0.80,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the end-to-end test with real Ollama embeddings.
    
    Args:
        model: Ollama model name (must be an embedding model)
        host: Ollama server URL (optional)
        threshold: Similarity threshold for cache hits
        verbose: Print detailed output
        
    Returns:
        Test results dictionary
    """
    from binary_semantic_cache.embeddings import OllamaEmbedder
    from binary_semantic_cache.core.encoder import BinaryEncoder
    from binary_semantic_cache.core.cache import BinarySemanticCache
    
    results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "threshold": threshold,
        "corpus_size": len(TEST_CORPUS),
        "query_count": len(TEST_QUERIES),
    }
    
    # Initialize embedder
    logger.info("=" * 60)
    logger.info("OLLAMA END-TO-END TEST")
    logger.info("=" * 60)
    
    embedder = OllamaEmbedder(model_name=model, host=host if host else None)
    logger.info(f"Embedder: {embedder}")
    
    # Get embedding dimension
    test_emb = embedder.embed_text("dimension test")
    embedding_dim = len(test_emb)
    logger.info(f"Embedding dimension: {embedding_dim}")
    results["embedding_dim"] = embedding_dim
    
    # Initialize encoder with detected dimension
    code_bits = min(256, embedding_dim)
    encoder = BinaryEncoder(
        embedding_dim=embedding_dim,
        code_bits=code_bits,
        seed=42,
    )
    logger.info(f"Encoder: {encoder}")
    
    # Initialize cache
    cache = BinarySemanticCache(
        encoder=encoder,
        max_entries=1000,
        similarity_threshold=threshold,
    )
    logger.info(f"Cache: max_entries=1000, threshold={threshold}")
    
    # Step 1: Generate embeddings for corpus
    logger.info("\n" + "-" * 60)
    logger.info("Step 1: Embedding corpus")
    logger.info("-" * 60)
    
    embed_times = []
    corpus_embeddings = []
    
    for i, text in enumerate(TEST_CORPUS):
        start = time.perf_counter()
        embedding = embedder.embed_text(text)
        elapsed = (time.perf_counter() - start) * 1000
        
        # SAFETY: Validate embedding shape
        if embedding.ndim != 1:
            raise ValueError(
                f"embed_text() must return 1D array, got shape {embedding.shape} for text #{i}"
            )
        if embedding.shape[0] != embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {embedding_dim}, got {embedding.shape[0]}"
            )
        
        embed_times.append(elapsed)
        corpus_embeddings.append(embedding)
        
        if verbose:
            preview = text[:50] + "..." if len(text) > 50 else text
            logger.info(f"  [{i+1:2d}] {elapsed:6.1f}ms | {preview}")
    
    avg_embed_time = np.mean(embed_times)
    logger.info(f"\nEmbedding stats:")
    logger.info(f"  Mean: {avg_embed_time:.1f} ms")
    logger.info(f"  Min: {min(embed_times):.1f} ms")
    logger.info(f"  Max: {max(embed_times):.1f} ms")
    
    results["embed_time_mean_ms"] = avg_embed_time
    results["embed_time_min_ms"] = min(embed_times)
    results["embed_time_max_ms"] = max(embed_times)
    
    # Step 2: Populate cache
    logger.info("\n" + "-" * 60)
    logger.info("Step 2: Populating cache")
    logger.info("-" * 60)
    
    insert_times = []
    for i, (text, embedding) in enumerate(zip(TEST_CORPUS, corpus_embeddings)):
        start = time.perf_counter()
        cache.put(embedding, {"text": text, "id": i})
        elapsed = (time.perf_counter() - start) * 1000
        insert_times.append(elapsed)
    
    logger.info(f"  Inserted {len(TEST_CORPUS)} entries")
    logger.info(f"  Mean insert time: {np.mean(insert_times):.3f} ms")
    
    results["insert_time_mean_ms"] = np.mean(insert_times)
    
    # Step 3: Run queries
    logger.info("\n" + "-" * 60)
    logger.info("Step 3: Running queries")
    logger.info("-" * 60)
    
    query_results = []
    lookup_times = []
    correct_predictions = 0
    
    for query_text, category, expected_hit in TEST_QUERIES:
        # Embed query
        query_embedding = embedder.embed_text(query_text)
        
        # Cache lookup
        start = time.perf_counter()
        result = cache.get(query_embedding)
        elapsed = (time.perf_counter() - start) * 1000
        lookup_times.append(elapsed)
        
        is_hit = result is not None
        is_correct = is_hit == expected_hit
        if is_correct:
            correct_predictions += 1
        
        status = "✓" if is_correct else "✗"
        hit_str = "HIT" if is_hit else "MISS"
        expected_str = "HIT" if expected_hit else "MISS"
        
        query_preview = query_text[:40] + "..." if len(query_text) > 40 else query_text
        
        logger.info(f"  {status} [{category:10s}] {hit_str:4s} (exp: {expected_str:4s}) | {query_preview}")
        
        if is_hit and result is not None:
            matched_text = result.response.get("text", "")[:30]
            logger.info(f"       → Matched: \"{matched_text}...\" (sim={result.similarity:.3f})")
        
        query_results.append({
            "query": query_text,
            "category": category,
            "expected_hit": expected_hit,
            "actual_hit": is_hit,
            "correct": is_correct,
            "lookup_time_ms": elapsed,
            "matched_text": result.response.get("text") if result else None,
            "similarity": result.similarity if result else None,
        })
    
    accuracy = correct_predictions / len(TEST_QUERIES)
    logger.info(f"\nQuery accuracy: {correct_predictions}/{len(TEST_QUERIES)} ({accuracy*100:.0f}%)")
    logger.info(f"Mean lookup time: {np.mean(lookup_times):.3f} ms")
    
    results["query_accuracy"] = accuracy
    results["correct_predictions"] = correct_predictions
    results["lookup_time_mean_ms"] = np.mean(lookup_times)
    results["query_results"] = query_results
    
    # Step 4: Similarity matrix analysis
    logger.info("\n" + "-" * 60)
    logger.info("Step 4: Similarity analysis")
    logger.info("-" * 60)
    
    # Compute pairwise cosine similarities for first 6 corpus items
    sample_embeddings = np.array(corpus_embeddings[:6])
    
    # SAFETY: Validate shape before processing
    assert sample_embeddings.ndim == 2, (
        f"Expected 2D array for batch embeddings, got shape {sample_embeddings.shape}"
    )
    assert sample_embeddings.shape[0] == 6, (
        f"Expected 6 sample embeddings, got {sample_embeddings.shape[0]}"
    )
    logger.info(f"  Sample embeddings shape: {sample_embeddings.shape}")
    
    cosine_sims = np.dot(sample_embeddings, sample_embeddings.T)
    
    # Compute Hamming similarities using batch encode
    # encode() auto-detects 2D input as batch
    codes = encoder.encode(sample_embeddings)
    
    # Validate codes shape
    assert codes.ndim == 2, f"Expected 2D codes array, got shape {codes.shape}"
    assert codes.shape[0] == 6, f"Expected 6 codes, got {codes.shape[0]}"
    logger.info(f"  Binary codes shape: {codes.shape}")
    
    from binary_semantic_cache.core.similarity import hamming_similarity
    
    # Compute pairwise Hamming similarities
    # hamming_similarity expects: query (1D), codes (2D)
    # So for each row i, we compute similarity against all other codes
    hamming_sims = np.zeros((6, 6))
    for i in range(6):
        # Compare codes[i] (1D query) against all codes (2D batch)
        row_sims = hamming_similarity(codes[i], codes, code_bits)
        hamming_sims[i, :] = row_sims
    
    logger.info("  Cosine vs Hamming similarity (first 6 corpus items):")
    logger.info("       " + "  ".join(f"{i:5d}" for i in range(6)))
    for i in range(6):
        row = [f"{cosine_sims[i,j]:.2f}/{hamming_sims[i,j]:.2f}" for j in range(6)]
        logger.info(f"  [{i}]  " + "  ".join(row))
    
    # Compute correlation
    upper_tri_idx = np.triu_indices(6, k=1)
    cosine_upper = cosine_sims[upper_tri_idx]
    hamming_upper = hamming_sims[upper_tri_idx]
    correlation = np.corrcoef(cosine_upper, hamming_upper)[0, 1]
    
    logger.info(f"\n  Cosine-Hamming correlation: {correlation:.4f}")
    results["similarity_correlation"] = correlation
    
    # Step 5: Cache stats
    logger.info("\n" + "-" * 60)
    logger.info("Step 5: Cache statistics")
    logger.info("-" * 60)
    
    stats = cache.stats()
    logger.info(f"  Size: {stats.size}")
    logger.info(f"  Hits: {stats.hits}")
    logger.info(f"  Misses: {stats.misses}")
    logger.info(f"  Hit rate: {stats.hit_rate:.1%}")
    logger.info(f"  Memory: {stats.memory_mb:.2f} MB")
    
    results["cache_stats"] = {
        "size": stats.size,
        "hits": stats.hits,
        "misses": stats.misses,
        "hit_rate": stats.hit_rate,
        "memory_mb": stats.memory_mb,
    }
    
    # Final verdict
    logger.info("\n" + "=" * 60)
    logger.info("VERDICT")
    logger.info("=" * 60)
    
    passed = accuracy >= 0.6 and correlation >= 0.5
    results["passed"] = passed
    
    if passed:
        logger.info("✓ END-TO-END TEST PASSED")
        logger.info(f"  - Query accuracy: {accuracy*100:.0f}% (≥60% required)")
        logger.info(f"  - Similarity correlation: {correlation:.2f} (≥0.50 required)")
    else:
        logger.info("✗ END-TO-END TEST FAILED")
        if accuracy < 0.6:
            logger.info(f"  - Query accuracy too low: {accuracy*100:.0f}% (<60%)")
        if correlation < 0.5:
            logger.info(f"  - Similarity correlation too low: {correlation:.2f} (<0.50)")
    
    # Cleanup
    embedder.close()
    
    # Save results
    output_path = _ROOT / "validation" / "results" / "ollama_e2e_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {output_path}")
    
    return results


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end test with Ollama EMBEDDING models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT: Use an embedding model, NOT a chat model.

Chat models (kimi, llama3, qwen, etc.) do NOT support embeddings.

Example usage:
    # First, pull an embedding model
    ollama pull nomic-embed-text
    
    # Then run the test
    python validation/ollama_end_to_end_test.py --model nomic-embed-text

Recommended embedding models:
    - nomic-embed-text (768 dim, fast)
    - mxbai-embed-large (1024 dim)
    - snowflake-arctic-embed (1024 dim)
""",
    )
    parser.add_argument(
        "--model", "-m",
        default="nomic-embed-text",
        help="Ollama EMBEDDING model name (default: nomic-embed-text). Must be an embedding model, not a chat model.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Ollama host URL (default: OLLAMA_HOST env or localhost:11434)",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.80,
        help="Similarity threshold for cache hits (default: 0.80)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Less verbose output",
    )
    
    args = parser.parse_args()
    
    # Check Ollama availability and embedding support
    logger.info("Checking Ollama availability...")
    available, message = check_ollama_available(args.host, args.model)
    
    if not available:
        # Check if it's an embedding model issue
        if "NOT support embeddings" in message or "chat-only" in message.lower():
            print_embedding_model_help()
            logger.error(f"\nDetails: {message}")
        else:
            logger.error(f"\n❌ {message}")
            logger.error("\nTo fix:")
            logger.error("  1. Make sure Ollama is running: ollama serve")
            logger.error("  2. Pull an embedding model: ollama pull nomic-embed-text")
            logger.error("  3. Run: python validation/ollama_end_to_end_test.py --model nomic-embed-text")
        return 1
    
    logger.info(f"✓ {message}\n")
    
    # Run test
    try:
        results = run_end_to_end_test(
            model=args.model,
            host=args.host,
            threshold=args.threshold,
            verbose=not args.quiet,
        )
        return 0 if results["passed"] else 1
    except Exception as e:
        logger.error(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
