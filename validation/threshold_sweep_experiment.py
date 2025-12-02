#!/usr/bin/env python3
"""
Threshold Sweep Experiment for Phase 1 Scientist Review

PURPOSE:
Diagnose the 60% query accuracy issue in the Ollama E2E test by:
1. Computing actual cosine similarity between queries and ALL corpus items
2. Showing best corpus match for each query (reveals if issue is embedding model or threshold)
3. Sweeping threshold from 0.6 to 0.9 to find optimal setting
4. Computing Hamming similarity for the same pairs to measure quantization error

USAGE:
    python validation/threshold_sweep_experiment.py --model nomic-embed-text
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from binary_semantic_cache.embeddings import OllamaEmbedder
from binary_semantic_cache.core.encoder import BinaryEncoder
from binary_semantic_cache.core.similarity import hamming_similarity

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Same corpus and queries as ollama_end_to_end_test.py
TEST_CORPUS = [
    "How do I reverse a string in Python?",
    "What's the best way to reverse a string using Python?",
    "Python string reversal method",
    "What's the weather like today?",
    "How is the weather outside?",
    "Tell me the current weather conditions",
    "How do I make pasta from scratch?",
    "What's the recipe for homemade pasta?",
    "Steps to prepare fresh pasta at home",
    "How do I create an index in PostgreSQL?",
    "PostgreSQL index creation syntax",
    "The quick brown fox jumps over the lazy dog",
    "Lorem ipsum dolor sit amet",
    "Hello",
    "This is a longer sentence with more context about artificial intelligence and machine learning applications in modern software development.",
]

TEST_QUERIES = [
    ("How can I reverse a string in Python?", "string reverse", True),
    ("What's the weather forecast?", "weather", True),
    ("Recipe for making fresh pasta", "pasta", True),
    ("What is quantum computing?", "quantum", False),
    ("How tall is Mount Everest?", "mountain", False),
]


def cosine_similarity_matrix(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two sets of embeddings."""
    # Normalize
    a_norm = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    b_norm = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)


def run_threshold_sweep(model: str = "nomic-embed-text", host: str = None) -> Dict[str, Any]:
    """Run the threshold sweep experiment."""
    
    logger.info("=" * 70)
    logger.info("THRESHOLD SWEEP EXPERIMENT - Scientist Phase 1 Review")
    logger.info("=" * 70)
    
    results = {
        "model": model,
        "corpus_size": len(TEST_CORPUS),
        "query_count": len(TEST_QUERIES),
    }
    
    # Initialize embedder
    embedder = OllamaEmbedder(model_name=model, host=host if host else None)
    logger.info(f"Model: {model}")
    
    # Get embedding dimension
    test_emb = embedder.embed_text("test")
    embedding_dim = len(test_emb)
    logger.info(f"Embedding dimension: {embedding_dim}")
    
    # Initialize encoder
    encoder = BinaryEncoder(
        embedding_dim=embedding_dim,
        code_bits=256,
        seed=42,
    )
    
    # Embed corpus
    logger.info(f"\n1. Embedding corpus ({len(TEST_CORPUS)} items)...")
    corpus_embeddings = []
    for text in TEST_CORPUS:
        emb = embedder.embed_text(text)
        corpus_embeddings.append(emb)
    corpus_embeddings = np.stack(corpus_embeddings)
    
    # Embed queries
    logger.info(f"2. Embedding queries ({len(TEST_QUERIES)} items)...")
    query_embeddings = []
    for text, _, _ in TEST_QUERIES:
        emb = embedder.embed_text(text)
        query_embeddings.append(emb)
    query_embeddings = np.stack(query_embeddings)
    
    # Encode to binary
    corpus_codes = encoder.encode(corpus_embeddings)
    query_codes = encoder.encode(query_embeddings)
    
    # Compute similarity matrices
    logger.info("3. Computing similarity matrices...")
    
    # Cosine similarity: (n_queries, n_corpus)
    cosine_sims = cosine_similarity_matrix(query_embeddings, corpus_embeddings)
    
    # Hamming similarity for each query vs all corpus
    hamming_sims = np.zeros_like(cosine_sims)
    for i in range(len(TEST_QUERIES)):
        hamming_sims[i, :] = hamming_similarity(query_codes[i], corpus_codes, 256)
    
    # =========================================================================
    # PART 1: Show actual similarities for each query
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PART 1: Query-to-Corpus Similarity Analysis")
    logger.info("=" * 70)
    
    query_analysis = []
    
    for i, (query_text, category, expected_hit) in enumerate(TEST_QUERIES):
        best_corpus_idx = np.argmax(cosine_sims[i])
        best_cosine = cosine_sims[i, best_corpus_idx]
        best_hamming = hamming_sims[i, best_corpus_idx]
        best_corpus_text = TEST_CORPUS[best_corpus_idx]
        
        logger.info(f"\nQuery {i+1}: \"{query_text[:50]}...\"")
        logger.info(f"  Category: {category}, Expected: {'HIT' if expected_hit else 'MISS'}")
        logger.info(f"  Best match: \"{best_corpus_text[:40]}...\"")
        logger.info(f"  Cosine similarity:  {best_cosine:.4f}")
        logger.info(f"  Hamming similarity: {best_hamming:.4f}")
        logger.info(f"  Quantization error: {abs(best_cosine - best_hamming):.4f}")
        
        # Would this hit at different thresholds?
        thresholds_that_hit = []
        for thresh in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
            if best_hamming >= thresh:
                thresholds_that_hit.append(thresh)
        
        hit_range = f"≤{max(thresholds_that_hit):.2f}" if thresholds_that_hit else "NEVER"
        logger.info(f"  Would HIT at thresholds: {hit_range}")
        
        query_analysis.append({
            "query": query_text,
            "category": category,
            "expected_hit": expected_hit,
            "best_corpus_match": best_corpus_text,
            "best_corpus_idx": int(best_corpus_idx),
            "cosine_similarity": float(best_cosine),
            "hamming_similarity": float(best_hamming),
            "quantization_error": float(abs(best_cosine - best_hamming)),
        })
    
    results["query_analysis"] = query_analysis
    
    # =========================================================================
    # PART 2: Threshold Sweep
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PART 2: Threshold Sweep (Binary Similarity)")
    logger.info("=" * 70)
    
    thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    sweep_results = []
    
    logger.info("\n  Threshold | Acc |  String | Weather |  Pasta | Quantum |  Mount |")
    logger.info("  " + "-" * 65)
    
    for thresh in thresholds:
        predictions = []
        for i, (_, category, expected_hit) in enumerate(TEST_QUERIES):
            best_hamming = np.max(hamming_sims[i])
            actual_hit = best_hamming >= thresh
            is_correct = actual_hit == expected_hit
            predictions.append((is_correct, actual_hit))
        
        accuracy = sum(p[0] for p in predictions) / len(predictions)
        
        # Format results
        pred_strs = ["✓" if p[0] else "✗" for p in predictions]
        logger.info(
            f"    {thresh:.2f}   | {accuracy*100:3.0f}% |"
            f"    {pred_strs[0]}    |    {pred_strs[1]}    |"
            f"   {pred_strs[2]}    |    {pred_strs[3]}    |   {pred_strs[4]}    |"
        )
        
        sweep_results.append({
            "threshold": thresh,
            "accuracy": accuracy,
            "predictions": [
                {"category": TEST_QUERIES[j][1], "correct": predictions[j][0], "hit": predictions[j][1]}
                for j in range(len(TEST_QUERIES))
            ],
        })
    
    results["threshold_sweep"] = sweep_results
    
    # Find best threshold
    best_sweep = max(sweep_results, key=lambda x: x["accuracy"])
    logger.info(f"\n  Best threshold: {best_sweep['threshold']:.2f} ({best_sweep['accuracy']*100:.0f}% accuracy)")
    results["best_threshold"] = best_sweep["threshold"]
    results["best_accuracy"] = best_sweep["accuracy"]
    
    # =========================================================================
    # PART 3: Quantization Error Analysis
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PART 3: Quantization Error Analysis")
    logger.info("=" * 70)
    
    all_cosines = cosine_sims.flatten()
    all_hammings = hamming_sims.flatten()
    
    correlation = np.corrcoef(all_cosines, all_hammings)[0, 1]
    mean_error = np.mean(np.abs(all_cosines - all_hammings))
    max_error = np.max(np.abs(all_cosines - all_hammings))
    
    # Signed error (hamming typically underestimates)
    mean_signed_error = np.mean(all_hammings - all_cosines)
    
    logger.info(f"\n  Cosine-Hamming correlation: {correlation:.4f}")
    logger.info(f"  Mean absolute error:        {mean_error:.4f}")
    logger.info(f"  Max absolute error:         {max_error:.4f}")
    logger.info(f"  Mean signed error:          {mean_signed_error:+.4f} (negative = underestimate)")
    
    results["quantization_analysis"] = {
        "correlation": float(correlation),
        "mean_abs_error": float(mean_error),
        "max_abs_error": float(max_error),
        "mean_signed_error": float(mean_signed_error),
    }
    
    # =========================================================================
    # PART 4: Diagnosis
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("DIAGNOSIS")
    logger.info("=" * 70)
    
    # Analyze the missed queries (weather, pasta)
    missed_queries = [qa for qa in query_analysis if qa["expected_hit"] and qa["hamming_similarity"] < 0.8]
    
    if missed_queries:
        logger.info("\n  MISSED EXPECTED HITS (threshold=0.8):")
        for qa in missed_queries:
            logger.info(f"    - {qa['category']}: cosine={qa['cosine_similarity']:.3f}, hamming={qa['hamming_similarity']:.3f}")
            if qa["cosine_similarity"] < 0.8:
                logger.info(f"      → Issue: Embedding model's cosine similarity is BELOW 0.8!")
                logger.info(f"      → This is an EMBEDDING MODEL limitation, not a cache issue.")
            else:
                logger.info(f"      → Issue: Quantization reduced {qa['cosine_similarity']:.3f} → {qa['hamming_similarity']:.3f}")
                logger.info(f"      → Mitigation: Lower threshold or increase code_bits")
    else:
        logger.info("\n  All expected HITs would pass at threshold=0.8")
    
    # Final recommendation
    logger.info("\n" + "-" * 70)
    logger.info("RECOMMENDATION")
    logger.info("-" * 70)
    
    # Calculate how many expected HITs fail at 0.8
    expected_hits = [qa for qa in query_analysis if qa["expected_hit"]]
    actual_hits_at_08 = sum(1 for qa in expected_hits if qa["hamming_similarity"] >= 0.8)
    
    if actual_hits_at_08 == len(expected_hits):
        logger.info("  VERDICT: Threshold 0.8 is appropriate for this embedding model.")
    elif best_sweep["accuracy"] > 0.6:
        logger.info(f"  VERDICT: Consider lowering threshold to {best_sweep['threshold']:.2f}")
        logger.info(f"           This would improve accuracy from 60% to {best_sweep['accuracy']*100:.0f}%")
    else:
        logger.info("  VERDICT: The test queries may not be semantically similar enough")
        logger.info("           according to the embedding model. This is a test design issue,")
        logger.info("           not a cache implementation issue.")
    
    # Save results
    results_path = project_root / "validation" / "results" / "threshold_sweep.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Threshold Sweep Experiment")
    parser.add_argument("--model", default="nomic-embed-text", help="Ollama model to use")
    parser.add_argument("--host", default=None, help="Ollama host URL")
    args = parser.parse_args()
    
    run_threshold_sweep(model=args.model, host=args.host)


if __name__ == "__main__":
    main()

