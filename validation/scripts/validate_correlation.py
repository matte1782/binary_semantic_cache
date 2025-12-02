#!/usr/bin/env python3
"""
Validate Similarity Correlation

Validates that Phase 1 production code achieves the same similarity
correlation as the PoC (r ≥ 0.93).

Requirements:
- Uses ONLY production code from src/
- Compares against PoC results
- Fails if deviation > 0.01
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import stats

# Add src to path
_ROOT = Path(__file__).resolve().parents[2]
_SRC_PATH = _ROOT / "src"
sys.path.insert(0, str(_SRC_PATH))

from binary_semantic_cache.core.encoder import BinaryEncoder
from binary_semantic_cache.core.similarity import hamming_similarity

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Validation thresholds
CORRELATION_TARGET = 0.93
DEVIATION_THRESHOLD = 0.01


def create_similar_embedding(
    base: np.ndarray, target_cosine: float, seed: int = None
) -> np.ndarray:
    """
    Create an embedding with a specific cosine similarity to base.
    Uses Gram-Schmidt orthogonalization.
    """
    rng = np.random.default_rng(seed)
    
    # Create random vector
    random = rng.standard_normal(len(base)).astype(np.float32)
    
    # Gram-Schmidt: make orthogonal to base
    random = random - np.dot(random, base) * base
    random = random / np.linalg.norm(random)
    
    # Combine with target angle
    theta = np.arccos(target_cosine)
    result = np.cos(theta) * base + np.sin(theta) * random
    return (result / np.linalg.norm(result)).astype(np.float32)


def validate_correlation() -> Dict[str, Any]:
    """Run correlation validation."""
    logger.info("=" * 60)
    logger.info("PHASE 1 CORRELATION VALIDATION")
    logger.info("=" * 60)
    
    # Initialize encoder with FROZEN parameters
    encoder = BinaryEncoder(embedding_dim=384, code_bits=256, seed=42)
    logger.info(f"Encoder: {encoder}")
    
    # Target cosine similarities to test
    targets = [0.99, 0.95, 0.90, 0.85, 0.80, 0.70, 0.50, 0.30]
    
    # Create base embedding
    rng = np.random.default_rng(42)
    base = rng.standard_normal(384).astype(np.float32)
    base = base / np.linalg.norm(base)
    base_code = encoder.encode(base)
    
    results: List[Dict[str, float]] = []
    cosines = []
    hammings = []
    
    logger.info("\nTesting similarity preservation:")
    logger.info("-" * 60)
    logger.info(f"{'Target':>8} | {'Actual':>8} | {'Hamming':>8} | {'Error':>8}")
    logger.info("-" * 60)
    
    for i, target in enumerate(targets):
        # Create embedding with target cosine similarity
        similar = create_similar_embedding(base, target, seed=100 + i)
        
        # Verify actual cosine similarity
        actual_cosine = float(np.dot(base, similar))
        
        # Encode and compute Hamming similarity
        similar_code = encoder.encode(similar)
        ham_sim = hamming_similarity(
            similar_code, base_code.reshape(1, -1), code_bits=256
        )[0]
        
        error = abs(actual_cosine - float(ham_sim))
        
        results.append({
            "target": target,
            "actual": actual_cosine,
            "hamming": float(ham_sim),
            "error": error,
        })
        
        cosines.append(actual_cosine)
        hammings.append(float(ham_sim))
        
        logger.info(f"{target:>8.2f} | {actual_cosine:>8.4f} | {ham_sim:>8.4f} | {error:>8.4f}")
    
    # Compute Pearson correlation
    correlation, p_value = stats.pearsonr(cosines, hammings)
    
    logger.info("-" * 60)
    logger.info(f"\nCorrelation (Pearson r): {correlation:.4f}")
    logger.info(f"P-value: {p_value:.2e}")
    logger.info(f"Target: ≥ {CORRELATION_TARGET}")
    
    # Check pass/fail
    passed = correlation >= CORRELATION_TARGET
    
    if passed:
        logger.info(f"Status: ✓ PASS (r={correlation:.4f} ≥ {CORRELATION_TARGET})")
    else:
        logger.error(f"Status: ✗ FAIL (r={correlation:.4f} < {CORRELATION_TARGET})")
    
    # Load PoC results for comparison
    poc_path = _ROOT / "validation" / "results" / "similarity_correlation_test.json"
    poc_correlation = None
    deviation = None
    
    if poc_path.exists():
        with open(poc_path) as f:
            poc_data = json.load(f)
        poc_correlation = poc_data["correlation"]
        deviation = abs(correlation - poc_correlation)
        
        logger.info(f"\nComparison with PoC:")
        logger.info(f"  PoC correlation: {poc_correlation:.4f}")
        logger.info(f"  Phase 1 correlation: {correlation:.4f}")
        logger.info(f"  Deviation: {deviation:.4f}")
        
        if deviation > DEVIATION_THRESHOLD:
            logger.warning(f"  ⚠ Deviation exceeds threshold ({DEVIATION_THRESHOLD})")
            passed = False
    
    # Prepare output
    output = {
        "timestamp": datetime.now().isoformat(),
        "correlation": correlation,
        "p_value": p_value,
        "target": CORRELATION_TARGET,
        "results": results,
        "avg_error": float(np.mean([r["error"] for r in results])),
        "max_error": float(np.max([r["error"] for r in results])),
        "poc_correlation": poc_correlation,
        "deviation": deviation,
        "pass": passed,
    }
    
    # Save results
    output_path = _ROOT / "validation" / "results" / "phase1_correlation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_path}")
    
    return output


def main() -> int:
    """Run validation and return exit code."""
    try:
        result = validate_correlation()
        return 0 if result["pass"] else 1
    except Exception as e:
        logger.error(f"Validation failed with exception: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

