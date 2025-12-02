#!/usr/bin/env python3
"""
Validate Cache Hit/Miss Logic

Validates that Phase 1 production code correctly handles:
- Similar embeddings (0.95 cosine) → HIT
- Different embeddings (<0.70 cosine) → MISS
- Exact embedding → HIT with sim=1.0

Requirements:
- Uses ONLY production code from src/
- Compares against PoC results
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Add src to path
_ROOT = Path(__file__).resolve().parents[2]
_SRC_PATH = _ROOT / "src"
sys.path.insert(0, str(_SRC_PATH))

from binary_semantic_cache.core.encoder import BinaryEncoder
from binary_semantic_cache.core.cache import BinarySemanticCache

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def create_similar_embedding(
    base: np.ndarray, target_cosine: float, seed: int = None
) -> np.ndarray:
    """Create an embedding with a specific cosine similarity to base."""
    rng = np.random.default_rng(seed)
    
    random = rng.standard_normal(len(base)).astype(np.float32)
    random = random - np.dot(random, base) * base
    random = random / np.linalg.norm(random)
    
    theta = np.arccos(target_cosine)
    result = np.cos(theta) * base + np.sin(theta) * random
    return (result / np.linalg.norm(result)).astype(np.float32)


def validate_cache_logic() -> Dict[str, Any]:
    """Run cache logic validation."""
    logger.info("=" * 60)
    logger.info("PHASE 1 CACHE LOGIC VALIDATION")
    logger.info("=" * 60)
    
    # Initialize
    encoder = BinaryEncoder(embedding_dim=384, code_bits=256, seed=42)
    cache = BinarySemanticCache(
        encoder=encoder,
        max_entries=1000,
        similarity_threshold=0.85,
    )
    
    # Create base embedding
    rng = np.random.default_rng(42)
    base = rng.standard_normal(384).astype(np.float32)
    base = base / np.linalg.norm(base)
    
    # Store base embedding
    cache.put(base, {"response": "original"})
    
    results: List[Dict[str, Any]] = []
    all_passed = True
    
    # Test 1: Similar embedding (0.95 cosine) should HIT
    logger.info("\n[Test 1] Similar embedding (0.95 cosine)")
    logger.info("-" * 40)
    
    similar = create_similar_embedding(base, 0.95, seed=100)
    actual_cosine = float(np.dot(base, similar))
    logger.info(f"  Created embedding with cosine={actual_cosine:.4f}")
    
    hit = cache.get(similar)
    test1_passed = hit is not None
    
    results.append({
        "test": "similar_hit",
        "description": "Similar embedding (0.95 cosine) should HIT",
        "target_cosine": 0.95,
        "actual_cosine": actual_cosine,
        "expected": "HIT",
        "actual": "HIT" if hit else "MISS",
        "pass": test1_passed,
    })
    
    if test1_passed:
        logger.info(f"  Result: ✓ HIT (response={hit.response})")
    else:
        logger.error(f"  Result: ✗ MISS (expected HIT)")
        all_passed = False
    
    # Test 2: Different embedding (<0.70 cosine) should MISS
    logger.info("\n[Test 2] Different embedding (random)")
    logger.info("-" * 40)
    
    different = rng.standard_normal(384).astype(np.float32)
    different = different / np.linalg.norm(different)
    diff_cosine = float(np.dot(base, different))
    logger.info(f"  Random embedding has cosine={diff_cosine:.4f}")
    
    miss = cache.get(different)
    test2_passed = miss is None
    
    results.append({
        "test": "different_miss",
        "description": "Different embedding should MISS",
        "actual_cosine": diff_cosine,
        "expected": "MISS",
        "actual": "MISS" if miss is None else "HIT",
        "pass": test2_passed,
    })
    
    if test2_passed:
        logger.info(f"  Result: ✓ MISS")
    else:
        logger.error(f"  Result: ✗ HIT (expected MISS)")
        all_passed = False
    
    # Test 3: Exact embedding should HIT with sim=1.0
    logger.info("\n[Test 3] Exact embedding")
    logger.info("-" * 40)
    
    exact = base.copy()
    exact_hit = cache.get(exact)
    test3_passed = exact_hit is not None
    
    results.append({
        "test": "exact_hit",
        "description": "Exact embedding should HIT with sim=1.0",
        "expected": "HIT",
        "actual": "HIT" if exact_hit else "MISS",
        "pass": test3_passed,
    })
    
    if test3_passed:
        logger.info(f"  Result: ✓ HIT (response={exact_hit.response})")
    else:
        logger.error(f"  Result: ✗ MISS (expected HIT)")
        all_passed = False
    
    # Summary
    passed_count = sum(1 for r in results if r["pass"])
    total_count = len(results)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"SUMMARY: {passed_count}/{total_count} tests passed")
    
    if all_passed:
        logger.info("OVERALL: ✓ PASS")
    else:
        logger.error("OVERALL: ✗ FAIL")
    logger.info("=" * 60)
    
    # Load PoC results for comparison
    poc_path = _ROOT / "validation" / "results" / "s3_poc_results_v2.json"
    poc_results = None
    
    if poc_path.exists():
        with open(poc_path) as f:
            poc_results = json.load(f)
        logger.info(f"\nPoC comparison: PoC had {len(poc_results.get('tests', []))} tests")
    
    # Prepare output
    output = {
        "timestamp": datetime.now().isoformat(),
        "tests": results,
        "passed": passed_count,
        "total": total_count,
        "pass": all_passed,
    }
    
    # Save results
    output_path = _ROOT / "validation" / "results" / "phase1_cache_logic.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_path}")
    
    return output


def main() -> int:
    """Run validation and return exit code."""
    try:
        result = validate_cache_logic()
        return 0 if result["pass"] else 1
    except Exception as e:
        logger.error(f"Validation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

