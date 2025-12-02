#!/usr/bin/env python3
"""
Manual End-to-End Test for OpenAI Integration.

This script verifies that the OpenAI integration works with a real API key.
It is designed to be run manually by a human operator.

Security:
- Checks for OPENAI_API_KEY in environment variables.
- NEVER logs or prints the API key.
- Skips gracefully if no key is found.
"""

import os
import sys
import logging
import numpy as np
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    print("=" * 60)
    print("MANUAL OPENAI INTEGRATION TEST")
    print("=" * 60)

    # 1. Check Environment Variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n[SKIP] OPENAI_API_KEY not found in environment variables.")
        print("       To run this test, set OPENAI_API_KEY and try again.")
        sys.exit(0)

    if not api_key.startswith("sk-"):
        print("\n[WARN] OPENAI_API_KEY does not start with 'sk-'. It might be invalid.")
    
    print("\n[INFO] OPENAI_API_KEY found. Proceeding with test...")

    try:
        # Import here to avoid hard dependency if not installed
        from binary_semantic_cache.embeddings.openai_backend import OpenAIEmbeddingBackend
        
        # 2. Initialize Backend
        print("[INFO] Initializing OpenAIEmbeddingBackend...")
        backend = OpenAIEmbeddingBackend()
        
        # 3. Run Test Query
        test_text = "This is a manual test of the binary semantic cache system."
        print(f"[INFO] Embedding text: '{test_text}'")
        
        embedding = backend.embed_text(test_text)
        
        # 4. Verify Output
        print(f"[INFO] Embedding received. Shape: {embedding.shape}")
        print(f"[INFO] Norm: {np.linalg.norm(embedding):.4f}")
        
        if embedding.shape[0] == 1536:
            print("\n[SUCCESS] Test passed! Embedding dimension is correct (1536).")
        else:
            print(f"\n[FAIL] Unexpected embedding dimension: {embedding.shape[0]}")
            sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()






