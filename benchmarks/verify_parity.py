#!/usr/bin/env python3
"""Quick verification of Rust/Python encoder parity."""

import numpy as np

def main():
    from binary_semantic_cache.binary_semantic_cache_rs import RustBinaryEncoder
    from binary_semantic_cache.core.encoder import BinaryEncoder
    
    rust_enc = RustBinaryEncoder(384, 256, seed=42)
    py_enc = BinaryEncoder(embedding_dim=384, code_bits=256, seed=42)
    
    emb = np.random.default_rng(123).standard_normal(384).astype(np.float32)
    
    rust_code = rust_enc.encode(emb)
    py_code = py_enc.encode(emb)
    
    print('Match:', np.array_equal(rust_code, py_code))
    print('Rust:', rust_code)
    print('Python:', py_code)

if __name__ == "__main__":
    main()


