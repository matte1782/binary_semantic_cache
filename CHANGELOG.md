# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-02

### Summary
First official production release. Features a hybrid Python/Rust architecture for sub-millisecond semantic caching.

### Known Issues
- **Full Cache Load Time:** Loading a 1M-entry cache takes ~300-350ms total. While the binary index is search-ready in <10ms, Python pickle deserialization of response objects adds ~300ms overhead. This is a known limitation and will be addressed in v1.1+.

### Added
- **Rust Storage Backend:** Core storage moved to Rust (`RustCacheStorage`), reducing memory usage to 44 bytes per entry (codes + metadata).
- **Persistence Format v3:** New binary format with SHA-256 integrity checks. Loading 1M entries now takes < 10ms (index ready).
- **OpenAI Integration:** Native `OpenAIEmbeddingBackend` with automatic batching, token bucket rate limiting, and cost tracking.
- **List-Based Response Storage:** Python-side response storage optimized from dict to list, reducing overhead by ~90%.
- **Migration Tool:** CLI tool to migrate legacy v2 caches to v3 format (`binary-semantic-cache migrate`).

### Changed
- **Performance:** Lookup latency for 100k entries reduced from 1.14ms to 0.16ms (7x speedup).
- **Memory:** Total memory footprint per entry reduced from ~119 bytes to ~52 bytes.
- **Dependency:** Added `maturin` build system for Rust extensions.

### Fixed
- **Parity:** Resolved bit-exactness issue between Rust and Python encoders (verified via OAI-08 regression test).
- **Windows Compatibility:** Fixed Unicode encoding issues in benchmark reporting.

## [0.2.1] - 2025-12-02 (Beta)
- *Pre-release version. All changes merged into 1.0.0.*

## [0.1.0] - 2025-11-25

### Added
- Initial release.
- Python-only implementation (NumPy + Numba).
- Binary quantization (256-bit Hamming distance).
- Ollama support (`OllamaEmbedder`).
- Basic `save`/`load` persistence (pickle).

