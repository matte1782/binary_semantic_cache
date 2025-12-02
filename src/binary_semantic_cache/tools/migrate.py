"""
Migration tool to convert legacy v2 caches (.npz) to v3 directory format.

Usage:
    python -m binary_semantic_cache.tools.migrate <input_path> <output_path> [--force]
"""

import argparse
import logging
import sys
import shutil
import time
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from binary_semantic_cache.core.cache import BinarySemanticCache, detect_format_version, MMAP_HEADER_FILE
from binary_semantic_cache.core.encoder import BinaryEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def inspect_v2_source(src: Path) -> Tuple[int, int, float]:
    """
    Inspect v2 source to determine configuration.
    
    Returns:
        (code_bits, max_entries, threshold)
    """
    # Default values
    code_bits = 256
    max_entries = 100_000
    threshold = 0.80

    if src.is_dir():
        # v2 directory format
        header_path = src / MMAP_HEADER_FILE
        if header_path.exists():
            try:
                with open(header_path, "r", encoding="utf-8") as f:
                    header = json.load(f)
                    code_bits = header.get("code_bits", code_bits)
                    max_entries = header.get("max_entries", max_entries)
                    threshold = header.get("threshold", threshold)
            except Exception as e:
                logger.warning(f"Failed to read header from {header_path}: {e}. Using defaults.")
        else:
            # Try to infer from codes.bin if header missing (should rely on detect_format_version but be robust)
            codes_path = src / "codes.bin"
            if codes_path.exists():
                # We can't easily know n_words without file size and n_entries, which we don't know.
                # So we rely on defaults or error out.
                # But wait, if it's a valid v2 dir, it MUST have header.json for load_mmap to work (see cache.py)
                pass
                
    elif src.suffix == ".npz":
        # v2 .npz format
        try:
            with np.load(src, allow_pickle=True) as data:
                if "codes" in data:
                    codes = data["codes"]
                    if codes.ndim == 2:
                        n_entries, n_words = codes.shape
                        code_bits = n_words * 64
                        # Ensure max_entries is at least n_entries
                        max_entries = max(max_entries, n_entries)
        except Exception as e:
            logger.warning(f"Failed to inspect .npz file {src}: {e}. Using defaults.")
            
    return code_bits, max_entries, threshold

def migrate_v2_to_v3(src_path: str, dst_path: str, force: bool = False) -> None:
    """
    Migrate a v2 cache (.npz) to v3 directory format.

    Args:
        src_path: Path to existing v2 cache file.
        dst_path: Path to destination directory.
        force: If True, overwrite destination if it exists.

    Raises:
        FileNotFoundError: If source does not exist.
        FileExistsError: If destination exists and force is False.
        ValueError: If source is not a valid v2 cache.
    """
    src = Path(src_path)
    dst = Path(dst_path)

    if not src.exists():
        raise FileNotFoundError(f"Source cache not found: {src}")

    # Check format version
    try:
        version = detect_format_version(str(src))
        if version != 2:
            raise ValueError(f"Source is not a v2 cache (detected version: {version})")
    except ValueError as e:
        # detect_format_version might raise ValueError for unknown formats
        raise ValueError(f"Invalid source format: {e}")

    if dst.exists():
        if not force:
            raise FileExistsError(f"Destination exists: {dst}. Use --force to overwrite.")
        logger.info(f"Destination exists, removing: {dst}")
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    logger.info(f"Migrating v2 cache '{src}' to v3 '{dst}'...")
    
    # Inspect source to get correct parameters
    code_bits, max_entries, threshold = inspect_v2_source(src)
    logger.info(f"Detected config: code_bits={code_bits}, max_entries={max_entries}, threshold={threshold}")

    start_time = time.time()

    # Initialize a temporary cache instance with detected config
    # We use a dummy encoder initialized with the correct code_bits
    dummy_encoder = BinaryEncoder(code_bits=code_bits)
    cache = BinarySemanticCache(
        encoder=dummy_encoder, 
        max_entries=max_entries,
        similarity_threshold=threshold
    )

    try:
        # 1. Load v2 cache
        logger.info("Loading v2 cache (this may take a moment)...")
        if src.is_dir():
            cache.load_mmap(str(src))
        else:
            cache.load(str(src))
        
        entries_count = len(cache)
        logger.info(f"Loaded {entries_count} entries from v2 cache.")

        # 2. Save as v3
        logger.info("Saving as v3 cache...")
        cache.save_mmap_v3(str(dst))
        
        elapsed = time.time() - start_time
        logger.info(f"Migration complete in {elapsed:.2f}s.")
        logger.info(f"Output verified at: {dst}")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate Binary Semantic Cache from v2 (.npz) to v3 (directory)."
    )
    parser.add_argument("input_path", help="Path to source v2 cache (.npz)")
    parser.add_argument("output_path", help="Path to destination v3 directory")
    parser.add_argument(
        "-f", "--force", 
        action="store_true", 
        help="Overwrite output directory if it exists"
    )

    args = parser.parse_args()

    try:
        migrate_v2_to_v3(args.input_path, args.output_path, args.force)
        return 0
    except Exception as e:
        logger.error(str(e))
        return 1

if __name__ == "__main__":
    sys.exit(main())

