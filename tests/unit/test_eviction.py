"""Unit tests for eviction policies.

Tests:
- LRU ordering
- Access updates order
- Eviction selection
"""

import pytest

import sys
from pathlib import Path

# Add src to path
_SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from binary_semantic_cache.core.eviction import (
    LRUEvictionPolicy,
    DiversityEvictionPolicy,
)


class TestLRUEvictionPolicy:
    """Test LRU eviction policy."""

    def test_empty_policy(self) -> None:
        """Empty policy should have zero length."""
        policy: LRUEvictionPolicy[str] = LRUEvictionPolicy()
        assert len(policy) == 0

    def test_insert_increases_length(self) -> None:
        """Insert should increase tracked items."""
        policy: LRUEvictionPolicy[str] = LRUEvictionPolicy()
        policy.record_insert("a")
        assert len(policy) == 1
        policy.record_insert("b")
        assert len(policy) == 2

    def test_evict_empty_raises(self) -> None:
        """Eviction on empty policy should raise."""
        policy: LRUEvictionPolicy[str] = LRUEvictionPolicy()
        with pytest.raises(ValueError, match="No items"):
            policy.select_for_eviction()

    def test_lru_order_insert(self) -> None:
        """First inserted should be evicted first."""
        policy: LRUEvictionPolicy[str] = LRUEvictionPolicy()
        policy.record_insert("a")
        policy.record_insert("b")
        policy.record_insert("c")

        assert policy.select_for_eviction() == "a"

    def test_access_updates_order(self) -> None:
        """Accessing an item should move it to end."""
        policy: LRUEvictionPolicy[str] = LRUEvictionPolicy()
        policy.record_insert("a")
        policy.record_insert("b")
        policy.record_insert("c")

        # Access 'a', making it most recent
        policy.record_access("a")

        # Now 'b' should be evicted first
        assert policy.select_for_eviction() == "b"

    def test_delete_removes_item(self) -> None:
        """Delete should remove item from tracking."""
        policy: LRUEvictionPolicy[str] = LRUEvictionPolicy()
        policy.record_insert("a")
        policy.record_insert("b")
        policy.record_insert("c")

        policy.record_delete("a")
        assert len(policy) == 2
        assert policy.select_for_eviction() == "b"

    def test_delete_nonexistent(self) -> None:
        """Deleting nonexistent key should be safe."""
        policy: LRUEvictionPolicy[str] = LRUEvictionPolicy()
        policy.record_insert("a")

        # Should not raise
        policy.record_delete("nonexistent")
        assert len(policy) == 1

    def test_access_nonexistent_adds(self) -> None:
        """Accessing nonexistent key should add it."""
        policy: LRUEvictionPolicy[str] = LRUEvictionPolicy()
        policy.record_access("new")
        assert len(policy) == 1

    def test_reinsert_updates_order(self) -> None:
        """Re-inserting should move to end."""
        policy: LRUEvictionPolicy[str] = LRUEvictionPolicy()
        policy.record_insert("a")
        policy.record_insert("b")
        policy.record_insert("c")

        # Re-insert 'a'
        policy.record_insert("a")

        # 'b' should be first now
        assert policy.select_for_eviction() == "b"

    def test_clear(self) -> None:
        """Clear should remove all items."""
        policy: LRUEvictionPolicy[str] = LRUEvictionPolicy()
        policy.record_insert("a")
        policy.record_insert("b")
        policy.record_insert("c")

        policy.clear()
        assert len(policy) == 0

    def test_repr(self) -> None:
        """Test string representation."""
        policy: LRUEvictionPolicy[str] = LRUEvictionPolicy()
        policy.record_insert("a")
        policy.record_insert("b")

        assert "size=2" in repr(policy)


class TestLRUWithIntegers:
    """Test LRU with integer keys."""

    def test_integer_keys(self) -> None:
        """Should work with integer keys."""
        policy: LRUEvictionPolicy[int] = LRUEvictionPolicy()
        policy.record_insert(1)
        policy.record_insert(2)
        policy.record_insert(3)

        assert policy.select_for_eviction() == 1


class TestDiversityEvictionPolicy:
    """Test diversity eviction policy placeholder."""

    def test_not_implemented(self) -> None:
        """Diversity policy should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Phase 2"):
            DiversityEvictionPolicy()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

