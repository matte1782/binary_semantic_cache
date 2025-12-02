"""
Eviction Policy Module

Implements cache eviction strategies. Phase 1 uses LRU (Least Recently Used).
Diversity-aware eviction is planned for Phase 2.

See: docs/DECISION_LOG_v1.md (D6)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Generic, Hashable, TypeVar

logger = logging.getLogger(__name__)

K = TypeVar("K", bound=Hashable)


class EvictionPolicy(ABC, Generic[K]):
    """
    Abstract base class for eviction policies.

    Subclasses must implement the eviction selection logic.
    """

    @abstractmethod
    def record_access(self, key: K) -> None:
        """
        Record that an item was accessed.

        Args:
            key: Identifier of the accessed item.
        """
        pass

    @abstractmethod
    def record_insert(self, key: K) -> None:
        """
        Record that an item was inserted.

        Args:
            key: Identifier of the inserted item.
        """
        pass

    @abstractmethod
    def record_delete(self, key: K) -> None:
        """
        Record that an item was deleted.

        Args:
            key: Identifier of the deleted item.
        """
        pass

    @abstractmethod
    def select_for_eviction(self) -> K:
        """
        Select an item for eviction.

        Returns:
            Key of the item to evict.

        Raises:
            ValueError: If no items available for eviction.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return number of tracked items."""
        pass


class LRUEvictionPolicy(EvictionPolicy[K]):
    """
    Least Recently Used (LRU) eviction policy.

    Uses OrderedDict to maintain access order with O(1) operations.

    Example:
        >>> policy = LRUEvictionPolicy()
        >>> policy.record_insert("a")
        >>> policy.record_insert("b")
        >>> policy.record_access("a")  # a is now most recent
        >>> policy.select_for_eviction()  # returns "b"
    """

    __slots__ = ("_order",)

    def __init__(self) -> None:
        """Initialize empty LRU tracker."""
        self._order: OrderedDict[K, None] = OrderedDict()

    def record_access(self, key: K) -> None:
        """
        Mark key as most recently used.

        Args:
            key: Identifier of the accessed item.
        """
        if key in self._order:
            self._order.move_to_end(key)
        else:
            # Key not tracked, add it
            self._order[key] = None

    def record_insert(self, key: K) -> None:
        """
        Add key as most recently used.

        Args:
            key: Identifier of the inserted item.
        """
        self._order[key] = None
        self._order.move_to_end(key)

    def record_delete(self, key: K) -> None:
        """
        Remove key from tracking.

        Args:
            key: Identifier of the deleted item.
        """
        if key in self._order:
            del self._order[key]

    def select_for_eviction(self) -> K:
        """
        Return the least recently used key.

        Returns:
            Key of oldest item (first in OrderedDict).

        Raises:
            ValueError: If no items are being tracked.
        """
        if not self._order:
            raise ValueError("No items to evict")
        # First item is the least recently used
        return next(iter(self._order))

    def clear(self) -> None:
        """Clear all tracked items."""
        self._order.clear()

    def __len__(self) -> int:
        """Return number of tracked items."""
        return len(self._order)

    def __repr__(self) -> str:
        return f"LRUEvictionPolicy(size={len(self)})"


# Phase 2 placeholder
class DiversityEvictionPolicy(EvictionPolicy[K]):
    """
    Diversity-aware eviction policy (Phase 2).

    Evicts items that minimize loss of semantic coverage,
    keeping a diverse set of cached embeddings.

    NOT IMPLEMENTED IN PHASE 1.
    """

    def __init__(self) -> None:
        raise NotImplementedError("Planned for Phase 2 - Diversity Eviction")

    def record_access(self, key: K) -> None:
        raise NotImplementedError("Planned for Phase 2 - Diversity Eviction")

    def record_insert(self, key: K) -> None:
        raise NotImplementedError("Planned for Phase 2 - Diversity Eviction")

    def record_delete(self, key: K) -> None:
        raise NotImplementedError("Planned for Phase 2 - Diversity Eviction")

    def select_for_eviction(self) -> K:
        raise NotImplementedError("Planned for Phase 2 - Diversity Eviction")

    def __len__(self) -> int:
        raise NotImplementedError("Planned for Phase 2 - Diversity Eviction")
