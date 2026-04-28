"""
CacheManager — caches deterministic LLM responses.

Phase 4. If the same tool-call produces the same result across
multiple runs, cache it to avoid redundant API calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

logger = logging.getLogger("nil.optimization.cache_manager")


class CacheManager:
    """In-memory response cache with hash-based lookup. Phase 4 stub."""

    def __init__(self, max_size: int = 1000) -> None:
        self._cache: dict[str, Any] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _hash_key(self, tool_name: str, arguments: dict) -> str:
        raw = json.dumps({"tool": tool_name, "args": arguments}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, tool_name: str, arguments: dict) -> Optional[Any]:
        key = self._hash_key(tool_name, arguments)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, tool_name: str, arguments: dict, result: Any) -> None:
        if len(self._cache) >= self.max_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        key = self._hash_key(tool_name, arguments)
        self._cache[key] = result
