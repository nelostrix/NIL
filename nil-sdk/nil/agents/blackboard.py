from __future__ import annotations
import copy
import fnmatch
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class BlackboardEntry:
    key: str = ""
    value: Any = None
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    writer: str = ""


class Blackboard:
    def __init__(self):
        self.store: dict[str, BlackboardEntry] = {}
        self._subscriptions: dict[str, list[Callable]] = {}
        self._lock = threading.Lock()
    def write(
        self, key: str, value: Any, metadata: Optional[dict] = None, writer: str = ""
    ) -> None:
        metadata = metadata or {}
        with self._lock:
            self.store[key] = BlackboardEntry(
                key=key,
                value=value,
                metadata=metadata,
                timestamp=time.time(),
                writer=writer,
            )
        self._notify(key)

    def append(self, key: str, value: Any, writer: str = "") -> None:
        with self._lock:
            if key not in self.store:
                self.store[key] = BlackboardEntry(
                    key=key, value=[], metadata={}, timestamp=time.time(), writer=writer
                )
            entry = self.store[key]
            if not isinstance(entry.value, list):
                entry.value = [entry.value]
            entry.value.append(value)
            entry.timestamp = time.time()
            if writer:
                entry.writer = writer
        self._notify(key)

    def read(self, key: str, default: Any = None) -> Any:
        with self._lock:
            entry = self.store.get(key)
            return copy.deepcopy(entry.value) if entry else default

    def read_entry(self, key: str) -> Optional[BlackboardEntry]:
        with self._lock:
            entry = self.store.get(key)
            return copy.deepcopy(entry) if entry else None

    def search(self, query: str) -> list[BlackboardEntry]:
        results = []
        q = query.lower()
        with self._lock:
            for entry in self.store.values():
                if q in entry.key.lower():
                    results.append(copy.deepcopy(entry))
                    continue
                if any(q in str(v).lower() for v in entry.metadata.values()):
                    results.append(copy.deepcopy(entry))
                    continue
                if isinstance(entry.value, str) and q in entry.value.lower():
                    results.append(copy.deepcopy(entry))
        return results

    def namespace(self, prefix: str) -> dict[str, BlackboardEntry]:
        with self._lock:
            return {
                k: copy.deepcopy(v)
                for k, v in self.store.items()
                if k.startswith(prefix)
            }

    def keys(self, prefix: str = "") -> list[str]:
        with self._lock:
            if prefix:
                return [k for k in self.store if k.startswith(prefix)]
            return list(self.store.keys())

    def subscribe(self, pattern: str, callback: Callable) -> str:
        sub_id = uuid.uuid4().hex[:8]
        with self._lock:
            if pattern not in self._subscriptions:
                self._subscriptions[pattern] = []
            self._subscriptions[pattern].append((sub_id, callback))
        return sub_id

    def unsubscribe(self, sub_id: str) -> None:
        with self._lock:
            for pattern in list(self._subscriptions):
                self._subscriptions[pattern] = [
                    (sid, cb)
                    for sid, cb in self._subscriptions[pattern]
                    if sid != sub_id
                ]
                if not self._subscriptions[pattern]:
                    del self._subscriptions[pattern]

    def clear(self) -> None:
        with self._lock:
            self.store.clear()
            self._subscriptions.clear()

    def to_dict(self) -> dict:
        with self._lock:
            return {
                k: {
                    "key": v.key,
                    "value": v.value,
                    "metadata": v.metadata,
                    "timestamp": v.timestamp,
                    "writer": v.writer,
                }
                for k, v in self.store.items()
            }

    def _notify(self, key: str) -> None:
        with self._lock:
            subscriptions = {p: list(cbs) for p, cbs in self._subscriptions.items()}
            entry = self.store.get(key)
        if not entry:
            return
        for pattern, callbacks in subscriptions.items():
            if fnmatch.fnmatch(key, pattern):
                for _, cb in callbacks:
                    cb(copy.deepcopy(entry))
