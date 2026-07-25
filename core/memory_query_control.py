import asyncio
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")


class MemoryQueryCooldownError(RuntimeError):
    def __init__(self, retry_after: float):
        super().__init__("memory query is cooling down")
        self.retry_after = max(0.0, float(retry_after))


class BoundedTTLCache(Generic[T]):
    """Small deterministic LRU+TTL cache with no background cleanup task."""

    def __init__(
        self,
        *,
        max_entries: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.max_entries = max(1, int(max_entries))
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self._clock = clock
        self._items: OrderedDict[object, tuple[float, T]] = OrderedDict()

    def get(self, key: object) -> T | None:
        item = self._items.pop(key, None)
        if item is None:
            return None
        created_at, value = item
        if self._clock() - created_at >= self.ttl_seconds:
            return None
        self._items[key] = item
        return value

    def put(self, key: object, value: T) -> None:
        self._items.pop(key, None)
        self._items[key] = (self._clock(), value)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class MemoryQueryControlStats:
    started: int = 0
    singleflight_reused: int = 0
    cooldown_rejected: int = 0


class MemoryQueryCoordinator(Generic[T]):
    """Per-process cooldown and single-flight control for expensive queries."""

    def __init__(
        self,
        *,
        user_cooldown_seconds: float,
        group_cooldown_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.user_cooldown_seconds = max(0.0, float(user_cooldown_seconds))
        self.group_cooldown_seconds = max(0.0, float(group_cooldown_seconds))
        self._clock = clock
        self._lock = asyncio.Lock()
        self._inflight: dict[object, asyncio.Task[T]] = {}
        self._last_user: dict[tuple[str, str], float] = {}
        self._last_group: dict[str, float] = {}
        self.stats = MemoryQueryControlStats()

    async def run(
        self,
        *,
        key: object,
        group_id: str,
        user_id: str,
        factory: Callable[[], Awaitable[T]],
    ) -> T:
        async with self._lock:
            existing = self._inflight.get(key)
            if existing is not None:
                self.stats.singleflight_reused += 1
                task = existing
            else:
                now = self._clock()
                user_key = (str(group_id), str(user_id))
                user_remaining = self.user_cooldown_seconds - (
                    now - self._last_user.get(user_key, float("-inf"))
                )
                group_remaining = self.group_cooldown_seconds - (
                    now - self._last_group.get(str(group_id), float("-inf"))
                )
                retry_after = max(user_remaining, group_remaining)
                if retry_after > 0:
                    self.stats.cooldown_rejected += 1
                    raise MemoryQueryCooldownError(retry_after)
                self._last_user[user_key] = now
                self._last_group[str(group_id)] = now
                task = asyncio.create_task(factory())
                self._inflight[key] = task
                self.stats.started += 1

        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                async with self._lock:
                    if self._inflight.get(key) is task:
                        self._inflight.pop(key, None)
