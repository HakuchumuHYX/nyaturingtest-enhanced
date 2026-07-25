import asyncio
from collections.abc import Awaitable, Callable


class PersistenceCoordinator:
    """Coalesce repeated save requests for one Session.

    The callback always reads current Session state at execution time. Versions
    therefore track dirty state rather than snapshots, and at most one callback
    is active for a Session.
    """

    def __init__(
        self,
        save_callback: Callable[[bool], Awaitable[bool]],
        *,
        task_factory: Callable[[Awaitable[bool]], asyncio.Task] = asyncio.create_task,
        debounce_seconds: float = 0.02,
    ):
        self._save_callback = save_callback
        self._task_factory = task_factory
        self._debounce_seconds = max(0.0, float(debounce_seconds))
        self._state_version = 0
        self._persisted_version = 0
        self._force_through_version = 0
        self._batch_depth = 0
        self._task: asyncio.Task | None = None
        self.save_requested_count = 0
        self.save_executed_count = 0

    @property
    def has_pending_changes(self) -> bool:
        return self._persisted_version < self._state_version

    def request(self, *, force_index: bool = False) -> int:
        self._state_version += 1
        self.save_requested_count += 1
        if force_index:
            self._force_through_version = self._state_version
        if self._batch_depth == 0:
            self._ensure_task()
        return self._state_version

    def begin_batch(self) -> None:
        self._batch_depth += 1

    async def end_batch(self, *, flush: bool = False) -> bool:
        if self._batch_depth <= 0:
            raise RuntimeError("persistence batch is not active")
        self._batch_depth -= 1
        if self._batch_depth == 0 and self.has_pending_changes:
            self._ensure_task()
        if flush and self._batch_depth == 0:
            return await self.flush()
        return True

    def mark_current_persisted(self) -> None:
        self._persisted_version = self._state_version

    def _ensure_task(self) -> asyncio.Task | None:
        if self._batch_depth > 0 or not self.has_pending_changes:
            return self._task
        if self._task is None or self._task.done():
            self._task = self._task_factory(self._run())
        return self._task

    async def _run(self) -> bool:
        failed = False
        try:
            if self._debounce_seconds:
                await asyncio.sleep(self._debounce_seconds)
            while self.has_pending_changes:
                target_version = self._state_version
                force_index = self._force_through_version > self._persisted_version
                self.save_executed_count += 1
                if not await self._save_callback(force_index):
                    failed = True
                    return False
                self._persisted_version = target_version
            return True
        finally:
            self._task = None
            # Do not hot-loop on a database failure. A later mutation or an
            # explicit flush will schedule the retry.
            if not failed and self._batch_depth == 0 and self.has_pending_changes:
                self._ensure_task()

    async def flush(self) -> bool:
        if self._batch_depth > 0:
            return False
        task = self._ensure_task()
        if task is None:
            return True
        result = await asyncio.shield(task)
        return bool(result) and not self.has_pending_changes
