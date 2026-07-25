import asyncio
from dataclasses import dataclass


@dataclass(frozen=True)
class InboxBatch:
    messages: list
    bot: object
    event: object


class DebouncedInbox:
    """Drain the existing priority-aware deque/list after a quiet window."""

    def __init__(self, state, *, debounce_seconds: float, idle_timeout: float = 20.0):
        self.state = state
        self.debounce_seconds = max(0.0, float(debounce_seconds))
        self.idle_timeout = max(0.1, float(idle_timeout))

    async def next_batch(self) -> InboxBatch | None:
        try:
            await asyncio.wait_for(
                self.state.new_message_signal.wait(),
                timeout=self.idle_timeout,
            )
        except asyncio.TimeoutError:
            return None

        await asyncio.sleep(self.debounce_seconds)
        self.state.new_message_signal.clear()
        async with self.state.data_lock:
            if (
                self.state.bot is None
                or self.state.event is None
                or not self.state.messages_chunk
            ):
                return None
            batch = InboxBatch(
                messages=list(self.state.messages_chunk),
                bot=self.state.bot,
                event=self.state.event,
            )
            self.state.messages_chunk.clear()
            return batch
