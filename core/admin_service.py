from collections.abc import Awaitable, Callable
from typing import Protocol


class _Session(Protocol):
    async def load_session(self) -> None: ...

    def bump_generation(self, reason: str = "") -> int: ...

    async def reset(self) -> None: ...


class _GroupState(Protocol):
    session: _Session
    session_lock: object


async def reset_session_with_backup(
    state: _GroupState,
    backup: Callable[[], Awaitable[bool]],
) -> bool:
    """Invalidate active turns, back up runtime data, then reset one session.

    The backup deliberately runs outside the session lock because it may spend a
    long time snapshotting and compressing global runtime data.
    """

    async with state.session_lock:  # type: ignore[attr-defined]
        await state.session.load_session()
        state.session.bump_generation("reset_requested")

    if not await backup():
        return False

    async with state.session_lock:  # type: ignore[attr-defined]
        await state.session.reset()
    return True
