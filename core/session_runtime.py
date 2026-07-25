import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionRuntime:
    """I/O resources and task coordination owned by a Session."""

    short_term_memory: Any = None
    vector_memory: Any = None
    http_client: Any = None
    owns_http_client: bool = False
    persistence: Any = None
    memory_writer: Any = None
    background_tasks: set[asyncio.Task] = field(default_factory=set)
    save_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
