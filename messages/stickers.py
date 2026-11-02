"""表情包库：内存索引、收图、计数、挑图、拉黑、每晚清理。
设计见 docs/nyaturingtest-sticker-2026-10-20.md，计数和清理见 docs/nyaturingtest-sticker-prune-2026-11-02.md。

所有群共用一个库。启动时整表载入内存，之后收图、计数、发送、拉黑、清理同时改内存和库。
"""

import hashlib
import io
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import anyio
import numpy as np
from nonebot import logger, require
from nonebot.utils import run_sync
from PIL import Image

from ..config import get_app_settings, get_data_dir
from ..memory.vector import EMBEDDING_BATCH_SIZE, embed_texts
from ..metrics import log_event
from ..storage import db
from .image import StickerCaption, cached_caption, forget_caption, join_sticker_caption

require("nonebot_plugin_apscheduler")
from nonebot_plugin_apscheduler import scheduler  # noqa: E402  必须在 require 之后导入

STICKER_DIR = get_data_dir() / "stickers"
# 查询是 Chat 写的 want（想发什么样的图），被检索的是各张图的 reaction 段
QUERY_INSTRUCTION = (
    "Instruct: Given the reaction a chat member wants to express, "
    "retrieve a sticker whose description matches it\nQuery: "
)
# 标签内按相似度取前几名，再挑其中最久没发过的一张，免得老发同一张
PICK_TOP_K = 3
# 每晚清理群里收的图：收进来满 3 天还只见过一次、bot 也没发过的删；群友和 bot 都满 7 天没发过的删
OBSERVATION_DAYS = 3
IDLE_DAYS = 7
_FORMAT_EXT = {"GIF": "gif", "PNG": "png", "JPEG": "jpg", "WEBP": "webp"}


@dataclass
class Sticker:
    digest: str
    ext: str
    reaction: str
    subject: str
    tags: list[str]
    vector: np.ndarray | None
    source: str
    created_at: datetime
    last_sent_at: datetime | None
    seen_count: int
    last_seen_at: datetime | None

    @property
    def path(self) -> Path:
        return STICKER_DIR / f"{self.digest}.{self.ext}"

    @property
    def caption(self) -> str:
        return join_sticker_caption(self.reaction, self.subject)


_stickers: dict[str, Sticker] = {}
# 正在收的 digest：同一张图几乎同时被看两次时只收一次
_collecting: set[str] = set()


async def load_stickers() -> None:
    """启动时载入；没有向量或向量是旧 embedding 模型算的，补算一遍。"""

    model = get_app_settings().memory.model
    _stickers.clear()
    for row in await db.load_stickers():
        fresh = row.embedding is not None and row.embedding_model == model
        _stickers[row.digest] = Sticker(
            digest=row.digest,
            ext=row.ext,
            reaction=row.reaction,
            subject=row.subject,
            tags=list(row.tags),
            vector=np.frombuffer(row.embedding, dtype=np.float32) if fresh else None,
            source=row.source,
            created_at=row.created_at,
            last_sent_at=row.last_sent_at,
            seen_count=row.seen_count,
            last_seen_at=row.last_seen_at,
        )

    pending = [sticker for sticker in _stickers.values() if sticker.vector is None]
    filled = 0
    for start in range(0, len(pending), EMBEDDING_BATCH_SIZE):
        batch = pending[start : start + EMBEDDING_BATCH_SIZE]
        try:
            vectors = await embed_texts([sticker.reaction for sticker in batch])
        except Exception as e:
            logger.error(f"表情包补算向量失败，下次启动再试: {e}")
            break
        for sticker, vector in zip(batch, vectors):
            sticker.vector = vector
            await db.save_sticker_embedding(sticker.digest, vector.tobytes(), model)
        filled += len(batch)
    logger.info(f"表情包库已载入: {len(_stickers)} 张，补算向量 {filled}/{len(pending)}")


@run_sync
def _image_ext(image_bytes: bytes) -> str | None:
    try:
        return _FORMAT_EXT.get(Image.open(io.BytesIO(image_bytes)).format or "")
    except Exception as e:
        logger.warning(f"识别表情包格式失败: {e}")
        return None


def _find_by_caption(caption: str) -> Sticker | None:
    return next((sticker for sticker in _stickers.values() if sticker.caption == caption), None)


async def _count_seen(sticker: Sticker) -> None:
    sticker.seen_count += 1
    sticker.last_seen_at = datetime.now()
    await db.mark_sticker_seen(sticker.digest, sticker.seen_count, sticker.last_seen_at)


async def note_seen(digest: str) -> None:
    """群友发了一张表情包：库里有就计一次数。

    按 digest 找不到再按描述找：去重时没收的那份文件描述相同、digest 不同，计数要算到库里那张头上。
    第一次出现的图还没有描述缓存，这里什么都不做，收进来时由 collect 记第一次。
    """

    sticker = _stickers.get(digest)
    if sticker is None:
        caption = await cached_caption(digest)
        sticker = _find_by_caption(caption) if caption else None
    if sticker is not None:
        await _count_seen(sticker)


async def collect(image_bytes: bytes, result: StickerCaption, source: str) -> bool:
    """收一张表情包：写原始字节、算向量、入库；返回是否真的收了。库里已有或拉黑过的跳过。

    库里已有描述完全相同的（同一张图被重新压缩过）不另收，算那张又被发了一次。
    向量没算成也照样入库，下次启动补算；这期间它只在 want 算不出向量时才可能被挑中。
    """

    digest = hashlib.sha1(image_bytes).hexdigest()
    if digest in _stickers or digest in _collecting:
        return False
    twin = _find_by_caption(result.caption)
    if twin is not None:
        await _count_seen(twin)
        return False
    _collecting.add(digest)
    try:
        if await db.sticker_known(digest):
            return False
        ext = await _image_ext(image_bytes)
        if ext is None:
            return False
        STICKER_DIR.mkdir(parents=True, exist_ok=True)
        path = STICKER_DIR / f"{digest}.{ext}"
        async with await anyio.open_file(path, "wb") as f:
            await f.write(image_bytes)

        model = get_app_settings().memory.model
        now = datetime.now()
        try:
            vector = (await embed_texts([result.reaction]))[0]
        except Exception as e:
            logger.warning(f"表情包向量没算成，先不带向量入库: {e}")
            vector = None
        await db.add_sticker(
            digest=digest,
            ext=ext,
            reaction=result.reaction,
            subject=result.subject,
            tags=result.tags,
            source=source,
            embedding=None if vector is None else vector.tobytes(),
            embedding_model=model if vector is not None else "",
            seen_at=now,
        )
        _stickers[digest] = Sticker(
            digest=digest,
            ext=ext,
            reaction=result.reaction,
            subject=result.subject,
            tags=result.tags,
            vector=vector,
            source=source,
            created_at=now,
            last_sent_at=None,
            seen_count=1,
            last_seen_at=now,
        )
        log_event(
            "sticker_collected", source=source, digest=digest[:8], tags=result.tags
        )
        return True
    finally:
        _collecting.discard(digest)


async def pick(tag: str, want: str) -> tuple[Sticker, float | None] | None:
    """在标签内按 want 的相似度挑一张，返回 (表情包, 相似度)；标签下没图返回 None。

    标签已经定了情绪方向，向量只比细节，所以不设相似度门槛。
    want 的向量算不出来时退回标签内最久没发过的一张，相似度为 None。
    """

    candidates = [sticker for sticker in _stickers.values() if tag in sticker.tags]
    if not candidates:
        return None
    scored = [sticker for sticker in candidates if sticker.vector is not None]
    scores: dict[str, float] = {}
    if want and scored:
        try:
            query = (await embed_texts([want], instruction=QUERY_INSTRUCTION))[0]
            similarities = np.stack([sticker.vector for sticker in scored]) @ query
            top = np.argsort(-similarities)[:PICK_TOP_K]
            candidates = [scored[index] for index in top]
            scores = {scored[index].digest: float(similarities[index]) for index in top}
        except Exception as e:
            logger.warning(f"表情包挑图算向量失败，退回标签内最久没发过的: {e}")
    chosen = min(candidates, key=lambda sticker: sticker.last_sent_at or datetime.min)
    return chosen, scores.get(chosen.digest)


async def mark_sent(sticker: Sticker) -> None:
    sticker.last_sent_at = datetime.now()
    await db.mark_sticker_sent(sticker.digest, sticker.last_sent_at)


def _delete_files(digest: str) -> bool:
    """删文件和描述缓存，返回有没有文件。

    按 digest 找文件而不是查内存：collect 先写文件再入库，入库失败时文件留下了、内存里却没有。
    描述缓存也要删：缓存命中的图不再走 collect，留着它，这张图以后就再也收不回来。
    """

    files = list(STICKER_DIR.glob(f"{digest}.*"))
    for path in files:
        try:
            path.unlink()
        except Exception as e:
            logger.warning(f"删除表情包文件失败 {path}: {e}")
    forget_caption(digest)
    return bool(files)


async def block(digest: str) -> bool:
    """删掉文件和描述并拉黑，以后再见到只描述不收；返回库里原来有没有这张。"""

    _stickers.pop(digest, None)
    existed = _delete_files(digest)
    await db.block_sticker(digest)
    log_event("sticker_blocked", digest=digest[:8], existed=existed)
    return existed


async def prune_stickers() -> None:
    """每晚清理群里收的图，素材库导入的不动。删行不拉黑，以后再有人发还会重新收。

    先从内存拿掉再删库和文件：中途挑图不会挑到正在删的图。
    """

    now = datetime.now()
    observed: list[Sticker] = []
    idle: list[Sticker] = []
    for sticker in _stickers.values():
        if not sticker.source.startswith("group:"):
            continue
        # 群里收的图收进来时就记了 last_seen_at，不会是 None
        last_used = max(sticker.last_seen_at, sticker.last_sent_at or datetime.min)
        if (
            sticker.seen_count <= 1
            and sticker.last_sent_at is None
            and now - sticker.created_at >= timedelta(days=OBSERVATION_DAYS)
        ):
            observed.append(sticker)
        elif now - last_used >= timedelta(days=IDLE_DAYS):
            idle.append(sticker)

    pruned = [sticker.digest for sticker in observed + idle]
    for digest in pruned:
        del _stickers[digest]
    if pruned:
        await db.delete_stickers(pruned)
    for digest in pruned:
        _delete_files(digest)
    log_event(
        "sticker_pruned", observation=len(observed), idle=len(idle), total=len(_stickers)
    )


def setup_sticker_prune_job():
    # 排在 03:00 图片缓存清理之后、04:00 备份之前
    scheduler.add_job(
        prune_stickers,
        "cron",
        hour=3,
        minute=15,
        id="nyaturingtest_sticker_prune",
        misfire_grace_time=3600,
        replace_existing=True,
    )
