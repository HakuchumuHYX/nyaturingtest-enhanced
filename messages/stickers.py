"""表情包库：内存索引、收图、挑图、拉黑。设计见 docs/nyaturingtest-sticker-2026-10-20.md。

所有群共用一个库。启动时整表载入内存，之后收图、发送、拉黑同时改内存和库；
导入脚本直接写库，要重启 bot 才进内存。
"""

import hashlib
import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import anyio
import numpy as np
from nonebot import logger
from nonebot.utils import run_sync
from PIL import Image

from ..config import get_app_settings, get_data_dir
from ..memory.vector import EMBEDDING_BATCH_SIZE, embed_texts
from ..metrics import log_event
from ..storage import db
from .image import StickerCaption, forget_caption, join_sticker_caption

STICKER_DIR = get_data_dir() / "stickers"
# 查询是 Chat 写的 want（想发什么样的图），被检索的是各张图的 reaction 段
QUERY_INSTRUCTION = (
    "Instruct: Given the reaction a chat member wants to express, "
    "retrieve a sticker whose description matches it\nQuery: "
)
# 标签内按相似度取前几名，再挑其中最久没发过的一张，免得老发同一张
PICK_TOP_K = 3
_FORMAT_EXT = {"GIF": "gif", "PNG": "png", "JPEG": "jpg", "WEBP": "webp"}


@dataclass
class Sticker:
    digest: str
    ext: str
    reaction: str
    subject: str
    tags: list[str]
    vector: np.ndarray | None
    last_sent_at: datetime | None

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
            last_sent_at=row.last_sent_at,
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


async def collect(image_bytes: bytes, result: StickerCaption, source: str) -> bool:
    """收一张表情包：写原始字节、算向量、入库；返回是否真的收了。库里已有或拉黑过的跳过。

    向量没算成也照样入库，下次启动补算；这期间它只在 want 算不出向量时才可能被挑中。
    """

    digest = hashlib.sha1(image_bytes).hexdigest()
    if digest in _stickers or digest in _collecting:
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
        )
        _stickers[digest] = Sticker(
            digest=digest,
            ext=ext,
            reaction=result.reaction,
            subject=result.subject,
            tags=result.tags,
            vector=vector,
            last_sent_at=None,
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


async def block(digest: str) -> bool:
    """删掉文件和描述并拉黑，以后再见到只描述不收；返回库里原来有没有这张。"""

    _stickers.pop(digest, None)
    # 按 digest 找文件而不是查内存：导入脚本刚写进库、还没重启载入的也要删到
    files = list(STICKER_DIR.glob(f"{digest}.*"))
    for path in files:
        try:
            path.unlink()
        except Exception as e:
            logger.warning(f"删除表情包文件失败 {path}: {e}")
    await db.block_sticker(digest)
    forget_caption(digest)
    log_event("sticker_blocked", digest=digest[:8], existed=bool(files))
    return bool(files)
