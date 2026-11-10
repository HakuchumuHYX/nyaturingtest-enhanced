"""她的相册：拍照、检索、清理。她就一部手机，同一个预设各群共用，只拍自己和眼前的东西，不带群友。

旧照片和印象一样，话题碰到才想起：决定开口后检索相册，重排通过的才给 Chat，由她决定发不发。
"""

import base64
import hashlib
import io
from dataclasses import dataclass
from datetime import datetime, timedelta

import anyio
import numpy as np
from nonebot import logger, require
from nonebot.utils import run_sync
from PIL import Image

from ..config import PRESET_DIR, get_app_settings, get_data_dir
from ..llm import VisionInput, generate_image
from ..memory.vector import embed_texts, rerank
from ..metrics import log_event
from ..prompts import RolePreset, get_photo_prompt
from ..storage import db
from .image import caption_image, forget_caption

require("nonebot_plugin_apscheduler")
from nonebot_plugin_apscheduler import scheduler  # noqa: E402  必须在 require 之后导入

PHOTO_DIR = get_data_dir() / "photos"
QUERY_INSTRUCTION = (
    "Instruct: Given a group chat message, retrieve a photo from the phone album "
    "that is relevant to it\nQuery: "
)
# 向量召回几张再重排，最后最多给 Chat 几张
RECALL_K = 8
ALBUM_K = 3
# 最后一次发送后这么多天没再发过的删掉
PHOTO_TTL_DAYS = 180
JPEG_QUALITY = 88
PHOTO_KINDS = {"selfie", "scene"}
_SEASONS = {12: "冬天", 1: "冬天", 2: "冬天", 3: "春天", 4: "春天", 5: "春天",
            6: "夏天", 7: "夏天", 8: "夏天", 9: "秋天", 10: "秋天", 11: "秋天"}


@dataclass
class NewPhoto:
    """新拍的、还没发出去的一张：发送成功才入库。"""

    kind: str
    describe: str
    caption: str
    digest: str
    taken_at: datetime


def enabled() -> bool:
    return bool(get_app_settings().image.model)


def photo_path(digest: str):
    return PHOTO_DIR / f"{digest}.jpg"


# 她发的照片写进短期记忆的样子；photodel 按它认出是哪张
PHOTO_MARK = "[图片: {}]"


def memory_text(caption: str) -> str:
    return PHOTO_MARK.format(caption)


def _search_text(caption: str, taken_at: datetime) -> str:
    # 「想念夏天」这种话要能碰到夏天拍的照片，描述里不一定有季节
    return f"{taken_at.year} 年 {taken_at.month} 月，{_SEASONS[taken_at.month]}拍的。{caption}"


@run_sync
def _to_jpeg(image: bytes) -> bytes:
    # 纯色背景时上游可能返回透明 PNG，不铺底色转出来就是黑底
    source = Image.open(io.BytesIO(image)).convert("RGBA")
    canvas = Image.new("RGBA", source.size, "white")
    canvas.alpha_composite(source)
    output = io.BytesIO()
    canvas.convert("RGB").save(output, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return output.getvalue()


async def take(
    session_id: str, preset: RolePreset, kind: str, describe: str, now: str
) -> tuple[str, NewPhoto | None, str]:
    """拍一张并写好文件，返回 (outcome, 照片, 错误)。outcome 是 ok / blocked / failed / no_portrait。"""

    reference = None
    if kind == "selfie":
        path = PRESET_DIR / preset.portrait
        if not preset.portrait or not path.is_file():
            return "no_portrait", None, ""
        async with await anyio.open_file(path, "rb") as f:
            reference = await f.read()

    taken_at = datetime.now()
    prompt = get_photo_prompt(
        kind=kind,
        appearance=preset.appearance,
        taken_at=taken_at,
        now=now,
        describe=describe,
    )
    result = await generate_image(prompt, reference, session_id=session_id)
    if result.outcome != "ok":
        return result.outcome, None, result.error
    try:
        jpeg = await _to_jpeg(result.image)
    except Exception as e:
        return "failed", None, f"{type(e).__name__}: {e}"[:300]

    digest = hashlib.sha1(jpeg).hexdigest()
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    async with await anyio.open_file(photo_path(digest), "wb") as f:
        await f.write(jpeg)
    # 发出去的图不一定和 describe 完全一样，短期记忆和相册都以看图的描述为准
    caption = await caption_image(
        VisionInput(
            ref_id="她自己拍的这张",
            data_url=f"data:image/jpeg;base64,{base64.b64encode(jpeg).decode()}",
            digest=digest,
        ),
        session_id,
    )
    caption = caption or " ".join(describe.replace("[", "【").replace("]", "】").split())
    return "ok", NewPhoto(kind, describe, caption, digest, taken_at), ""


def discard(photo: NewPhoto) -> None:
    """拍了但没发出去：文件和描述缓存都不留。"""

    _delete_files(photo.digest)


async def save(session_id: str, preset: str, photo: NewPhoto) -> None:
    model = get_app_settings().embedding.model
    try:
        vector = (await embed_texts([_search_text(photo.caption, photo.taken_at)]))[0]
    except Exception as e:
        logger.warning(f"照片向量没算成，先不带向量入库: {e}")
        vector = None
    await db.add_photo(
        session_id=session_id,
        preset=preset,
        kind=photo.kind,
        request=photo.describe,
        caption=photo.caption,
        digest=photo.digest,
        embedding=None if vector is None else vector.tobytes(),
        embedding_model=model if vector is not None else "",
        created_at=photo.taken_at,
        last_sent_at=photo.taken_at,
    )


async def _vectors(rows: list) -> list[np.ndarray]:
    """每张的向量；没算成或换过 embedding 模型的这里补算，补算失败抛异常。"""

    model = get_app_settings().embedding.model
    stale = [row for row in rows if row.embedding is None or row.embedding_model != model]
    if stale:
        vectors = await embed_texts([_search_text(row.caption, row.created_at) for row in stale])
        for row, vector in zip(stale, vectors):
            row.embedding = vector.tobytes()
            await db.save_photo_embedding(row.id, row.embedding, model)
    return [np.frombuffer(row.embedding, dtype=np.float32) for row in rows]


async def candidates(preset: str, queries: list[str]) -> list[dict]:
    """相册里和这段聊天有关的照片，给 Chat 的 album；没有或检索失败都是空列表。"""

    rows = await db.load_photos(preset)
    if not rows or not queries:
        return []
    try:
        matrix = np.stack(await _vectors(rows))
        query_vectors = await embed_texts(queries, instruction=QUERY_INSTRUCTION)
        scores = (matrix @ query_vectors.T).max(axis=1)
        top = [rows[index] for index in np.argsort(-scores)[:RECALL_K]]
        # 向量分在闲聊上分不出相关和无关，和记忆检索一样靠重排；第一条 query 是最新的消息
        reranked = await rerank(
            queries[0], [_search_text(row.caption, row.created_at) for row in top]
        )
    except Exception as e:
        logger.warning(f"检索相册失败，这轮不给旧照片: {e}")
        return []
    picked = [top[index] for index, _ in reranked or []][:ALBUM_K]
    return [
        {
            "ref": f"p{row.id}",
            "taken": f"{row.created_at:%Y-%m-%d}",
            "last_sent": f"{row.last_sent_at:%Y-%m-%d}",
            "caption": row.caption,
        }
        for row in picked
    ]


async def mark_sent(photo_id: int) -> None:
    await db.mark_photo_sent(photo_id, datetime.now())


def _delete_files(digest: str) -> None:
    try:
        photo_path(digest).unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"删除照片文件失败 {digest}: {e}")
    forget_caption(digest)


async def _delete(rows: list) -> None:
    if not rows:
        return
    await db.delete_photos([row.id for row in rows])
    for row in rows:
        _delete_files(row.digest)


async def remove_by_message(session_id: str, content: str | None) -> bool:
    """photodel：普通图片经 QQ 后字节可能变，按被回复那条消息记的 [图片: 描述] 找到相册里那张。"""

    prefix, suffix = PHOTO_MARK.split("{}")
    if not content or not content.startswith(prefix) or not content.endswith(suffix):
        return False
    row = await db.find_photo_by_caption(content[len(prefix) : -len(suffix)])
    if row is None:
        return False
    await _delete([row])
    log_event("photo_removed", session_id=session_id, photo_id=row.id)
    return True


async def prune_photos() -> None:
    expired = await db.expired_photos(datetime.now() - timedelta(days=PHOTO_TTL_DAYS))
    await _delete(expired)
    log_event("photo_pruned", count=len(expired))


def setup_photo_prune_job():
    # 不看开关：关掉生图后，相册里已有的照片照样到期删除。排在 04:00 备份之前
    scheduler.add_job(
        prune_photos,
        "cron",
        hour=3,
        minute=20,
        id="nyaturingtest_photo_prune",
        misfire_grace_time=3600,
        replace_existing=True,
    )
