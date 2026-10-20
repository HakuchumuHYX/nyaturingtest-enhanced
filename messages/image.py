# 图片下载、压缩与原生多模态输入准备

import asyncio
import base64
import hashlib
import io
import math
import os
import re
import time
from dataclasses import dataclass

import anyio
from nonebot import logger, require
from nonebot.utils import run_sync
from PIL import Image

from ..config import IMAGE_CACHE_DIR
from ..llm import (
    VisionInput,
    extract_and_parse_json,
    feedback_client,
    get_http_client,
)
from ..metrics import log_event
from ..prompts import (
    STICKER_TAGS,
    get_image_caption_prompt,
    get_sticker_caption_prompt,
)

require("nonebot_plugin_apscheduler")
from nonebot_plugin_apscheduler import scheduler  # noqa: E402  必须在 require 之后导入

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 4096 * 4096
MAX_IMAGE_SIDE = 1280
MAX_CACHE_KEY_LEN = 128
SAFE_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
_CACHE_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

_IMG_SEMAPHORE = asyncio.Semaphore(3)

# 原图缓存从写入起留 48 小时；描述缓存命中续租，30 天没被命中才删
RAW_RETENTION_SECONDS = 48 * 3600
CAPTION_RETENTION_SECONDS = 30 * 24 * 3600
CAPTION_TEMPERATURE = 0.1
CAPTION_CHARS = 120
_CAPTION_SEMAPHORE = asyncio.Semaphore(4)
STICKER_MAX_TAGS = 3
# 表情包动图最多取 GRID_SIDE × GRID_SIDE 帧拼图；每格边长、格间距
GRID_SIDE = 3
GRID_CELL = 320
GRID_GAP = 4


def _sanitize_cache_key(key: str) -> str:
    value = str(key or "").strip()
    if not value or len(value) > MAX_CACHE_KEY_LEN:
        return ""
    if "/" in value or "\\" in value or ".." in value or not _CACHE_KEY_RE.match(value):
        return ""
    return value


def _cache_key(url: str, file_unique: str) -> str:
    match = re.search(r"[?&]fileid=([A-Za-z0-9_-]+)", url or "")
    if match:
        return _sanitize_cache_key(match.group(1))
    return _sanitize_cache_key(file_unique)


async def _read_cached(key: str) -> bytes | None:
    path = IMAGE_CACHE_DIR.joinpath("raw", key)
    if not path.exists():
        return None
    try:
        async with await anyio.open_file(path, "rb") as f:
            return await f.read()
    except Exception as e:
        logger.warning(f"读取图片缓存失败: {e}")
        return None


async def _write_cached(key: str, data: bytes) -> None:
    path = IMAGE_CACHE_DIR.joinpath("raw", key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        async with await anyio.open_file(path, "wb") as f:
            await f.write(data)
    except Exception as e:
        logger.warning(f"写入图片缓存失败: {e}")


async def fetch_image_input(
    url: str,
    file_unique: str,
    *,
    is_sticker: bool,
    ref_id: str,
) -> tuple[str, VisionInput | None, bytes | None]:
    """下载图片并准备原生多模态输入。

    返回 (消息文本占位符, 图片输入, 原始字节)；拿不到图时占位写明原因、后两项为 None。
    原始字节给表情包看图拼九宫格和收进库用：图片输入是压缩过的首帧。
    """

    placeholder = "[表情包]" if is_sticker else "[图片]"
    if not url:
        return ("[无效图片]", None, None)

    async with _IMG_SEMAPHORE:
        try:
            key = _cache_key(url, file_unique)
            image_bytes = await _read_cached(key) if key else None

            if not image_bytes:
                client = get_http_client()
                for _ in range(2):
                    try:
                        resp = await client.get(url, timeout=10.0)
                        resp.raise_for_status()
                        content_type = (
                            (resp.headers.get("content-type") or "")
                            .split(";")[0]
                            .strip()
                            .lower()
                        )
                        if (
                            content_type
                            and content_type not in SAFE_IMAGE_CONTENT_TYPES
                        ):
                            logger.warning(f"拒绝非图片响应: {content_type}")
                            return ("[图片类型不支持]", None, None)
                        if len(resp.content) > MAX_IMAGE_BYTES:
                            logger.warning(f"拒绝过大图片: {len(resp.content)} bytes")
                            return ("[图片过大]", None, None)
                        image_bytes = resp.content
                        break
                    except Exception:
                        await asyncio.sleep(0.5)
                if image_bytes and key:
                    await _write_cached(key, image_bytes)

            if not image_bytes:
                return ("[图片下载失败]", None, None)

            payload = await _prepare_native_image_payload(
                image_bytes, max_side=MAX_IMAGE_SIDE
            )
            if not payload:
                return (placeholder, None, None)
            payload_bytes, image_format = payload
            encoded = base64.b64encode(payload_bytes).decode("utf-8")
            return (
                placeholder,
                VisionInput(
                    ref_id=ref_id,
                    data_url=f"data:image/{image_format};base64,{encoded}",
                    is_sticker=is_sticker,
                    digest=hashlib.sha1(image_bytes).hexdigest(),
                ),
                image_bytes,
            )
        except Exception as e:
            logger.error(f"Image fetch error: {e}")
            return ("[图片处理出错]", None, None)


def _caption_path(digest: str):
    return IMAGE_CACHE_DIR.joinpath("caption", f"{digest}.txt")


async def cached_caption(digest: str) -> str | None:
    """查描述缓存；命中就刷新文件时间续租。"""

    path = _caption_path(digest)
    if not digest or not path.exists():
        return None
    try:
        async with await anyio.open_file(path, encoding="utf-8") as f:
            caption = await f.read()
        os.utime(path)
    except Exception as e:
        logger.warning(f"读取图片描述缓存失败: {e}")
        return None
    return caption or None


def _clean_caption(raw) -> str:
    # 方括号会和消息里的标记混在一起
    text = str(raw or "")
    return " ".join(text.replace("[", "【").replace("]", "】").split())[:CAPTION_CHARS]


@dataclass(frozen=True)
class StickerCaption:
    # 表情、动作、语气、配字；挑图的向量只按这段算
    reaction: str
    # 外观，不含角色名
    subject: str
    tags: list[str]
    keep: bool

    @property
    def caption(self) -> str:
        return join_sticker_caption(self.reaction, self.subject)


def join_sticker_caption(reaction: str, subject: str) -> str:
    """写进消息的表情包描述：反应在前、外观在后，顺序由这里定，不靠模型。"""

    return f"{reaction}；{subject}" if subject else reaction


async def _write_caption(digest: str, caption: str) -> None:
    path = _caption_path(digest)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        async with await anyio.open_file(path, "w", encoding="utf-8") as f:
            await f.write(caption)
    except Exception as e:
        logger.warning(f"写入图片描述缓存失败: {e}")


def forget_caption(digest: str) -> None:
    """删掉一张图的描述缓存；stickerdel 删坏图时连描述一起清，下次见到重新看。"""

    try:
        _caption_path(digest).unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"删除图片描述缓存失败: {e}")


def _log_caption(session_id: str, ok: bool, started_at: float) -> None:
    log_event(
        "image_caption",
        "DEBUG",
        session_id=session_id,
        outcome="ok" if ok else "failed",
        latency_ms=int((time.perf_counter() - started_at) * 1000),
    )


async def caption_image(vision_input: VisionInput, session_id: str) -> str:
    """单独看一张普通图片写描述并缓存；失败返回空串。表情包走 caption_sticker。

    描述只写图本身、和发图的人与上下文无关，所以能按内容跨消息、跨群复用。
    """

    started_at = time.perf_counter()
    async with _CAPTION_SEMAPHORE:
        parsed = extract_and_parse_json(
            await feedback_client.generate(
                get_image_caption_prompt(),
                session_id=session_id,
                temperature=CAPTION_TEMPERATURE,
                images=[vision_input],
            )
        )
    caption = _clean_caption(parsed.get("caption") if isinstance(parsed, dict) else "")
    if caption and vision_input.digest:
        await _write_caption(vision_input.digest, caption)
    _log_caption(session_id, bool(caption), started_at)
    return caption


async def caption_sticker(
    image_bytes: bytes, ref_id: str, session_id: str
) -> StickerCaption | None:
    """看一张表情包（动图拼九宫格）、打标并缓存描述；失败返回 None。"""

    started_at = time.perf_counter()
    vision_input = await sticker_vision_input(image_bytes, ref_id)
    result = await describe_sticker(vision_input, session_id) if vision_input else None
    if result is not None:
        await _write_caption(vision_input.digest, result.caption)
    _log_caption(session_id, result is not None, started_at)
    return result


async def describe_sticker(
    vision_input: VisionInput, session_id: str
) -> StickerCaption | None:
    """看一张表情包；模型没给出可用结果时返回 None。"""

    async with _CAPTION_SEMAPHORE:
        parsed = extract_and_parse_json(
            await feedback_client.generate(
                get_sticker_caption_prompt(),
                session_id=session_id,
                temperature=CAPTION_TEMPERATURE,
                images=[vision_input],
            )
        )
    if not isinstance(parsed, dict):
        return None
    reaction = _clean_caption(parsed.get("reaction"))
    if not reaction:
        return None
    raw_tags = parsed.get("tags")
    if not isinstance(raw_tags, list):
        raw_tags = []
    tags = list(dict.fromkeys(tag for tag in raw_tags if tag in STICKER_TAGS))
    tags = tags[:STICKER_MAX_TAGS]
    return StickerCaption(
        reaction=reaction,
        subject=_clean_caption(parsed.get("subject")),
        tags=tags,
        keep=parsed.get("keep") is True and bool(tags),
    )


async def sticker_vision_input(image_bytes: bytes, ref_id: str) -> VisionInput | None:
    """看表情包用的图：动图拼九宫格，静图照常压缩。Chat 附的原图仍是第一帧，不走这里。"""

    payload = await _sticker_grid_payload(image_bytes)
    if payload is None:
        payload = await _prepare_native_image_payload(
            image_bytes, max_side=MAX_IMAGE_SIDE
        )
    if not payload:
        return None
    payload_bytes, image_format = payload
    encoded = base64.b64encode(payload_bytes).decode("utf-8")
    return VisionInput(
        ref_id=ref_id,
        data_url=f"data:image/{image_format};base64,{encoded}",
        is_sticker=True,
        digest=hashlib.sha1(image_bytes).hexdigest(),
    )


@run_sync
def _sticker_grid_payload(image_bytes: bytes) -> tuple[bytes, str] | None:
    """动图均匀取最多 9 帧，按时间从左到右、从上到下拼成 JPEG。

    静图和拼图失败都返回 None，调用方退回只看第一帧。
    上游不收 GIF，而很多动图第一帧还没做出表情。
    """

    try:
        image = Image.open(io.BytesIO(image_bytes))
        frame_count = getattr(image, "n_frames", 1)
        if frame_count < 2:
            return None
        if image.width * image.height > MAX_IMAGE_PIXELS:
            logger.warning(f"拒绝像素过大的图片: {image.width}x{image.height}")
            return None
        count = min(frame_count, GRID_SIDE * GRID_SIDE)
        indices = [round(i * (frame_count - 1) / (count - 1)) for i in range(count)]
        # 不足 9 帧时排成接近正方形，免得空出大片格子
        columns = math.ceil(math.sqrt(count))
        rows = math.ceil(count / columns)
        step = GRID_CELL + GRID_GAP
        sheet = Image.new(
            "RGB", (columns * step - GRID_GAP, rows * step - GRID_GAP), (200, 200, 200)
        )
        for slot, index in enumerate(indices):
            image.seek(index)
            frame = image.convert("RGBA")
            frame.thumbnail((GRID_CELL, GRID_CELL), Image.Resampling.LANCZOS)
            # 透明底铺白，不然转 RGB 后是黑底
            cell = Image.new("RGB", (GRID_CELL, GRID_CELL), (255, 255, 255))
            cell.paste(
                frame,
                ((GRID_CELL - frame.width) // 2, (GRID_CELL - frame.height) // 2),
                frame,
            )
            sheet.paste(cell, ((slot % columns) * step, (slot // columns) * step))
        output = io.BytesIO()
        sheet.save(output, format="JPEG", quality=90)
        return output.getvalue(), "jpeg"
    except Exception as exc:
        logger.warning(f"动图拼九宫格失败: {exc}")
        return None


@run_sync
def _prepare_native_image_payload(
    image_bytes: bytes,
    *,
    max_side: int,
) -> tuple[bytes, str] | None:
    """校验、缩放并转码图片，产出可直接发送的 data URL 负载。"""

    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.width * image.height > MAX_IMAGE_PIXELS:
            logger.warning(f"拒绝像素过大的图片: {image.width}x{image.height}")
            return None
        raw_format = (image.format or "JPEG").lower()
        if raw_format == "jpg":
            raw_format = "jpeg"
        # 动图只取第一帧：上游（grok）只接受 JPG/PNG/WebP，原样透传 GIF 会让整轮请求 400。
        # 只有 GIF/WebP 这类多帧插件才有 is_animated，JPEG 上这个属性不存在
        if getattr(image, "is_animated", False):
            image.seek(0)

        w, h = image.size
        if max(w, h) > max_side:
            ratio = max_side / max(w, h)
            image = image.resize(
                (max(1, int(w * ratio)), max(1, int(h * ratio))),
                Image.Resampling.LANCZOS,
            )

        output = io.BytesIO()
        if raw_format == "png":
            if image.mode not in {"RGB", "RGBA", "L", "LA"}:
                image = image.convert("RGBA")
            image.save(output, format="PNG", optimize=True)
            return output.getvalue(), "png"

        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(output, format="JPEG", quality=90)
        return output.getvalue(), "jpeg"
    except Exception as exc:
        logger.warning(f"图片预处理失败: {exc}")
        return None


@run_sync
def _clean_old_image_caches_sync():
    """清理过期的原图缓存（写入起 48 小时）和描述缓存（30 天没被命中）。"""

    try:
        now = time.time()
        for name, retention_seconds in (
            ("raw", RAW_RETENTION_SECONDS),
            ("caption", CAPTION_RETENTION_SECONDS),
        ):
            cache_dir = IMAGE_CACHE_DIR.joinpath(name)
            if not cache_dir.exists():
                continue
            count = 0
            for file_path in cache_dir.iterdir():
                if file_path.is_file() and (
                    now - file_path.stat().st_mtime > retention_seconds
                ):
                    try:
                        file_path.unlink()
                        count += 1
                    except Exception as e:
                        logger.warning(f"删除过期图片缓存失败 {file_path}: {e}")
            if count > 0:
                logger.info(f"清理了 {count} 个过期的图片缓存文件（{name}）")
    except Exception as e:
        logger.error(f"清理图片缓存任务执行异常: {e}")


async def cleanup_image_cache_task():
    await _clean_old_image_caches_sync()


def setup_image_cache_job():
    scheduler.add_job(
        cleanup_image_cache_task,
        "cron",
        hour=3,
        minute=0,
        id="nyabot_image_cache_cleanup",
        replace_existing=True,
    )
