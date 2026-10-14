# 图片下载、压缩与原生多模态输入准备

import asyncio
import base64
import hashlib
import io
import os
import re
import time

import anyio
from nonebot import logger, require
from nonebot.utils import run_sync
from PIL import Image

from ..config import IMAGE_CACHE_DIR
from ..core.llm import (
    VisionInput,
    extract_and_parse_json,
    feedback_client,
    get_http_client,
)
from ..core.metrics import log_event
from ..core.prompts import get_image_caption_prompt

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
) -> tuple[str, VisionInput | None]:
    """下载图片并准备原生多模态输入。返回 (消息文本占位符, 图片输入)；拿不到图时占位写明原因。"""

    placeholder = "[表情包]" if is_sticker else "[图片]"
    if not url:
        return ("[无效图片]", None)

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
                            return ("[图片类型不支持]", None)
                        if len(resp.content) > MAX_IMAGE_BYTES:
                            logger.warning(f"拒绝过大图片: {len(resp.content)} bytes")
                            return ("[图片过大]", None)
                        image_bytes = resp.content
                        break
                    except Exception:
                        await asyncio.sleep(0.5)
                if image_bytes and key:
                    await _write_cached(key, image_bytes)

            if not image_bytes:
                return ("[图片下载失败]", None)

            payload = await _prepare_native_image_payload(
                image_bytes, max_side=MAX_IMAGE_SIDE
            )
            if not payload:
                return (placeholder, None)
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
            )
        except Exception as e:
            logger.error(f"Image fetch error: {e}")
            return ("[图片处理出错]", None)


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


async def caption_image(vision_input: VisionInput, session_id: str) -> str:
    """单独看一张图写描述并缓存；失败返回空串。

    描述只写图本身、和发图的人与上下文无关，所以能按内容跨消息、跨群复用。
    """

    started_at = time.perf_counter()
    async with _CAPTION_SEMAPHORE:
        parsed = extract_and_parse_json(
            await feedback_client.generate(
                get_image_caption_prompt(is_sticker=vision_input.is_sticker),
                session_id=session_id,
                temperature=CAPTION_TEMPERATURE,
                images=[vision_input],
            )
        )
    raw = str(parsed.get("caption") or "") if isinstance(parsed, dict) else ""
    # 方括号会和消息里的标记混在一起
    caption = " ".join(raw.replace("[", "【").replace("]", "】").split())[:CAPTION_CHARS]
    if caption and vision_input.digest:
        path = _caption_path(vision_input.digest)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            async with await anyio.open_file(path, "w", encoding="utf-8") as f:
                await f.write(caption)
        except Exception as e:
            logger.warning(f"写入图片描述缓存失败: {e}")
    log_event(
        "image_caption",
        session_id=session_id,
        outcome="ok" if caption else "failed",
        latency_ms=int((time.perf_counter() - started_at) * 1000),
    )
    return caption


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
    logger.info("触发图片缓存自动清理任务...")
    await _clean_old_image_caches_sync()


scheduler.add_job(
    cleanup_image_cache_task,
    "cron",
    hour=3,
    minute=0,
    id="nyabot_image_cache_cleanup",
    replace_existing=True,
)
