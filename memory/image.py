# 图片下载、压缩与原生多模态输入准备
import asyncio
import base64
import io
import re
import time

import anyio
from nonebot import logger, require
from nonebot.utils import run_sync
from PIL import Image

try:
    require("nonebot_plugin_apscheduler")
    from nonebot_plugin_apscheduler import scheduler
except Exception:
    scheduler = None

from ..config import get_image_cache_dir
from ..core.llm import get_http_client
from ..core.llm import VisionInput


MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 4096 * 4096
MAX_IMAGE_SIDE = 1280
MAX_CACHE_KEY_LEN = 128
SAFE_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
_CACHE_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

IMAGE_CACHE_DIR = get_image_cache_dir()
_IMG_SEMAPHORE = asyncio.Semaphore(3)


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
    source: str,
) -> tuple[str, VisionInput | None]:
    """下载图片并准备原生多模态输入。返回 (消息文本占位符, 图片输入)。"""

    placeholder = "\n[表情包]\n" if is_sticker else "\n[图片]\n"
    if not url:
        return ("\n[无效图片]\n", None)

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
                            (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
                        )
                        if content_type and content_type not in SAFE_IMAGE_CONTENT_TYPES:
                            logger.warning(f"拒绝非图片响应: {content_type}")
                            return ("\n[图片类型不支持]\n", None)
                        if len(resp.content) > MAX_IMAGE_BYTES:
                            logger.warning(f"拒绝过大图片: {len(resp.content)} bytes")
                            return ("\n[图片过大]\n", None)
                        image_bytes = resp.content
                        break
                    except Exception:
                        await asyncio.sleep(0.5)
                if image_bytes and key:
                    await _write_cached(key, image_bytes)

            if not image_bytes:
                return ("\n[图片下载失败]\n", None)

            payload = await _prepare_native_image_payload(image_bytes, max_side=MAX_IMAGE_SIDE)
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
                    source=source,
                ),
            )
        except Exception as e:
            logger.error(f"Image fetch error: {e}")
            return ("\n[图片处理出错]\n", None)


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
        if image.is_animated and image.n_frames > 1:
            if raw_format == "gif":
                return image_bytes, "gif"
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
    """清理超过 48 小时的图片缓存文件。"""

    try:
        now = time.time()
        retention_seconds = 48 * 3600
        count = 0

        if IMAGE_CACHE_DIR.exists():
            raw_dir = IMAGE_CACHE_DIR.joinpath("raw")
            if raw_dir.exists():
                for file_path in raw_dir.iterdir():
                    if file_path.is_file() and (now - file_path.stat().st_mtime > retention_seconds):
                        try:
                            file_path.unlink()
                            count += 1
                        except Exception as e:
                            logger.warning(f"删除过期原图缓存失败 {file_path}: {e}")

        if count > 0:
            logger.info(f"成功清理 {count} 个超过 48 小时的旧图片缓存文件。")
    except Exception as e:
        logger.error(f"清理图片缓存任务执行异常: {e}")


async def cleanup_image_cache_task():
    logger.info("触发图片缓存自动清理任务...")
    await _clean_old_image_caches_sync()


if scheduler:
    scheduler.add_job(
        cleanup_image_cache_task,
        "cron",
        hour=3,
        minute=0,
        id="nyabot_image_cache_cleanup",
        replace_existing=True,
    )
