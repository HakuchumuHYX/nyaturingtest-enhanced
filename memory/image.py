# nyaturingtest/image_manager.py
import asyncio
import base64
from collections import OrderedDict
import hashlib
import io
import math
from pathlib import Path
import re

from typing import Callable
import anyio
from nonebot import logger, require
import nonebot_plugin_localstore as store
from PIL import Image, ImageDraw, ImageFont
from nonebot.utils import run_sync
import time

try:
    require("nonebot_plugin_apscheduler")
    from nonebot_plugin_apscheduler import scheduler
except Exception:
    scheduler = None

from ..config import (
    plugin_config,
    get_effective_vlm_api_key,
    get_effective_vlm_base_url,
    get_effective_vlm_model,
    should_use_standalone_vlm,
)
from ..llm.vlm import VLM
from ..llm.vision import VisionInput
from ..core.metrics import metrics
from ..utils import get_http_client
from .image_policy import (
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS,
    MEM_CACHE_MAX_ITEMS,
    MEM_CACHE_TTL_SECONDS,
    SAFE_IMAGE_CONTENT_TYPES,
    sanitize_image_cache_key,
)
from .image_schema import (
    ImageWithDescription,
    parse_vlm_response,
    render_image_text,
    gif_target_count,
)

IMAGE_CACHE_DIR = Path(f"{store.get_plugin_cache_dir()}/image_cache")
_IMG_SEMAPHORE = asyncio.Semaphore(3)
IMAGE_OBSERVATION_SCHEMA_VERSION = "3"


class ImageManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._vlm: VLM | None = None
            IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self._initialized = True
            self._mem_cache: OrderedDict[str, tuple[float, ImageWithDescription]] = OrderedDict()

    def _get_vlm(self) -> VLM:
        if self._vlm is None:
            if not should_use_standalone_vlm():
                raise RuntimeError("Standalone VLM is disabled by the effective image route.")
            vlm_provider = plugin_config.get("vlm", {}).get("provider", "openai_compatible").strip().lower()
            self._vlm = VLM(
                api_key=get_effective_vlm_api_key(),
                endpoint=get_effective_vlm_base_url(),
                model=get_effective_vlm_model(),
                provider=vlm_provider,
                timeout=int(plugin_config.get("vlm", {}).get("timeout") or 60),
            )
        return self._vlm

    def get_from_cache(self, key: str) -> ImageWithDescription | None:
        safe_key = sanitize_image_cache_key(key)
        if not safe_key or safe_key not in self._mem_cache:
            return None
        created_at, data = self._mem_cache[safe_key]
        if time.time() - created_at > MEM_CACHE_TTL_SECONDS:
            self._mem_cache.pop(safe_key, None)
            return None
        self._mem_cache.move_to_end(safe_key)
        return data

    def save_to_cache(self, key: str, data: ImageWithDescription):
        safe_key = sanitize_image_cache_key(key)
        if safe_key and data:
            self._mem_cache[safe_key] = (time.time(), data)
            self._mem_cache.move_to_end(safe_key)
            while len(self._mem_cache) > MEM_CACHE_MAX_ITEMS:
                self._mem_cache.popitem(last=False)

    def _description_cache_key(
        self,
        identifier: str,
        *,
        is_sticker: bool,
        context_text: str,
    ) -> str:
        material = "|".join([
            IMAGE_OBSERVATION_SCHEMA_VERSION,
            get_effective_vlm_model(),
            str(identifier or ""),
            "sticker" if is_sticker else "image",
            str(context_text or ""),
        ])
        return hashlib.sha256(material.encode("utf-8", "ignore")).hexdigest()

    async def resolve_image_from_url(self, url: str, file_unique: str, is_sticker: bool, context_text: str = "",
                                     on_usage: Callable[[dict], None] | None = None,
                                     *,
                                     describe: bool = True,
                                     include_native: bool = False,
                                     ref_id: str = "",
                                     source: str = "primary",
                                     ) -> tuple[str, dict | None, VisionInput | None]:
        """
        高层接口：下载图片，并按路由选择独立 VLM 描述和/或原生图片输入。
        元数据为 None 表示无结构化观测（占位/失败/未识别）。
        """
        if not url:
            return ("[无效图片]", None, None)

        async with _IMG_SEMAPHORE:
            try:
                description_cache_key = self._description_cache_key(
                    file_unique or url,
                    is_sticker=is_sticker,
                    context_text=context_text,
                )
                cached_desc = self.get_from_cache(description_cache_key) if describe else None
                if cached_desc and not include_native:
                    return (render_image_text(cached_desc, is_sticker), cached_desc.to_meta(), None)

                # 2. 准备文件缓存路径
                cache_path = IMAGE_CACHE_DIR.joinpath("raw")
                cache_path.mkdir(parents=True, exist_ok=True)

                # 尝试从 URL 或 file_unique 提取文件名
                key = None
                key_match = re.search(r"[?&]fileid=([a-zA-Z0-9_-]+)", url)
                if key_match:
                    key = sanitize_image_cache_key(key_match.group(1))
                elif file_unique:
                    key = sanitize_image_cache_key(file_unique)

                image_bytes = None

                # 3. 尝试读取本地文件缓存
                if key and cache_path.joinpath(key).exists():
                    try:
                        async with await anyio.open_file(cache_path.joinpath(key), "rb") as f:
                            image_bytes = await f.read()
                    except Exception as e:
                        logger.warning(f"读取图片缓存失败: {e}")

                if not image_bytes:
                    # 4. 下载图片
                    client = get_http_client()
                    for _ in range(2):  # 重试2次
                        try:
                            resp = await client.get(url, timeout=10.0)  # 稍微增加超时
                            resp.raise_for_status()
                            content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
                            if content_type and content_type not in SAFE_IMAGE_CONTENT_TYPES:
                                logger.warning(f"拒绝非图片响应: {content_type}")
                                return ("\n[图片类型不支持]\n", None, None)
                            if len(resp.content) > MAX_IMAGE_BYTES:
                                logger.warning(f"拒绝过大图片: {len(resp.content)} bytes")
                                return ("\n[图片过大]\n", None, None)
                            image_bytes = resp.content
                            break
                        except Exception:
                            await asyncio.sleep(0.5)

                    # 下载成功后写入缓存
                    if image_bytes and key:
                        try:
                            async with await anyio.open_file(cache_path.joinpath(key), "wb") as f:
                                await f.write(image_bytes)
                        except Exception as e:
                            logger.warning(f"写入图片缓存失败: {e}")

                if not image_bytes:
                    return ("\n[图片下载失败]\n", None, None)

                vision_input = None
                if include_native:
                    vision_input = await self._build_native_vision_input(
                        image_bytes,
                        is_sticker=is_sticker,
                        ref_id=ref_id,
                        source=source,
                    )

                if not describe:
                    placeholder = "\n[表情包]\n" if is_sticker else "\n[图片]\n"
                    return (placeholder, None, vision_input)

                if cached_desc:
                    return (
                        render_image_text(cached_desc, is_sticker),
                        cached_desc.to_meta(),
                        vision_input,
                    )

                # 5. 调用 VLM 进行识别
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                description = await self.get_image_description(
                    image_base64=image_base64, is_sticker=is_sticker,
                    cache_key=description_cache_key,
                    context_text=context_text, on_usage=on_usage
                )

                if description:
                    return (
                        render_image_text(description, is_sticker),
                        description.to_meta(),
                        vision_input,
                    )
                return ("\n[图片识别无结果]\n", None, vision_input)

            except Exception as e:
                logger.error(f"Image resolve error: {e}")
                return ("\n[图片处理出错]\n", None, None)

    async def _build_native_vision_input(
        self,
        image_bytes: bytes,
        *,
        is_sticker: bool,
        ref_id: str,
        source: str,
    ) -> VisionInput | None:
        normalized = await _prepare_native_image_payload(
            image_bytes,
            max_side=_configured_max_image_side(),
        )
        if not normalized:
            return None
        payload_bytes, image_format = normalized
        encoded = base64.b64encode(payload_bytes).decode("utf-8")
        return VisionInput(
            ref_id=ref_id,
            data_url=f"data:image/{image_format};base64,{encoded}",
            is_sticker=is_sticker,
            source=source,
        )

    async def get_image_description(self, image_base64: str, is_sticker: bool,
                                    cache_key: str | None = None,
                                    context_text: str | None = None,
                                    on_usage: Callable[[dict], None] | None = None) -> ImageWithDescription | None:
        # 1. 缓存检查
        cached = self.get_from_cache(cache_key or "")
        if cached:
            return cached

        image_bytes = base64.b64decode(image_base64)
        if len(image_bytes) > MAX_IMAGE_BYTES:
            logger.warning(f"拒绝过大图片: {len(image_bytes)} bytes")
            return None
        image_hash = await _calculate_image_hash(image_bytes)

        disk_cache_material = "|".join([
            IMAGE_OBSERVATION_SCHEMA_VERSION,
            get_effective_vlm_model(),
            image_hash,
            str(cache_key or ""),
            "sticker" if is_sticker else "image",
            str(context_text or ""),
        ])
        disk_cache_key = hashlib.sha256(
            disk_cache_material.encode("utf-8", "ignore")
        ).hexdigest()
        cache = IMAGE_CACHE_DIR.joinpath(f"{disk_cache_key}.json")
        if cache.exists():
            try:
                async with await anyio.open_file(cache, encoding="utf-8") as f:
                    image_with_desc = ImageWithDescription.from_json(await f.read())
                    if cache_key:
                        self.save_to_cache(cache_key, image_with_desc)
                    return image_with_desc
            except ValueError:
                cache.unlink(missing_ok=True)

        # 2. 图片加载与初步格式判断
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.width * image.height > MAX_IMAGE_PIXELS:
                logger.warning(f"拒绝像素过大的图片: {image.width}x{image.height}")
                return None
            raw_format = (image.format or "JPEG").lower()
        except Exception:
            logger.error("无法识别的图片格式")
            return None

        target_image_base64 = image_base64
        target_format = raw_format

        code_mark = "```"

        # 构建基础 Prompt（群聊语境理解任务，非通用标注）
        base_prompt = "你正在观察一个群聊里的图片。请从「群聊语境理解」角度分析这张图片。\n"
        if context_text:
            base_prompt += (
                f"【背景信息】这张图片是在对话中发送的，相关文本内容是：“{context_text}”。"
                "请判断这张图是对哪句话的回应、传达什么意图。\n"
            )

        base_prompt += f"""请输出以下字段：
1. visual_description：用中文完整描述与群聊理解有关的画面、动作、空间关系和显著细节，最多160字。
2. ocr_text：按自然阅读顺序原样提取图片里的文字；截图、表格、聊天记录需尽量保留换行和区块关系，没有文字就留空字符串。
3. entities：凭你已有的知识判断图中出现的角色/IP/真人/品牌/meme 名称，给出数组，每项含 name(名称)、type(取值: character/real_person/meme/brand/object)、confidence(0~1)。不认识就返回空数组 []，禁止编造不存在的名字。
4. pragmatic_intent：这张图在对话里的语用功能，从以下封闭标签里选一个：嘲讽/自嘲/附和/破冰/卖萌/终结话题/否认/求助/感叹/无。
5. affect：图片表达的情感，用 VAD 三元组：{{"valence":-1~1(愉悦度),"arousal":0~1(兴奋度),"dominance":-1~1(支配度)}}。
请直接输出纯 JSON，不要包含任何其他内容：
{code_mark}json
{{
    "visual_description": "...",
    "ocr_text": "...",
    "entities": [{{"name": "...", "type": "character", "confidence": 0.0}}],
    "pragmatic_intent": "无",
    "affect": {{"valence": 0.0, "arousal": 0.0, "dominance": 0.0}}
}}
{code_mark}
"""
        prompt = base_prompt

        # === 3. 核心逻辑：GIF 专门处理 ===
        # 判定条件：是动图 且 帧数 > 1 (避免单帧 GIF 误判)
        if getattr(image, "is_animated", False) and image.n_frames > 1:
            if image.n_frames > 80:
                logger.warning(f"GIF 帧数过多，降级首帧处理: {image.n_frames}")
                target_format = "jpeg"
            else:
                # 调用专门的九宫格处理函数
                grid_info = await _process_gif_to_grid(image_base64)
                if grid_info:
                    grid_b64, frame_count = grid_info
                    # 更新 Prompt，告诉 LLM 这是一张拼图，并要求 temporal 时序输出
                    prompt = (
                                 f"这是一张包含 {frame_count} 个关键帧的动图分解拼图。"
                                 "图片左上角标有数字序号（1, 2, 3...），代表时间顺序。"
                                 "请结合这些关键帧，分析这个动图发生了什么动作或情节，"
                                 "并在 JSON 中额外输出 temporal 字段：按帧序号给出动作序列，"
                                 "形如 [{\"frame\":1,\"action\":\"举手\"}, ...]。"
                             ) + base_prompt
                    target_image_base64 = grid_b64
                    target_format = "jpeg"
                else:
                    # 如果处理失败，降级为第一帧
                    target_format = "jpeg"

        # === 4. 格式最终清洗 + 静态图压缩 ===
        try:
            img = Image.open(io.BytesIO(base64.b64decode(target_image_base64)))
            max_side = _configured_max_image_side()
            w, h = img.size
            if max(w, h) > max_side:
                ratio = max_side / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            preserve_png = target_format == "png"
            if preserve_png:
                if img.mode not in {"RGB", "RGBA", "L", "LA"}:
                    img = img.convert("RGBA")
                img.save(buffer, format="PNG", optimize=True)
                target_format = "png"
            else:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(buffer, format="JPEG", quality=90)
                target_format = "jpeg"
            target_image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"图片压缩/格式转换失败: {e}")
            if target_format not in ["jpeg", "png", "webp"]:
                return None

        # 5. 发送请求（分级 detail：表情包走 high 提升配字/角色细节识别率）
        high_detail_for_sticker = bool(
            plugin_config.get("vlm", {}).get("high_detail_for_sticker", True)
        )
        high_detail_for_png = bool(
            plugin_config.get("vlm", {}).get("high_detail_for_png", True)
        )
        detail = "high" if (
            (is_sticker and high_detail_for_sticker)
            or (target_format == "png" and high_detail_for_png)
        ) else "low"
        response = await self._get_vlm().request(
            prompt=prompt,
            image_base64=target_image_base64,
            image_format=target_format,
            on_usage=on_usage,
            detail=detail,
            response_format={"type": "json_object"},
        )

        if not response:
            metrics.vlm_failure += 1
            return None
        metrics.vlm_success += 1

        # 解析 VLM 响应为多正交槽结构（逐字段缺省降级，解析失败也不抛错）
        result = parse_vlm_response(response, is_sticker=is_sticker)

        async with await anyio.open_file(cache, "w", encoding="utf-8") as f:
            await f.write(result.to_json())

        if cache_key:
            self.save_to_cache(cache_key, result)

        return result


@run_sync
def _process_gif_to_grid(gif_base64: str) -> tuple[str, int] | None:
    """
    GIF 转带序号的九宫格拼图
    返回: (base64_str, 实际抽取的帧数)
    """
    try:
        gif_data = base64.b64decode(gif_base64)
        if len(gif_data) > MAX_IMAGE_BYTES:
            logger.warning(f"拒绝过大 GIF: {len(gif_data)} bytes")
            return None
        gif = Image.open(io.BytesIO(gif_data))

        total_frames = gif.n_frames
        if total_frames <= 1:
            return None
        if total_frames > 80:
            logger.warning(f"拒绝帧数过多 GIF: {total_frames}")
            return None

        # --- 策略：根据帧数决定抽多少帧 ---
        # 2-4帧 -> 4 (2x2)；5-6帧 -> 6 (2x3)；7-9帧 -> 9 (3x3)；>9帧 -> 16 (4x4)
        target_count = gif_target_count(total_frames)

        # 计算采样索引 (均匀分布)
        # step = (total - 1) / (target - 1)
        indices = []
        if total_frames <= target_count:
            indices = list(range(total_frames))
        else:
            step = (total_frames - 1) / (target_count - 1)
            indices = [int(i * step) for i in range(target_count)]
            # 去重并排序 (防止计算误差)
            indices = sorted(list(set(indices)))

        # 抽取帧
        selected_frames = []
        for i in indices:
            gif.seek(i)
            # 必须转 RGBA 再转 RGB，处理透明背景
            frame = gif.convert("RGBA")
            bg = Image.new("RGB", frame.size, (255, 255, 255))
            bg.paste(frame, mask=frame.split()[3])
            selected_frames.append(bg)

        if not selected_frames:
            return None

        # --- 拼接逻辑 ---
        real_count = len(selected_frames)
        cols = math.ceil(math.sqrt(real_count))  # 列数
        rows = math.ceil(real_count / cols)  # 行数

        # 调整单帧大小 (兼顾清晰度和总Token)
        # 单帧高度 256px，平衡文字可读性与 token 消耗
        target_h = 256
        w, h = selected_frames[0].size
        if h == 0: return None
        target_w = int((target_h / h) * w)

        resized_frames = [
            f.resize((target_w, target_h), Image.Resampling.LANCZOS) for f in selected_frames
        ]

        # 创建大画布
        grid_w = cols * target_w
        grid_h = rows * target_h
        if grid_w * grid_h > MAX_IMAGE_PIXELS:
            logger.warning(f"拒绝输出尺寸过大 GIF 拼图: {grid_w}x{grid_h}")
            return None
        combined_image = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

        # 尝试加载字体 (如果失败则不画或者画简单的)
        try:
            # 尝试加载默认字体，稍微放大一点
            # PIL 默认字体无法调整大小，所以这里用简单的矩形+默认字体，或者画大一点
            # 为了通用性，这里直接用默认 draw.text，它虽然小但能够着
            font = ImageFont.load_default()
            font_available = True
        except Exception:
            font_available = False

        for idx, frame in enumerate(resized_frames):
            r = idx // cols
            c = idx % cols
            x = c * target_w
            y = r * target_h

            # 贴图
            combined_image.paste(frame, (x, y))

            # --- 标号逻辑 ---
            draw = ImageDraw.Draw(combined_image)
            text = str(idx + 1)

            # 在左上角画一个红色半透明小背景，方便看清数字
            # 矩形位置
            box_w, box_h = 20, 20
            draw.rectangle([x, y, x + box_w, y + box_h], fill=(255, 0, 0))
            # 写数字 (白色)
            draw.text((x + 6, y + 4), text, fill=(255, 255, 255))

            # 画个边框隔开每一帧，避免视觉混淆
            draw.rectangle([x, y, x + target_w, y + target_h], outline=(200, 200, 200), width=2)

        buffer = io.BytesIO()
        combined_image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8"), real_count

    except Exception as e:
        logger.error(f"GIF转九宫格失败: {e}")
        return None


def _configured_max_image_side() -> int:
    raw_value = plugin_config.get("vlm", {}).get("max_image_side", 1280)
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = 1280
    return max(512, min(value, 4096))


@run_sync
def _prepare_native_image_payload(
    image_bytes: bytes,
    *,
    max_side: int,
) -> tuple[bytes, str] | None:
    """Validate and resize a native multimodal payload without semantic analysis."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.width * image.height > MAX_IMAGE_PIXELS:
            logger.warning(f"拒绝像素过大的原生视觉图片: {image.width}x{image.height}")
            return None
        raw_format = (image.format or "JPEG").lower()
        if raw_format == "jpg":
            raw_format = "jpeg"
        if getattr(image, "is_animated", False) and getattr(image, "n_frames", 1) > 1:
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
        logger.warning(f"原生视觉图片预处理失败: {exc}")
        return None


@run_sync
def _calculate_image_hash(image: bytes) -> str:
    return hashlib.sha256(image).hexdigest()


image_manager = ImageManager()


@run_sync
def _clean_old_image_caches_sync():
    """同步清理超过48小时的图片缓存文件"""
    try:
        now = time.time()
        retention_seconds = 48 * 3600
        count = 0

        # 清理生成的 json 文件
        if IMAGE_CACHE_DIR.exists():
            for file_path in IMAGE_CACHE_DIR.glob("*.json"):
                if file_path.is_file() and (now - file_path.stat().st_mtime > retention_seconds):
                    try:
                        file_path.unlink()
                        count += 1
                    except Exception as e:
                        logger.warning(f"删除过期图片缓存JSON失败 {file_path}: {e}")

        # 清理 raw 目录下下载的原始图片
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
            logger.info(f"成功清理 {count} 个超过48小时的旧图片缓存文件。")
    except Exception as e:
        logger.error(f"清理图片缓存任务执行异常: {e}")


async def cleanup_image_cache_task():
    """异步包装器，用于被 APScheduler 调用"""
    logger.info("触发图片缓存自动清理任务...")
    await _clean_old_image_caches_sync()


if scheduler:
    # 每天凌晨 03:00 执行
    scheduler.add_job(
        cleanup_image_cache_task,
        "cron",
        hour=3,
        minute=0,
        id="nyabot_image_cache_cleanup",
        replace_existing=True,
    )
