# nyaturingtest/image_manager.py
import asyncio
import base64
from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import io
import json
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
)
from ..llm.vlm import VLM
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

IMAGE_CACHE_DIR = Path(f"{store.get_plugin_cache_dir()}/image_cache")
_IMG_SEMAPHORE = asyncio.Semaphore(3)


@dataclass
class ImageWithDescription:
    description: str
    emotion: str
    is_sticker: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(json_str: str) -> "ImageWithDescription":
        image_with_desc = ImageWithDescription("", "", False)
        try:
            data = json.loads(json_str)
            if not all(key in data for key in ["description", "emotion", "is_sticker"]):
                raise ValueError("缺少必要的字段")
            image_with_desc.description = data["description"]
            image_with_desc.emotion = data["emotion"]
            image_with_desc.is_sticker = data["is_sticker"]
            return image_with_desc
        except Exception:
            raise ValueError("JSON解析失败")


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
            vlm_provider = plugin_config.get("vlm", {}).get("provider", "openai_compatible").strip().lower()

            self._vlm = VLM(
                api_key=get_effective_vlm_api_key(),
                endpoint=get_effective_vlm_base_url(),
                model=get_effective_vlm_model(),
                provider=vlm_provider,
                timeout=int(plugin_config.get("vlm", {}).get("timeout") or 60),
            )
            IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self._initialized = True
            self._mem_cache: OrderedDict[str, tuple[float, ImageWithDescription]] = OrderedDict()

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

    def _extract_json(self, response: str) -> dict | None:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        match = re.search(r"(\{[\s\S]*\})", response)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return None

    async def resolve_image_from_url(self, url: str, file_unique: str, is_sticker: bool, context_text: str = "",
                                     on_usage: Callable[[dict], None] | None = None) -> str:
        """
        高层接口：下载并分析图片，返回格式化的描述文本
        """
        if not url:
            return "[无效图片]"

        async with _IMG_SEMAPHORE:
            try:
                # 1. 尝试从内存缓存获取
                if file_unique:
                    cached_desc = self.get_from_cache(file_unique)
                    if cached_desc:
                        if is_sticker:
                            return f"\n[表情包] [情感:{cached_desc.emotion}] [内容:{cached_desc.description}]\n"
                        else:
                            return f"\n[图片] {cached_desc.description}\n"

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
                                return "\n[图片类型不支持]\n"
                            if len(resp.content) > MAX_IMAGE_BYTES:
                                logger.warning(f"拒绝过大图片: {len(resp.content)} bytes")
                                return "\n[图片过大]\n"
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
                    return "\n[图片下载失败]\n"

                # 5. 调用 VLM 进行识别
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                description = await self.get_image_description(
                    image_base64=image_base64, is_sticker=is_sticker, cache_key=file_unique,
                    context_text=context_text, on_usage=on_usage
                )

                if description:
                    if is_sticker:
                        return f"\n[表情包] [情感:{description.emotion}] [内容:{description.description}]\n"
                    else:
                        return f"\n[图片] {description.description}\n"
                return "\n[图片识别无结果]\n"

            except Exception as e:
                logger.error(f"Image resolve error: {e}")
                return "\n[图片处理出错]\n"

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

        cache = IMAGE_CACHE_DIR.joinpath(f"{image_hash}.json")
        if cache.exists():
            try:
                async with await anyio.open_file(cache, encoding="utf-8") as f:
                    image_with_desc = ImageWithDescription.from_json(await f.read())
                    if image_with_desc.is_sticker != is_sticker:
                        image_with_desc.is_sticker = is_sticker
                        async with await anyio.open_file(cache, "w", encoding="utf-8") as f:
                            await f.write(image_with_desc.to_json())
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
        
        # 构建基础 Prompt
        base_prompt = "请分析这张图片。\n"
        if context_text:
            base_prompt += f"【背景信息】这张图片是在对话中发送的，相关文本内容是：“{context_text}”。请结合此背景理解图片的含义。\n"
            
        base_prompt += f"""1. 用中文详细描述图片内容（'description'），如果是表情包请把上面的文字也识别出来，最多100字。
2. 分析图片表达的情感（'emotion'），格式为'情感, 类型, 含义'，例如"开心, 表情包, 嘲讽"。
请直接输出纯 JSON 格式，不要包含其他内容：
{code_mark}json
{{
    "description": "...",
    "emotion": "..."
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
                    # 更新 Prompt，告诉 LLM 这是一张拼图
                    prompt = (
                                 f"这是一张包含 {frame_count} 个关键帧的动图分解拼图。"
                                 "图片左上角标有数字序号（1, 2, 3...），代表时间顺序。"
                                 "请结合这些关键帧，分析这个动图发生了什么动作或情节。"
                             ) + base_prompt
                    target_image_base64 = grid_b64
                    target_format = "jpeg"
                else:
                    # 如果处理失败，降级为第一帧
                    target_format = "jpeg"

        # === 4. 格式最终清洗 + 静态图压缩 ===
        try:
            img = Image.open(io.BytesIO(base64.b64decode(target_image_base64)))
            max_side = 512
            w, h = img.size
            if max(w, h) > max_side:
                ratio = max_side / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
            if img.mode != "RGB":
                img = img.convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            target_image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            target_format = "jpeg"
        except Exception as e:
            logger.error(f"图片压缩/格式转换失败: {e}")
            if target_format not in ["jpeg", "png", "webp"]:
                return None

        # 5. 发送请求
        response = await self._vlm.request(
            prompt=prompt,
            image_base64=target_image_base64,
            image_format=target_format,
            on_usage=on_usage,
            response_format={"type": "json_object"},
        )

        if not response:
            metrics.vlm_failure += 1
            return None
        metrics.vlm_success += 1

        data = self._extract_json(response)
        if not data:
            description = response[:100]
            emotion = "未知, 图片, 未知"
        else:
            description = data.get("description", "图片识别失败")
            emotion = data.get("emotion", "未知, 图片, 未知")

        result = ImageWithDescription(
            description=description,
            emotion=emotion,
            is_sticker=is_sticker,
        )

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
        # 规则：
        # 2-4帧 -> 全取 (2x2)
        # 5-6帧 -> 取 6 帧 (2x3)
        # 7-9帧 -> 取 9 帧 (3x3)
        # >9帧  -> 均匀抽取 16 帧 (4x4)

        target_count = 9
        if total_frames <= 4:
            target_count = 4
        elif total_frames <= 6:
            target_count = 6
        else:
            target_count = 9

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


@run_sync
def _calculate_image_hash(image: bytes) -> str:
    sha256_hash = hashlib.md5(image).hexdigest()
    return sha256_hash


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
