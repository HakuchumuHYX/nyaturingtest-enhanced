# nyaturingtest/image_manager.py
import base64
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import re

import anyio
from nonebot import logger
import nonebot_plugin_localstore as store
import numpy as np
from PIL import Image, ImageSequence
from nonebot.utils import run_sync

from .config import plugin_config
from .vlm import VLM

IMAGE_CACHE_DIR = Path(f"{store.get_plugin_cache_dir()}/image_cache")


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
            self._vlm = VLM(
                api_key=plugin_config.nyaturingtest_chat_openai_api_key,
                endpoint=plugin_config.nyaturingtest_chat_openai_base_url,
                model=plugin_config.nyaturingtest_chat_openai_model,
            )
            IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self._initialized = True
            self._mem_cache: dict[str, ImageWithDescription] = {}

    def get_from_cache(self, key: str) -> ImageWithDescription | None:
        """从内存缓存或 ID 映射文件中尝试获取描述，避免文件 I/O"""
        return self._mem_cache.get(key)

    def save_to_cache(self, key: str, data: ImageWithDescription):
        """保存到内存缓存"""
        if key and data:
            self._mem_cache[key] = data

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

    async def get_image_description(self, image_base64: str, is_sticker: bool,
                                    cache_key: str | None = None) -> ImageWithDescription | None:
        # 1. 如果提供了 cache_key，先检查内存缓存
        if cache_key and cache_key in self._mem_cache:
            return self._mem_cache[cache_key]

        image_bytes = base64.b64decode(image_base64)
        image_hash = await _calculate_image_hash(image_bytes)

        # 2. 检查基于内容的磁盘缓存
        cache = IMAGE_CACHE_DIR.joinpath(f"{image_hash}.json")
        if cache.exists():
            try:
                async with await anyio.open_file(cache, encoding="utf-8") as f:
                    image_with_desc_raw = await f.read()
                    image_with_desc = ImageWithDescription.from_json(image_with_desc_raw)
                    if image_with_desc.is_sticker != is_sticker:
                        image_with_desc.is_sticker = is_sticker
                        # 更新缓存中的贴纸标记
                        async with await anyio.open_file(cache, "w", encoding="utf-8") as f:
                            await f.write(image_with_desc.to_json())

                    # 命中磁盘缓存后，也更新到内存缓存和 ID 映射
                    if cache_key:
                        self._mem_cache[cache_key] = image_with_desc

                    return image_with_desc
            except ValueError:
                logger.warning(f"缓存损坏 {cache}，将重新生成")
                cache.unlink(missing_ok=True)

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image_format = image.format or "JPEG"
        except Exception:
            logger.error("无法识别的图片格式")
            return None

        target_image_base64 = image_base64
        target_format = image_format

        code_mark = "```"
        base_prompt = f"""请分析这张图片。
1. 用中文详细描述图片内容（'description'），如果是表情包请把上面的文字也识别出来，最多100字。
2. 分析图片表达的情感（'emotion'），格式为'情感, 类型, 含义'，例如"开心, 表情包, 嘲讽"。
请直接输出纯 JSON 格式，不要包含其他内容：
{code_mark}json
{{
    "description": "...",
    "emotion": "..."
}}
{code_mark}
"""

        if getattr(image, "is_animated", False):
            gif_transfromed = await _transform_gif(image_base64)
            if gif_transfromed:
                prompt = "这是一个动态图的拼接帧（每一张图代表某一帧，黑色背景代表透明）。" + base_prompt
                target_image_base64 = gif_transfromed
                target_format = "jpeg"
            else:
                prompt = base_prompt
        else:
            prompt = base_prompt

        response = await self._vlm.request(
            prompt=prompt,
            image_base64=target_image_base64,
            image_format=target_format,
            # 这里传入 extra_body，vlm.py 修改后会透传给 API
            extra_body={
                "top_k": 64,  # Gemini 识图也需要这个来保证稳定性
                "google": {
                    "model_safety_settings": {
                        "enabled": False  # 关掉识图的安全过滤，防止把用户的普通表情包误杀
                    }
                }
            }
        )

        if not response:
            return None

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

        # 写入磁盘缓存
        async with await anyio.open_file(cache, "w", encoding="utf-8") as f:
            await f.write(result.to_json())

        # 写入内存缓存
        if cache_key:
            self._mem_cache[cache_key] = result

        return result


@run_sync
def _transform_gif(gif_base64: str, similarity_threshold: float = 1000.0, max_frames: int = 15) -> str | None:
    """
    将GIF转换为水平拼接的静态图像，流式读取防止内存溢出
    """
    try:
        gif_data = base64.b64decode(gif_base64)
        gif = Image.open(io.BytesIO(gif_data))

        selected_frames = []
        last_selected_array = None

        # 使用 Iterator 遍历，避免一次性加载所有帧
        # enumerate 确保我们可以获取第一帧
        for i, frame in enumerate(ImageSequence.Iterator(gif)):
            # 必须 copy，因为 frame 是迭代器生成的临时对象
            current_frame = frame.convert("RGB")

            should_keep = False
            if i == 0:
                should_keep = True
            elif last_selected_array is not None:
                current_array = np.array(current_frame)
                mse = np.mean((current_array - last_selected_array) ** 2)
                if mse > similarity_threshold:
                    should_keep = True

            if should_keep:
                # 只有决定保留时才将图片数据存入内存列表
                selected_frames.append(current_frame.copy())
                last_selected_array = np.array(current_frame)

                if len(selected_frames) >= max_frames:
                    break

        if not selected_frames:
            return None

        # 拼接逻辑
        frame_width, frame_height = selected_frames[0].size
        target_height = 200
        if frame_height == 0: return None
        target_width = int((target_height / frame_height) * frame_width)
        if target_width == 0: target_width = 1

        resized_frames = [
            f.resize((target_width, target_height), Image.Resampling.LANCZOS) for f in selected_frames
        ]

        total_width = target_width * len(resized_frames)
        combined_image = Image.new("RGB", (total_width, target_height))

        for idx, frame in enumerate(resized_frames):
            combined_image.paste(frame, (idx * target_width, 0))

        buffer = io.BytesIO()
        combined_image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    except Exception as e:
        logger.error(f"GIF转换失败: {e}")
        return None


@run_sync
def _calculate_image_hash(image: bytes) -> str:
    sha256_hash = hashlib.md5(image).hexdigest()
    return sha256_hash


image_manager = ImageManager()
