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
from PIL import Image
from nonebot.utils import run_sync

from .config import plugin_config
from .vlm import SiliconFlowVLM

IMAGE_CACHE_DIR = Path(f"{store.get_plugin_cache_dir()}/image_cache")


@dataclass
class ImageWithDescription:
    """
    图片和描述
    """

    description: str
    """
    图像内容简述
    """
    emotion: str
    """
    图像情感关键词
    """
    is_sticker: bool = False
    """
    是否是贴图
    """

    def to_json(self) -> str:
        """
        将对象转换为JSON字符串
        """
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(json_str: str) -> "ImageWithDescription":
        """
        从JSON字符串转换为对象，错误时抛出
        """
        image_with_desc = ImageWithDescription("", "", False)
        data = json.loads(json_str)
        # 检查数据完整性
        if not all(key in data for key in ["description", "emotion", "is_sticker"]):
            raise ValueError("缺少必要的字段")
        image_with_desc.description = data["description"]
        image_with_desc.emotion = data["emotion"]
        image_with_desc.is_sticker = data["is_sticker"]
        return image_with_desc


class ImageManager:
    """
    图片管理
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            # [关键修改] 使用账单中确认可用的模型，确保稳定且支持 JSON 指令
            self._vlm = SiliconFlowVLM(
                api_key=plugin_config.nyaturingtest_siliconflow_api_key,
                model="Qwen/Qwen3-VL-32B-Instruct",
            )
            IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self._initialized = True

    def _extract_json(self, response: str) -> dict | None:
        """提取 JSON 的辅助函数"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        # 尝试提取 markdown 代码块
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # 尝试提取最外层 {}
        match = re.search(r"(\{[\s\S]*\})", response)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return None

    async def get_image_description(self, image_base64: str, is_sticker: bool) -> ImageWithDescription | None:
        """
        获取图片描述
        """
        image_bytes = base64.b64decode(image_base64)
        # 计算图片的SHA256哈希值
        image_hash = await _calculate_image_hash(image_bytes)
        # 检查缓存
        cache = IMAGE_CACHE_DIR.joinpath(f"{image_hash}.json")
        if cache.exists():
            async with await anyio.open_file(cache, encoding="utf-8") as f:
                image_with_desc_raw = await f.read()
                try:
                    image_with_desc = ImageWithDescription.from_json(image_with_desc_raw)
                    if image_with_desc.is_sticker != is_sticker:
                        image_with_desc.is_sticker = is_sticker
                        # 修改缓存文件
                        async with await anyio.open_file(cache, "w", encoding="utf-8") as f:
                            await f.write(image_with_desc.to_json())
                    return image_with_desc
                except ValueError as e:
                    logger.error(f"缓存文件({cache})格式错误，重新生成")
                    logger.error(e)
                    cache.unlink(missing_ok=True)  # 删除缓存文件

        # 获取图片描述
        # 获取图片类型
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image_format = image.format
        except Exception:
            logger.error("无法识别的图片格式")
            return None

        if not image_format:
            image_format = "JPEG"  # Fallback

        target_image_base64 = image_base64
        target_format = image_format

        # [优化] 统一提示词逻辑，一次调用获取两个结果
        # 使用变量拼接防止 markdown 解析错误
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

        if image_format.upper() == "GIF":
            gif_transfromed = await _transform_gif(image_base64)
            if not gif_transfromed:
                logger.error("GIF转换失败")
                return None

            # GIF 提示词微调
            prompt = "这是一个动态图的拼接帧（每一张图代表某一帧，黑色背景代表透明）。" + base_prompt
            target_image_base64 = gif_transfromed
            target_format = "jpeg"
        else:
            prompt = base_prompt

        # [修改 2] 只请求一次 VLM，大幅提升速度
        response = await self._vlm.request(
            prompt=prompt,
            image_base64=target_image_base64,
            image_format=target_format,
        )

        if not response:
            logger.error("VLM请求失败")
            return None

        # 解析 JSON
        data = self._extract_json(response)

        if not data:
            logger.warning(f"VLM 返回了非 JSON 格式: {response}，尝试降级处理")
            # 降级处理：整个作为描述，情感设为未知
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

        # 缓存结果
        async with await anyio.open_file(cache, "w", encoding="utf-8") as f:
            await f.write(result.to_json())

        return result


@run_sync
def _transform_gif(gif_base64: str, similarity_threshold: float = 1000.0, max_frames: int = 15) -> str | None:
    """将GIF转换为水平拼接的静态图像, 跳过相似的帧"""
    try:
        # 解码base64
        gif_data = base64.b64decode(gif_base64)
        gif = Image.open(io.BytesIO(gif_data))

        # 收集所有帧
        all_frames = []
        try:
            while True:
                gif.seek(len(all_frames))
                # 确保是RGB格式方便比较
                frame = gif.convert("RGB")
                all_frames.append(frame.copy())
        except EOFError:
            pass  # 读完啦

        if not all_frames:
            logger.warning("GIF中没有找到任何帧")
            return None  # 空的GIF直接返回None

        # --- 新的帧选择逻辑 ---
        selected_frames = []
        last_selected_frame_np = None

        for i, current_frame in enumerate(all_frames):
            current_frame_np = np.array(current_frame)

            # 第一帧总是要选的
            if i == 0:
                selected_frames.append(current_frame)
                last_selected_frame_np = current_frame_np
                continue

            # 计算和上一张选中帧的差异（均方误差 MSE）
            if last_selected_frame_np is not None:
                mse = np.mean((current_frame_np - last_selected_frame_np) ** 2)
                if mse > similarity_threshold:
                    selected_frames.append(current_frame)
                    last_selected_frame_np = current_frame_np
                    if len(selected_frames) >= max_frames:
                        break

        if not selected_frames:
            logger.warning("处理后没有选中任何帧")
            return None

        # 获取选中的第一帧的尺寸
        frame_width, frame_height = selected_frames[0].size

        # 计算目标尺寸，保持宽高比
        target_height = 200  # 固定高度
        if frame_height == 0: return None
        target_width = int((target_height / frame_height) * frame_width)
        if target_width == 0: target_width = 1

        # 调整所有选中帧的大小
        resized_frames = [
            frame.resize((target_width, target_height), Image.Resampling.LANCZOS) for frame in selected_frames
        ]

        # 创建拼接图像
        total_width = target_width * len(resized_frames)
        if total_width == 0: return None

        combined_image = Image.new("RGB", (total_width, target_height))

        # 水平拼接图像
        for idx, frame in enumerate(resized_frames):
            combined_image.paste(frame, (idx * target_width, 0))

        # 转换为base64
        buffer = io.BytesIO()
        combined_image.save(buffer, format="JPEG", quality=85)
        result_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return result_base64

    except MemoryError:
        logger.error("GIF转换失败: 内存不足")
        return None
    except Exception as e:
        logger.error(f"GIF转换失败: {e}")
        return None


@run_sync
def _calculate_image_hash(image: bytes) -> str:
    """
    计算图片的SHA256哈希值
    """
    sha256_hash = hashlib.md5(image).hexdigest()
    return sha256_hash


image_manager = ImageManager()