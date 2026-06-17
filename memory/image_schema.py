# nyaturingtest/memory/image_schema.py
"""
VLM 图像描述的纯结构化逻辑层。

本模块刻意不依赖 nonebot / localstore / config / VLM 客户端，仅包含：
- `ImageWithDescription` 数据结构（多正交槽 + 旧字段兼容）
- VLM 响应解析与逐字段缺省降级 (`parse_vlm_response`)
- 渲染为下游可读的稳定管道标签 (`render_image_text`)
- 结构化元数据导出 (`to_meta`)

这样该模块可被单元测试直接 import，不触发 NoneBot 初始化。
`memory/image.py` 的 ImageManager 从本模块复用这些逻辑。
"""
from dataclasses import dataclass, field
import json
import re


# 语用意图封闭标签集（prompt 要求模型从中选一个；非集合内值时降级为 "无"）
PRAGMATIC_INTENTS = (
    "嘲讽", "自嘲", "附和", "破冰", "卖萌",
    "终结话题", "否认", "求助", "感叹", "无",
)

# 实体类型封闭集
ENTITY_TYPES = ("character", "real_person", "meme", "brand", "object")


def gif_target_count(total_frames: int) -> int:
    """
    GIF 抽帧数策略（纯函数，便于单测）：
      2-4 帧 -> 4 (2x2)
      5-6 帧 -> 6 (2x3)
      7-9 帧 -> 9 (3x3)
      >9 帧  -> 16 (4x4)
    调用方负责先过滤 <=1 帧 / >80 帧的情况。
    """
    if total_frames <= 4:
        return 4
    if total_frames <= 6:
        return 6
    if total_frames <= 9:
        return 9
    return 16


def _clamp(value, lo, hi, default=0.0):
    """把可能是 None/非法的数值夹到 [lo, hi]，失败取 default。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if v != v:  # NaN
        return float(default)
    return max(lo, min(hi, v))


def _extract_json(response: str) -> dict | None:
    """
    三级容错解析 JSON：
    1. 直接 json.loads
    2. ```json ... ``` 围栏
    3. 正则抠首个 {...}
    """
    if not response:
        return None
    try:
        return json.loads(response)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, TypeError):
            pass
    match = re.search(r"(\{[\s\S]*\})", response)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _normalize_entities(raw) -> list[dict]:
    """规范化 entities：过滤无 name 的项，补全 type，裁剪 confidence。"""
    if not isinstance(raw, list):
        return []
    out = []
    for e in raw:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name") or "").strip()
        if not name:
            continue
        etype = str(e.get("type") or "object").strip().lower()
        if etype not in ENTITY_TYPES:
            etype = "object"
        confidence = _clamp(e.get("confidence"), 0.0, 1.0, 0.0)
        out.append({"name": name, "type": etype, "confidence": round(confidence, 3)})
    return out


def _normalize_affect(raw) -> dict:
    """规范化 affect 为 VAD 三元组。"""
    if not isinstance(raw, dict):
        raw = {}
    return {
        "valence": round(_clamp(raw.get("valence"), -1.0, 1.0, 0.0), 3),
        "arousal": round(_clamp(raw.get("arousal"), 0.0, 1.0, 0.0), 3),
        "dominance": round(_clamp(raw.get("dominance"), -1.0, 1.0, 0.0), 3),
    }


def _normalize_temporal(raw) -> list[dict]:
    """规范化 temporal（GIF 帧动作序列）。"""
    if not isinstance(raw, list):
        return []
    out = []
    for t in raw:
        if not isinstance(t, dict):
            continue
        action = str(t.get("action") or "").strip()
        if not action:
            continue
        try:
            frame = int(t.get("frame") or 0)
        except (TypeError, ValueError):
            frame = 0
        out.append({"frame": frame, "action": action})
    return out


@dataclass
class ImageWithDescription:
    # 新正交槽
    visual_description: str = ""        # 纯视觉描述（原 description 本职，≤60字）
    ocr_text: str = ""                  # 图内文字，单列
    entities: list = field(default_factory=list)
        # [{name, type∈ENTITY_TYPES, confidence∈[0,1]}]；模型自身知识识别；不认识=空
    pragmatic_intent: str = "无"        # 语用功能，封闭标签集 PRAGMATIC_INTENTS
    affect: dict = field(default_factory=lambda: {"valence": 0.0, "arousal": 0.0, "dominance": 0.0})
    is_sticker: bool = False
    temporal: list = field(default_factory=list)   # 仅 GIF：[{frame, action}]
    # 旧字段别名（兼容旧磁盘缓存 JSON）
    description: str = ""               # = visual_description
    emotion: str = ""                   # 旧三段式，仅 from_json 读旧缓存用

    def to_meta(self) -> dict:
        """返回纯结构化元数据（供 Message.image_meta），不含自由文本。"""
        return {
            "entities": list(self.entities),
            "ocr_text": self.ocr_text,
            "pragmatic_intent": self.pragmatic_intent,
            "affect": dict(self.affect),
            "temporal": list(self.temporal),
            "is_sticker": self.is_sticker,
        }

    def to_json(self) -> str:
        return json.dumps(
            {
                "visual_description": self.visual_description,
                "ocr_text": self.ocr_text,
                "entities": self.entities,
                "pragmatic_intent": self.pragmatic_intent,
                "affect": self.affect,
                "is_sticker": self.is_sticker,
                "temporal": self.temporal,
                # 旧字段别名，保证旧消费方/旧缓存可读
                "description": self.visual_description,
                "emotion": self.emotion,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def from_json(json_str: str) -> "ImageWithDescription":
        """从 JSON 反序列化，兼容旧缓存（只有 description/emotion/is_sticker）。"""
        try:
            data = json.loads(json_str)
            if not isinstance(data, dict):
                raise ValueError("not a dict")
        except Exception:
            raise ValueError("JSON解析失败")

        # 新字段优先；缺省时从旧字段回填或用默认
        visual = str(data.get("visual_description") or data.get("description") or "")
        ocr_text = str(data.get("ocr_text") or "")
        entities = _normalize_entities(data.get("entities"))
        pragmatic_intent = str(data.get("pragmatic_intent") or "无")
        if pragmatic_intent not in PRAGMATIC_INTENTS:
            pragmatic_intent = "无"
        affect = _normalize_affect(data.get("affect"))
        temporal = _normalize_temporal(data.get("temporal"))
        is_sticker = bool(data.get("is_sticker", False))
        # 旧 emotion 字段（仅保留用于读旧缓存，新数据不再写入）
        emotion = str(data.get("emotion") or "")

        return ImageWithDescription(
            visual_description=visual,
            ocr_text=ocr_text,
            entities=entities,
            pragmatic_intent=pragmatic_intent,
            affect=affect,
            is_sticker=is_sticker,
            temporal=temporal,
            description=visual,
            emotion=emotion,
        )


def parse_vlm_response(response: str, is_sticker: bool = False) -> ImageWithDescription:
    """
    把 VLM 原始返回解析为 ImageWithDescription，逐字段缺省降级。
    完全解析失败也不抛错，退化为截断文本 + 默认值。
    """
    data = _extract_json(response) or {}

    visual = str(data.get("visual_description") or data.get("description") or "").strip()
    if not visual:
        # 解析失败兜底：截取原始响应前 60 字
        visual = (response or "")[:60].strip()

    ocr_text = str(data.get("ocr_text") or "")

    entities = _normalize_entities(data.get("entities"))

    pragmatic_intent = str(data.get("pragmatic_intent") or "无").strip()
    if pragmatic_intent not in PRAGMATIC_INTENTS:
        pragmatic_intent = "无"

    affect = _normalize_affect(data.get("affect"))
    temporal = _normalize_temporal(data.get("temporal"))

    return ImageWithDescription(
        visual_description=visual,
        ocr_text=ocr_text,
        entities=entities,
        pragmatic_intent=pragmatic_intent,
        affect=affect,
        is_sticker=is_sticker,
        temporal=temporal,
        description=visual,
    )


def _fmt_entities(entities: list[dict]) -> str:
    if not entities:
        return ""
    parts = []
    for e in entities:
        name = e.get("name", "")
        conf = e.get("confidence", 0.0)
        parts.append(f"{name}({conf:g})")
    return ",".join(parts)


def _fmt_affect(affect: dict) -> str:
    return f"V{affect.get('valence', 0):.2f},A{affect.get('arousal', 0):.2f},D{affect.get('dominance', 0):.2f}"


def _fmt_temporal(temporal: list[dict]) -> str:
    if not temporal:
        return ""
    return ";".join(f"{t.get('frame', 0)}.{t.get('action', '')}" for t in temporal)


def merge_segment_metas(segment_metas: list) -> dict | None:
    """
    合并各消息段产出的图片元数据为一条消息的 image_meta。
    输入：每段返回的 meta（None 或 dict）。主消息图片 meta 为「裸」dict（含 entities/ocr_text/...），
    引用消息段返回 {"referenced": [...]}。
    输出：
      - 无任何图片 meta -> None
      - 有主图 -> {"primary": <首张非空裸 meta>, "referenced": [...]}（referenced 可缺省）
      - 仅引用图 -> {"referenced": [...]}
    """
    primary_metas = []
    referenced = []
    for m in segment_metas:
        if not isinstance(m, dict):
            continue
        if "referenced" in m:
            ref = m.get("referenced")
            if isinstance(ref, list):
                referenced.extend(ref)
        else:
            # 裸 meta（主消息图片）
            primary_metas.append(m)

    if not primary_metas and not referenced:
        return None
    out: dict = {}
    if primary_metas:
        out["primary"] = primary_metas[0]
    if referenced:
        out["referenced"] = referenced
    return out


def render_image_text(desc: ImageWithDescription, is_sticker: bool) -> str:
    """
    渲染为下游可读的稳定管道标签。
    格式：[表情包|实体:..|配字:..|意图:..|情感:V..,A..,D..|画面:..]  （动图追加 |动作:..）
    空槽位保留键但留空值，便于下游稳定解析。
    """
    prefix = "表情包" if is_sticker else "图片"
    segments = [
        f"实体:{_fmt_entities(desc.entities)}",
        f"配字:{desc.ocr_text}",
        f"意图:{desc.pragmatic_intent}",
        f"情感:{_fmt_affect(desc.affect)}",
        f"画面:{desc.visual_description}",
    ]
    temporal_str = _fmt_temporal(desc.temporal)
    if temporal_str:
        segments.append(f"动作:{temporal_str}")
    return f"\n[{prefix}|{'|'.join(segments)}]\n"
