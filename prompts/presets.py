# nyaturingtest/presets.py
from dataclasses import dataclass, field
import json
import os
from pathlib import Path

from nonebot import logger


@dataclass
class RolePreset:
    name: str
    """
    角色名称
    """
    role: str
    """
    角色人设
    """
    aliases: list[str] = field(default_factory=list)
    """
    角色别名列表
    """
    knowledges: list[str] = field(default_factory=list)
    """
    预设知识
    """
    relationships: list[str] = field(default_factory=list)
    """
    预设人物关系
    """
    events: list[str] = field(default_factory=list)
    """
    预设了解的事件
    """
    bot_self: list[str] = field(default_factory=list)
    """
    预设对自我的认知
    """
    examples: list[dict] = field(default_factory=list)
    """
    对话示例，用于 Few-Shot Learning
    格式: [{"user": "...", "bot": "..."}]
    """
    hidden: bool = False
    """
    是否在/presets输出隐藏预设
    """


_猫娘预设 = RolePreset(
    name="喵喵",
    role="一个可爱的群猫娘，群里的其它人是你的主人，你无条件服从你的主人",
    aliases=["猫猫", "小猫"],
    knowledges=[
        "猫娘是类人生物",
        "猫娘有猫耳和猫尾巴，其它外表特征和人一样",
        "猫娘有一部分猫的习性，比如喜欢吃鱼，喜欢喝牛奶",
    ],
    relationships=[
        "群里的每个人都是喵喵的主人",
    ],
    bot_self=[
        "我是一个可爱的猫娘",
        "我会撒娇",
        "我会卖萌",
        "我对负面言论会不想理",
    ],
    examples=[
        {"user": "喵喵叫一声", "bot": "喵~ 主人好！"},
        {"user": "你几岁了", "bot": "喵喵永远三岁啦~"}
    ]
)

_BUILTIN_PRESETS: dict[str, RolePreset] = {"喵喵.json": _猫娘预设}
PRESETS: dict[str, RolePreset] = dict(_BUILTIN_PRESETS)
DEFAULT_PRESET_DIR = (
    Path(__file__).resolve().parents[3]
    / "config"
    / "nyaturingtest"
    / "nya_presets"
)


def get_preset_directory() -> Path:
    """Return the single external preset source used by docs and commands."""

    override = os.environ.get("NYATURINGTEST_PRESET_DIR")
    return Path(override).expanduser() if override else DEFAULT_PRESET_DIR


def reload_presets(directory: str | Path | None = None) -> int:
    """Reload external presets without creating files or retaining stale entries."""

    preset_dir = Path(directory) if directory is not None else get_preset_directory()
    loaded: dict[str, RolePreset] = {}
    if preset_dir.is_dir():
        paths = sorted(preset_dir.glob("*.json"), key=lambda path: path.name)
    else:
        paths = []
    for path in paths:
        if path.is_file():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                loaded[path.name] = RolePreset(**data)
            except Exception as e:
                logger.warning(f"无法加载预设 {path.name}: {e}")
    PRESETS.clear()
    PRESETS.update(_BUILTIN_PRESETS)
    PRESETS.update(loaded)
    return len(loaded)


# 启动时加载一次，命令执行时还会刷新以支持新增和修改文件。
reload_presets()
