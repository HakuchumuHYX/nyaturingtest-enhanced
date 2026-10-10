"""QQ 系统表情目录。

发表情、贴表情和收到的表情转文字都按 SnowLuma 的 fetch_sys_faces 来：
SnowLuma 发 face 段时查的是同一份目录，目录里没有的编号会直接发送失败，所以不另写白名单。
"""

from dataclasses import dataclass

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot

# 连着发的拼接部件，单独发没有意义
_COMBO_PARTS = {"中龙舟", "大龙舟", "新年中龙", "新年大龙", "中火车", "大火车", "蛇身", "蛇尾"}


@dataclass(frozen=True)
class Face:
    id: str
    name: str
    big: bool


@dataclass(frozen=True)
class Reaction:
    target_id: str
    emoji_id: str
    name: str


@dataclass(frozen=True)
class FaceCatalog:
    sendable: dict[str, Face]
    # 编号 → 名字，含不能发的拼接部件，收到时转文字用
    names: dict[str, str]
    # emoji 字符 → 十进制码点，只能用来贴表情
    emojis: dict[str, str]
    small_line: str
    big_line: str
    emoji_line: str


_catalog: FaceCatalog | None = None


def _build_catalog(packs: list[dict]) -> FaceCatalog:
    sendable: dict[str, Face] = {}
    names: dict[str, str] = {}
    emojis: dict[str, str] = {}
    aliases: list[tuple[str, Face]] = []
    for pack in packs:
        for entry in pack["emojis"]:
            face_id = entry["q_sid"]
            name = entry["q_des"].lstrip("/")
            # emoji 分组的 q_sid 是字符本身，不能当 face 发，只能拿码点贴表情
            if not face_id.isdigit():
                emojis[face_id] = str(ord(face_id))
                continue
            # 同一编号会出现在两个分组里
            if face_id in names:
                continue
            names[face_id] = name
            # 微笑、撇嘴等 8 个名字还有一份新版（450–457），同样是小表情，发送只留经典版
            if name in _COMBO_PARTS or name in sendable:
                continue
            face = Face(id=face_id, name=name, big=entry["is_super"])
            sendable[name] = face
            aliases.extend((alias, face) for alias in entry["emoji_name_alias"])
    # 别名和正式名字冲突时以正式名字为准（「害羞」是 466 的别名，但首先是 6 的名字）
    listed = list(sendable.values())
    for alias, face in aliases:
        sendable.setdefault(alias, face)
    return FaceCatalog(
        sendable=sendable,
        names=names,
        emojis=emojis,
        small_line="、".join(face.name for face in listed if not face.big),
        big_line="、".join(face.name for face in listed if face.big),
        emoji_line="".join(emojis),
    )


async def ensure_faces(bot: Bot) -> None:
    """第一次用时拉一次目录，进程内常驻；失败就等下一条消息再试。"""

    global _catalog
    if _catalog is not None:
        return
    try:
        result = await bot.call_api("fetch_sys_faces")
    except Exception as e:
        logger.warning(f"获取 QQ 表情目录失败: {e}")
        return
    _catalog = _build_catalog(result["packs"])
    logger.info(
        f"QQ 表情目录已加载: 可发 {len(_catalog.sendable)} 个名字, emoji {len(_catalog.emojis)} 个"
    )


def _clean(name: str) -> str:
    # 模型可能照抄输入里的 [表情:捂脸] 或 QQ 的 /捂脸 写法
    name = name.strip().strip("[]/").strip()
    return name.removeprefix("表情:").removeprefix("表情：").strip()


def face_name(face_id: str) -> str:
    """收到的表情编号 → 名字；超过 3 位的是 emoji 码点。查不到返回空字符串。"""

    if len(face_id) > 3 and face_id.isdigit():
        return chr(int(face_id))
    if _catalog is None:
        return ""
    return _catalog.names.get(face_id, "")


def sendable_face(name: str) -> Face | None:
    if _catalog is None:
        return None
    return _catalog.sendable.get(_clean(name))


def reaction_emoji(name: str) -> tuple[str, str] | None:
    """贴表情用的 (emoji_id, 名字)：系统表情给编号，emoji 字符给码点。"""

    if _catalog is None:
        return None
    name = _clean(name).replace("️", "")
    face = _catalog.sendable.get(name)
    if face is not None:
        return face.id, face.name
    if name in _catalog.emojis:
        return _catalog.emojis[name], name
    return None


def face_lines() -> tuple[str, str, str]:
    """给模型的小表情、大表情、emoji 三行名单；目录没加载时都是空字符串。"""

    if _catalog is None:
        return "", "", ""
    return _catalog.small_line, _catalog.big_line, _catalog.emoji_line
