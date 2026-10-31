"""合并转发：后台读完文字和图、写成摘要，再交给主流程。

截断加不读图会丢信息，放进一轮对话里读又会拖住这一轮，所以单独读：
占位消息照常进队列（保住先后顺序），读完原地改写成摘要，再把同一条消息放回队列触发一轮，
像真人先看到有个转发、点开读完再说话。
"""

import time
from dataclasses import dataclass, field
from datetime import datetime

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, Event
from nonebot.adapters.onebot.v11.exception import ActionFailed
from nonebot.exception import NetworkError

from ..chat.groups import GroupState
from ..llm import VisionInput, feedback_client
from ..metrics import log_event
from ..prompts import get_forward_prompt
from .convert import FORWARD_PLACEHOLDER, is_sticker_segment, segment_text
from .image import fetch_image_input
from .message import ChatMessage

# 上限只为控制单次请求的大小；超出的在提示词里注明，不悄悄丢
FORWARD_MAX_CHARS = 8000
FORWARD_MAX_IMAGES = 20
# 嵌套转发展开到第二层，再往里只留占位
FORWARD_MAX_DEPTH = 2
FORWARD_TEMPERATURE = 0.1
FORWARD_NOTE = "这是转发的聊天记录，不要将其认定为发送者的事实"


@dataclass
class _Expansion:
    lines: list[str] = field(default_factory=list)
    images: list[VisionInput] = field(default_factory=list)
    chars: int = 0
    skipped_nodes: int = 0
    skipped_images: int = 0


async def _node_text(bot: Bot, node: dict, depth: int, out: _Expansion) -> str:
    parts = []
    for seg in node.get("message") or []:
        seg_type, data = seg.get("type"), seg.get("data") or {}
        if seg_type == "forward" and depth < FORWARD_MAX_DEPTH:
            indent = "  " * (depth - 1)
            out.lines.append(f"{indent}{_node_name(node)}：转发了一份聊天记录：")
            try:
                await _expand(bot, str(data.get("id") or ""), depth + 1, out)
            except (ActionFailed, NetworkError) as e:
                logger.warning(f"[Forward] 拉取嵌套转发失败，留占位: {e}")
                out.lines.append(f"{indent}  {FORWARD_PLACEHOLDER}")
            continue
        text = segment_text(seg_type, data)
        if text is not None:
            parts.append(text)
        elif seg_type == "image":
            parts.append(await _image_text(data, out))
    return "".join(parts).strip()


async def _image_text(data: dict, out: _Expansion) -> str:
    if len(out.images) >= FORWARD_MAX_IMAGES:
        out.skipped_images += 1
        return "[图片]"
    ref_id = f"图{len(out.images) + 1}"
    text, vision_input, _ = await fetch_image_input(
        data.get("url", ""),
        data.get("file_unique", ""),
        is_sticker=is_sticker_segment(data),
        ref_id=ref_id,
    )
    if vision_input is None:
        return text
    out.images.append(vision_input)
    return f"[{ref_id}]"


def _node_name(node: dict) -> str:
    sender = node.get("sender") or {}
    return str(
        sender.get("card") or sender.get("nickname") or sender.get("user_id") or "未知"
    )


async def _expand(bot: Bot, forward_id: str, depth: int, out: _Expansion) -> int:
    """把一份转发的节点按顺序写进 out，返回这一层的条数。"""

    data = await bot.call_api("get_forward_msg", id=forward_id)
    nodes = data["messages"]
    for node in nodes:
        if out.chars >= FORWARD_MAX_CHARS:
            out.skipped_nodes += 1
            continue
        text = await _node_text(bot, node, depth, out)
        if not text:
            continue
        when = f"{datetime.fromtimestamp(node['time']):%m-%d %H:%M} " if node.get("time") else ""
        line = f"{'  ' * (depth - 1)}{when}{_node_name(node)}：{text}"
        out.lines.append(line)
        out.chars += len(line)
    return len(nodes)


async def digest_forward(
    group: GroupState,
    bot: Bot,
    event: Event,
    message: ChatMessage,
    forward_id: str,
    generation: int,
) -> None:
    """读完一份转发，把占位改写成摘要，再放回队列触发一轮。"""

    session = group.session
    started_at = time.perf_counter()
    out = _Expansion()
    node_count = 0
    try:
        node_count = await _expand(bot, forward_id, 1, out)
        parsed = await feedback_client.generate_json(
            get_forward_prompt(
                bot_name=session.state.name,
                sender_name=message.user_name,
                node_count=node_count,
                lines=out.lines,
                skipped_nodes=out.skipped_nodes,
                skipped_images=out.skipped_images,
            ),
            session_id=session.id,
            temperature=FORWARD_TEMPERATURE,
            images=out.images,
        )
        summary = str(parsed.get("summary") or "").strip() if parsed else ""
        if not summary:
            logger.warning(f"[Forward] 群 {session.id} 转发摘要输出无效，留着占位")
            outcome = "failed"
        elif session.stale(generation, "forward_digest"):
            outcome = "stale"
        elif message.content != FORWARD_PLACEHOLDER:
            # 读的时候被撤回了
            outcome = "recalled"
        else:
            message.content = f"[合并转发 {node_count}条（{FORWARD_NOTE}）: {summary}]"
            session.short_term_memory.move_to_latest(message)
            session.schedule_save()
            await group.enqueue(message, bot, event)
            outcome = "ok"
    except (ActionFailed, NetworkError) as e:
        logger.warning(f"[Forward] 群 {session.id} 拉取转发内容失败，留着占位: {e}")
        outcome = "failed"
    log_event(
        "forward_digest",
        "DEBUG",
        session_id=session.id,
        nodes=node_count,
        images=len(out.images),
        skipped_nodes=out.skipped_nodes,
        skipped_images=out.skipped_images,
        latency_ms=int((time.perf_counter() - started_at) * 1000),
        outcome=outcome,
    )
