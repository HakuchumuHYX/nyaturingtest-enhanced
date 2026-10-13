"""印象卡片：bot 手账里关于某个群友的一页，填进 impression.html，用 Chromium 截图。"""

import base64
from datetime import datetime
from html import escape
from pathlib import Path

from nonebot import logger

from ..core.llm import get_http_client
from ..core.memory_query import Impression
from .browser import screenshot

_TEMPLATE = Path(__file__).with_name("impression.html")
# 版式按 680×962（A4 比例）排，3 倍像素密度截图，成图 2040×2886
PAGE_WIDTH = 680
PAGE_SCALE = 3
AVATAR_TIMEOUT_SECONDS = 5.0
_WEEKDAYS = "一二三四五六日"
# 荧光笔颜色跟着关系档位走，档位见 prompts.describe_relation
_HIGHLIGHTS = {
    "很讨厌": "#cfcfd4",
    "有点反感": "#b9d3ea",
    "普通群友": "#c8e6c1",
    "比较熟、有好感": "#f6e7a1",
    "关系很好": "#f8cda5",
    "非常亲近": "#f6c1cf",
}


async def _avatar_data_uri(user_id: str) -> str:
    """头像先拉到本地再嵌进页面，截图不用等远程图片；拉不到就不贴相片，版式照常。"""

    try:
        response = await get_http_client().get(
            f"https://q1.qlogo.cn/g?b=qq&nk={user_id}&s=640",
            timeout=AVATAR_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except Exception as e:
        logger.warning(f"印象卡片头像获取失败 {user_id}: {e}")
        return ""
    mime = response.headers.get("content-type", "image/jpeg").split(";")[0]
    return f"data:{mime};base64,{base64.b64encode(response.content).decode()}"


def _status_html(page: Impression, now: datetime) -> str:
    parts = [f"{page.relation}（{page.affinity}）"]
    if page.first_interaction_at:
        days = (now - page.first_interaction_at.replace(tzinfo=None)).days + 1
        parts.append(f"认识 {days} 天")
    parts.append(f"见过 {page.interactions} 条发言")
    return "".join(f"<span>{escape(part)}</span>" for part in parts)


async def render_impression_card(page: Impression) -> bytes:
    now = datetime.now()
    avatar = await _avatar_data_uri(page.target_id)
    memories = "".join(f"<li><span>{escape(item)}</span></li>" for item in page.memories)
    values = {
        "highlight": _HIGHLIGHTS[page.relation],
        "page_class": "has-photo" if avatar else "",
        "photo": f'<div class="photo"><img src="{avatar}" alt=""></div>' if avatar else "",
        "date": f"{now.month}.{now.day}",
        "weekday": f"周{_WEEKDAYS[now.weekday()]}",
        "year": str(now.year),
        "name": escape(page.target_name),
        "headline": escape(page.headline),
        "status": _status_html(page, now),
        "impression": escape(page.impression),
        "memories": (
            '<div class="gap"></div>'
            f'<section class="memories"><h2>记得的事</h2><ul>{memories}</ul></section>'
            if memories
            else ""
        ),
        "tags": "".join(f"<span>#{escape(tag)}</span>" for tag in page.tags),
        "memo": (
            f'<div class="memo"><p><b>最近</b>　{escape(page.recent)}</p></div>'
            if page.recent
            else ""
        ),
        "bot_name": escape(page.bot_name),
    }
    html = _TEMPLATE.read_text(encoding="utf-8")
    for key, value in values.items():
        html = html.replace(f"{{{{{key}}}}}", value)
    return await screenshot(html, PAGE_WIDTH, scale=PAGE_SCALE)
