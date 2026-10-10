"""Token 用量卡片：数据由 storage.db.get_token_stats 聚合好，填进 token_stats.html 截图。"""

from datetime import datetime
from html import escape
from pathlib import Path

from .browser import screenshot

_TEMPLATE = Path(__file__).with_name("token_stats.html")


def _amount(value: int) -> tuple[str, str]:
    """数字和单位分开，单位在卡片上排小一号。"""

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}", "B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}", "M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}", "K"
    return str(value), ""


def _text(value: int) -> str:
    number, unit = _amount(value)
    return number + unit


def _model_html(item: dict) -> str:
    number, unit = _amount(item["total"])
    total = item["total"] or 1
    # 条形图：深色输入、红色输出
    split = (
        f'<div class="split"><i style="width:{item["prompt"] / total:.1%}"></i>'
        f'<i style="width:{item["completion"] / total:.1%}"></i></div>'
    )
    details = [
        f'<span class="in">输入 {_text(item["prompt"])}</span>',
        f'<span class="out">输出 {_text(item["completion"])}</span>',
    ]
    # 有的上游把推理算在输出里，有的另算，所以不写「其中」
    if item["reasoning"]:
        details.append(f"<span>推理 {_text(item['reasoning'])}</span>")
    return (
        '<div class="model"><div class="headline">'
        f'<span class="name">{escape(item["model"])}</span>'
        f'<span class="total">{number}<small>{unit}</small></span></div>'
        f'{split}<p class="detail">{"".join(details)}</p></div>'
    )


def _scope_html(label: str, note: str, items: list[dict]) -> str:
    items = sorted(items, key=lambda item: item["total"], reverse=True)
    body = "".join(_model_html(item) for item in items) or '<p class="empty">暂无</p>'
    return (
        f'<div class="scope"><div class="scope-label"><b>{label}</b><small>{note}</small></div>'
        f'<div class="models">{body}</div></div>'
    )


def _period_html(title: str, local: list[dict], global_: list[dict]) -> str:
    local_total = sum(item["total"] for item in local)
    global_total = sum(item["total"] for item in global_)
    share = f"占 {local_total / global_total:.0%}" if global_total else ""
    return (
        f"<section><h2>{title}</h2><div>"
        f'{_scope_html("本群", share, local)}{_scope_html("全部", "所有群", global_)}'
        "</div></section>"
    )


async def render_token_stats_card(stats: dict) -> bytes:
    now = datetime.now()
    date = f"统计至 {now.month} 月 {now.day} 日 {now:%H:%M}"
    sections = (
        _period_html("今日", stats["1d_local"], stats["1d_global"])
        + _period_html("近七日", stats["7d_local"], stats["7d_global"])
        + "<section><h2>累计</h2><div>"
        + _scope_html("全部", "所有群", stats["all_global"])
        + "</div></section>"
    )
    html = (
        _TEMPLATE.read_text(encoding="utf-8")
        .replace("{{date}}", date)
        .replace("{{sections}}", sections)
    )
    return await screenshot(html, 520)
