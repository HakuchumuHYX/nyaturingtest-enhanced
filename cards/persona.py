"""人设卡片：人生线、本群的 ta 和最近的改动，填进 persona.html，用 Chromium 截图。"""

import re
from dataclasses import dataclass
from datetime import date, datetime
from html import escape
from pathlib import Path

from .browser import screenshot
from .notes import parse_sections

_TEMPLATE = Path(__file__).with_name("persona.html")
PAGE_WIDTH = 600
# 尺子在页面里的宽度（页宽减左右边距）
_RULER_WIDTH = PAGE_WIDTH - 64
# 群层的风格每条以「（自 YYYY-MM-DD）」结尾，约定写在 prompts.get_persona_group_prompt
_SINCE_RE = re.compile(r"（自\s*(\d{4})-(\d{2})-(\d{2})）")
_DATED_RE = re.compile(r"(\d{2})-(\d{2})\s*(.*)", re.S)
# 一条约定或说过的话以「MM-DD」开头，前一条以句号或分号结束
_DATED_SPLIT_RE = re.compile(r"(?<=[。；])\s*(?=\d{2}-\d{2})")
# 段名是写给整理模型看的，卡片上换成读着顺的说法
_GROUP_TITLES = {"在这个群的样子": "沾上的说法"}


@dataclass(frozen=True)
class Change:
    day: date
    scope: str  # life / group
    text: str


def _short(day: date) -> str:
    return f"{day.month}.{day.day}"


def _para(title: str, lines: list[str]) -> str:
    """段名放在段首；「暂无」也照样占一段，让人知道这一项还空着。"""

    text = "".join(lines)
    body = f'<span class="empty">{escape(text)}</span>' if lines == ["暂无"] else escape(text)
    return f'<p class="para"><b>{escape(title)}</b>{body}</p>'


def _list(title: str, items: str, css: str) -> str:
    return f'<span class="list-title">{escape(title)}</span><ul class="list {css}">{items}</ul>'


def _styles_html(lines: list[str]) -> str:
    """一条说法一行，末尾写从哪天沾上的；没按格式写就当普通段落。"""

    title = "沾上的说法"
    parts = _SINCE_RE.split("".join(lines))
    if len(parts) == 1:
        return _para(title, lines)
    rows = []
    # split 后是 [正文, 年, 月, 日, 正文, 年, 月, 日, ..., 结尾]
    for index in range(0, len(parts) - 1, 4):
        phrase = parts[index].strip(" 。；")
        month, day = int(parts[index + 2]), int(parts[index + 3])
        if phrase:
            rows.append(f"<li>{escape(phrase)}<time>{month}.{day} 起</time></li>")
    tail = parts[-1].strip(" 。；")
    if tail:
        rows.append(f"<li>{escape(tail)}</li>")
    return _list(title, "".join(rows), "styles")


def _dated_html(title: str, lines: list[str]) -> str:
    """说过的话、答应的事：每条以 MM-DD 开头，日期放左栏。"""

    if lines == ["暂无"]:
        return _para(title, lines)
    items = [
        item.strip()
        for line in lines
        for item in _DATED_SPLIT_RE.split(line)
        if item.strip()
    ]
    rows = []
    for item in items:
        match = _DATED_RE.match(item)
        when = f"{int(match[1])}.{int(match[2])}" if match else ""
        rows.append(
            f"<li><time>{when}</time><span>{escape(match[3] if match else item)}</span></li>"
        )
    return _list(title, "".join(rows), "dated")


def _group_html(text: str) -> str:
    if not text:
        return '<p class="para empty">还没整理过，每天 03:30 整理一次</p>'
    blocks = []
    for title, lines in parse_sections(text):
        if title == "在这个群的样子":
            blocks.append(_styles_html(lines))
        elif title == "说过的话、答应的事":
            blocks.append(_dated_html(title, lines))
        else:
            blocks.append(_para(title, lines))
    return "".join(blocks)


def _life_html(text: str) -> str:
    """【现状】当开头一段，不写段名；其余段名放段首。"""

    blocks = []
    for title, lines in parse_sections(text):
        if title == "现状":
            blocks.append(f'<p class="lead">{escape("".join(lines))}</p>')
        else:
            blocks.append(_para(title, lines))
    return "".join(blocks)


def _months_between(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + end.month - start.month


def _ruler_html(start: date, today: date, marks: list[Change]) -> str:
    """出发点到今天的尺子：每月一格，每年一月加长；改动日记一道铅笔印。"""

    span = max((today - start).days, 1)

    def left(day: date) -> str:
        return f"{min(max((day - start).days / span, 0), 1) * 100:.3f}%"

    parts = []
    for offset in range(_months_between(start, today) + 1):
        month_start = date(start.year + (start.month - 1 + offset) // 12, (start.month - 1 + offset) % 12 + 1, 1)
        if month_start < start or month_start > today:
            continue
        year_tick = month_start.month == 1
        parts.append(f'<i class="tick{" year" if year_tick else ""}" style="left:{left(month_start)}"></i>')
        # 年份标在一月，离两头太近就不标，免得和出发点、今天挤在一起
        position = (month_start - start).days / span
        if year_tick and 0.12 < position < 0.82:
            parts.append(
                f'<span class="label" style="left:{left(month_start)}">{month_start.year}</span>'
            )
    marks_html = "".join(
        f'<i class="mark {mark.scope}" style="left:{left(mark.day)}"></i>' for mark in marks
    )
    return (
        f'<div class="marks">{marks_html}</div><div class="line"></div>{"".join(parts)}'
        f'<i class="now" style="left:100%"></i>'
        f'<span class="label start" style="left:0">{start.year}.{start.month}</span>'
        f'<span class="label now-label">今天</span>'
    )


def _changes_html(changes: list[Change]) -> str:
    if not changes:
        return ""
    rows = "".join(
        f"<li><time>{_short(change.day)}</time>"
        f'<span class="scope scope-{change.scope}">{"人生线" if change.scope == "life" else "本群"}</span>'
        f"<span>{escape(change.text)}</span></li>"
        for change in changes
    )
    return f'<div class="changes"><span class="list-title">最近的改动</span><ul class="log">{rows}</ul></div>'


async def render_persona_card(
    *,
    name: str,
    as_of: str,
    life: str,
    group_text: str,
    group_name: str,
    changes: list[Change],
    marks: list[Change],
) -> bytes:
    """changes 是最近几次改动（新的在前），marks 是所有真改了东西的日子，画在尺子上。"""

    today = datetime.now().date()
    start = datetime.strptime(as_of, "%Y-%m").date()
    months = _months_between(start, today) + 1
    values = {
        "name": escape(name),
        "since": f"从 {start.year} 年 {start.month} 月出发，现在是第 {months} 个月",
        "ruler": _ruler_html(start, today, marks),
        "life": _life_html(life),
        "group_name": escape(group_name),
        "group": _group_html(group_text),
        "changes": _changes_html(changes),
    }
    html = _TEMPLATE.read_text(encoding="utf-8")
    for key, value in values.items():
        html = html.replace(f"{{{{{key}}}}}", value)
    return await screenshot(html, PAGE_WIDTH)
