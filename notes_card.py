"""群志卡片：群志按【段名】切段，填进 notes_card.html，用 Chromium 截图。

查看群志不常用，内存又紧，所以每次现开浏览器、截完就关，不常驻。
"""

import os
import re
from datetime import datetime
from html import escape
from pathlib import Path

from playwright.async_api import async_playwright

from .config import get_app_settings

_TEMPLATE = Path(__file__).with_name("notes_card.html")
_SECTION_RE = re.compile(r"^【(.+?)】\s*(.*)$")
_BULLET_RE = re.compile(r"^[-•·*]\s*")
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})\S*\s+(.+)$")
_ALIAS_SPLIT_RE = re.compile(r"(?<=[／/、])")
_MONTHS = "一二三四五六七八九十"


def parse_sections(notes: str) -> list[tuple[str, list[str]]]:
    """按【段名】切段；段名前的内容或没按格式写的群志归入无名段，保证总能渲染。"""

    sections: list[tuple[str, list[str]]] = []
    for raw in notes.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _SECTION_RE.match(line)
        if match:
            sections.append((match[1], []))
            line = match[2].strip()
            if not line:
                continue
        if not sections:
            sections.append(("", []))
        sections[-1][1].append(_BULLET_RE.sub("", line))
    return sections


def _month_name(month: int) -> str:
    if month <= 10:
        return f"{_MONTHS[month - 1]}月"
    return f"十{_MONTHS[month - 11]}月"


def _terms_html(lines: list[str]) -> str:
    rows = []
    for line in lines:
        term, _, meaning = line.replace(":", "：").partition("：")
        # 一条常列好几个别名：每个别名连同后面的分隔符成块，只在块之间换行
        aliases = "".join(
            f"<span>{escape(alias)}</span>"
            for alias in _ALIAS_SPLIT_RE.split(term.strip())
            if alias
        )
        rows.append(f"<dt>{aliases}</dt><dd>{escape(meaning.strip())}</dd>")
    return f'<dl class="terms">{"".join(rows)}</dl>'


def _recent_html(lines: list[str]) -> str:
    """同一个月的事并到一组，月份只写一次；没写日期的行单独成组。"""

    groups: list[tuple[str, str, list[str]]] = []
    for line in lines:
        match = _DATE_RE.match(line)
        year, month, event = (
            (match[1], _month_name(int(match[2])), match[3]) if match else ("", "", line)
        )
        if groups and groups[-1][:2] == (year, month):
            groups[-1][2].append(event)
        else:
            groups.append((year, month, [event]))
    return "".join(
        f'<div class="month"><div class="month-label"><b>{month}</b><small>{year}</small></div>'
        f'<ul>{"".join(f"<li>{escape(event)}</li>" for event in events)}</ul></div>'
        for year, month, events in groups
    )


def _section_html(title: str, lines: list[str]) -> str:
    if lines == ["暂无"]:
        body = '<p class="empty">暂无</p>'
    elif title == "群像":
        body = "".join(f'<p class="lead">{escape(line)}</p>' for line in lines)
    elif title == "称呼与梗":
        body = _terms_html(lines)
    elif title == "近期":
        body = _recent_html(lines)
    else:
        body = "".join(f'<p class="text">{escape(line)}</p>' for line in lines)
    return f'<section><h2>{escape(title)}</h2><div class="body">{body}</div></section>'


async def render_group_notes_card(
    *, notes: str, group_name: str, updated_at: datetime | None
) -> bytes:
    date = (
        f"整理至 {updated_at.year} 年 {updated_at.month} 月 {updated_at.day} 日 "
        f"{updated_at:%H:%M}"
        if updated_at
        else ""
    )
    html = (
        _TEMPLATE.read_text(encoding="utf-8")
        .replace("{{title}}", escape(group_name))
        .replace("{{date}}", date)
        .replace(
            "{{sections}}",
            "".join(_section_html(title, lines) for title, lines in parse_sections(notes)),
        )
    )

    # 和 HakuBot 共用同一份 Chromium（playwright 版本一致），不用再下载
    browsers_path = get_app_settings().playwright_browsers_path
    if browsers_path:
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = browsers_path
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(args=["--no-sandbox"])
        try:
            page = await browser.new_page(
                viewport={"width": 520, "height": 200}, device_scale_factor=2
            )
            await page.set_content(html)
            await page.evaluate("document.fonts.ready")
            return await page.screenshot(full_page=True)
        finally:
            await browser.close()
