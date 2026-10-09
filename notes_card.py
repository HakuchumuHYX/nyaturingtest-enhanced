"""群志卡片：按【段名】切段，每段一张圆角卡片，称呼与梗、近期各有专门排版。"""

import re
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import ImageFont

from .config import get_data_dir

CARD_WIDTH = 750
_SECTION_RE = re.compile(r"^【(.+?)】\s*(.*)$")
_BULLET_RE = re.compile(r"^[-•·*]\s*")
_DATE_RE = re.compile(r"^(\d{4}-\d{2}\S*)\s+(.+)$")

_TEXT = (30, 40, 50, 255)
_SUB = (90, 105, 120, 255)
_MUTED = (140, 155, 170, 255)
_WHITE = (255, 255, 255, 255)
_ACCENTS = {
    "群像": (88, 101, 242),
    "日常": (16, 150, 130),
    "称呼与梗": (226, 120, 30),
    "近期": (219, 68, 98),
}
_DEFAULT_ACCENT = (100, 116, 139)
# 不能出现在行首的标点：绘图库逐字换行，句号常被单独挤到下一行
_NO_LINE_START = set("，。、；：？！）」』】》…—,.;:?!)")


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


def _wrap(content: str, font: ImageFont.FreeTypeFont, width: int) -> str:
    """按宽度预先断行：禁则标点不放行首（带上一行末字下来），英文单词不从中间拆开。"""

    lines = []
    for paragraph in content.split("\n"):
        line = ""
        for char in paragraph:
            if not line or font.getlength(line + char) <= width:
                line += char
                continue
            if char in _NO_LINE_START and len(line) > 1:
                cut = len(line) - 1
            elif _is_word_char(char) and _is_word_char(line[-1]):
                cut = len(line)
                while cut > 0 and _is_word_char(line[cut - 1]):
                    cut -= 1
                # 整行就是一个超长单词时只能硬拆
                cut = cut or len(line)
            else:
                cut = len(line)
            lines.append(line[:cut])
            line = line[cut:] + char
        lines.append(line)
    return "\n".join(lines)


def _is_word_char(char: str) -> bool:
    return char.isascii() and (char.isalnum() or char in "-_'./")


def _tint(accent: tuple[int, int, int], ratio: float) -> tuple[int, int, int, int]:
    return (*(round(255 - (255 - c) * ratio) for c in accent), 255)


async def render_group_notes_card(
    *, notes: str, group_name: str, updated_at: datetime | None
) -> bytes:
    # plugins/ 目录，utils.draw 是同级插件目录下的公共绘图库
    plugins_dir = Path(__file__).resolve().parents[1]
    if str(plugins_dir) not in sys.path:
        sys.path.insert(0, str(plugins_dir))
    from utils.draw.plot import (
        Canvas,
        FillBg,
        HSplit,
        RoundRectBg,
        Spacer,
        TextBox,
        TextStyle,
        VSplit,
    )

    font_dir = get_data_dir()

    def style(weight: str, size: int, color: tuple) -> TextStyle:
        return TextStyle(
            font=str(font_dir / f"SourceHanSansCN-{weight}.ttf"), size=size, color=color
        )

    def text(content: str, text_style: TextStyle, width: int) -> TextBox:
        font = ImageFont.truetype(text_style.font, text_style.size)
        # 预留 2px：库内测宽与这里可能有亚像素差，避免它再按自己的规则折一次
        return (
            TextBox(
                _wrap(content, font, width - 2),
                style=text_style,
                use_real_line_count=True,
                line_sep=7,
            )
            .set_w(width)
            .set_padding(0)
        )

    outer_margin = 24
    card_padding = 26
    section_padding = 20
    content_width = CARD_WIDTH - outer_margin * 2 - card_padding * 2
    inner_width = content_width - section_padding * 2

    body_style = style("Regular", 19, _TEXT)

    def section_items(title: str, lines: list[str], accent: tuple) -> list:
        accent_dark = (*(round(c * 0.8) for c in accent), 255)
        if title == "群像":
            return [
                text(line, style("Regular", 21, _TEXT), inner_width) for line in lines
            ]
        if title == "称呼与梗":
            items = []
            for line in lines:
                term, sep, meaning = line.replace(":", "：").partition("：")
                entry = [
                    text(term.strip(), style("Bold", 20, accent_dark), inner_width)
                ]
                if sep and meaning.strip():
                    entry.append(
                        text(
                            meaning.strip(),
                            style("Regular", 17, _SUB),
                            inner_width - 18,
                        ).set_margin((18, 0))
                    )
                items.append(VSplit(items=entry, sep=3, item_align="lt"))
            return items
        if title == "近期":
            items = []
            badge_width = 104
            for line in lines:
                match = _DATE_RE.match(line)
                if not match:
                    items.append(text(line, body_style, inner_width))
                    continue
                badge = (
                    TextBox(match[1], style=style("Bold", 15, _WHITE))
                    .set_w(badge_width)
                    .set_content_align("c")
                    .set_padding((0, 4))
                    .set_bg(RoundRectBg(fill=(*accent, 255), radius=10))
                )
                items.append(
                    HSplit(
                        items=[
                            badge,
                            text(match[2], body_style, inner_width - badge_width - 14),
                        ],
                        sep=14,
                        item_align="lt",
                    )
                )
            return items
        bullet_width = 24
        return [
            HSplit(
                items=[
                    TextBox("•", style=style("Heavy", 19, accent_dark), overflow="clip")
                    .set_w(bullet_width)
                    .set_padding(0),
                    text(line, body_style, inner_width - bullet_width - 4),
                ],
                sep=4,
                item_align="lt",
            )
            for line in lines
        ]

    def section(title: str, lines: list[str]):
        accent = _ACCENTS.get(title, _DEFAULT_ACCENT)
        items = []
        if title:
            items.append(
                TextBox(title, style=style("Bold", 17, _WHITE))
                .set_padding((14, 5))
                .set_bg(RoundRectBg(fill=(*accent, 255), radius=14))
            )
            items.append(Spacer(1, 4))
        items.extend(section_items(title, lines, accent))
        return (
            VSplit(items=items, sep=12, item_align="lt")
            .set_w(content_width)
            .set_padding(section_padding)
            .set_bg(
                RoundRectBg(
                    fill=_tint(accent, 0.06),
                    radius=18,
                    stroke=_tint(accent, 0.22),
                    stroke_width=2,
                )
            )
        )

    subtitle = group_name
    if updated_at is not None:
        subtitle += f" · 整理至 {updated_at:%Y-%m-%d}"
    items = [
        text("群志", style("Heavy", 38, _TEXT), content_width),
        text(subtitle, style("Regular", 17, _MUTED), content_width),
        Spacer(1, 10),
        *(section(title, lines) for title, lines in parse_sections(notes)),
        Spacer(1, 4),
        TextBox("Generated by HakuBot", style=style("Regular", 14, _MUTED))
        .set_w(content_width)
        .set_content_align("r")
        .set_padding(0),
    ]
    card = (
        VSplit(items=items, sep=12, item_align="lt")
        .set_w(CARD_WIDTH - outer_margin * 2)
        .set_padding(card_padding)
        .set_margin(outer_margin)
        .set_bg(
            RoundRectBg(
                fill=_WHITE, radius=26, stroke=(200, 215, 230, 255), stroke_width=2
            )
        )
    )
    canvas = Canvas(w=CARD_WIDTH, h=None, bg=FillBg((240, 245, 250, 255)))
    canvas.set_items([card]).set_content_align("c")
    image = await canvas.get_img()
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
