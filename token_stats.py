# 由多个模块合并而来：database/token_stats_aggregation.py, presenters/token_stats_card.py

from collections.abc import Mapping
import unicodedata
from io import BytesIO
from pathlib import Path
import sys


# ======== from database/token_stats_aggregation.py ========
TOKEN_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
    "reasoning_tokens",
)


def _clean_model_name(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def _model_key(value: object) -> str:
    return _clean_model_name(value).casefold()


def _empty_totals() -> dict[str, int]:
    return {field: 0 for field in TOKEN_FIELDS}


def _format_totals(totals: Mapping[str, int]) -> dict[str, int | float]:
    prompt = int(totals.get("prompt_tokens", 0) or 0)
    completion = int(totals.get("completion_tokens", 0) or 0)
    cache_hit = int(totals.get("prompt_cache_hit_tokens", 0) or 0)
    cache_miss = int(totals.get("prompt_cache_miss_tokens", 0) or 0)
    cache_total = cache_hit + cache_miss
    return {
        "prompt": prompt,
        "completion": completion,
        "reasoning": int(totals.get("reasoning_tokens", 0) or 0),
        "cache_hit": cache_hit,
        "cache_miss": cache_miss,
        "cache_hit_ratio": cache_hit / cache_total if cache_total else 0.0,
        "total": prompt + completion,
    }


def merge_token_stats_by_model(
    aggregate: Mapping[tuple[str, str], Mapping[str, int]],
) -> list[dict]:
    """按规范化模型名合并（大小写不敏感），保留每个模型的汇总。"""

    models: dict[str, dict] = {}
    for (raw_model, _raw_provider), totals in aggregate.items():
        model_name = _clean_model_name(raw_model)
        key = _model_key(model_name)
        entry = models.setdefault(key, {"model": model_name, "totals": _empty_totals()})
        for field in TOKEN_FIELDS:
            entry["totals"][field] += int(totals.get(field, 0) or 0)

    return [
        {"model": models[key]["model"], **_format_totals(models[key]["totals"])}
        for key in sorted(models)
    ]


# ======== from presenters/token_stats_card.py ========
async def render_token_stats_card(
    *,
    stats: dict,
    watermark: str | None = None,
    scope_label: str = "当前模型",
    width: int = 750,
) -> bytes:
    """Render a compact token-usage PNG without coupling handlers to drawing."""

    plugins_dir = Path(__file__).resolve().parents[2]
    if str(plugins_dir) not in sys.path:
        sys.path.insert(0, str(plugins_dir))
    from utils.draw.plot import (
        Canvas,
        FillBg,
        RoundRectBg,
        Spacer,
        TextBox,
        TextStyle,
        VSplit,
    )

    font_dir = Path(__file__).resolve().parents[3] / "data" / "nyaturingtest"
    colors = {
        "canvas": (240, 245, 250, 255),
        "card": (255, 255, 255, 255),
        "border": (200, 215, 230, 255),
        "section": (248, 251, 255, 255),
        "text": (30, 40, 50, 255),
        "sub": (90, 105, 120, 255),
        "muted": (140, 155, 170, 255),
    }

    def style(filename: str, size: int, color: tuple) -> TextStyle:
        return TextStyle(font=str(font_dir / filename), size=size, color=color)

    title_style = style("SourceHanSansCN-Heavy.ttf", 36, colors["text"])
    section_style = style("SourceHanSansCN-Bold.ttf", 24, colors["text"])
    label_style = style("SourceHanSansCN-Regular.ttf", 18, colors["sub"])
    value_style = style("SourceHanSansCN-Bold.ttf", 20, colors["text"])
    model_style = style("SourceHanSansCN-Bold.ttf", 18, colors["text"])
    watermark_style = style(
        "SourceHanSansCN-Regular.ttf",
        14,
        colors["muted"],
    )
    outer_margin = 24
    padding = 22
    content_width = width - outer_margin * 2 - padding * 2

    def rows(values: list[dict]) -> list:
        result = []
        for item in values:
            result.append(
                TextBox(
                    f"模型: {item['model']}",
                    style=model_style,
                ).set_w(content_width).set_padding((8, 2))
            )
            result.append(
                TextBox(
                    f"  Prompt: {item['prompt']:,}  |  "
                    f"Completion: {item['completion']:,}  |  "
                    f"Total: {item['total']:,}",
                    style=value_style,
                ).set_w(content_width).set_padding((16, 2))
            )
            if (
                item.get("reasoning")
                or item.get("cache_hit")
                or item.get("cache_miss")
            ):
                result.append(
                    TextBox(
                        f"  Reasoning: {item.get('reasoning', 0):,}  |  "
                        f"Cache hit: {item.get('cache_hit', 0):,}  |  "
                        f"Cache miss: {item.get('cache_miss', 0):,}  |  "
                        f"Hit ratio: {item.get('cache_hit_ratio', 0.0):.1%}",
                        style=label_style,
                    ).set_w(content_width).set_padding((16, 2))
                )
        return result

    def period(title: str, local: list, global_: list):
        items = [
            TextBox(title, style=section_style)
            .set_w(content_width)
            .set_padding((0, 8)),
            TextBox(
                "【本群消耗】" if local else "【本群消耗】无数据",
                style=label_style,
            ).set_w(content_width).set_padding((0, 4)),
            *rows(local),
            Spacer(1, 8),
            TextBox(
                "【全局所有群消耗】"
                if global_
                else "【全局所有群消耗】无数据",
                style=label_style,
            ).set_w(content_width).set_padding((0, 4)),
            *rows(global_),
        ]
        return (
            VSplit(items=items, sep=4, item_size_mode="fixed")
            .set_w(content_width)
            .set_padding(16)
            .set_bg(RoundRectBg(fill=colors["section"], radius=16))
        )

    items = [
        TextBox(
            f"Token 使用统计（{scope_label}）",
            style=title_style,
        ).set_w(content_width).set_padding(0),
        Spacer(1, 16),
        period(
            "今日统计",
            stats.get("1d_local", []),
            stats.get("1d_global", []),
        ),
        Spacer(1, 12),
        period(
            "7天统计",
            stats.get("7d_local", []),
            stats.get("7d_global", []),
        ),
    ]
    historical = stats.get("all_global", [])
    if historical:
        items.extend(
            [
                Spacer(1, 12),
                VSplit(
                    items=[
                        TextBox("历史总消耗", style=section_style)
                        .set_w(content_width)
                        .set_padding((0, 8)),
                        *rows(historical),
                    ],
                    sep=4,
                    item_size_mode="fixed",
                )
                .set_w(content_width)
                .set_padding(16)
                .set_bg(RoundRectBg(fill=colors["section"], radius=16)),
            ]
        )
    items.append(Spacer(1, 12))
    watermark = "Generated by HakuBot" if watermark is None else watermark
    if watermark:
        items.append(
            TextBox(watermark, style=watermark_style)
            .set_w(content_width)
            .set_content_align("r")
            .set_padding(0)
        )
    card = (
        VSplit(items=items, sep=4, item_size_mode="fixed")
        .set_w(width - outer_margin * 2)
        .set_padding(padding)
        .set_margin(outer_margin)
        .set_bg(
            RoundRectBg(
                fill=colors["card"],
                radius=26,
                stroke=colors["border"],
                stroke_width=2,
            )
        )
    )
    canvas = Canvas(w=width, h=None, bg=FillBg(colors["canvas"]))
    canvas.set_items([card]).set_content_align("c")
    image = await canvas.get_img()
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
