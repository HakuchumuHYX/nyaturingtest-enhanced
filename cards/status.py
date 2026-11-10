"""/status 的状态卡片：画成她手机的锁屏，填进 status.html，用 Chromium 截图。

壁纸是她相册里最近的自拍；她在做的事像锁屏签名一样直接写在壁纸上，本群和打算是两条通知，今天的各段是一条时间轨。
她自己的话用文楷，时间和系统字用黑体。
"""

import base64
import colorsys
import io
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path

from nonebot.utils import run_sync
from PIL import Image

from ..prompts import SOURCE_TAGS
from .browser import screenshot

_TEMPLATE = Path(__file__).with_name("status.html")
PAGE_WIDTH = 600
_WEEKDAYS = "一二三四五六日"
# 没有照片时的底色
_DEFAULT_BASE = "#1f2433"
# 壁纸按原图比例铺满宽度，高度限在这个范围里，超出才裁
PHOTO_MIN_HEIGHT = 520
PHOTO_MAX_HEIGHT = 760
# 竖图要裁时上面少裁、下面多裁，脸一般在上半部分
PHOTO_CROP_TOP = 0.3
# 取底色的那一段：照片淡进底色的地方，占照片高度的比例
BASE_SAMPLE_BAND = (0.45, 0.85)
# 时间线上的状态，时钟下面那行用同一套词
_MODES = {"free": "有空", "busy": "在忙", "away": "放下手机", "sleep": "睡觉"}
_FOCUS = {"free": "有空", "busy": "在忙", "away": "放下手机了", "sleep": "在睡觉"}


@dataclass(frozen=True)
class GroupView:
    name: str
    unread: int
    last_look: datetime | None
    mood: str
    summary: str
    muted_until: datetime | None


def _kind(segment: dict) -> str:
    return "sleep" if segment.get("sleep") else segment["mode"]


def _fit(image: Image.Image) -> Image.Image:
    """裁成壁纸显示的样子：宽 PAGE_WIDTH 的 2 倍（截图是 2 倍像素密度），高按比例、限在范围里。"""

    width, height = image.size
    shown = min(max(round(PAGE_WIDTH * height / width), PHOTO_MIN_HEIGHT), PHOTO_MAX_HEIGHT)
    ratio = PAGE_WIDTH / shown
    if width / height > ratio:
        kept = round(height * ratio)
        left = (width - kept) // 2
        box = (left, 0, left + kept, height)
    else:
        kept = round(width / ratio)
        top = round((height - kept) * PHOTO_CROP_TOP)
        box = (0, top, width, top + kept)
    return image.crop(box).resize((PAGE_WIDTH * 2, shown * 2), Image.Resampling.LANCZOS)


def _base_color(image: Image.Image) -> str:
    """照片淡出那一段的主色压暗：量化成几种颜色，占得够多的里面挑最鲜艳的。

    不用平均色：衣服、桌面一平均就成了灰褐，每张照片得出的底色都差不多。
    本来就是灰的照片照样是灰，饱和度只放大、不凭空加。
    """

    top, bottom = (round(image.height * edge) for edge in BASE_SAMPLE_BAND)
    band = image.crop((0, top, image.width, bottom)).resize((64, 32))
    quantized = band.quantize(colors=6)
    palette = quantized.getpalette()
    total = band.width * band.height
    colors = [
        tuple(channel / 255 for channel in palette[index * 3 : index * 3 + 3])
        for count, index in quantized.getcolors()
        if count / total >= 0.12
    ]
    # 彩度（最大减最小通道）兼顾饱和度和亮度，很暗的噪点得分自然低
    r, g, b = max(colors, key=lambda color: max(color) - min(color))
    hue, saturation, _ = colorsys.rgb_to_hsv(r, g, b)
    r, g, b = colorsys.hsv_to_rgb(hue, min(saturation * 1.5, 0.45), 0.17)
    return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"


@run_sync
def _load_wallpaper(path: Path) -> tuple[str, str, int]:
    """返回 (data URI, 底色, 显示高度)。显示的和取色的是同一张裁好的图。"""

    with Image.open(path) as source:
        image = _fit(source.convert("RGB"))
    output = io.BytesIO()
    image.save(output, "JPEG", quality=88)
    data_uri = f"data:image/jpeg;base64,{base64.b64encode(output.getvalue()).decode()}"
    return data_uri, _base_color(image), image.height // 2


def _until(until: datetime, now: datetime) -> str:
    return f"{until:%H:%M}" if until.date() == now.date() else f"{until:%m-%d %H:%M}"


def _tag(segment: dict) -> str:
    tag = SOURCE_TAGS.get(segment["source"], "").strip("（）")
    return f'<span class="tag">{escape(tag)}</span>' if tag else ""


def _focus_html(current: dict | None, calls: int) -> str:
    if current is None:
        return ""
    called = f"，被叫了 {calls} 次" if calls else ""
    return f'<p class="focus">{_FOCUS[_kind(current)]}{called}</p>'


def _now_html(current: dict | None, until: datetime, now: datetime) -> str:
    if current is None:
        return '<section class="now"><p class="doing">还没定在做什么</p></section>'
    start = current["start"]
    total = (until - start).total_seconds()
    progress = 1.0 if total <= 0 else min(max((now - start).total_seconds() / total, 0.0), 1.0)
    left = max(int((until - now).total_seconds() // 60), 0)
    if left >= 60:
        left_text = f"还有 {left // 60} 小时 {left % 60} 分"
    else:
        # 到点后要等下一次检查才定下一段，中间最多差一分钟
        left_text = f"还有 {left} 分钟" if left else "到点了"
    return f"""<section class="now">
  <p class="doing">{escape(current["doing"])}{_tag(current)}</p>
  <div class="bar"><span style="width: {progress * 100:.1f}%"></span></div>
  <p class="span"><span>{start:%H:%M} 开始</span><span>{left_text}，到 {_until(until, now)}</span></p>
</section>"""


def _group_html(group: GroupView) -> str:
    seen = f"{group.last_look:%H:%M} 看过" if group.last_look else "还没看过"
    unread = f"{group.unread} 条新消息" if group.unread else "没有新消息"
    summary = f'<p class="body">{escape(group.summary)}</p>' if group.summary else ""
    muted = (
        f'<p class="muted">被禁言到 {group.muted_until:%m-%d %H:%M}</p>' if group.muted_until else ""
    )
    return f"""<section class="note">
  <p class="meta"><b>{escape(group.name)}</b><time>{seen}</time></p>
  <p class="count">{unread}</p>
  {summary}
  <p class="mood">{escape(group.mood)}</p>
  {muted}
</section>"""


def _plan_html(plan: str) -> str:
    if not plan:
        return ""
    return f"""<section class="note memo">
  <p class="meta"><b>这几天的打算</b></p>
  <p class="body">{escape(plan)}</p>
</section>"""


def _row(segment: dict, previous: dict | None, is_now: bool) -> str:
    """状态和城市都是变了才标，和 segment_line 标城市一样；现在这一段总是标状态。"""

    kind = _kind(segment)
    changed = previous is None or _kind(previous) != kind
    mode_html = f'<span class="mode">{_MODES[kind]}</span>' if changed or is_now else ""
    city = segment["city"]
    city_html = (
        f'<span class="city">到{escape(city)}</span>'
        if previous is not None and previous["city"] != city
        else ""
    )
    classes = f"m-{kind} is-now" if is_now else f"m-{kind}"
    return (
        f'<li class="{classes}"><time>{segment["start"]:%H:%M}</time>'
        f"<p>{mode_html}{city_html}{escape(segment['doing'])}{_tag(segment)}</p></li>"
    )


def _today_html(today: list[tuple[dict, datetime]], current: dict | None) -> str:
    segments = [segment for segment, _ in today] + ([current] if current else [])
    if not segments:
        return ""
    last = len(segments) - 1
    rows = [
        _row(
            segment,
            segments[index - 1] if index else None,
            current is not None and index == last,
        )
        for index, segment in enumerate(segments)
    ]
    return f"""<section class="today">
  <p class="head">今天</p>
  <ol>{"".join(rows)}</ol>
</section>"""


async def render_status_card(
    *,
    name: str,
    current: dict | None,
    until: datetime,
    calls: int,
    plan: str,
    today: list[tuple[dict, datetime]],
    group: GroupView,
    wallpaper: Path | None,
) -> bytes:
    """current 是现在这一段，today 是今天醒来以后、现在之前的各段；wallpaper 是相册里的照片，可为 None。"""

    now = datetime.now()
    image, base, photo_height = "", _DEFAULT_BASE, 0
    if wallpaper is not None and wallpaper.exists():
        image, base, photo_height = await _load_wallpaper(wallpaper)
    values = {
        "base": base,
        "photo_height": str(photo_height),
        "photo": f'<img class="photo" src="{image}">' if image else "",
        "photo_class": "has-photo" if image else "no-photo",
        "date": f"{now.month}月{now.day}日 星期{_WEEKDAYS[now.weekday()]}",
        "city": escape(current["city"]) if current else "",
        "clock": f"{now:%H:%M}",
        "focus": _focus_html(current, calls),
        "now": _now_html(current, until, now),
        "notes": _group_html(group) + _plan_html(plan),
        "today": _today_html(today, current),
        "owner": escape(f"{name} 的手机"),
    }
    html = _TEMPLATE.read_text(encoding="utf-8")
    for key, value in values.items():
        html = html.replace(f"{{{{{key}}}}}", value)
    return await screenshot(html, PAGE_WIDTH)
