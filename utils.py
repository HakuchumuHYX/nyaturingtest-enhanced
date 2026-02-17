# nyaturingtest/utils.py
import json
import re
import ssl
from datetime import datetime

import chinese_calendar as chinesecalendar
import httpx
from nonebot import logger
from .mem import Message

# 全局客户端变量
_GLOBAL_HTTP_CLIENT: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """获取全局优化的 HTTP 客户端"""
    global _GLOBAL_HTTP_CLIENT
    if _GLOBAL_HTTP_CLIENT is None:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        ssl_context.set_ciphers("ALL:@SECLEVEL=1")
        _GLOBAL_HTTP_CLIENT = httpx.AsyncClient(
            verify=ssl_context,
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=50, max_connections=100)
        )
    return _GLOBAL_HTTP_CLIENT


async def close_http_client():
    """关闭全局 HTTP 客户端"""
    global _GLOBAL_HTTP_CLIENT
    if _GLOBAL_HTTP_CLIENT:
        await _GLOBAL_HTTP_CLIENT.aclose()
        _GLOBAL_HTTP_CLIENT = None
        logger.info("全局 HTTP 客户端已关闭")


def sanitize_text(text: str) -> str:
    """清洗文本，移除无法编码的字符"""
    if not text: return ""
    try:
        return text.encode('utf-8', 'ignore').decode('utf-8')
    except:
        return ""


def escape_for_prompt(text: str) -> str:
    """转义文本以安全放入 JSON 或 Prompt"""
    if not text: return ""
    return text.replace('"', '\\"').replace('\n', ' ')


def smart_split_text(text: str, max_chars: int = 40) -> list[str]:
    """
    严格断句逻辑：
    只要遇到句号、问号、感叹号等标点，强制进行切分，不合并短句。
    """
    text = text.strip()
    if not text:
        return []

    # 正则：匹配标点符号 [。！？!?~\n]，(?<=...) 为后视断言，保留标点在前半句
    raw_parts = re.split(r'(?<=[。！？!?~\n])\s*', text)

    final_parts = []

    for part in raw_parts:
        part = part.strip()
        if not part:
            continue

        final_parts.append(part)

    return final_parts if final_parts else [text]


def extract_and_parse_json(text: str) -> dict | list | None:
    """
    提取并解析 JSON，自动去除 Markdown 代码块和思考过程标签
    """
    if not text:
        return None

    # 1. 强力去除 <think>...</think> 标签及其内容 (支持跨行)
    # flag=re.DOTALL 让 . 可以匹配换行符
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 2. 去除 Markdown 代码块包裹
    # 匹配 ```json ... ``` 或 ``` ... ```，捕获中间的内容
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if match:
        text = match.group(1)
    else:
        # 如果没匹配到成对的 ```，尝试单边清理
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)

    # 3. 寻找最外层的 { } 或 [ ]
    try:
        start_idx = text.find("{")
        list_start_idx = text.find("[")

        # 判断是对象还是列表
        if start_idx != -1 and (list_start_idx == -1 or start_idx < list_start_idx):
            # 提取对象
            end_idx = text.rfind("}")
            if end_idx != -1:
                json_str = text[start_idx: end_idx + 1]
                return json.loads(json_str)
        elif list_start_idx != -1:
            # 提取列表
            end_idx = text.rfind("]")
            if end_idx != -1:
                json_str = text[list_start_idx: end_idx + 1]
                return json.loads(json_str)

    except json.JSONDecodeError as e:
        logger.warning(f"JSON 解析失败: {e}\n原文: {text}")
    except Exception as e:
        logger.error(f"JSON 提取未知错误: {e}")

    return None


def estimate_split_count(text: str) -> int:
    """
    估算实际发送的消息条数
    """
    if not text:
        return 0

    raw_parts = re.split(r'(?<=[。！？!?.~\n])\s*', text)
    final_parts = []

    for part in raw_parts:
        part = part.strip()
        if not part: continue
        final_parts.append(part)

    return len(final_parts) if final_parts else 1


def check_relevance(bot_name: str, aliases: list[str], messages: list[Message]) -> bool:
    """
    检查这一批消息中是否有与机器人强相关的内容
    支持检查 bot_name 和 aliases (别名)
    增加忽略大小写的提及判定
    """
    # 合并主名和别名作为所有触发词
    triggers = [bot_name]
    if aliases:
        triggers.extend(aliases)

    # 过滤掉空字符串，防止误触
    triggers = [t for t in triggers if t and t.strip()]

    for msg in messages:
        # 统一转为小写，实现忽略大小写匹配
        content = msg.content.lower()

        for trigger in triggers:
            t = trigger.lower()
            if t in content:
                return True


    return False


def should_store_memory(content: str) -> bool:
    """
    判断记忆是否值得存储
    过滤低质量内容，避免存储无意义的语气词
    """
    if not content:
        return False
    
    content = content.strip()
    
    # 太短不存（少于10字符）
    if len(content) < 10:
        return False
    
    # 纯语气词/无意义内容不存
    noise_words = {
        "好的", "好", "嗯", "嗯嗯", "哦", "哦哦", 
        "ok", "OK", "Ok", "收到", "了解", "明白",
        "哈哈", "哈哈哈", "233", "666", "厉害",
        "是的", "对", "对的", "是啊", "好吧",
        "行", "可以", "没问题", "好呀", "好哒",
        "谢谢", "感谢", "辛苦了", "拜拜", "再见",
        "早", "晚安", "午安", "早安", "晚上好",
    }
    
    if content.lower() in noise_words:
        return False
    
    return True


def calculate_dynamic_k(interaction_count: int, memory_count: int, days_since_first: int) -> int:
    """
    综合计算检索条数，上限根据记忆量自动调整
    
    Args:
        interaction_count: 交互次数
        memory_count: 记忆总量
        days_since_first: 距离首次交互的天数
    
    Returns:
        检索条数 k
    """
    # 1. 根据记忆量决定上限
    if memory_count <= 10:
        max_limit = memory_count  # 记忆少，全取
    elif memory_count <= 30:
        max_limit = 20
    elif memory_count <= 50:
        max_limit = 30
    else:
        max_limit = 40  # 记忆非常丰富，取 Top 40
    
    # 2. 计算综合分数
    base = 5
    
    # 交互贡献：每 50 次 +1，最多 +6
    interaction_bonus = min(interaction_count // 50, 6)
    
    # 记忆贡献：每 10 条 +1，最多 +8
    memory_bonus = min(memory_count // 10, 8)
    
    # 时间贡献
    if days_since_first > 90:
        time_bonus = 4
    elif days_since_first > 30:
        time_bonus = 3
    elif days_since_first > 7:
        time_bonus = 2
    else:
        time_bonus = 0
    
    # 3. 求和并限制范围
    k = base + interaction_bonus + memory_bonus + time_bonus
    return max(5, min(k, max_limit))


def get_time_description(dt: datetime) -> str:
    """
    生成详细的时间描述，包含节假日判断
    """
    # 星期映射
    weekday_map = {0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五", 5: "周六", 6: "周日"}
    weekday_str = weekday_map[dt.weekday()]

    # 格式化基础时间
    time_str = dt.strftime("%Y年%m月%d日 %H:%M")

    # 判断时段
    hour = dt.hour
    period = ""
    if 0 <= hour < 6:
        period = "深夜"
    elif 6 <= hour < 9:
        period = "清晨"
    elif 9 <= hour < 12:
        period = "上午"
    elif 12 <= hour < 14:
        period = "中午"
    elif 14 <= hour < 18:
        period = "下午"
    elif 18 <= hour < 23:
        period = "晚上"
    else:
        period = "深夜"

    # 判断节假日
    try:
        # chinesecalendar.is_holiday: True if holiday (including weekends)
        # chinesecalendar.get_holiday_detail: (bool, name)
        is_rest = chinesecalendar.is_holiday(dt.date())
        on_holiday, holiday_name = chinesecalendar.get_holiday_detail(dt.date())

        status_str = "工作日"
        if is_rest:
            if holiday_name:
                status_str = f"节假日({holiday_name})"
            else:
                status_str = "周末休息" if dt.weekday() >= 5 else "休息日"
        else:
            status_str = "工作日"

    except Exception as e:
        logger.warning(f"节假日判断失败: {e}")
        status_str = "周末" if dt.weekday() >= 5 else "工作日"

    return f"{time_str} {weekday_str} [{period}] [{status_str}]"


async def render_token_stats_card(
    *,
    stats: dict,
    watermark: str | None = None,
    width: int = 750
) -> bytes:
    """渲染 Token 统计卡片
    
    Args:
        stats: 统计数据字典（包含 1d_local, 1d_global, 7d_local, 7d_global, all_global）
        watermark: 右下角水印文字（可选）
        width: 图片宽度
    
    Returns:
        PNG bytes
    """
    from io import BytesIO
    import sys
    sys.path.insert(0, str(__file__).replace("nyaturingtest\\utils.py", ""))
    from utils.draw.plot import Canvas, VSplit, HSplit, TextBox, TextStyle, FillBg, RoundRectBg, Spacer
    
    # 配色方案
    canvas_bg = (240, 245, 250, 255)  # 浅灰蓝
    card_bg = (255, 255, 255, 255)    # 白色
    card_border = (200, 215, 230, 255)
    
    section_bg = (248, 251, 255, 255)  # 超浅蓝
    text_main = (30, 40, 50, 255)
    text_sub = (90, 105, 120, 255)
    text_muted = (140, 155, 170, 255)
    
    # 模型颜色映射（根据模型名称）
    model_colors = {
        "chat": (100, 150, 255, 255),     # 蓝色
        "vlm": (150, 100, 255, 255),      # 紫色  
        "feedback": (255, 150, 100, 255), # 橙色
    }
    
    # 文本样式（使用绝对路径）
    import os
    font_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "nyaturingtest")
    
    title_style = TextStyle(font=os.path.join(font_dir, "SourceHanSansCN-Heavy.ttf"), size=36, color=text_main)
    section_title_style = TextStyle(font=os.path.join(font_dir, "SourceHanSansCN-Bold.ttf"), size=24, color=text_main)
    label_style = TextStyle(font=os.path.join(font_dir, "SourceHanSansCN-Regular.ttf"), size=18, color=text_sub)
    value_style = TextStyle(font=os.path.join(font_dir, "SourceHanSansCN-Bold.ttf"), size=20, color=text_main)
    model_style = TextStyle(font=os.path.join(font_dir, "SourceHanSansCN-Bold.ttf"), size=18, color=text_main)
    watermark_style = TextStyle(font=os.path.join(font_dir, "SourceHanSansCN-Regular.ttf"), size=14, color=text_muted)
    
    outer_margin = 24
    card_padding = 22
    content_w = width - outer_margin * 2 - card_padding * 2
    
    items = []
    
    # 标题
    items.append(TextBox("Token 使用统计", style=title_style).set_w(content_w).set_padding(0))
    items.append(Spacer(1, 16))
    
    # 辅助函数：格式化单个时间段的统计
    def render_period(title: str, local_stats: list, global_stats: list):
        period_items = []
        
        # 标题
        period_items.append(
            TextBox(title, style=section_title_style).set_w(content_w).set_padding((0, 8))
        )
        
        # 本群消耗
        if local_stats:
            period_items.append(TextBox("【本群消耗】", style=label_style).set_w(content_w).set_padding((0, 4)))
            for item in local_stats:
                model_rows = []
                model_rows.append(
                    TextBox(f"模型: {item['model']}", style=model_style).set_w(content_w).set_padding((8, 2))
                )
                model_rows.append(
                    TextBox(
                        f"  Prompt: {item['prompt']:,}  |  Completion: {item['completion']:,}  |  Total: {item['total']:,}",
                        style=value_style
                    ).set_w(content_w).set_padding((16, 2))
                )
                period_items.extend(model_rows)
        else:
            period_items.append(TextBox("【本群消耗】无数据", style=label_style).set_w(content_w).set_padding((0, 4)))
        
        period_items.append(Spacer(1, 8))
        
        # 全局消耗
        if global_stats:
            period_items.append(TextBox("【全局所有群消耗】", style=label_style).set_w(content_w).set_padding((0, 4)))
            for item in global_stats:
                model_rows = []
                model_rows.append(
                    TextBox(f"模型: {item['model']}", style=model_style).set_w(content_w).set_padding((8, 2))
                )
                model_rows.append(
                    TextBox(
                        f"  Prompt: {item['prompt']:,}  |  Completion: {item['completion']:,}  |  Total: {item['total']:,}",
                        style=value_style
                    ).set_w(content_w).set_padding((16, 2))
                )
                period_items.extend(model_rows)
        else:
            period_items.append(TextBox("【全局所有群消耗】无数据", style=label_style).set_w(content_w).set_padding((0, 4)))
        
        # 打包成一个圆角区域
        section = VSplit(items=period_items, sep=4, item_size_mode="fixed") \
            .set_w(content_w) \
            .set_padding(16) \
            .set_bg(RoundRectBg(fill=section_bg, radius=16))
        
        return section
    
    # 24小时统计
    items.append(render_period("24小时统计", stats.get("1d_local", []), stats.get("1d_global", [])))
    items.append(Spacer(1, 12))
    
    # 7天统计
    items.append(render_period("7天统计", stats.get("7d_local", []), stats.get("7d_global", [])))
    items.append(Spacer(1, 12))
    
    # 总计统计
    all_stats = stats.get("all_global", [])
    if all_stats:
        total_items = []
        total_items.append(
            TextBox("历史总消耗", style=section_title_style).set_w(content_w).set_padding((0, 8))
        )
        for item in all_stats:
            model_rows = []
            model_rows.append(
                TextBox(f"模型: {item['model']}", style=model_style).set_w(content_w).set_padding((8, 2))
            )
            model_rows.append(
                TextBox(
                    f"  Prompt: {item['prompt']:,}  |  Completion: {item['completion']:,}  |  Total: {item['total']:,}",
                    style=value_style
                ).set_w(content_w).set_padding((16, 2))
            )
            total_items.extend(model_rows)
        
        total_section = VSplit(items=total_items, sep=4, item_size_mode="fixed") \
            .set_w(content_w) \
            .set_padding(16) \
            .set_bg(RoundRectBg(fill=section_bg, radius=16))
        
        items.append(total_section)
    
    # 水印
    items.append(Spacer(1, 12))
    if watermark is None:
        watermark = "Generated by HakuBot"
    if watermark:
        items.append(
            TextBox(watermark, style=watermark_style).set_w(content_w).set_content_align("r").set_padding(0)
        )
    
    # 卡片容器
    card = VSplit(items=items, sep=4, item_size_mode="fixed") \
        .set_w(width - outer_margin * 2) \
        .set_padding(card_padding) \
        .set_margin(outer_margin) \
        .set_bg(RoundRectBg(fill=card_bg, radius=26, stroke=card_border, stroke_width=2))
    
    # 画布
    canvas = Canvas(w=width, h=None, bg=FillBg(canvas_bg))
    canvas.set_items([card]).set_content_align("c")
    
    img = await canvas.get_img()
    
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

