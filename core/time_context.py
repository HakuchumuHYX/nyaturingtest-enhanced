from datetime import datetime

import chinese_calendar
from nonebot import logger


def get_time_description(value: datetime) -> str:
    weekday = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")[
        value.weekday()
    ]
    hour = value.hour
    if hour < 6 or hour >= 23:
        period = "深夜"
    elif hour < 9:
        period = "清晨"
    elif hour < 12:
        period = "上午"
    elif hour < 14:
        period = "中午"
    elif hour < 18:
        period = "下午"
    else:
        period = "晚上"
    try:
        is_rest = chinese_calendar.is_holiday(value.date())
        _, holiday_name = chinese_calendar.get_holiday_detail(value.date())
        if is_rest:
            status = (
                f"节假日({holiday_name})"
                if holiday_name
                else "周末休息" if value.weekday() >= 5 else "休息日"
            )
        else:
            status = "工作日"
    except Exception as e:
        logger.warning(f"节假日判断失败: {e}")
        status = "周末" if value.weekday() >= 5 else "工作日"
    return f"{value:%Y年%m月%d日 %H:%M} {weekday} [{period}] [{status}]"
