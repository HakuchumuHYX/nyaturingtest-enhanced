from datetime import datetime, timedelta

from nonebot import logger

from ..models.database import GlobalMessageModel, InteractionLogModel, TokenUsageModel


RETENTION_DISABLED_DAYS = 0

# 原始明细保留天数；0 表示永不清理
RAW_MESSAGE_RETENTION_DAYS = 180
RAW_INTERACTION_RETENTION_DAYS = 180
TOKEN_USAGE_RETENTION_DAYS = 90


async def _delete_older_than(model, field_name: str, days: int) -> int:
    if days <= RETENTION_DISABLED_DAYS:
        return 0
    cutoff = datetime.now() - timedelta(days=days)
    return await model.filter(**{f"{field_name}__lt": cutoff}).delete()


async def cleanup_raw_data_retention() -> dict[str, int]:
    """按保留期清理原始数据库行。

    刻意不触碰长期向量记忆：语义记忆的生命周期由向量库清理路径负责。
    """

    result = {
        "messages": 0,
        "interactions": 0,
        "token_usage": 0,
    }

    try:
        result["messages"] = await _delete_older_than(
            GlobalMessageModel,
            "time",
            RAW_MESSAGE_RETENTION_DAYS,
        )
        result["interactions"] = await _delete_older_than(
            InteractionLogModel,
            "timestamp",
            RAW_INTERACTION_RETENTION_DAYS,
        )
        result["token_usage"] = await _delete_older_than(
            TokenUsageModel,
            "timestamp",
            TOKEN_USAGE_RETENTION_DAYS,
        )
    except Exception as e:
        logger.error(f"[Retention] 原始数据库行清理失败: {e}")
        raise

    if any(result.values()):
        logger.info(
            "[Retention] 清理原始数据库行: "
            f"messages={result['messages']}, "
            f"interactions={result['interactions']}, "
            f"token_usage={result['token_usage']}"
        )
    return result
