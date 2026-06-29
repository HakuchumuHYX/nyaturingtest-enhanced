from datetime import datetime, timedelta

from nonebot import logger

from ..config import get_runtime_settings
from ..models.database import GlobalMessageModel, InteractionLogModel, TokenUsageModel


RETENTION_DISABLED_DAYS = 0


def _retention_days(settings: dict, key: str) -> int:
    try:
        days = int(settings.get(key, RETENTION_DISABLED_DAYS) or RETENTION_DISABLED_DAYS)
    except (TypeError, ValueError):
        return RETENTION_DISABLED_DAYS
    return max(RETENTION_DISABLED_DAYS, days)


async def _delete_older_than(model, field_name: str, days: int) -> int:
    if days <= RETENTION_DISABLED_DAYS:
        return 0
    cutoff = datetime.now() - timedelta(days=days)
    return await model.filter(**{f"{field_name}__lt": cutoff}).delete()


async def cleanup_raw_data_retention(settings: dict | None = None) -> dict[str, int]:
    """Delete old raw database rows according to opt-in retention settings.

    This deliberately does not touch long-term vector memory. Semantic memory
    lifecycle remains owned by the vector store cleanup path.
    """
    runtime = get_runtime_settings() if settings is None else settings
    result = {
        "messages": 0,
        "interactions": 0,
        "token_usage": 0,
    }

    try:
        result["messages"] = await _delete_older_than(
            GlobalMessageModel,
            "time",
            _retention_days(runtime, "raw_message_retention_days"),
        )
        result["interactions"] = await _delete_older_than(
            InteractionLogModel,
            "timestamp",
            _retention_days(runtime, "raw_interaction_retention_days"),
        )
        result["token_usage"] = await _delete_older_than(
            TokenUsageModel,
            "timestamp",
            _retention_days(runtime, "token_usage_retention_days"),
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
