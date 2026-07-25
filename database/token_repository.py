from collections import defaultdict
from datetime import date, datetime, timedelta

from nonebot import logger
from tortoise.transactions import in_transaction

from ..core.metrics import metrics
from ..models.database import DailyTokenUsageModel, TokenUsageModel
from .token_stats_aggregation import TOKEN_FIELDS, merge_token_stats_by_model


def _empty_stats() -> dict[str, list[dict]]:
    return {
        "1d_local": [],
        "1d_global": [],
        "7d_local": [],
        "7d_global": [],
        "all_global": [],
    }


def _format_aggregate_rows(
    aggregate: dict[tuple[str, str], dict[str, int]],
) -> list[dict]:
    return merge_token_stats_by_model(aggregate)


class TokenUsageRepository:
    @staticmethod
    async def log_token_usage(
        session_id: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        prompt_cache_hit_tokens: int = 0,
        prompt_cache_miss_tokens: int = 0,
        reasoning_tokens: int = 0,
        finish_reason: str = "",
        provider: str = "",
    ):
        await TokenUsageRepository.log_token_usages(
            [
                {
                    "session_id": session_id,
                    "model_name": model_name,
                    "provider": provider,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "prompt_cache_hit_tokens": prompt_cache_hit_tokens,
                    "prompt_cache_miss_tokens": prompt_cache_miss_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "finish_reason": finish_reason,
                }
            ]
        )

    @staticmethod
    async def log_token_usages(rows: list[dict]):
        """In one transaction, append raw usage and atomically increment daily totals."""

        if not rows:
            return
        now = datetime.now().astimezone()
        day = now.date().isoformat()
        grouped: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
            lambda: {field: 0 for field in TOKEN_FIELDS} | {"request_count": 0}
        )
        for row in rows:
            key = (
                str(row.get("session_id") or ""),
                str(row.get("model_name") or ""),
                str(row.get("provider") or ""),
            )
            for field in TOKEN_FIELDS:
                grouped[key][field] += int(row.get(field, 0) or 0)
            grouped[key]["request_count"] += 1

        try:
            async with in_transaction("default") as conn:
                await TokenUsageModel.bulk_create(
                    [
                        TokenUsageModel(
                            session_id=row.get("session_id", ""),
                            model_name=row.get("model_name", ""),
                            provider=row.get("provider", ""),
                            prompt_tokens=row.get("prompt_tokens", 0),
                            completion_tokens=row.get("completion_tokens", 0),
                            prompt_cache_hit_tokens=row.get(
                                "prompt_cache_hit_tokens", 0
                            ),
                            prompt_cache_miss_tokens=row.get(
                                "prompt_cache_miss_tokens", 0
                            ),
                            reasoning_tokens=row.get("reasoning_tokens", 0),
                            finish_reason=row.get("finish_reason", ""),
                        )
                        for row in rows
                    ],
                    using_db=conn,
                )
                await conn.execute_many(
                    """
                    INSERT INTO nyabot_daily_token_usage (
                        day, session_id, model_name, provider,
                        prompt_tokens, completion_tokens,
                        prompt_cache_hit_tokens, prompt_cache_miss_tokens,
                        reasoning_tokens, request_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(day, session_id, model_name, provider)
                    DO UPDATE SET
                        prompt_tokens = prompt_tokens + excluded.prompt_tokens,
                        completion_tokens = completion_tokens + excluded.completion_tokens,
                        prompt_cache_hit_tokens = prompt_cache_hit_tokens + excluded.prompt_cache_hit_tokens,
                        prompt_cache_miss_tokens = prompt_cache_miss_tokens + excluded.prompt_cache_miss_tokens,
                        reasoning_tokens = reasoning_tokens + excluded.reasoning_tokens,
                        request_count = request_count + excluded.request_count
                    """,
                    [
                        [
                            day,
                            session_id,
                            model_name,
                            provider,
                            totals["prompt_tokens"],
                            totals["completion_tokens"],
                            totals["prompt_cache_hit_tokens"],
                            totals["prompt_cache_miss_tokens"],
                            totals["reasoning_tokens"],
                            totals["request_count"],
                        ]
                        for (
                            session_id,
                            model_name,
                            provider,
                        ), totals in grouped.items()
                    ],
                )
        except Exception as e:
            metrics.db_write_failure += 1
            logger.error(f"[Repo] 记录 Token 消耗失败: {e}")

    @staticmethod
    async def get_token_stats(
        group_id: str | int,
        model_names: list[str] | None = None,
    ) -> dict:
        """Read all five views from one compact daily-aggregate query."""

        result = _empty_stats()
        group_id_str = str(group_id)
        today = date.today()
        one_day_cutoff = today
        seven_day_cutoff = today - timedelta(days=6)
        try:
            query = DailyTokenUsageModel.all()
            if model_names:
                query = query.filter(model_name__in=model_names)
            rows = await query.values(
                "day",
                "session_id",
                "model_name",
                "provider",
                *TOKEN_FIELDS,
            )

            buckets = {
                name: defaultdict(
                    lambda: {field: 0 for field in TOKEN_FIELDS}
                )
                for name in result
            }
            for row in rows:
                row_day = row["day"]
                if isinstance(row_day, str):
                    row_day = date.fromisoformat(row_day)
                key = (
                    str(row.get("model_name") or ""),
                    str(row.get("provider") or ""),
                )
                is_local = str(row.get("session_id") or "") == group_id_str
                targets = ["all_global"]
                if row_day >= seven_day_cutoff:
                    targets.append("7d_global")
                    if is_local:
                        targets.append("7d_local")
                if row_day >= one_day_cutoff:
                    targets.append("1d_global")
                    if is_local:
                        targets.append("1d_local")
                for bucket in targets:
                    for field in TOKEN_FIELDS:
                        buckets[bucket][key][field] += int(row.get(field, 0) or 0)

            for name in result:
                result[name] = _format_aggregate_rows(buckets[name])
        except Exception as e:
            logger.error(f"[Repo] 查询 Token 统计失败: {e}")
        return result
