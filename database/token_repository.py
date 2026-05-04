from datetime import datetime, timedelta

from nonebot import logger
from tortoise.functions import Sum

from ..core.metrics import metrics
from ..models.database import TokenUsageModel


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
        """记录 Token 消耗"""
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
        """批量记录 Token 消耗"""
        try:
            if not rows:
                return
            await TokenUsageModel.bulk_create(
                [
                    TokenUsageModel(
                        session_id=row.get("session_id", ""),
                        model_name=row.get("model_name", ""),
                        provider=row.get("provider", ""),
                        prompt_tokens=row.get("prompt_tokens", 0),
                        completion_tokens=row.get("completion_tokens", 0),
                        prompt_cache_hit_tokens=row.get("prompt_cache_hit_tokens", 0),
                        prompt_cache_miss_tokens=row.get("prompt_cache_miss_tokens", 0),
                        reasoning_tokens=row.get("reasoning_tokens", 0),
                        finish_reason=row.get("finish_reason", ""),
                    )
                    for row in rows
                ]
            )
        except Exception as e:
            metrics.db_write_failure += 1
            logger.error(f"[Repo] 记录 Token 消耗失败: {e}")

    @staticmethod
    async def get_token_stats(group_id: str | int, model_names: list[str] | None = None) -> dict:
        result = {
            "1d_local": [],
            "1d_global": [],
            "7d_local": [],
            "7d_global": [],
            "all_global": [],
        }
        group_id_str = str(group_id)
        now = datetime.now()
        one_day_ago = now - timedelta(days=1)
        seven_days_ago = now - timedelta(days=7)

        async def _query(filter_kwargs):
            if model_names:
                filter_kwargs["model_name__in"] = model_names

            stats = (
                await TokenUsageModel.filter(**filter_kwargs)
                .annotate(
                    total_prompt=Sum("prompt_tokens"),
                    total_completion=Sum("completion_tokens"),
                    total_cache_hit=Sum("prompt_cache_hit_tokens"),
                    total_reasoning=Sum("reasoning_tokens"),
                )
                .group_by("model_name", "provider")
                .values(
                    "model_name",
                    "provider",
                    "total_prompt",
                    "total_completion",
                    "total_cache_hit",
                    "total_reasoning",
                )
            )

            rows = []
            for s in stats:
                prompt = s["total_prompt"] or 0
                completion = s["total_completion"] or 0
                cache_hit = s["total_cache_hit"] or 0
                rows.append(
                    {
                        "model": s["model_name"],
                        "provider": s.get("provider") or "",
                        "prompt": prompt,
                        "completion": completion,
                        "reasoning": s["total_reasoning"] or 0,
                        "cache_hit": cache_hit,
                        "total": prompt + completion,
                    }
                )
            return rows

        try:
            result["1d_local"] = await _query({"session_id": group_id_str, "timestamp__gte": one_day_ago})
            result["1d_global"] = await _query({"timestamp__gte": one_day_ago})
            result["7d_local"] = await _query({"session_id": group_id_str, "timestamp__gte": seven_days_ago})
            result["7d_global"] = await _query({"timestamp__gte": seven_days_ago})
            result["all_global"] = await _query({})
        except Exception as e:
            logger.error(f"[Repo] 查询 Token 统计失败: {e}")

        return result
