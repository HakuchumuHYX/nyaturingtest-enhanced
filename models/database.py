# nyaturingtest/models.py
from tortoise import fields
from tortoise.indexes import Index
from tortoise.models import Model


class SessionModel(Model):
    id = fields.CharField(pk=True, max_length=255)
    name = fields.CharField(max_length=255, default="terminus")
    role = fields.TextField(default="一个普通的AI助手")
    aliases = fields.JSONField(default=list)

    valence = fields.FloatField(default=0.0)
    arousal = fields.FloatField(default=0.0)
    dominance = fields.FloatField(default=0.0)

    chat_summary = fields.TextField(default="")
    last_speak_time = fields.DatetimeField(null=True)
    last_consolidated_time = fields.DatetimeField(null=True)
    chatting_state = fields.IntField(default=0)

    class Meta:
        table = "nyabot_sessions"


class UserProfileModel(Model):
    id = fields.IntField(pk=True)
    session = fields.ForeignKeyField("models.SessionModel", related_name="users")
    user_id = fields.CharField(max_length=255)

    valence = fields.FloatField(default=0.0)
    arousal = fields.FloatField(default=0.0)
    dominance = fields.FloatField(default=0.0)

    last_update_time = fields.DatetimeField(auto_now=True)
    interaction_count = fields.IntField(default=0)
    first_interaction_at = fields.DatetimeField(null=True)
    last_interaction_at = fields.DatetimeField(null=True)

    class Meta:
        table = "nyabot_user_profiles"
        unique_together = (("session", "user_id"),)


class InteractionLogModel(Model):
    id = fields.IntField(pk=True)
    user = fields.ForeignKeyField("models.UserProfileModel", related_name="interactions")
    timestamp = fields.DatetimeField()
    delta_valence = fields.FloatField()
    delta_arousal = fields.FloatField()
    delta_dominance = fields.FloatField()

    class Meta:
        table = "nyabot_interactions"
        indexes = (Index(fields=("timestamp",), name="idx_interactions_timestamp"),)


class GlobalMessageModel(Model):
    id = fields.IntField(pk=True)
    session = fields.ForeignKeyField("models.SessionModel", related_name="messages")
    user_name = fields.CharField(max_length=255)
    user_id = fields.CharField(max_length=255, default="")
    content = fields.TextField()
    time = fields.DatetimeField()
    msg_id = fields.CharField(max_length=255, default="")

    class Meta:
        table = "nyabot_global_messages"
        unique_together = (("session", "msg_id"),)
        indexes = (
            Index(fields=("time",), name="idx_messages_time"),
            Index(fields=("session_id", "time"), name="idx_messages_session_time"),
            Index(
                fields=("session_id", "user_id", "time"),
                name="idx_messages_session_user_time",
            ),
        )


class EnabledGroupModel(Model):
    """存储启用的群组ID"""
    group_id = fields.BigIntField(pk=True)

    class Meta:
        table = "nyabot_enabled_groups"


class TokenUsageModel(Model):
    """记录 Token 消耗"""
    id = fields.IntField(pk=True)
    session_id = fields.CharField(max_length=255)  # 群号
    model_name = fields.CharField(max_length=255)  # 模型名称
    provider = fields.CharField(max_length=64, default="")
    prompt_tokens = fields.IntField()
    completion_tokens = fields.IntField()
    prompt_cache_hit_tokens = fields.IntField(default=0)
    prompt_cache_miss_tokens = fields.IntField(default=0)
    reasoning_tokens = fields.IntField(default=0)
    finish_reason = fields.CharField(max_length=64, default="")
    timestamp = fields.DatetimeField(auto_now_add=True)  # 自动记录时间

    class Meta:
        table = "nyabot_token_usage"
        indexes = (
            Index(
                fields=("session_id", "timestamp"),
                name="idx_token_usage_session_time",
            ),
            Index(
                fields=("model_name", "timestamp"),
                name="idx_token_usage_model_time",
            ),
        )


class DailyTokenUsageModel(Model):
    """按自然日汇总的 Token 消耗，用于长期统计和安全清理明细。"""

    id = fields.IntField(pk=True)
    day = fields.DateField()
    session_id = fields.CharField(max_length=255)
    model_name = fields.CharField(max_length=255)
    provider = fields.CharField(max_length=64, default="")
    prompt_tokens = fields.BigIntField(default=0)
    completion_tokens = fields.BigIntField(default=0)
    prompt_cache_hit_tokens = fields.BigIntField(default=0)
    prompt_cache_miss_tokens = fields.BigIntField(default=0)
    reasoning_tokens = fields.BigIntField(default=0)
    request_count = fields.IntField(default=0)

    class Meta:
        table = "nyabot_daily_token_usage"
        unique_together = (("day", "session_id", "model_name", "provider"),)
        indexes = (
            Index(
                fields=("session_id", "day"),
                name="idx_daily_token_session_day",
            ),
            Index(fields=("model_name", "day"), name="idx_daily_token_model_day"),
        )
