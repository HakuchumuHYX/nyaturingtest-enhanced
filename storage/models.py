from datetime import datetime

from tortoise import fields
from tortoise.indexes import Index
from tortoise.models import Model


class SessionModel(Model):
    id = fields.CharField(pk=True, max_length=255)
    # 预设文件名，空串是没设人设（不说话）；名字、别名等运行时从预设读，不存快照，免得改了 json 不同步
    preset = fields.CharField(max_length=255, default="")

    # 角色在群里的心情；读取前按 emotion_updated_at 结算回落，见 chat/domain.py
    valence = fields.FloatField(default=0.0)
    arousal = fields.FloatField(default=0.3)
    dominance = fields.FloatField(default=0.0)
    emotion_updated_at = fields.DatetimeField(default=datetime.now)

    chat_summary = fields.TextField(default="")
    last_speak_time = fields.DatetimeField(null=True)
    # 已读水位：Feedback 读到的最后一条消息时间
    last_read_time = fields.DatetimeField(null=True)
    # 群志只由每日整理任务写入；水位是已整理到的最新碎片 created_at
    group_notes = fields.TextField(default="")
    notes_summarized_until = fields.DatetimeField(null=True)
    # 分段整理已读到的最后一条消息时间，只由 memory/episodes.py 写入
    episodes_until = fields.DatetimeField(null=True)

    class Meta:
        table = "nyabot_sessions"


class UserProfileModel(Model):
    id = fields.IntField(pk=True)
    session = fields.ForeignKeyField("models.SessionModel", related_name="users")
    user_id = fields.CharField(max_length=255)

    # 长期好感度 -100～100；负值读取前按 affinity_updated_at 结算回升
    affinity = fields.FloatField(default=0.0)
    affinity_updated_at = fields.DatetimeField(default=datetime.now)
    interaction_count = fields.IntField(default=0)
    first_interaction_at = fields.DatetimeField(null=True)
    last_interaction_at = fields.DatetimeField(null=True)

    # 长期档案只由每日整理任务写入；水位是已整理到的最新碎片 created_at
    summary = fields.TextField(default="")
    summarized_until = fields.DatetimeField(null=True)

    class Meta:
        table = "nyabot_user_profiles"
        unique_together = (("session", "user_id"),)


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


class MemoryModel(Model):
    """长期记忆碎片；向量与元数据同表，检索时整群加载成矩阵做暴力余弦。"""

    id = fields.CharField(pk=True, max_length=36)
    session = fields.ForeignKeyField("models.SessionModel", related_name="memories")
    content = fields.TextField()
    # 归一化后的 float32 字节；NULL 表示 embedding 调用失败，由每日维护补算
    embedding = fields.BinaryField(null=True)
    embedding_model = fields.CharField(max_length=255, default="")
    category = fields.CharField(max_length=32)
    subject_user_id = fields.CharField(max_length=255, default="")
    subject_user_name = fields.CharField(max_length=255, default="")
    speaker_user_id = fields.CharField(max_length=255, default="")
    speaker_user_name = fields.CharField(max_length=255, default="")
    confidence = fields.FloatField(default=1.0)
    importance = fields.FloatField(default=0.0)
    date = fields.IntField()  # YYYYMMDD，检索衰减按它算
    expires_at = fields.DatetimeField()
    # 依据的新消息 msg_id，空格分隔；事后核对时据此找回原话（历史行为空）
    source_msg_ids = fields.TextField(default="")
    # 和这条记忆有关的 QQ 号，空格分隔：事实是主体和说话人，episode 是参与聊天的群友；按人检索只看它
    participant_ids = fields.TextField(default="")
    # 更正条：替换掉被纠正的旧记忆，整理档案/群志时据此删改旧说法
    is_correction = fields.BooleanField(default=False)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "nyabot_memories"
        indexes = (
            Index(
                fields=("session_id", "subject_user_id"),
                name="idx_memories_session_subject",
            ),
            Index(fields=("expires_at",), name="idx_memories_expires_at"),
        )


class PersonaLifeModel(Model):
    """人生线：每个预设一份，各群共享，只由每日整理写入（memory/persona.py）。"""

    preset = fields.CharField(pk=True, max_length=255)
    text = fields.TextField(default="")
    last_digest = fields.DateField(null=True)
    # 各群抽出、还没整理进人生线的自我陈述；人生线整理失败时留到下次
    pending_facts = fields.JSONField(default=list)

    class Meta:
        table = "nyabot_persona_life"


class PersonaGroupModel(Model):
    """群里的她：每个（群, 预设）一份，切走预设再切回来还在。"""

    id = fields.IntField(pk=True)
    session = fields.ForeignKeyField("models.SessionModel", related_name="persona_groups")
    preset = fields.CharField(max_length=255)
    text = fields.TextField(default="")
    # 最近几天的观察日记，跨天反复出现的才收进群层
    journal = fields.JSONField(default=list)
    # 已整理到的最后一个自然日
    digested_until = fields.DateField(null=True)

    class Meta:
        table = "nyabot_persona_group"
        unique_together = (("session", "preset"),)


class PersonaHistoryModel(Model):
    """人生线和群层的每次重写，只追加，用于回看和回滚。"""

    id = fields.IntField(pk=True)
    scope = fields.CharField(max_length=16)  # life / group
    key = fields.CharField(max_length=255)  # 预设文件名，或「群号:预设文件名」
    date = fields.DateField()
    text = fields.TextField()
    changes = fields.TextField(default="")

    class Meta:
        table = "nyabot_persona_history"
        indexes = (Index(fields=("scope", "key", "date"), name="idx_persona_history_key"),)


class StickerModel(Model):
    """表情包库，兼做黑名单；文件在 data 目录的 stickers/<digest>.<ext>，所有群共用。"""

    # 原始字节 SHA-1，和图片描述缓存同一个 key
    digest = fields.CharField(pk=True, max_length=40)
    ext = fields.CharField(max_length=8, default="")
    # 表情、动作、语气、配字；挑图的向量只按这段算
    reaction = fields.TextField(default="")
    # 外观，不含角色名
    subject = fields.TextField(default="")
    tags = fields.JSONField(default=list)
    # library:<素材库文件夹> / group:<群号>
    source = fields.CharField(max_length=255, default="")
    # 归一化后的 float32 字节；NULL 表示 embedding 调用失败，启动时补算
    embedding = fields.BinaryField(null=True)
    embedding_model = fields.CharField(max_length=255, default="")
    # 被 stickerdel 删过：文件、描述、向量都已清掉，再见到只描述不收
    blocked = fields.BooleanField(default=False)
    created_at = fields.DatetimeField(auto_now_add=True)
    last_sent_at = fields.DatetimeField(null=True)

    class Meta:
        table = "nyabot_stickers"


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
    prompt_tokens = fields.BigIntField(default=0)
    completion_tokens = fields.BigIntField(default=0)
    prompt_cache_hit_tokens = fields.BigIntField(default=0)
    prompt_cache_miss_tokens = fields.BigIntField(default=0)
    reasoning_tokens = fields.BigIntField(default=0)
    request_count = fields.IntField(default=0)

    class Meta:
        table = "nyabot_daily_token_usage"
        unique_together = (("day", "session_id", "model_name"),)
        indexes = (
            Index(
                fields=("session_id", "day"),
                name="idx_daily_token_session_day",
            ),
            Index(fields=("model_name", "day"), name="idx_daily_token_model_day"),
        )
