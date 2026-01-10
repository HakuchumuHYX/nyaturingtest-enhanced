# nyaturingtest/models.py
from tortoise import fields, models


class SessionModel(models.Model):
    """会话表 (对应 Session)"""
    id = fields.CharField(pk=True, max_length=255)  # 例如 "group_123456"
    name = fields.CharField(max_length=255, default="terminus")
    role = fields.TextField(default="一个男性人类")
    # 全局情绪 VAD
    valence = fields.FloatField(default=0.0)
    arousal = fields.FloatField(default=0.0)
    dominance = fields.FloatField(default=0.0)

    chat_summary = fields.TextField(default="")
    last_speak_time = fields.DatetimeField(null=True)
    # 状态 (存储枚举的 value, 0, 1, 2)
    chatting_state = fields.IntField(default=0)

    class Meta:
        table = "nyabot_sessions"


class UserProfileModel(models.Model):
    """用户档案表 (对应 PersonProfile)"""
    # 联合主键难以处理，这里使用自增ID，加唯一索引
    id = fields.IntField(pk=True)
    session = fields.ForeignKeyField("models.SessionModel", related_name="profiles")
    user_id = fields.CharField(max_length=255)  # QQ号

    # 情绪 VAD
    valence = fields.FloatField(default=0.0)
    arousal = fields.FloatField(default=0.0)
    dominance = fields.FloatField(default=0.0)

    last_update_time = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "nyabot_users"
        unique_together = (("session", "user_id"),)


class InteractionLogModel(models.Model):
    """
    交互记录表 (对应 Impression)
    """
    id = fields.IntField(pk=True)
    user = fields.ForeignKeyField("models.UserProfileModel", related_name="interactions")
    timestamp = fields.DatetimeField(auto_now_add=True)

    # 记录当时带来的情绪冲击 delta
    delta_valence = fields.FloatField(default=0.0)
    delta_arousal = fields.FloatField(default=0.0)
    delta_dominance = fields.FloatField(default=0.0)

    class Meta:
        table = "nyabot_interactions"


class GlobalMessageModel(models.Model):
    """全局短时记忆表 (替代 global_memory 中的 messages)"""
    id = fields.IntField(pk=True)
    session = fields.ForeignKeyField("models.SessionModel", related_name="messages")
    user_name = fields.CharField(max_length=255)
    user_id = fields.CharField(max_length=255, default="")
    content = fields.TextField()
    time = fields.DatetimeField(auto_now_add=True)
    msg_id = fields.CharField(max_length=255, default="")

    class Meta:
        table = "nyabot_global_messages"
