# nyaturingtest/models.py
from tortoise import fields
from tortoise.models import Model


class SessionModel(Model):
    id = fields.CharField(pk=True, max_length=255)
    name = fields.CharField(max_length=255, default="terminus")
    role = fields.TextField(default="一个普通的AI助手")
    # [新增] 存储别名，使用 JSON 列表存储
    aliases = fields.JSONField(default=list)

    valence = fields.FloatField(default=0.5)
    arousal = fields.FloatField(default=0.5)
    dominance = fields.FloatField(default=0.5)

    chat_summary = fields.TextField(default="")
    last_speak_time = fields.DatetimeField(null=True)
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
