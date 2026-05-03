from nonebot import logger
from tortoise import Tortoise


SCHEMA_VERSION = 2


async def _execute_ignore_duplicate(conn, statement: str):
    try:
        await conn.execute_query(statement)
    except Exception as e:
        text = str(e).lower()
        if "duplicate column" not in text and "already exists" not in text:
            logger.warning(f"[Migration] skipped statement: {e}")


async def ensure_schema_version():
    conn = Tortoise.get_connection("default")
    await conn.execute_query(
        "CREATE TABLE IF NOT EXISTS nyabot_schema_version (id INT PRIMARY KEY, version INT NOT NULL)"
    )
    rows = await conn.execute_query_dict("SELECT version FROM nyabot_schema_version WHERE id=1")
    current = int(rows[0]["version"]) if rows else 0

    if current < 1:
        await conn.execute_query("INSERT OR REPLACE INTO nyabot_schema_version (id, version) VALUES (1, 1)")
        current = 1

    if current < 2:
        for statement in [
            "ALTER TABLE nyabot_token_usage ADD COLUMN provider VARCHAR(64) NOT NULL DEFAULT ''",
            "ALTER TABLE nyabot_token_usage ADD COLUMN prompt_cache_hit_tokens INT NOT NULL DEFAULT 0",
            "ALTER TABLE nyabot_token_usage ADD COLUMN prompt_cache_miss_tokens INT NOT NULL DEFAULT 0",
            "ALTER TABLE nyabot_token_usage ADD COLUMN reasoning_tokens INT NOT NULL DEFAULT 0",
            "ALTER TABLE nyabot_token_usage ADD COLUMN finish_reason VARCHAR(64) NOT NULL DEFAULT ''",
            "CREATE INDEX IF NOT EXISTS idx_token_usage_model_time ON nyabot_token_usage(model_name, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_messages_session_time ON nyabot_global_messages(session_id, time)",
            "CREATE INDEX IF NOT EXISTS idx_messages_session_user_time ON nyabot_global_messages(session_id, user_id, time)",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_messages_session_msg_id ON nyabot_global_messages(session_id, msg_id)",
            "CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON nyabot_interactions(timestamp)",
        ]:
            await _execute_ignore_duplicate(conn, statement)
        await conn.execute_query("UPDATE nyabot_schema_version SET version=2 WHERE id=1")
