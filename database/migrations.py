from collections import defaultdict

from nonebot import logger
from tortoise import Tortoise


SCHEMA_VERSION = 6

CANONICAL_INDEXES = {
    "nyabot_global_messages": {
        ("time",): "idx_messages_time",
        ("session_id", "time"): "idx_messages_session_time",
        ("session_id", "user_id", "time"): "idx_messages_session_user_time",
    },
    "nyabot_interactions": {
        ("timestamp",): "idx_interactions_timestamp",
    },
    "nyabot_token_usage": {
        ("session_id", "timestamp"): "idx_token_usage_session_time",
        ("model_name", "timestamp"): "idx_token_usage_model_time",
    },
    "nyabot_daily_token_usage": {
        ("session_id", "day"): "idx_daily_token_session_day",
        ("model_name", "day"): "idx_daily_token_model_day",
    },
}


def _is_duplicate_schema_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "duplicate column" in text or "already exists" in text


async def _execute_ignore_duplicate(conn, statement: str):
    try:
        await conn.execute_query(statement)
    except Exception as e:
        if _is_duplicate_schema_error(e):
            logger.warning(f"[Migration] skipped already-applied statement: {e}")
            return
        logger.error(f"[Migration] failed statement: {e}")
        raise


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


async def _index_columns(conn, index_name: str) -> tuple[str, ...]:
    rows = await conn.execute_query_dict(
        f"PRAGMA index_info({_quote_identifier(index_name)})"
    )
    return tuple(
        str(row["name"])
        for row in sorted(rows, key=lambda row: int(row.get("seqno", 0)))
        if row.get("name")
    )


async def _drop_equivalent_indexes(conn) -> list[str]:
    """Drop redundant indexes only after PRAGMA confirms identical columns."""

    dropped: list[str] = []
    for table, canonical_by_columns in CANONICAL_INDEXES.items():
        rows = await conn.execute_query_dict(
            f"PRAGMA index_list({_quote_identifier(table)})"
        )
        grouped: dict[tuple[bool, tuple[str, ...]], list[str]] = defaultdict(list)
        for row in rows:
            name = str(row.get("name") or "")
            if not name:
                continue
            columns = await _index_columns(conn, name)
            grouped[(bool(row.get("unique")), columns)].append(name)

        for (is_unique, columns), names in grouped.items():
            if len(names) <= 1:
                continue
            canonical = canonical_by_columns.get(columns)
            if is_unique:
                keep = next(
                    (name for name in names if name.startswith("sqlite_autoindex_")),
                    sorted(names)[0],
                )
            else:
                keep = canonical if canonical in names else sorted(names)[0]
            for name in names:
                if name == keep or name.startswith("sqlite_autoindex_"):
                    continue
                # Re-read immediately before deletion; never trust an index name
                # without confirming its exact column combination.
                if await _index_columns(conn, name) != columns:
                    continue
                await conn.execute_query(
                    f"DROP INDEX IF EXISTS {_quote_identifier(name)}"
                )
                dropped.append(name)
    return dropped


async def ensure_schema_version():
    conn = Tortoise.get_connection("default")
    await conn.execute_query(
        "CREATE TABLE IF NOT EXISTS nyabot_schema_version (id INT PRIMARY KEY, version INT NOT NULL)"
    )
    rows = await conn.execute_query_dict(
        "SELECT version FROM nyabot_schema_version WHERE id=1"
    )
    current = int(rows[0]["version"]) if rows else 0

    if current < 1:
        await conn.execute_query(
            "INSERT OR REPLACE INTO nyabot_schema_version (id, version) VALUES (1, 1)"
        )
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
        await conn.execute_query(
            "UPDATE nyabot_schema_version SET version=2 WHERE id=1"
        )
        current = 2

    if current < 3:
        await _execute_ignore_duplicate(
            conn,
            "ALTER TABLE nyabot_sessions ADD COLUMN last_consolidated_time TIMESTAMP",
        )
        await conn.execute_query(
            "UPDATE nyabot_schema_version SET version=3 WHERE id=1"
        )
        current = 3

    if current < 4:
        for statement in [
            "ALTER TABLE nyabot_user_profiles ADD COLUMN interaction_count INT NOT NULL DEFAULT 0",
            "ALTER TABLE nyabot_user_profiles ADD COLUMN first_interaction_at TIMESTAMP",
            "ALTER TABLE nyabot_user_profiles ADD COLUMN last_interaction_at TIMESTAMP",
        ]:
            await _execute_ignore_duplicate(conn, statement)
        await conn.execute_query(
            """
            UPDATE nyabot_user_profiles
            SET interaction_count = (
                    SELECT COUNT(*) FROM nyabot_interactions
                    WHERE nyabot_interactions.user_id = nyabot_user_profiles.id
                ),
                first_interaction_at = (
                    SELECT MIN(timestamp) FROM nyabot_interactions
                    WHERE nyabot_interactions.user_id = nyabot_user_profiles.id
                ),
                last_interaction_at = (
                    SELECT MAX(timestamp) FROM nyabot_interactions
                    WHERE nyabot_interactions.user_id = nyabot_user_profiles.id
                )
            """
        )
        await conn.execute_query(
            "UPDATE nyabot_schema_version SET version=4 WHERE id=1"
        )
        current = 4

    if current < 5:
        for statement in [
            """
            CREATE TABLE IF NOT EXISTS nyabot_daily_token_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                day DATE NOT NULL,
                session_id VARCHAR(255) NOT NULL,
                model_name VARCHAR(255) NOT NULL,
                provider VARCHAR(64) NOT NULL DEFAULT '',
                prompt_tokens BIGINT NOT NULL DEFAULT 0,
                completion_tokens BIGINT NOT NULL DEFAULT 0,
                prompt_cache_hit_tokens BIGINT NOT NULL DEFAULT 0,
                prompt_cache_miss_tokens BIGINT NOT NULL DEFAULT 0,
                reasoning_tokens BIGINT NOT NULL DEFAULT 0,
                request_count INT NOT NULL DEFAULT 0,
                UNIQUE(day, session_id, model_name, provider)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_daily_token_session_day ON nyabot_daily_token_usage(session_id, day)",
            "CREATE INDEX IF NOT EXISTS idx_daily_token_model_day ON nyabot_daily_token_usage(model_name, day)",
            """
            INSERT INTO nyabot_daily_token_usage (
                day, session_id, model_name, provider,
                prompt_tokens, completion_tokens,
                prompt_cache_hit_tokens, prompt_cache_miss_tokens,
                reasoning_tokens, request_count
            )
            SELECT DATE(timestamp), session_id, model_name, provider,
                   SUM(prompt_tokens), SUM(completion_tokens),
                   SUM(prompt_cache_hit_tokens), SUM(prompt_cache_miss_tokens),
                   SUM(reasoning_tokens), COUNT(*)
            FROM nyabot_token_usage
            GROUP BY DATE(timestamp), session_id, model_name, provider
            ON CONFLICT(day, session_id, model_name, provider) DO NOTHING
            """,
        ]:
            await _execute_ignore_duplicate(conn, statement)
        await conn.execute_query(
            "UPDATE nyabot_schema_version SET version=5 WHERE id=1"
        )
        current = 5

    if current < 6:
        for statement in [
            "CREATE INDEX IF NOT EXISTS idx_messages_time ON nyabot_global_messages(time)",
            "CREATE INDEX IF NOT EXISTS idx_messages_session_time ON nyabot_global_messages(session_id, time)",
            "CREATE INDEX IF NOT EXISTS idx_messages_session_user_time ON nyabot_global_messages(session_id, user_id, time)",
            "CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON nyabot_interactions(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_token_usage_session_time ON nyabot_token_usage(session_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_token_usage_model_time ON nyabot_token_usage(model_name, timestamp)",
        ]:
            await _execute_ignore_duplicate(conn, statement)
        dropped = await _drop_equivalent_indexes(conn)
        await conn.execute_query("ANALYZE")
        await conn.execute_query(
            "UPDATE nyabot_schema_version SET version=6 WHERE id=1"
        )
        logger.info(f"[Migration] removed {len(dropped)} equivalent indexes")
