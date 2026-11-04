"""写进 WebConsole 库的历史：LLM 请求、看群决策、每日整理，以及 WARNING 起的日志。

当前状态不在这里，由 WebConsole 拉 webconsole_api.py。表由 WebConsole 建，列的含义以它的 schema.sql 为准。

没有 WebConsole 时只在启动时打一条警告，之后什么都不做：监控不该拖垮聊天，也不该刷屏。
写入用一个后台线程独占连接，因为 loguru sink 可能在任意线程里被调用（比如 to_thread 里打的 warning）。
"""

import gzip
import json
import queue
import sqlite3
import threading
import time
from urllib.parse import quote

from nonebot import logger

from .config import get_app_settings

TABLES = ("diagnostic_logs", "autochat_llm_calls", "autochat_looks", "autochat_maintenance_runs")
SUMMARY_LIMIT = 2_000
PLUGIN_NAME = "nyaturingtest"

INSERT_DIAGNOSTIC = """
    INSERT INTO diagnostic_logs (
        created_at_ms, level, logger_name, plugin_name, message_summary, full_log_gzip
    ) VALUES (
        :created_at_ms, :level, :logger_name, :plugin_name, :message_summary, :full_log_gzip
    )
"""
INSERT_LLM_CALL = """
    INSERT INTO autochat_llm_calls (
        created_at_ms, endpoint, model, session_id, ok, latency_ms, error_type
    ) VALUES (
        :created_at_ms, :endpoint, :model, :session_id, :ok, :latency_ms, :error_type
    )
"""
INSERT_LOOK = """
    INSERT INTO autochat_looks (
        created_at_ms, group_id, trigger, unread, willing, threshold, self_share,
        muted, speak, focus, react, poke
    ) VALUES (
        :created_at_ms, :group_id, :trigger, :unread, :willing, :threshold, :self_share,
        :muted, :speak, :focus, :react, :poke
    )
"""
INSERT_MAINTENANCE = """
    INSERT INTO autochat_maintenance_runs (
        started_at_ms, finished_at_ms, status, detail_json
    ) VALUES (
        :started_at_ms, :finished_at_ms, :status, :detail_json
    )
"""

_STOP = object()
_queue: queue.Queue = queue.Queue()
_thread: threading.Thread | None = None
_sink_id: int | None = None
# 带这个标记的日志 sink 不收，免得写库失败的 warning 又要写库
_internal_logger = logger.bind(webconsole_internal=True)


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def _connect(path: str) -> sqlite3.Connection:
    """库打不开或缺表都抛 sqlite3.Error；缺表多半是 WebConsole 还没升级。"""

    connection = sqlite3.connect(
        f"file:{quote(path, safe='/')}?mode=rw", uri=True, timeout=5, check_same_thread=False
    )
    try:
        connection.execute("PRAGMA synchronous = NORMAL")
        for table in TABLES:
            connection.execute(f"SELECT 1 FROM {table} LIMIT 0")
    except sqlite3.Error:
        connection.close()
        raise
    return connection


def open_store() -> None:
    """只在启动时判断一次；先起 autochat、后装 WebConsole 时要重启 autochat 才开始写。"""

    global _thread, _sink_id
    path = get_app_settings().webconsole_db
    if not path:
        logger.warning("没有配置 webconsole_db，不向 WebConsole 记录历史和警告")
        return
    try:
        connection = _connect(path)
    except sqlite3.Error as e:
        logger.warning(f"WebConsole 的库不可用，不记录历史和警告: {path}: {e}")
        return
    _thread = threading.Thread(
        target=_write_loop, args=(connection,), name="webconsole-writer", daemon=True
    )
    _thread.start()
    _sink_id = logger.add(
        _sink,
        level="WARNING",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
        "{name}:{function}:{line} - {message}",
        diagnose=False,
        backtrace=True,
    )


def close_store() -> None:
    global _thread
    if _thread is None:
        return
    logger.remove(_sink_id)
    _queue.put(_STOP)
    _thread.join(timeout=5)
    _thread = None


def _write_loop(connection: sqlite3.Connection) -> None:
    # 库出问题时每天有上千次写入，逐行报警就成了噪音：只记第一条和恢复，中间的行丢弃
    failing = False
    while (item := _queue.get()) is not _STOP:
        sql, row = item
        try:
            with connection:
                connection.execute(sql, row)
        except sqlite3.Error as e:
            if not failing:
                failing = True
                _internal_logger.warning(
                    f"写 WebConsole 的库失败，恢复前不再提示，这期间的记录丢弃: {type(e).__name__}: {e}"
                )
            continue
        if failing:
            failing = False
            _internal_logger.info("写 WebConsole 的库恢复了")
    connection.close()


def _put(sql: str, row: dict) -> None:
    if _thread is not None:
        _queue.put((sql, row))


def _sink(message) -> None:
    record = message.record
    if record["extra"].get("webconsole_internal"):
        return
    name = record["name"] or ""
    # 其他来源单独分组，不和主 bot 的 nonebot 混在一起
    plugin = name if name == PLUGIN_NAME else f"autochat/{name.split('.')[0]}"
    _put(
        INSERT_DIAGNOSTIC,
        {
            "created_at_ms": now_ms(),
            "level": record["level"].name,
            "logger_name": name,
            "plugin_name": plugin,
            "message_summary": str(record["message"])[:SUMMARY_LIMIT],
            "full_log_gzip": gzip.compress(str(message).encode("utf-8"), compresslevel=6, mtime=0),
        },
    )


def record_llm_call(
    endpoint: str,
    model: str,
    session_id: str | None,
    ok: bool,
    latency_ms: int,
    error_type: str | None = None,
) -> None:
    _put(
        INSERT_LLM_CALL,
        {
            "created_at_ms": now_ms(),
            "endpoint": endpoint,
            "model": model,
            "session_id": session_id,
            "ok": int(ok),
            "latency_ms": latency_ms,
            "error_type": error_type,
        },
    )


def record_look(
    *,
    group_id: str,
    trigger: str,
    unread: int,
    willing: float | None,
    threshold: float,
    self_share: float,
    muted: bool,
    speak: bool,
    focus: str | None,
    react: str | None,
    poke: str | None,
) -> None:
    _put(
        INSERT_LOOK,
        {
            "created_at_ms": now_ms(),
            "group_id": group_id,
            "trigger": trigger,
            "unread": unread,
            "willing": willing,
            "threshold": threshold,
            "self_share": self_share,
            "muted": int(muted),
            "speak": int(speak),
            "focus": focus,
            "react": react,
            "poke": poke,
        },
    )


def record_maintenance(started_at_ms: int, status: str, detail: dict) -> None:
    _put(
        INSERT_MAINTENANCE,
        {
            "started_at_ms": started_at_ms,
            "finished_at_ms": now_ms(),
            "status": status,
            "detail_json": json.dumps(detail, ensure_ascii=False, separators=(",", ":")),
        },
    )
