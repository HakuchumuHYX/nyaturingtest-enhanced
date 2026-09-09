# 数据备份、保留期清理与定时任务

import asyncio
import os
import shutil
import sqlite3
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from nonebot import logger, require

require("nonebot_plugin_apscheduler")
from nonebot_plugin_apscheduler import scheduler  # noqa: E402  必须在 require 之后导入

from .config import BACKUP_DIR, get_data_dir
from .core.state_manager import maintain_vector_memories
from .memory.vector import BACKUP_IO_LOCK
from .models import GlobalMessageModel, InteractionLogModel, TokenUsageModel

# 原始明细保留天数
RAW_MESSAGE_RETENTION_DAYS = 180
RAW_INTERACTION_RETENTION_DAYS = 180
TOKEN_USAGE_RETENTION_DAYS = 90

DEFAULT_BACKUP_RETENTION_COUNT = 7
SQLITE_FILENAME = "nyabot.sqlite"


async def _delete_older_than(model, field_name: str, days: int) -> int:
    cutoff = datetime.now() - timedelta(days=days)
    return await model.filter(**{f"{field_name}__lt": cutoff}).delete()


async def cleanup_raw_data_retention() -> dict[str, int]:
    """按保留期清理原始数据库行。

    刻意不触碰长期向量记忆：语义记忆的生命周期由向量库清理路径负责。
    """

    try:
        result = {
            "messages": await _delete_older_than(
                GlobalMessageModel, "time", RAW_MESSAGE_RETENTION_DAYS
            ),
            "interactions": await _delete_older_than(
                InteractionLogModel, "timestamp", RAW_INTERACTION_RETENTION_DAYS
            ),
            "token_usage": await _delete_older_than(
                TokenUsageModel, "timestamp", TOKEN_USAGE_RETENTION_DAYS
            ),
        }
    except Exception as e:
        logger.error(f"[Retention] 原始数据库行清理失败: {e}")
        raise

    if any(result.values()):
        logger.info(f"[Retention] 清理原始数据库行: {result}")
    return result


def _copy_sqlite_snapshot(source: Path, target: Path):
    if not source.exists():
        return
    with sqlite3.connect(str(source)) as src_conn:
        with sqlite3.connect(str(target)) as dst_conn:
            src_conn.backup(dst_conn)


def _copy_data_to_staging(data_dir: Path, staging_dir: Path):
    sqlite_path = data_dir / SQLITE_FILENAME
    for root, dirs, files in os.walk(data_dir):
        root_path = Path(root)
        target_root = staging_dir / root_path.relative_to(data_dir)
        target_root.mkdir(parents=True, exist_ok=True)

        for dirname in list(dirs):
            if dirname == "__pycache__":
                dirs.remove(dirname)

        for file in files:
            file_path = root_path / file
            if file_path == sqlite_path or file_path.name in {
                f"{SQLITE_FILENAME}-wal",
                f"{SQLITE_FILENAME}-shm",
            }:
                continue
            # 字体是静态资源，每个备份包重复约 24MB，没必要打包
            if file_path.suffix.lower() in {".ttf", ".otf", ".ttc"}:
                continue
            shutil.copy2(file_path, target_root / file)

    _copy_sqlite_snapshot(sqlite_path, staging_dir / SQLITE_FILENAME)


def _backup_data_sync() -> bool:
    """同步的备份执行函数"""
    data_dir = get_data_dir()
    backup_dir = BACKUP_DIR

    if not data_dir.exists():
        logger.warning(f"备份失败：数据目录 {data_dir} 不存在。")
        return False

    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_filepath = backup_dir / f"nyabot_backup_{datetime.now():%Y%m%d_%H%M%S}.zip"
    logger.info(f"开始备份 NyaTuringTest 数据到: {backup_filepath}")

    try:
        with BACKUP_IO_LOCK:
            with TemporaryDirectory(prefix="nyaturingtest_backup_") as tmp:
                staging_dir = Path(tmp) / "data"
                staging_dir.mkdir(parents=True, exist_ok=True)
                _copy_data_to_staging(data_dir, staging_dir)

                with zipfile.ZipFile(backup_filepath, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(staging_dir):
                        for file in files:
                            file_path = Path(root) / file
                            zipf.write(file_path, file_path.relative_to(staging_dir))
        logger.info(f"备份完成: {backup_filepath}")
    except Exception as e:
        logger.error(f"备份过程发生异常: {e}")
        if backup_filepath.exists():
            backup_filepath.unlink(missing_ok=True)
        return False

    _clean_old_backups_sync()
    return True


def _clean_old_backups_sync():
    """只保留最近 DEFAULT_BACKUP_RETENTION_COUNT 个备份。"""

    backup_dir = BACKUP_DIR
    if not backup_dir.exists():
        return
    try:
        backups = [
            (file.stat().st_mtime, file)
            for file in backup_dir.glob("nyabot_backup_*.zip")
            if file.is_file()
        ]
        backups.sort(key=lambda item: item[0], reverse=True)
        for _, old_file in backups[DEFAULT_BACKUP_RETENTION_COUNT:]:
            logger.info(f"删除过期的备份文件: {old_file}")
            old_file.unlink()
    except Exception as e:
        logger.error(f"清理过期备份时发生异常: {e}")


async def backup_task() -> bool:
    """执行一次数据备份；仅备份文件创建成功时返回 ``True``。"""

    logger.info("触发自动备份任务...")
    if not await asyncio.to_thread(_backup_data_sync):
        return False

    try:
        await cleanup_raw_data_retention()
    except Exception as e:
        # 备份文件已经成功落盘。Retention 是独立的后置维护任务，失败不应
        # 把一次有效备份重新标记为失败，也不应阻止管理员随后执行 reset。
        logger.error(f"备份后清理原始数据库行失败: {e}")
    return True


def setup_backup_job():
    """注册定时备份与向量记忆维护任务"""

    scheduler.add_job(
        backup_task,
        "cron",
        hour=4,
        minute=0,
        id="nyaturingtest_daily_backup",
        misfire_grace_time=3600,  # 允许误差一小时（比如刚好四点时机器人没开机）
        replace_existing=True,
    )
    scheduler.add_job(
        maintain_vector_memories,
        "cron",
        hour=3,
        minute=30,
        id="nyaturingtest_vector_maintenance",
        misfire_grace_time=3600,
        replace_existing=True,
    )
    logger.info("已注册自动备份定时任务: 每天凌晨 04:00")
