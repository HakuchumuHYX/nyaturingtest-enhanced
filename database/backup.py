import os
import shutil
import sqlite3
from tempfile import TemporaryDirectory
import zipfile
import asyncio
from datetime import datetime
from pathlib import Path
from nonebot import logger, require
import nonebot_plugin_localstore as store

from .backup_lock import BACKUP_IO_LOCK

# 确保调度器插件已加载
require("nonebot_plugin_apscheduler")
from nonebot_plugin_apscheduler import scheduler

# 保留最近多少天的备份
MAX_BACKUP_DAYS = 7
SQLITE_FILENAME = "nyabot.sqlite"


def get_backup_dirs() -> tuple[Path, Path]:
    # 获取插件的数据目录（包含了 nyabot.sqlite, 字体文件，向量数据库索引等）
    data_dir = Path(store.get_plugin_data_dir())
    # 设置备份的存放目录（放在数据目录上一级，避免无限循环打包）
    backup_dir = data_dir.parent / "nyaturingtest_backups"
    return data_dir, backup_dir


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
        rel_root = root_path.relative_to(data_dir)
        target_root = staging_dir / rel_root
        target_root.mkdir(parents=True, exist_ok=True)

        for dirname in list(dirs):
            if dirname == "__pycache__":
                dirs.remove(dirname)

        for file in files:
            file_path = root_path / file
            if file_path == sqlite_path or file_path.name in {f"{SQLITE_FILENAME}-wal", f"{SQLITE_FILENAME}-shm"}:
                continue
            target_path = target_root / file
            shutil.copy2(file_path, target_path)

    _copy_sqlite_snapshot(sqlite_path, staging_dir / SQLITE_FILENAME)


def _backup_data_sync():
    """同步的备份执行函数"""
    data_dir, backup_dir = get_backup_dirs()
    
    if not data_dir.exists():
        logger.warning(f"备份失败：数据目录 {data_dir} 不存在。")
        return

    # 确保备份目录存在
    backup_dir.mkdir(parents=True, exist_ok=True)

    # 构造备份文件名: nyabot_backup_YYYYMMDD_HHMMSS.zip
    now = datetime.now()
    backup_filename = f"nyabot_backup_{now.strftime('%Y%m%d_%H%M%S')}.zip"
    backup_filepath = backup_dir / backup_filename

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
                            arcname = file_path.relative_to(staging_dir)
                            zipf.write(file_path, arcname)

        logger.info(f"备份完成: {backup_filepath}")

    except Exception as e:
        logger.error(f"备份过程发生异常: {e}")
        # 如果出错，尝试清理不完整的备份文件
        if backup_filepath.exists():
            try:
                backup_filepath.unlink()
            except:
                pass
        return

    # 清理过期的备份文件
    _clean_old_backups_sync()


def _clean_old_backups_sync():
    """同步清理旧的备份文件"""
    _, backup_dir = get_backup_dirs()
    if not backup_dir.exists():
        return

    try:
        backups = []
        for file in backup_dir.glob("nyabot_backup_*.zip"):
            if file.is_file():
                # 获取文件的最后修改时间
                mtime = file.stat().st_mtime
                backups.append((mtime, file))

        # 按时间从新到旧排序
        backups.sort(key=lambda x: x[0], reverse=True)

        # 保留最近 MAX_BACKUP_DAYS 个备份，删除多余的
        if len(backups) > MAX_BACKUP_DAYS:
            for _, old_file in backups[MAX_BACKUP_DAYS:]:
                logger.info(f"删除过期的备份文件: {old_file}")
                old_file.unlink()

    except Exception as e:
        logger.error(f"清理过期备份时发生异常: {e}")


async def backup_task():
    """异步包装器，用于被 APScheduler 调用"""
    logger.info("触发自动备份任务...")
    # 由于文件压缩可能比较耗时且是阻塞的 I/O 操作，将其放入 asyncio 线程池中运行
    await asyncio.to_thread(_backup_data_sync)


def setup_backup_job():
    """注册定时备份任务"""
    # 每天凌晨 04:00 执行
    scheduler.add_job(
        backup_task,
        "cron",
        hour=4,
        minute=0,
        id="nyaturingtest_daily_backup",
        misfire_grace_time=3600, # 允许误差一小时（比如刚好四点时机器人没开机）
        replace_existing=True
    )
    logger.info("已注册自动备份定时任务: 每天凌晨 04:00")
