"""悬着的事：之后可以接上的话头。分段整理时写入，看群时到了时候就提起，过了 forget_at 算她忘了。

了结有两处：Chat 回复里接上了哪件（问了、兑现了）当场了结；本人自己说了结果、取消了，由分段整理了结。
只靠分段整理不够：一段要安静 15 分钟或攒满 80 条才整理，她问过的话早就滚出 recent_msgs 的 20 条，
那件事还挂着，她会再问一遍；整理时漏判了，更会每次见到那个人都问。

和长期记忆分开放：这里没有向量，按时间查，会被了结；长期记忆按相似度检索、按重要度过期。
"""

import uuid
from datetime import date, datetime, timedelta

from tortoise.expressions import Q

from ..messages.message import ChatMessage
from ..storage.db import message_final_id
from ..storage.models import ThreadModel
from .candidates import source_messages

# 类型 → 给模型看的说明
KINDS = {
    "matter": "ta的事",
    "state": "ta的状况",
    "promise": "你答应ta的",
    "group": "群里约好的",
}
# 日子最远记到说起后多少天：再远的真人也不会掐着日子问，只当没日子
DUE_MAX_DAYS = 60
# 有日子的事到日子后还会想起来几天；群里约好的过两天就没人提了
DUE_REMEMBER_DAYS = 7
GROUP_REMEMBER_DAYS = 2
# 没日子的事：隔几天再见到才问，一个月后就忘了
MATTER_ASK_AFTER_DAYS = 2
MATTER_REMEMBER_DAYS = 30
# 状况只管这两天：第二天起问一句好点没
STATE_REMEMBER_DAYS = 3
PROMISE_REMEMBER_DAYS = 14
CONTENT_MIN_CHARS = 6
PROMPT_LIMIT = 10


def _day_start(day: date) -> datetime:
    return datetime.combine(day, datetime.min.time())


def _schedule(kind: str, said: datetime, due: date | None) -> tuple[datetime, datetime]:
    """(ask_from, forget_at)：从什么时候起提起、什么时候忘掉。group 一定有 due。"""

    if kind == "group":
        return _day_start(due), _day_start(due) + timedelta(days=GROUP_REMEMBER_DAYS + 1)
    if due is not None:
        return _day_start(due), _day_start(due) + timedelta(days=DUE_REMEMBER_DAYS + 1)
    if kind == "state":
        return _day_start(said.date() + timedelta(days=1)), said + timedelta(
            days=STATE_REMEMBER_DAYS
        )
    if kind == "promise":
        return said, said + timedelta(days=PROMISE_REMEMBER_DAYS)
    return said + timedelta(days=MATTER_ASK_AFTER_DAYS), said + timedelta(
        days=MATTER_REMEMBER_DAYS
    )


def _parse_due(raw, said: datetime) -> date | None:
    """MM-DD 推成说起之后第一个对得上的日子；格式不对、超过 DUE_MAX_DAYS 返回 None。"""

    try:
        month, day = (int(part) for part in str(raw).strip().split("-"))
        due = date(said.year, month, day)
        if due < said.date():
            due = date(said.year + 1, month, day)
    except (TypeError, ValueError):
        return None
    if (due - said.date()).days > DUE_MAX_DAYS:
        return None
    return due


def parse_threads(
    items, messages: list[ChatMessage], session_id: str, members: dict[str, str]
) -> list[ThreadModel]:
    """分段整理输出的 threads 校验成待写入的行；members 是这段发言的群友（QQ 号 -> 名字）。

    关于谁由代码按 members 对上号、名字取当前群名片；群里约好的没有日子就不收。
    """

    threads = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip()
        content = str(item.get("content") or "").strip()
        sources = source_messages(item, messages)
        if kind not in KINDS or len(content) < CONTENT_MIN_CHARS or not sources:
            continue
        user_id = "" if kind == "group" else str(item.get("user_id") or "").strip()
        if kind != "group" and user_id not in members:
            continue
        said = sources[0].time
        # 状况只管这两天，不按日子算
        due = _parse_due(item.get("due"), said) if item.get("due") and kind != "state" else None
        if kind == "group" and due is None:
            continue
        ask_from, forget_at = _schedule(kind, said, due)
        threads.append(
            ThreadModel(
                id=str(uuid.uuid4()),
                session_id=session_id,
                kind=kind,
                user_id=user_id,
                user_name=members.get(user_id, ""),
                content=content,
                due=int(due.strftime("%Y%m%d")) if due else None,
                ask_from=ask_from,
                forget_at=forget_at,
                source_msg_ids=" ".join(dict.fromkeys(message_final_id(m) for m in sources)),
            )
        )
    return threads


def _about(user_ids) -> Q:
    return Q(user_id__in=list(user_ids)) | Q(user_id="")


async def open_threads(session_id: str, user_ids, now: datetime) -> list[ThreadModel]:
    """看群时该提起的：关于这些人或全群、到了提起时间还没忘掉的。"""

    return (
        await ThreadModel.filter(session_id=session_id, ask_from__lte=now, forget_at__gt=now)
        .filter(_about(user_ids))
        .order_by("ask_from")
        .limit(PROMPT_LIMIT)
    )


async def pending_threads(session_id: str, user_ids, now: datetime) -> list[ThreadModel]:
    """分段整理时还挂着的：没到提起时间的也算，提前有了结果、取消了也要能了结。"""

    return (
        await ThreadModel.filter(session_id=session_id, forget_at__gt=now)
        .filter(_about(user_ids))
        .order_by("created_at")
        .limit(PROMPT_LIMIT)
    )


async def add_threads(rows: list[ThreadModel]) -> None:
    if rows:
        await ThreadModel.bulk_create(rows)


async def close_threads(rows: list[ThreadModel], raw_refs) -> int:
    """模型按编号（t1、t2…，就是 rows 的顺序）指出的了结掉；编号不对的当没写。"""

    refs = {f"t{index + 1}": row.id for index, row in enumerate(rows)}
    ids = [
        refs[ref]
        for ref in (raw_refs if isinstance(raw_refs, list) else [])
        if isinstance(ref, str) and ref in refs
    ]
    if not ids:
        return 0
    return await ThreadModel.filter(id__in=ids).delete()


async def drop_forgotten() -> int:
    return await ThreadModel.filter(forget_at__lte=datetime.now()).delete()


def thread_items(rows: list[ThreadModel]) -> list[dict]:
    """给模型看的：编号、谁的、什么类型、什么事、哪天说的、日子。编号按 rows 的顺序，了结时用。"""

    return [{"ref": f"t{index + 1}", **_thread_item(row)} for index, row in enumerate(rows)]


def _thread_item(row: ThreadModel) -> dict:
    item = {
        "who": row.user_name or "全群",
        "kind": KINDS[row.kind],
        "content": row.content,
        "said": f"{row.created_at:%m-%d}",
    }
    if row.due:
        item["due"] = f"{str(row.due)[4:6]}-{str(row.due)[6:]}"
    return item
