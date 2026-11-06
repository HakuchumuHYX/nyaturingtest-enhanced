"""角色自我成长：人生线（每个预设一份，各群共享）和群里的ta（每个群一份）。

群里的事不出群，也改不了人生事实。群层每天写一条观察日记，跨天反复出现的才收进群层。
性格变化只认外部证据：ta说的话是按档案生成的，拿来证明性格会自我强化。ta的话只用来固定事实。
ta自己的日记（时间线写成的）同理：是ta自己生成的，只算发生过的事。
"""

from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from nonebot import logger

from ..llm import background_client
from ..prompts import (
    JOURNAL_DAYS,
    MAX_LIFE_FACTS,
    PRESETS,
    RolePreset,
    diary_line,
    get_diary_prompt,
    get_persona_group_prompt,
    get_persona_life_prompt,
    segment_line,
)
from ..storage.db import load_persona_status, save_persona_diary
from ..storage.models import (
    GlobalMessageModel,
    PersonaGroupModel,
    PersonaHistoryModel,
    PersonaLifeModel,
    SessionModel,
    UserProfileModel,
)

GROUP_MATERIAL_CHARS = 12000
# ta参与的对话最多占素材的这么多，剩下留给群里其他时候的聊天（口癖和梗的来源）
OWN_CONTEXT_SHARE = 0.6
CONTEXT_BEFORE_TURN = 6
LINE_CHARS = 120
# 当天消息少于这么多、ta又没开口，就不整理这个群
GROUP_MIN_MESSAGES = 30
# 没有新的自我陈述时，隔这么久也要整理一次人生线，让时间推进
LIFE_INTERVAL = timedelta(days=7)
# ta自己的日记留几天、写日记时看前几天
DIARY_DAYS = 14
DIARY_PREVIOUS = 7

_NOISE = {"[图片]", "[表情包]"}


@dataclass(frozen=True)
class GroupDigest:
    text: str
    observation: str
    life_facts: list[str]
    changes: str


@dataclass(frozen=True)
class LifeDigest:
    text: str
    changes: str


def _line(row: dict) -> str:
    content = " ".join(row["content"].split())
    if len(content) > LINE_CHARS:
        content = content[:LINE_CHARS] + "…"
    return f"{row['time']:%H:%M} {row['user_name']}: {content}"


def _evenly(items: list, count: int) -> list:
    if count >= len(items):
        return items
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]


def group_due(rows: list[dict], bot_name: str) -> bool:
    return len(rows) >= GROUP_MIN_MESSAGES or any(
        row["user_name"] == bot_name for row in rows
    )


def group_material(rows: list[dict], bot_name: str) -> str:
    """一天的群聊素材：ta参与的对话（每次开口前几条 + ta的话），剩余预算均匀抽样群里其他时候的聊天。

    rows 按时间升序，每项含 time、user_name、content。
    """

    # ta连续发的几条算一次开口
    turns: list[list[int]] = []
    for index, row in enumerate(rows):
        if row["user_name"] != bot_name:
            continue
        if turns and turns[-1][-1] == index - 1:
            turns[-1].append(index)
        else:
            turns.append([index])

    windows = [
        list(range(max(0, turn[0] - CONTEXT_BEFORE_TURN), turn[-1] + 1)) for turn in turns
    ]
    own_budget = int(GROUP_MATERIAL_CHARS * OWN_CONTEXT_SHARE)
    size = sum(len(_line(rows[i])) for window in windows for i in window)
    if size > own_budget:
        windows = _evenly(windows, max(1, len(windows) * own_budget // size))

    used = {i for window in windows for i in window}
    own_blocks = ["\n".join(_line(rows[i]) for i in window) for window in windows]
    own_size = sum(len(block) for block in own_blocks)

    others = [
        i for i, row in enumerate(rows) if i not in used and row["content"].strip() not in _NOISE
    ]
    other_size = sum(len(_line(rows[i])) for i in others)
    other_budget = GROUP_MATERIAL_CHARS - own_size
    if other_size > other_budget:
        others = _evenly(others, max(0, len(others) * other_budget // other_size))

    parts = []
    if own_blocks:
        parts.append("[ta参与的对话]（每段是ta开口前的几条消息和ta说的话）\n" + "\n---\n".join(own_blocks))
    if others:
        parts.append(
            "[群里其他时候的聊天]（均匀抽样，不连续）\n" + "\n".join(_line(rows[i]) for i in others)
        )
    return "\n\n".join(parts)


def life_due(last_digest: date | None, day: date, has_facts: bool) -> bool:
    return last_digest is None or has_facts or day - last_digest >= LIFE_INTERVAL


async def digest_group(
    persona: RolePreset,
    *,
    life: str,
    group_text: str,
    journal: list[str],
    days_in_group: int,
    rows: list[dict],
    day: date,
    session_id: str,
) -> GroupDigest | None:
    """整理一个群一天的聊天，失败返回 None（外层不推进水位）。

    journal 是最近的观察日记，每条形如「MM-DD 观察」；外层把返回的 observation 追加进去并只留 JOURNAL_DAYS 条。
    days_in_group 从本群群友档案里最早的 first_interaction_at 算起（和查询记忆的认识天数同一口径）。
    """

    data = await background_client.generate_json(
        get_persona_group_prompt(
            persona,
            life,
            group_text,
            journal,
            days_in_group,
            group_material(rows, persona.name),
            day,
        ),
        session_id=session_id,
    )
    if (
        data is None
        or not isinstance(data.get("group_self"), str)
        or not isinstance(data.get("observation"), str)
        or not isinstance(data.get("life_facts"), list)
    ):
        logger.warning(f"[Persona] 群 {session_id} {day} 群层整理输出无效")
        return None
    return GroupDigest(
        text=data["group_self"].strip(),
        observation=f"{day:%m-%d} {data['observation'].strip()}",
        life_facts=[str(fact).strip() for fact in data["life_facts"] if str(fact).strip()][
            :MAX_LIFE_FACTS
        ],
        changes=str(data.get("changes") or "").strip(),
    )


async def digest_life(
    persona: RolePreset,
    *,
    life: str,
    life_facts: list[str],
    group_styles: list[str],
    diary: list[str],
    day: date,
    session_id: str,
) -> LifeDigest | None:
    """整理人生线，失败返回 None。diary 是上次整理以来ta自己的日记。"""

    data = await background_client.generate_json(
        get_persona_life_prompt(persona, life, life_facts, group_styles, diary, day),
        session_id=session_id,
    )
    if data is None or not isinstance(data.get("life"), str):
        logger.warning(f"[Persona] {persona.name} {day} 人生线整理输出无效")
        return None
    return LifeDigest(
        text=data["life"].strip(), changes=str(data.get("changes") or "").strip()
    )


def day_segments(timeline: list[dict], until: datetime, day: date) -> list[str]:
    """时间线里这一天开始的各段；每段到下一段开始为止，最后一段到 until。"""

    lines = []
    for index, segment in enumerate(timeline):
        if segment["start"].date() != day:
            continue
        end = timeline[index + 1]["start"] if index + 1 < len(timeline) else until
        lines.append(segment_line(segment, end))
    return lines


async def write_diary(
    persona: RolePreset,
    *,
    life: str,
    day: date,
    previous: list[str],
    timeline: list[str],
    life_facts: list[str],
    plan: str,
    session_id: str,
) -> tuple[str, str] | None:
    """写ta一天的日记和接下来的打算，返回（日记, 打算），失败返回 None。打算可以为空。"""

    data = await background_client.generate_json(
        get_diary_prompt(
            persona,
            life=life,
            day=day,
            previous=previous,
            timeline=timeline,
            life_facts=life_facts,
            plan=plan,
        ),
        session_id=session_id,
    )
    if (
        data is None
        or not isinstance(data.get("diary"), str)
        or not data["diary"].strip()
        or not isinstance(data.get("plan"), str)
    ):
        logger.warning(f"[Persona] {persona.name} {day} 日记和打算输出无效")
        return None
    return data["diary"].strip(), data["plan"].strip()


def group_style(group_text: str) -> str:
    """群层里【在这个群的样子】那一段，交给人生线判断风格提升。"""

    start = group_text.find("【在这个群的样子】")
    if start < 0:
        return ""
    end = group_text.find("【", start + 1)
    return group_text[start : end if end > 0 else None].strip()


async def _retry(make_call: Callable[[], Awaitable]):
    """整理偶尔输出无效 JSON，重试一次；还失败就留到下一晚。"""

    for _ in range(2):
        result = await make_call()
        if result is not None:
            return result
    return None


async def _day_rows(session_id: str, day: date) -> list[dict]:
    start = datetime.combine(day, datetime.min.time())
    return (
        await GlobalMessageModel.filter(
            session_id=session_id, time__gte=start, time__lt=start + timedelta(days=1)
        )
        .order_by("time")
        .values("time", "user_name", "content")
    )


async def _joined(session_id: str) -> date | None:
    """进群日：本群群友档案里最早的 first_interaction_at；消息表只留 180 天，不能拿最早消息算。"""

    first = (
        await UserProfileModel.filter(session_id=session_id, first_interaction_at__not_isnull=True)
        .order_by("first_interaction_at")
        .first()
    )
    return first.first_interaction_at.date() if first else None


async def _history(scope: str, key: str, day: date, text: str, changes: str) -> None:
    await PersonaHistoryModel.create(scope=scope, key=key, date=day, text=text, changes=changes)


async def _maintain_preset(
    filename: str,
    session_ids: list[str],
    yesterday: date,
    guards: dict[str, Callable[[], bool]],
) -> tuple[PersonaLifeModel, dict[str, PersonaGroupModel], dict]:
    """返回（人生线, 这次更新的群层, 这个预设的整理结果）；结果只给控制台看。"""

    persona = PRESETS[filename]
    life_session = f"persona:{filename}"
    life, _ = await PersonaLifeModel.get_or_create(preset=filename)
    layers = {
        session_id: (await PersonaGroupModel.get_or_create(session_id=session_id, preset=filename))[0]
        for session_id in session_ids
    }
    joined = {session_id: await _joined(session_id) for session_id in session_ids}
    # 还没定过状态（这个预设从没启用过）就没有日记可写
    status = await load_persona_status(filename)

    if not life.text:
        result = await _retry(
            lambda: digest_life(
                persona,
                life="",
                life_facts=[],
                group_styles=[],
                diary=[],
                day=yesterday,
                session_id=life_session,
            )
        )
        if result is None:
            return life, {}, {
                "life_until": None,
                "life_failed": True,
                "groups_updated": [],
                "groups_blocked": [],
                "diary_failed": False,
            }
        life.text, life.last_digest = result.text, yesterday
        await life.save()
        await _history("life", filename, yesterday, result.text, f"[初始化] {result.changes}")

    # 新群层不补历史，从昨天开始
    def next_day(layer: PersonaGroupModel) -> date:
        return layer.digested_until + timedelta(days=1) if layer.digested_until else yesterday

    # 某个群某天整理失败就停在那天，后面几天等下一晚，保证日记和水位按天连续
    blocked: set[str] = set()
    diary_blocked = False
    life_failed = False
    updated: dict[str, PersonaGroupModel] = {}
    day = min((next_day(layer) for layer in layers.values()), default=yesterday + timedelta(days=1))
    while day <= yesterday:
        day_facts: list[str] = []
        for session_id, layer in layers.items():
            if session_id in blocked or next_day(layer) > day:
                continue
            rows = await _day_rows(session_id, day)
            if rows and group_due(rows, persona.name):
                result = await _retry(
                    lambda: digest_group(
                        persona,
                        life=life.text,
                        group_text=layer.text,
                        journal=layer.journal,
                        days_in_group=(day - joined[session_id]).days if joined[session_id] else 0,
                        rows=rows,
                        day=day,
                        session_id=session_id,
                    )
                )
                # 整理途中群被 reset 或换了预设，结果作废
                if result is None or not guards[session_id]():
                    blocked.add(session_id)
                    continue
                layer.text = result.text
                layer.journal = (layer.journal + [result.observation])[-JOURNAL_DAYS:]
                life.pending_facts = life.pending_facts + result.life_facts
                day_facts += result.life_facts
                await life.save()
                await _history("group", f"{session_id}:{filename}", day, result.text, result.changes)
                updated[session_id] = layer
            layer.digested_until = day
            await layer.save()

        # 日记在群层之后：要用这天各群抽出的自我陈述；在人生线之前：人生线要读它。失败就停在这天，下一晚接着写
        if status is not None and not diary_blocked and (
            status["diary_until"] is None or day > status["diary_until"]
        ):
            diary_blocked = not await _write_day_diary(persona, filename, status, life.text, day, day_facts)

        if life_due(life.last_digest, day, bool(life.pending_facts)):
            # 风格提升只看现在启用的群：停用、退出的群里沾上的说法不该继续「变老」被提升
            styles = [style for style in (group_style(layer.text) for layer in layers.values()) if style]
            since = life.last_digest.isoformat() if life.last_digest else ""
            diary = [
                diary_line(entry)
                for entry in (status["diary"] if status else [])
                if since < entry["day"] <= day.isoformat()
            ]
            result = await _retry(
                lambda: digest_life(
                    persona,
                    life=life.text,
                    life_facts=life.pending_facts,
                    group_styles=styles,
                    diary=diary,
                    day=day,
                    session_id=life_session,
                )
            )
            if result is not None:
                life.text, life.last_digest, life.pending_facts = result.text, day, []
                await life.save()
                await _history("life", filename, day, result.text, result.changes)
            else:
                life_failed = True
        day += timedelta(days=1)

    logger.info(
        f"[Persona] {filename} 整理到 {yesterday}：群层更新 {len(updated)} 个群，"
        f"整理失败停在半路的群 {sorted(blocked) or '无'}"
    )
    return life, updated, {
        "life_until": life.last_digest.isoformat() if life.last_digest else None,
        "life_failed": life_failed,
        "groups_updated": sorted(updated),
        "groups_blocked": sorted(blocked),
        "diary_failed": diary_blocked,
    }


async def _write_day_diary(
    persona: RolePreset, filename: str, status: dict, life: str, day: date, facts: list[str]
) -> bool:
    """写这一天的日记和打算并推进日记水位；这天一段时间线都没有（服务停了一整天）就跳过。失败返回 False。"""

    timeline = day_segments(status["timeline"], status["until"], day)
    if timeline:
        written = await _retry(
            lambda: write_diary(
                persona,
                life=life,
                day=day,
                previous=[diary_line(entry) for entry in status["diary"][-DIARY_PREVIOUS:]],
                timeline=timeline,
                life_facts=facts,
                plan=status["plan"],
                session_id=f"persona:{filename}",
            )
        )
        if written is None:
            return False
        text, status["plan"] = written
        status["diary"] = (status["diary"] + [{"day": day.isoformat(), "text": text}])[-DIARY_DAYS:]
    status["diary_until"] = day
    await save_persona_diary(filename, status["diary"], status["plan"], day)
    return True


async def maintain_personas(
    group_ids: list[int], guards: dict[str, Callable[[], bool]]
) -> tuple[dict[str, str], dict[str, str], dict[str, dict]]:
    """每日整理：设了人设的启用群把群层推到昨天，再整理对应预设的人生线。

    返回（{预设文件名: 人生线}, {群号: 群层}, {预设文件名: 整理结果}），前两个外层用来同步已加载 session 的只读副本。
    guards 是 {群号: 会话还没被作废的检查}，整理途中作废的群结果丢弃。
    """

    by_preset: dict[str, list[str]] = defaultdict(list)
    for session in await SessionModel.filter(id__in=[str(g) for g in group_ids]).exclude(preset=""):
        if session.preset in PRESETS:
            by_preset[session.preset].append(session.id)
        else:
            logger.warning(f"[Persona] 群 {session.id} 的预设 {session.preset} 不存在，跳过整理")

    yesterday = date.today() - timedelta(days=1)
    lives: dict[str, str] = {}
    groups: dict[str, str] = {}
    reports: dict[str, dict] = {}
    for filename, session_ids in by_preset.items():
        life, updated, reports[filename] = await _maintain_preset(
            filename, session_ids, yesterday, guards
        )
        lives[filename] = life.text
        groups.update({session_id: layer.text for session_id, layer in updated.items()})
    return lives, groups, reports
