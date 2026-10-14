"""角色自我成长：人生线（每个预设一份，各群共享）和群里的ta（每个群一份）。

人生线是ta这个人：身份阶段、近况、性格与说话、喜好习惯、生活里的事，按日期推进，收各群抽出的自我陈述。
群里的ta只管在这个群的说话方式、和群的关系、对群友说过的话，群里的事不出群，也改不了人生事实。
群层每天还写一条观察日记，留最近 JOURNAL_DAYS 天：单看一天什么都收不进来，跨天反复出现才算数，证据靠日记攒。

性格变化只认外部证据（群友的影响、生活里的事）：ta说的话是按档案生成的，拿来证明性格会自我强化，
档案写「说话冲」→ ta说话冲 → 整理看到ta说话冲 → 更冲。ta的话只用来固定事实。

「旧文本 + 材料 → 新文本」的整理函数不碰数据库（prompt 在 prompts.py），回放脚本（persona_replay/）和线上共用；
线上每日整理 maintain_personas 在文件末尾，天循环和回放脚本一致。
"""

from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from nonebot import logger

from ..llm import extract_and_parse_json, feedback_client
from ..prompts import (
    JOURNAL_DAYS,
    MAX_LIFE_FACTS,
    PRESETS,
    RolePreset,
    get_persona_group_prompt,
    get_persona_life_prompt,
)
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
PERSONA_TEMPERATURE = 0.4

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


async def _generate(prompt: str, session_id: str) -> dict | None:
    data = extract_and_parse_json(
        await feedback_client.generate(
            prompt, session_id=session_id, temperature=PERSONA_TEMPERATURE
        )
    )
    return data if isinstance(data, dict) else None


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

    data = await _generate(
        get_persona_group_prompt(
            persona,
            life,
            group_text,
            journal,
            days_in_group,
            group_material(rows, persona.name),
            day,
        ),
        session_id,
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
    day: date,
    session_id: str,
) -> LifeDigest | None:
    """整理人生线，失败返回 None。"""

    data = await _generate(
        get_persona_life_prompt(persona, life, life_facts, group_styles, day), session_id
    )
    if data is None or not isinstance(data.get("life"), str):
        logger.warning(f"[Persona] {persona.name} {day} 人生线整理输出无效")
        return None
    return LifeDigest(
        text=data["life"].strip(), changes=str(data.get("changes") or "").strip()
    )


def group_style(group_text: str) -> str:
    """群层里【在这个群的样子】那一段，交给人生线判断风格提升。"""

    start = group_text.find("【在这个群的样子】")
    if start < 0:
        return ""
    end = group_text.find("【", start + 1)
    return group_text[start : end if end > 0 else None].strip()


# ==================== 线上每日整理 ====================


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
    still_current: Callable[[str], bool],
) -> tuple[PersonaLifeModel, dict[str, PersonaGroupModel]]:
    persona = PRESETS[filename]
    life_session = f"persona:{filename}"
    life, _ = await PersonaLifeModel.get_or_create(preset=filename)
    layers = {
        session_id: (await PersonaGroupModel.get_or_create(session_id=session_id, preset=filename))[0]
        for session_id in session_ids
    }
    joined = {session_id: await _joined(session_id) for session_id in session_ids}

    if not life.text:
        result = await _retry(
            lambda: digest_life(
                persona, life="", life_facts=[], group_styles=[], day=yesterday, session_id=life_session
            )
        )
        if result is None:
            return life, {}
        life.text, life.last_digest = result.text, yesterday
        await life.save()
        await _history("life", filename, yesterday, result.text, f"[初始化] {result.changes}")

    # 新群层不补历史，从昨天开始
    def next_day(layer: PersonaGroupModel) -> date:
        return layer.digested_until + timedelta(days=1) if layer.digested_until else yesterday

    # 某个群某天整理失败就停在那天，后面几天等下一晚，保证日记和水位按天连续
    blocked: set[str] = set()
    updated: dict[str, PersonaGroupModel] = {}
    day = min((next_day(layer) for layer in layers.values()), default=yesterday + timedelta(days=1))
    while day <= yesterday:
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
                if result is None or not still_current(session_id):
                    blocked.add(session_id)
                    continue
                layer.text = result.text
                layer.journal = (layer.journal + [result.observation])[-JOURNAL_DAYS:]
                life.pending_facts = life.pending_facts + result.life_facts
                await life.save()
                await _history("group", f"{session_id}:{filename}", day, result.text, result.changes)
                updated[session_id] = layer
            layer.digested_until = day
            await layer.save()

        if life_due(life.last_digest, day, bool(life.pending_facts)):
            # 风格提升只看现在启用的群：停用、退出的群里沾上的说法不该继续「变老」被提升
            styles = [style for style in (group_style(layer.text) for layer in layers.values()) if style]
            result = await _retry(
                lambda: digest_life(
                    persona,
                    life=life.text,
                    life_facts=life.pending_facts,
                    group_styles=styles,
                    day=day,
                    session_id=life_session,
                )
            )
            if result is not None:
                life.text, life.last_digest, life.pending_facts = result.text, day, []
                await life.save()
                await _history("life", filename, day, result.text, result.changes)
        day += timedelta(days=1)

    logger.info(
        f"[Persona] {filename} 整理到 {yesterday}：群层更新 {len(updated)} 个群，"
        f"整理失败停在半路的群 {sorted(blocked) or '无'}"
    )
    return life, updated


async def maintain_personas(
    group_ids: list[int], still_current: Callable[[str], bool]
) -> tuple[dict[str, str], dict[str, str]]:
    """每日整理：设了人设的启用群把群层推到昨天，再整理对应预设的人生线。

    返回（{预设文件名: 人生线}, {群号: 群层}），外层用来同步已加载 session 的只读副本。
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
    for filename, session_ids in by_preset.items():
        life, updated = await _maintain_preset(filename, session_ids, yesterday, still_current)
        lives[filename] = life.text
        groups.update({session_id: layer.text for session_id, layer in updated.items()})
    return lives, groups
