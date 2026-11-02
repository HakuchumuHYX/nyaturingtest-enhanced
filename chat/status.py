"""她自己的时间线：现在在做什么、一段过完定下一段、放下手机时被叫到、最近几晚几点睡。

一个预设一份，所有群共用：她是一个人，在 A 群说去开组会了，B 群也不能还在水。
一次只定眼下这一段，每段都从上一段接着走，不在一天开始时排好。状态只在这几种时候变：
到点（due）、她在群里自己说了（said）、away 时被叫到并拿起了手机（called）、启动或断了很久（start）、/calm（calm）。
状态里只有她自己在做什么，不带任何群的内容：被叫到时不给群名，状态模型只看她开口的时间、不看说了什么。
"""

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from nonebot import logger, require

require("nonebot_plugin_apscheduler")
from nonebot_plugin_apscheduler import scheduler  # noqa: E402  必须在 require 之后导入

from ..llm import feedback_client  # noqa: E402
from ..metrics import log_event  # noqa: E402
from ..prompts import (  # noqa: E402
    PRESETS,
    diary_line,
    get_status_called_prompt,
    get_status_prompt,
    get_time_description,
    segment_line,
)
from ..storage.db import load_life_text, load_persona_status, save_persona_status  # noqa: E402
from .groups import is_shutting_down, sessions  # noqa: E402

MODES = ("free", "busy", "away")
CHECK_INTERVAL_SECONDS = 60
# 一段的长短由模型定，这里只挡离谱的值；睡觉也是一段，所以上限要够一整晚
MIN_MINUTES = 10
MAX_MINUTES = 720
DOING_CHARS = 40
# 模型失败或输出无效时，原状态先延长这么久再定
RETRY_MINUTES = 10
# 到点后过了这么久才定下一段，说明服务停过，要告诉模型中间断了多久
GAP_NOTE_MINUTES = 30
# 被叫到时最多这么久问一次状态模型；节流期间攒下的到点一起问
CALL_THROTTLE = timedelta(minutes=10)
# 给状态模型看的：时间线最近几段、几天日记、她最近几次开口（只给时间）
TIMELINE_PROMPT_SEGMENTS = 16
STATUS_DIARY_DAYS = 7
SPEAK_TIMES_LIMIT = 6
SPEAK_TIMES_WINDOW = timedelta(hours=2)
# 时间线写进日记后只留最近这么久；还没写日记的那几天整段留着
TIMELINE_KEEP = timedelta(hours=18)
NIGHTS_KEEP = 7
STATUS_TEMPERATURE = 0.5

SOURCE_DUE = "due"
SOURCE_SAID = "said"
SOURCE_CALLED = "called"
SOURCE_START = "start"
SOURCE_CALM = "calm"


@dataclass
class PersonaStatus:
    preset: str
    # 每段 {"start", "mode", "doing", "sleep", "source"}，最后一段是现在
    timeline: list[dict] = field(default_factory=list)
    until: datetime = datetime.min
    # 每晚 {"sleep", "wake"}，还没醒时 wake 为 None
    nights: list[dict] = field(default_factory=list)
    # 每天 {"day": "YYYY-MM-DD", "text"}，只由每日整理写
    diary: list[dict] = field(default_factory=list)
    diary_until: date | None = None
    # 每次改状态加一；调模型前记下，回来时变了说明这期间她在群里自己说了新情况，以她说的为准
    version: int = 0
    deciding: asyncio.Task | None = None
    # 这次放下手机以来叫她的消息，问过的条数，上次问的时间
    calls: list[str] = field(default_factory=list)
    calls_asked: int = 0
    last_call_ask: datetime = datetime.min
    call_task: asyncio.Task | None = None
    save_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def current(self) -> dict | None:
        return self.timeline[-1] if self.timeline else None

    @property
    def away(self) -> bool:
        return self.current is not None and self.current["mode"] == "away"


_statuses: dict[str, PersonaStatus] = {}


def get_status(preset: str) -> PersonaStatus | None:
    return _statuses.get(preset)


def is_away(preset: str) -> bool:
    status = _statuses.get(preset)
    return status is not None and status.away


async def _load(preset: str) -> PersonaStatus:
    status = _statuses.get(preset)
    if status is not None:
        return status
    status = PersonaStatus(preset=preset)
    data = await load_persona_status(preset)
    if data is not None:
        status.timeline = data["timeline"]
        status.until = data["until"] or datetime.min
        status.nights = data["nights"]
        status.diary = data["diary"]
        status.diary_until = data["diary_until"]
    # await 期间可能已经有别的调用载入过
    return _statuses.setdefault(preset, status)


# ---------- 给提示词的文字 ----------


def _segment_end(status: PersonaStatus, index: int) -> datetime:
    following = status.timeline[index + 1 :]
    return following[0]["start"] if following else status.until


def _timeline_lines(status: PersonaStatus) -> list[str]:
    start = max(0, len(status.timeline) - TIMELINE_PROMPT_SEGMENTS)
    lines = []
    for index in range(start, len(status.timeline)):
        segment = status.timeline[index]
        if index == len(status.timeline) - 1:
            lines.append(f"{segment_line(segment, None)}（原定到 {status.until:%H:%M}）")
        else:
            lines.append(segment_line(segment, _segment_end(status, index)))
    return lines


def _night_lines(status: PersonaStatus) -> list[str]:
    return [
        f"{night['sleep']:%m-%d %H:%M} 睡，" + (f"{night['wake']:%H:%M} 起" if night["wake"] else "还在睡")
        for night in status.nights
    ]


def recent_diary(preset: str, days: int) -> list[str]:
    status = _statuses.get(preset)
    return [diary_line(entry) for entry in status.diary[-days:]] if status else []


def _just_back(status: PersonaStatus, last_look: datetime) -> str:
    """上次看这个群之后才结束的最近一段 away：她放下手机时这个群还没看过，回来第一眼要知道自己刚干嘛去了。"""

    for index in range(len(status.timeline) - 2, -1, -1):
        segment = status.timeline[index]
        end = _segment_end(status, index)
        if end <= last_look:
            return ""
        if segment["mode"] == "away":
            return segment_line(segment, end)
    return ""


def describe_now(preset: str, last_look: datetime) -> str:
    """Feedback / Chat / 主动话题里的 now：现在这一段；这个群还没在她回来后看过时，带上刚结束的那段 away。"""

    status = _statuses.get(preset)
    if status is None or status.current is None:
        return ""
    now = f"{segment_line(status.current, None)}，大概到 {status.until:%H:%M}"
    back = _just_back(status, last_look)
    return f"刚结束：{back}；现在：{now}" if back else now


def today_lines(preset: str) -> list[str]:
    """今天醒来以后的各段（还没醒过就是今天 0 点以后的），不含现在这一段。"""

    status = _statuses.get(preset)
    if status is None:
        return []
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    woke = [night["wake"] for night in status.nights if night["wake"] and night["wake"] >= today]
    since = woke[-1] if woke else today
    return [
        segment_line(segment, _segment_end(status, index))
        for index, segment in enumerate(status.timeline[:-1])
        if _segment_end(status, index) > since
    ]


# ---------- 定下一段 ----------


def _speak_times(preset: str, now: datetime) -> list[str]:
    times = {
        message.time
        for session in list(sessions.values())
        if session.state.preset == preset and session.state.name
        for message in session.short_term_memory.messages_after(now - SPEAK_TIMES_WINDOW, 200)
        if message.user_name == session.state.name and not message.is_action
    }
    return [f"{moment:%H:%M}" for moment in sorted(times)[-SPEAK_TIMES_LIMIT:]]


def parse_segment(data) -> dict | None:
    """状态模型和 Chat 的 status 字段共用的入口校验；无效返回 None。返回 {"mode", "doing", "minutes", "sleep"}。"""

    if not isinstance(data, dict):
        return None
    mode = data.get("mode")
    doing = str(data.get("doing") or "").strip()[:DOING_CHARS]
    sleep = data.get("sleep") is True
    try:
        minutes = int(data.get("minutes"))
    except (TypeError, ValueError):
        return None
    if mode not in MODES or not doing or (sleep and mode != "away"):
        return None
    return {
        "mode": mode,
        "doing": doing,
        "minutes": max(MIN_MINUTES, min(MAX_MINUTES, minutes)),
        "sleep": sleep,
    }


def _note_night(status: PersonaStatus) -> None:
    """按刚加上的这段维护 nights：被叫起来看一眼又接着睡的算同一晚，起床时间按醒着的第一段算。"""

    segment = status.timeline[-1]
    awake_night = status.nights and status.nights[-1]["wake"] is None
    if segment["sleep"]:
        if not awake_night:
            status.nights = (status.nights + [{"sleep": segment["start"], "wake": None}])[-NIGHTS_KEEP:]
        return
    if awake_night and segment["source"] != SOURCE_CALLED:
        previous = status.timeline[-2] if len(status.timeline) > 1 else segment
        woke_by_call = previous["source"] == SOURCE_CALLED
        status.nights[-1]["wake"] = previous["start"] if woke_by_call else segment["start"]


def _prune(status: PersonaStatus, now: datetime) -> None:
    """写进日记的只留最近 TIMELINE_KEEP；还没写日记的那几天整段留着，日记要用。"""

    keep_after = now - TIMELINE_KEEP
    kept = [
        segment
        for segment in status.timeline[:-1]
        if segment["start"] >= keep_after
        or status.diary_until is None
        or segment["start"].date() > status.diary_until
    ]
    status.timeline = kept + status.timeline[-1:]


def _wake_workers(preset: str) -> None:
    """away 结束了：同一个预设的各群 worker 醒来按未读重新定时，开始补看。"""

    for session in list(sessions.values()):
        if session.state.preset == preset:
            session.new_message_signal.set()


async def _save(status: PersonaStatus) -> None:
    async with status.save_lock:
        await save_persona_status(status.preset, status.timeline, status.until, status.nights)


async def _apply(status: PersonaStatus, segment: dict, source: str, reason: str) -> None:
    """新的一段成为现在。改内存的这几行之间没有 await，和版本号检查一起就不会被别的改动插进来。"""

    now = datetime.now()
    was_away = status.away
    status.timeline.append(
        {
            "start": now,
            "mode": segment["mode"],
            "doing": segment["doing"],
            "sleep": segment["sleep"],
            "source": source,
        }
    )
    status.until = now + timedelta(minutes=segment["minutes"])
    status.version += 1
    _note_night(status)
    _prune(status, now)
    if was_away != status.away:
        status.calls, status.calls_asked = [], 0
    log_event(
        "status",
        preset=status.preset,
        source=source,
        mode=segment["mode"],
        doing=segment["doing"],
        until=f"{status.until:%m-%d %H:%M}",
        sleep=segment["sleep"],
        reason=reason,
    )
    if was_away and not status.away:
        _wake_workers(status.preset)
    await _save(status)


async def _status_background(status: PersonaStatus, now: datetime) -> dict:
    return {
        "life": await load_life_text(status.preset),
        "diary": [diary_line(entry) for entry in status.diary[-STATUS_DIARY_DAYS:]],
        "nights": _night_lines(status),
        "timeline": _timeline_lines(status),
        "speak_times": _speak_times(status.preset, now),
        "time_info": get_time_description(now),
    }


async def _decide(status: PersonaStatus, source: str) -> None:
    """到点、启动、/calm：让状态模型定下一段。"""

    persona = PRESETS.get(status.preset)
    if persona is None:
        return
    version = status.version
    now = datetime.now()
    gap = None
    if status.current is not None and (now - status.until) > timedelta(minutes=GAP_NOTE_MINUTES):
        gap = int((now - status.until).total_seconds() // 60)
    data = await feedback_client.generate_json(
        get_status_prompt(persona, **await _status_background(status, now), gap_minutes=gap),
        session_id=f"status:{status.preset}",
        temperature=STATUS_TEMPERATURE,
    )
    if status.version != version:
        log_event("status_discarded", "DEBUG", preset=status.preset, source=source)
        return
    segment = parse_segment(data)
    if segment is None:
        log_event("status_invalid", "WARNING", preset=status.preset, source=source, data=data)
        status.until = max(status.until, now) + timedelta(minutes=RETRY_MINUTES)
        await _save(status)
        return
    if gap is not None:
        source = SOURCE_START
    await _apply(status, segment, source, str(data.get("reason") or ""))


async def _answer_calls(status: PersonaStatus) -> None:
    """away 时有人叫她：节流到点后把攒下的一起问状态模型，拿不拿起手机由它判断。"""

    while not is_shutting_down():
        wait = (status.last_call_ask + CALL_THROTTLE - datetime.now()).total_seconds()
        if wait > 0:
            await asyncio.sleep(wait)
        if not status.away or len(status.calls) <= status.calls_asked:
            return
        persona = PRESETS.get(status.preset)
        if persona is None:
            return
        status.calls_asked = len(status.calls)
        status.last_call_ask = now = datetime.now()
        version = status.version
        data = await feedback_client.generate_json(
            get_status_called_prompt(
                persona, **await _status_background(status, now), calls=list(status.calls)
            ),
            session_id=f"status:{status.preset}",
            temperature=STATUS_TEMPERATURE,
        )
        if status.version != version:
            return
        reason = str(data.get("reason") or "") if isinstance(data, dict) else ""
        segment = parse_segment(data) if isinstance(data, dict) and data.get("pick_up") else None
        if segment is None or segment["mode"] == "away":
            log_event(
                "status_call_ignored",
                preset=status.preset,
                calls=len(status.calls),
                reason=reason or f"invalid: {data}",
            )
            continue
        await _apply(status, segment, SOURCE_CALLED, reason)
        return


def note_call(preset: str, lines: list[str]) -> None:
    """away 时群里有人 @、回复、戳她：记下来交给 _answer_calls；不写是哪个群。"""

    status = _statuses.get(preset)
    if status is None or not status.away or not lines:
        return
    status.calls.extend(lines)
    if status.call_task is None or status.call_task.done():
        status.call_task = asyncio.create_task(_answer_calls(status))
        status.call_task.add_done_callback(_log_task_error)


async def apply_said(preset: str, raw) -> None:
    """Chat 的 status 字段：她在群里说出了和现在不同的新情况，直接成为新的一段。"""

    segment = parse_segment(raw)
    if segment is None:
        if raw is not None:
            logger.debug(f"Chat 的 status 无效，丢弃: {raw}")
        return
    await _apply(await _load(preset), segment, SOURCE_SAID, "她在群里说的")


async def redecide(preset: str) -> None:
    """/calm：不管这一段到没到点，现在重新定一次。"""

    status = await _load(preset)
    await _decide(status, SOURCE_CALM)


async def reload_diaries() -> None:
    """每日整理写完日记后同步内存副本，顺带按新的日记水位裁掉时间线。"""

    for status in list(_statuses.values()):
        data = await load_persona_status(status.preset)
        if data is None:
            continue
        status.diary, status.diary_until = data["diary"], data["diary_until"]
        _prune(status, datetime.now())
        await _save(status)


def _log_task_error(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception():
        logger.opt(exception=task.exception()).error(f"状态任务异常: {task.exception()}")


async def check_statuses() -> None:
    """每分钟：启用群用到的每个预设，没有状态或这一段到点了就定下一段。"""

    if is_shutting_down():
        return
    presets = {
        session.state.preset for session in list(sessions.values()) if session.state.name
    }
    now = datetime.now()
    for preset in presets:
        status = await _load(preset)
        # 还没定过状态时 until 是 datetime.min；第一次定失败也会被延后 RETRY_MINUTES
        if (status.deciding is not None and not status.deciding.done()) or now < status.until:
            continue
        source = SOURCE_START if status.current is None else SOURCE_DUE
        status.deciding = asyncio.create_task(_decide(status, source))
        status.deciding.add_done_callback(_log_task_error)


def status_text(preset: str) -> list[str]:
    """/status 里的几行：现在在干嘛、今天的各段。"""

    status = _statuses.get(preset)
    if status is None or status.current is None:
        return ["现在：还没定"]
    lines = [f"现在：{segment_line(status.current, None)}，到 {status.until:%H:%M}"]
    if status.calls:
        lines.append(f"放下手机以来被叫了 {len(status.calls)} 次")
    today = today_lines(preset)
    if today:
        lines.append("今天：")
        lines.extend(f"- {line}" for line in today)
    return lines


def setup_status_job() -> None:
    # 启动时马上跑一次：没有状态或停服期间到点了的，先定下来
    scheduler.add_job(
        check_statuses,
        "interval",
        seconds=CHECK_INTERVAL_SECONDS,
        id="nyaturingtest_status",
        next_run_time=datetime.now(),
        replace_existing=True,
    )
    logger.info(f"已注册时间线检查: 每 {CHECK_INTERVAL_SECONDS} 秒")
