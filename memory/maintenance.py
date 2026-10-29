"""长期记忆每日维护：03:30 把新碎片整理进档案、群志和人设成长，再删过期碎片。

读 chat.groups 的注册表，对已加载的群同步内存副本，所以不放进 digest / persona。
"""

from collections.abc import Callable

from nonebot import logger, require

require("nonebot_plugin_apscheduler")
from nonebot_plugin_apscheduler import scheduler  # noqa: E402  必须在 require 之后导入

from ..chat.groups import group_states, runtime_enabled_groups  # noqa: E402
from .digest import digest_group_notes, digest_user_profiles  # noqa: E402
from .persona import maintain_personas  # noqa: E402
from .vector import maintain_memories  # noqa: E402


def _generation(group_id: int) -> int:
    """未加载的群按 0 算：加载不改 generation，只有 reset/load_preset 等会递增。"""

    group = group_states.get(group_id)
    return group.session.state.generation if group else 0


def _generation_guard(group_id: int) -> Callable[[], bool]:
    """记下此刻的 generation，返回「会话还没被 reset/load_preset 作废」的检查。"""

    generation = _generation(group_id)
    return lambda: _generation(group_id) == generation


async def refresh_group_notes(group_id: int, *, min_new: int = 1) -> str | None:
    """把新碎片整理进群志；群志不随 save_session 回写，已加载的群同步内存副本。"""

    still_current = _generation_guard(group_id)
    notes = await digest_group_notes(str(group_id), still_current, min_new=min_new)
    group = group_states.get(group_id)
    if notes is not None and group is not None and still_current():
        group.session.state.group_notes = notes
    return notes


async def maintain_long_term_memory() -> None:
    """长期记忆每日维护：先把新碎片整理进档案与群志，再整理人设成长，最后删过期碎片，
    这样即将过期的碎片一定先被归纳过；最后让已加载的群下次重新读矩阵。"""

    for group_id in list(runtime_enabled_groups):
        session_id = str(group_id)
        still_current = _generation_guard(group_id)
        try:
            summaries = await digest_user_profiles(session_id, still_current)
            await refresh_group_notes(group_id)
        except Exception as e:
            logger.warning(f"群 {group_id} 档案/群志整理失败: {e}")
            continue
        # 档案不随 save_session 回写，已加载的群要同步内存副本
        group = group_states.get(group_id)
        if group is not None and still_current():
            group.session.state.user_summaries.update(summaries)

    await _maintain_personas()

    try:
        await maintain_memories()
    except Exception as e:
        logger.warning(f"长期记忆定时维护失败: {e}")
    for group in group_states.values():
        group.session.vector_memory.drop_cache()


async def _maintain_personas() -> None:
    group_ids = list(runtime_enabled_groups)
    guards = {str(group_id): _generation_guard(group_id) for group_id in group_ids}
    try:
        lives, group_texts = await maintain_personas(group_ids, guards)
    except Exception as e:
        logger.warning(f"人设成长整理失败: {e}")
        return
    # 人生线和群层不随 save_session 回写，已加载的群要同步内存副本
    for group_id in group_ids:
        group = group_states.get(group_id)
        if group is None or not guards[str(group_id)]():
            continue
        state = group.session.state
        if state.preset in lives:
            state.self_life = lives[state.preset]
        if str(group_id) in group_texts:
            state.self_in_group = group_texts[str(group_id)]


def setup_maintenance_job():
    scheduler.add_job(
        maintain_long_term_memory,
        "cron",
        hour=3,
        minute=30,
        id="nyaturingtest_vector_maintenance",
        misfire_grace_time=3600,
        replace_existing=True,
    )
