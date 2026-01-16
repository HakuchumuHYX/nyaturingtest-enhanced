# nyaturingtest/memory_query.py
import json
import asyncio
from nonebot import on_command, logger
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message
from nonebot.params import CommandArg
from nonebot.utils import run_sync
from nonebot.exception import FinishedException

from .state_manager import ensure_group_state
from .utils import extract_and_parse_json
from .models import GlobalMessageModel

# 定义命令
query_memory = on_command("查询记忆", aliases={"memory", "印象"}, priority=5, block=True)


@query_memory.handle()
async def handle_query_memory(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    group_id = event.group_id
    sender_id = str(event.user_id)

    # 1. 确定目标用户
    target_id = sender_id
    target_name = event.sender.card or event.sender.nickname or sender_id

    # 检查是否有 @其他人
    for seg in args:
        if seg.type == "at":
            at_id = str(seg.data.get("qq", ""))
            if at_id:
                target_id = at_id
                # 尝试获取被@人的昵称
                try:
                    info = await bot.get_group_member_info(group_id=group_id, user_id=int(target_id))
                    target_name = info.get("card") or info.get("nickname") or target_id
                except Exception:
                    target_name = target_id
                break

    # 2. 获取群组状态
    state = ensure_group_state(group_id)
    if not state:
        await query_memory.finish("本群尚未启用 AI 功能。")
        return

    # 3. 收集数据 (加锁读取)
    profile_data = None
    bot_name = "Bot"
    bot_role = "AI助手"
    recent_user_msgs = []
    vector_records = []

    async with state.session_lock:
        await state.session.load_session()
        bot_name = state.session.name()
        bot_role = state.session.role()

        # --- 用户画像查找逻辑 ---
        profile = state.session.profiles.get(target_id)
        if not profile:
            profile = state.session.profiles.get(target_name)

        if profile:
            profile_data = {
                "valence": profile.emotion.valence,
                "arousal": profile.emotion.arousal,
                "dominance": profile.emotion.dominance,
                "interactions": len(profile.interactions),
                "last_seen": profile.last_update_time.strftime("%Y-%m-%d %H:%M")
            }
        else:
            profile_data = {
                "valence": 0.0,
                "arousal": 0.0,
                "dominance": 0.0,
                "interactions": 0,
                "last_seen": "未知"
            }

        # --- B. 读取 向量数据库 (双重过滤) ---
        try:
            search_queries = [
                f"关于{target_name}的记忆",
                f"我对{target_name}的看法",
                f"{target_name}做过的事",
                f"{target_name}的性格特点"
            ]

            # [关键] 构造过滤条件
            # 基础条件：只查记忆 (source="memory")，排除预设
            where_conditions = [{"source": {"$eq": "memory"}}]

            # 增强条件：如果已知 target_id，且不为空，则强制匹配 user_id
            if target_id and target_id.strip():
                where_conditions.append({"user_id": {"$eq": target_id}})

            # 组合条件 (ChromaDB 的 $and 语法)
            # 最终形式: {"$and": [{"source": {"$eq": "memory"}}, {"user_id": {"$eq": "12345"}}]}
            search_filter = {"$and": where_conditions} if len(where_conditions) > 1 else where_conditions[0]

            if hasattr(state.session, 'long_term_memory'):
                vector_records = await run_sync(state.session.long_term_memory.retrieve)(
                    search_queries,
                    k=5,
                    where=search_filter
                )
                if vector_records:
                    unique_recs = []
                    seen = set()
                    for rec in vector_records:
                        content = rec.get('content', '')
                        if content and content not in seen:
                            seen.add(content)
                            unique_recs.append(content)
                    vector_records = unique_recs

                    logger.debug(f"查询记忆: 检索到 {len(vector_records)} 条记录 (Filter: {search_filter})")
        except Exception as e:
            logger.error(f"向量记忆检索失败: {e}")

        # --- C. 读取 SQL 数据库 (最近发言) ---
        try:
            from .models import SessionModel
            session_db = await SessionModel.get_or_none(id=state.session.id)

            if session_db:
                db_msgs = []
                # 优先 ID 查
                if target_id and target_id.strip():
                    db_msgs = await GlobalMessageModel.filter(
                        session=session_db,
                        user_id=target_id
                    ).order_by("-time").limit(10)
                # 兜底 名字 查
                if not db_msgs:
                    db_msgs = await GlobalMessageModel.filter(
                        session=session_db,
                        user_name=target_name
                    ).order_by("-time").limit(10)

                recent_user_msgs = [m.content for m in reversed(db_msgs)]

            if not recent_user_msgs:
                recent_user_msgs = ["(暂无最近发言记录)"]
        except Exception as e:
            logger.error(f"查询历史消息失败: {e}")
            recent_user_msgs = ["(查询历史失败)"]

    # 4. 判断逻辑
    if profile_data['interactions'] == 0 and not vector_records:
        msg = "我对你还没有形成具体的印象呢，多和我聊聊天吧！" if target_id == sender_id else f"我的记忆中暂时没有关于 {target_name} 的印象。"
        await query_memory.finish(msg)
        return

    # 5. 构建 Prompt
    await query_memory.send("正在回溯记忆深处...")

    vector_memory_str = "\n".join([f"- {rec}" for rec in vector_records]) if vector_records else "(暂无深层记忆)"

    prompt = f"""
你现在的名字是"{bot_name}"，设定是"{bot_role}"。
请根据以下数据，生成你对用户"{target_name}"的印象评价。

[用户数据]
- 情感定位(VAD模型):
  - 愉悦度(Valence, -1讨厌~1喜欢): {profile_data['valence']:.2f}
  - 关注度(Arousal, 0无感~1兴奋): {profile_data['arousal']:.2f}
  - 支配度(Dominance, -1你畏惧他~1你掌控他): {profile_data['dominance']:.2f}
- 交互深度: {profile_data['interactions']} 次
- 长期记忆碎片(重要参考 - 这些是发生在他身上的真实事件):
{vector_memory_str}
- 他最近说过的话: {json.dumps(recent_user_msgs, ensure_ascii=False)}

[任务]
请模仿你的角色语气，以第一人称输出一个 JSON 对象：
{{
    "description": "结合'长期记忆碎片'和'最近说过的话'，评价这个用户。描述你们的过往经历（如果有）、关系（朋友、陌生人、死对头等），以及他对你的态度。100字以内。",
    "emotion": "3-5个关键词，概括你对他的感觉（例如：'信赖, 亲密, 有趣' 或 '冷漠, 警惕, 陌生'）"
}}
"""

    # 6. 调用 LLM (带重试)
    from .config import plugin_config
    max_retries = 2

    for attempt in range(max_retries + 1):
        try:
            # 使用 JSON Mode
            response = await state.client.generate_response(
                prompt=prompt,
                model=plugin_config.nyaturingtest_chat_openai_model,
                temperature=0.8 + (attempt * 0.2),
                response_format={"type": "json_object"}
            )

            result = extract_and_parse_json(response)

            if result and "description" in result:
                description = result.get("description", "数据解析错误")
                emotion = result.get("emotion", "未知")

                msg = f"=== {target_name} 的印象档案 ===\n\n"
                msg += f"「{description}」\n\n"
                msg += f"标签: {emotion}\n"
                msg += f"------------------\n"
                msg += f"记忆深度: {profile_data['interactions']} | VAD: {profile_data['valence']:.1f}/{profile_data['arousal']:.1f}/{profile_data['dominance']:.1f}"

                await query_memory.finish(msg)
                return

            else:
                logger.warning(f"印象生成 JSON 解析失败: {response}")

        except FinishedException:
            raise
        except Exception as e:
            logger.error(f"LLM 请求异常: {e}")

        if attempt < max_retries:
            await asyncio.sleep(1)

    await query_memory.finish("大脑处理过载，记忆读取失败，请稍后再试。")
