# nyaturingtest/memory_query.py
import json
import asyncio
from datetime import datetime
from nonebot import on_command, logger
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message
from nonebot.params import CommandArg
from nonebot.utils import run_sync
from nonebot.exception import FinishedException

from ..core.state_manager import ensure_group_state
from ..utils import extract_and_parse_json, calculate_dynamic_k, should_store_memory
from ..database.repository import SessionRepository
from ..core.logic import llm_response
from ..config import plugin_config, get_effective_chat_model

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

        # --- 用户画像与交互统计逻辑 ---
        # 使用 Repository 获取真实的交互次数
        interaction_count = await SessionRepository.get_interaction_count(state.session.id, target_id)

        # 获取内存中的情绪数据
        profile = state.session.profiles.get(target_id)

        # 构造显示数据
        valence = profile.emotion.valence if profile else 0.0
        arousal = profile.emotion.arousal if profile else 0.0
        dominance = profile.emotion.dominance if profile else 0.0
        last_seen = profile.last_update_time.strftime("%Y-%m-%d %H:%M") if profile else "未知"

        profile_data = {
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
            "interactions": interaction_count,
            "last_seen": last_seen
        }

        # --- 向量记忆检索 (RAG) ---
        try:
            search_queries = [
                f"关于{target_name}的记忆",
                f"我对{target_name}的看法",
                f"{target_name}做过的事",
                f"{target_name}的性格特点"
            ]

            # 构造过滤条件：匹配 target_id 或者 user_id 为空 (全局记忆/未标记记忆)
            user_filter = []
            if target_id and target_id.strip():
                user_filter = [{"user_id": {"$eq": target_id}}, {"user_id": {"$eq": ""}}]
            else:
                user_filter = [{"user_id": {"$eq": ""}}]

            search_filter = {
                "$and": [
                    {"source": {"$eq": "memory"}},
                    {"$or": user_filter}
                ]
            }

            if hasattr(state.session, 'long_term_memory'):
                # 获取用户记忆数量
                memory_count = await run_sync(state.session.long_term_memory.count_by_user)(target_id)
                
                # 获取首次交互时间，计算天数
                first_interaction_time = await SessionRepository.get_first_interaction_time(
                    state.session.id, target_id
                )
                if first_interaction_time:
                    # 统一为 naive datetime，避免时区问题
                    if first_interaction_time.tzinfo is not None:
                        first_interaction_time = first_interaction_time.replace(tzinfo=None)
                    days_since_first = (datetime.now() - first_interaction_time).days
                else:
                    days_since_first = 0
                
                # 使用综合评分计算动态 k 值
                dynamic_k = calculate_dynamic_k(interaction_count, memory_count, days_since_first)
                
                # 使用时间衰减检索（近期记忆权重更高）
                vector_records = await run_sync(state.session.long_term_memory.retrieve_with_decay)(
                    search_queries,
                    k=dynamic_k,
                    where=search_filter,
                    use_rerank=True,
                    decay_rate=0.02  # 衰减率：约35天后权重减半
                )
                
                # 后过滤：去重 + 质量过滤
                if vector_records:
                    unique_recs = []
                    seen = set()
                    for rec in vector_records:
                        content = rec.get('content', '')
                        # 质量过滤：排除低质量内容
                        if content and content not in seen and should_store_memory(content):
                            seen.add(content)
                            unique_recs.append(content)
                    vector_records = unique_recs

                    logger.debug(
                        f"查询记忆: 交互={interaction_count}, 记忆量={memory_count}, "
                        f"天数={days_since_first}, 目标k={dynamic_k}, 实际={len(vector_records)}条"
                    )
        except Exception as e:
            logger.error(f"向量记忆检索失败: {e}")

        # --- 获取最近聊天记录 ---
        recent_user_msgs = await SessionRepository.get_recent_messages_by_user(
            state.session.id, 
            user_id=target_id, 
            user_name=target_name, 
            limit=10
        )
        if not recent_user_msgs:
            recent_user_msgs = ["(暂无最近发言记录)"]

    # 4. 判断逻辑 (如果没有交互且没有记忆，直接返回)
    if profile_data['interactions'] == 0 and not vector_records:
        msg = "我对你还没有形成具体的印象呢，多和我聊聊天吧！" if target_id == sender_id else f"我的记忆中暂时没有关于 {target_name} 的印象。"
        await query_memory.finish(msg)
        return

    # 5. 构建 Prompt
    await query_memory.send("正在回溯记忆深处...")

    vector_memory_str = "\n".join([f"- {rec}" for rec in vector_records]) if vector_records else "(暂无深层记忆)"

    # Prompt 增加甄别指令，防止张冠李戴
    prompt = f"""
你现在的名字是"{bot_name}"，设定是"{bot_role}"。
请根据以下数据，生成你对用户"{target_name}"的印象评价。

[用户数据]
- 情感定位(VAD模型):
  - 愉悦度(Valence, -1讨厌~1喜欢): {profile_data['valence']:.2f}
  - 关注度(Arousal, 0无感~1兴奋): {profile_data['arousal']:.2f}
  - 支配度(Dominance, -1你畏惧他~1你掌控他): {profile_data['dominance']:.2f}
- 交互深度: {profile_data['interactions']} 次
- 长期记忆碎片(重要参考):
{vector_memory_str}
**(注意: 记忆碎片可能包含群聊中其他人的信息。请仔细甄别，只提取主语是"{target_name}"或明显关于他的事件，忽略无关人员的记忆。)**

- 他最近说过的话: {json.dumps(recent_user_msgs, ensure_ascii=False)}

[任务]
请模仿你的角色语气，以第一人称输出一个 JSON 对象：
{{
    "description": "结合'长期记忆碎片'和'最近说过的话'，评价这个用户。描述你们的过往经历（如果有）、关系（朋友、陌生人、死对头等），以及他对你的态度。100字以内。",
    "emotion": "3-5个关键词，概括你对他的感觉（例如：'信赖, 亲密, 有趣' 或 '冷漠, 警惕, 陌生'）"
}}
"""

    # 6. 调用 LLM
    max_retries = 2

    def make_usage_recorder(model_name_record: str):
        def _recorder(usage: dict):
            asyncio.create_task(
                SessionRepository.log_token_usage(
                    session_id=str(state.session.id),
                    model_name=model_name_record,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0)
                )
            )
        return _recorder

    for attempt in range(max_retries + 1):
        try:
            # 使用统一的 llm_response 封装
            response = await llm_response(
                state.client,
                prompt,
                model=get_effective_chat_model(),
                temperature=0.8 + (attempt * 0.2),
                json_mode=True,
                on_usage=make_usage_recorder(get_effective_chat_model())
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
