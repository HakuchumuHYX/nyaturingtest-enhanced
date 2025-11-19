import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
import json
import os
import pickle
import random
import re
import traceback

import anyio  # 用于异步文件操作
from nonebot import logger
import nonebot_plugin_localstore as store
from nonebot.utils import run_sync
from openai import AsyncOpenAI
import httpx

from .client import LLMClient
from .config import plugin_config
from .emotion import EmotionState
from .hippo_mem import HippoMemory
from .impression import Impression
from .mem import Memory, Message
from .presets import PRESETS
from .profile import PersonProfile


@dataclass
class _SearchResult:
    """
    检索阶段的结果
    """

    mem_history: list[str]
    """
    记忆记录
    """


class _ChattingState(Enum):
    ILDE = 0
    """
    潜水状态
    """
    BUBBLE = 1
    """
    冒泡状态
    """
    ACTIVE = 2
    """
    对话状态
    """

    def __str__(self):
        match self:
            case _ChattingState.ILDE:
                return "潜水状态"
            case _ChattingState.BUBBLE:
                return "冒泡状态"
            case _ChattingState.ACTIVE:
                return "对话状态"


class Session:
    """
    群聊会话
    """

    def __init__(self, siliconflow_api_key: str, id: str = "global", name: str = "terminus"):
        self.id = id
        """
        会话ID，用于持久化时的标识
        """

        # [优化] 为记忆压缩模块也配置高性能的 HTTP 客户端
        memory_http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            timeout=60.0
        )

        self.global_memory: Memory = Memory(
            llm_client=LLMClient(
                client=AsyncOpenAI(
                    api_key=plugin_config.nyaturingtest_siliconflow_api_key,
                    base_url="https://api.siliconflow.cn/v1",
                    http_client=memory_http_client
                )
            )
        )
        """
        全局短时记忆
        """
        self.long_term_memory: HippoMemory = HippoMemory(
            llm_model=plugin_config.nyaturingtest_chat_openai_model,
            llm_api_key=plugin_config.nyaturingtest_chat_openai_api_key,
            llm_base_url=plugin_config.nyaturingtest_chat_openai_base_url,
            embedding_api_key=siliconflow_api_key,
            persist_directory=f"{store.get_plugin_data_dir()}/hippo_index_{id}",
        )
        """
        对聊天记录的长期记忆 (基于HippoRAG)
        """
        self.__name = name
        """
        我的名称
        """
        self.profiles: dict[str, PersonProfile] = {}
        """
        人物记忆
        """
        self.global_emotion: EmotionState = EmotionState()
        """
        全局情感状态
        """
        self.last_response: list[Message] = []
        """
        上次回复
        """
        self.chat_summary = ""
        """
        对话总结
        """
        self.__role = "一个男性人类"
        """
        我的角色
        """
        self.__chatting_state = _ChattingState.ILDE
        """
        对话状态
        """
        self.__bubble_willing_sum = 0.0
        """
        冒泡意愿总和（冒泡意愿会累积）
        """
        self.__update_hippo = False
        """
        是否重新索引，检索HippoRAG
        """
        self.__search_result = None

        # 从文件加载会话状态（如果存在）
        self.load_session()

    async def set_role(self, name: str, role: str):
        """
        设置角色
        """
        await self.reset()
        self.__role = role
        self.__name = name
        await self.save_session()  # 保存角色设置变更

    def role(self) -> str:
        """
        获取角色
        """
        return f"{self.__name}（{self.__role}）"

    def name(self) -> str:
        """
        获取名称
        """
        return self.__name

    async def reset(self):
        """
        重置会话
        """
        self.__name = "terminus"
        self.__role = "一个男性人类"
        await self.global_memory.clear()
        self.long_term_memory.clear()
        self.profiles = {}
        self.global_emotion = EmotionState()
        self.last_response = []
        self.chat_summary = ""
        await self.save_session()  # 保存重置后的状态

    async def calm_down(self):
        """
        冷静下来
        """
        self.global_emotion.valence = 0.0
        self.global_emotion.arousal = 0.0
        self.global_emotion.dominance = 0.0
        self.profiles = {}
        await self.save_session()  # 保存冷静后的状态

    def get_session_file_path(self) -> str:
        """
        获取会话文件路径
        """
        # 确保会话目录存在
        os.makedirs(f"{store.get_plugin_data_dir()}/yaturningtest_sessions", exist_ok=True)
        return f"{store.get_plugin_data_dir()}/yaturningtest_sessions/session_{self.id}.json"

    async def save_session(self):
        """
        保存会话状态到文件 (异步优化版)
        """
        try:
            # 准备要保存的数据
            session_data = {
                "id": self.id,
                "name": self.__name,
                "role": self.__role,
                "global_memory": {
                    "compressed_history": self.global_memory.access().compressed_history,
                    "messages": [msg.to_json() for msg in self.global_memory.access().messages],
                },
                "global_emotion": asdict(self.global_emotion),
                "chat_summary": self.chat_summary,
                "profiles": {
                    user_id: {
                        "user_id": profile.user_id,
                        "emotion": asdict(profile.emotion),
                        # interactions 是一个 deque，直接序列化
                        "interactions": pickle.dumps(profile.interactions).hex(),
                    }
                    for user_id, profile in self.profiles.items()
                },
                "last_response": [
                    {"time": msg.time.isoformat(), "user_name": msg.user_name, "content": msg.content}
                    for msg in self.last_response
                ],
                "chatting_state": self.__chatting_state.value,
            }

            # 使用 anyio 异步写入文件，防止阻塞事件循环
            file_path = self.get_session_file_path()
            async with await anyio.open_file(file_path, "w", encoding="utf-8") as f:
                json_str = json.dumps(session_data, ensure_ascii=False, indent=2)
                await f.write(json_str)

            logger.debug(f"[Session {self.id}] 会话状态已保存")
        except Exception as e:
            logger.error(f"[Session {self.id}] 保存会话状态失败: {e}")

    def load_session(self):
        """
        从文件加载会话状态
        """
        file_path = self.get_session_file_path()
        if not os.path.exists(file_path):
            logger.debug(f"[Session {self.id}] 会话文件不存在，使用默认状态")
            return

        try:
            with open(file_path, encoding="utf-8") as f:
                session_data = json.load(f)

            # 恢复会话状态
            self.__name = session_data.get("name", self.__name)
            self.__role = session_data.get("role", self.__role)

            # 恢复全局情绪状态
            emotion_data = session_data.get("global_emotion", {})
            self.global_emotion.valence = emotion_data.get("valence", 0.0)
            self.global_emotion.arousal = emotion_data.get("arousal", 0.0)
            self.global_emotion.dominance = emotion_data.get("dominance", 0.0)

            # 恢复全局短时记忆
            if "global_memory" in session_data:
                # [优化] 恢复时也注入优化后的客户端
                memory_http_client = httpx.AsyncClient(
                    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                    timeout=60.0
                )
                try:
                    self.global_memory = Memory(
                        compressed_message=session_data["global_memory"].get("compressed_history", ""),
                        messages=[Message.from_json(msg) for msg in session_data["global_memory"].get("messages", [])],
                        llm_client=LLMClient(
                            client=AsyncOpenAI(
                                api_key=plugin_config.nyaturingtest_siliconflow_api_key,
                                base_url="https://api.siliconflow.cn/v1",
                                http_client=memory_http_client
                            )
                        ),
                    )
                except Exception as e:
                    logger.error(f"[Session {self.id}] 恢复全局短时记忆失败: {e}")
                    # 重新初始化
                    self.global_memory = Memory(
                        llm_client=LLMClient(
                            client=AsyncOpenAI(
                                api_key=plugin_config.nyaturingtest_siliconflow_api_key,
                                base_url="https://api.siliconflow.cn/v1",
                                http_client=memory_http_client
                            )
                        )
                    )

            # 恢复聊天总结
            self.chat_summary = str(session_data.get("chat_summary", ""))

            # 恢复用户档案
            self.profiles = {}
            for user_id, profile_data in session_data.get("profiles", {}).items():
                profile = PersonProfile(user_id=profile_data.get("user_id", user_id))

                # 设置情绪
                emotion_data = profile_data.get("emotion", {})
                profile.emotion.valence = emotion_data.get("valence", 0.0)
                profile.emotion.arousal = emotion_data.get("arousal", 0.0)
                profile.emotion.dominance = emotion_data.get("dominance", 0.0)

                # 恢复交互记录
                if "interactions" in profile_data:
                    try:
                        profile.interactions = pickle.loads(bytes.fromhex(profile_data["interactions"]))
                        if not isinstance(profile.interactions, deque):
                            profile.interactions = deque(profile.interactions)
                    except Exception as e:
                        logger.error(f"[Session {self.id}] 恢复用户 {user_id} 交互记录失败: {e}")

                self.profiles[user_id] = profile

            # 恢复最后一次回复
            self.last_response = []
            for msg_data in session_data.get("last_response", []):
                try:
                    time = datetime.fromisoformat(msg_data.get("time"))
                except ValueError:
                    time = datetime.now()

                self.last_response.append(
                    Message(time=time, user_name=msg_data.get("user_name", ""), content=msg_data.get("content", ""))
                )

            # 恢复对话状态
            self.__chatting_state = _ChattingState(session_data.get("chatting_state", _ChattingState.ILDE.value))

            logger.info(f"[Session {self.id}] 会话状态已加载")
        except Exception as e:
            logger.error(f"[Session {self.id}] 加载会话状态失败: {e}")
            # 加载失败时使用默认状态，不需要额外操作

    def presets(self) -> list[str]:
        """
        获取可选预设
        """
        return [f"{filename}: {preset.name} {preset.role}" for filename, preset in PRESETS.items() if not preset.hidden]

    async def load_preset(self, filename: str) -> bool:
        """
        加载预设
        """
        # 自动补全 .json 后缀
        if not filename.endswith(".json") and f"{filename}.json" in PRESETS.keys():
            filename = f"{filename}.json"

        if filename not in PRESETS.keys():
            logger.error(f"不存在的预设：{filename}")
            return False

        preset = PRESETS[filename]
        await self.set_role(preset.name, preset.role)
        self.long_term_memory.add_texts(preset.knowledges)
        self.long_term_memory.add_texts(preset.relationships)
        self.long_term_memory.add_texts(preset.events)
        self.long_term_memory.add_texts(preset.bot_self)

        try:
            # 并发安全：使用 run_sync 防止阻塞
            await run_sync(self.long_term_memory.index)()
        except Exception as e:
            logger.error(f"预设索引构建失败: {e}")

        logger.info(f"加载预设：{filename} 成功")
        return True

    def status(self) -> str:
        """
        获取机器人状态
        """

        recent_messages = self.global_memory.access().messages
        recent_messages_str = (
            "\n".join([f"{msg.user_name}: {msg.content}" for msg in recent_messages]) if recent_messages else "没有消息"
        )

        return f"""
名字：{self.__name}

设定：{self.__role}

情感状态：V:{self.global_emotion.valence:.2f} A:{self.global_emotion.arousal:.2f} D:{self.global_emotion.dominance:.2f}

最近消息：
{recent_messages_str}

过去总结：
{self.global_memory.access().compressed_history}

现状认识：{self.chat_summary}
"""

    async def __search_stage(self):
        """
        检索阶段 (并行优化版)
        """
        logger.debug("检索阶段开始")

        # 准备要检索的 Query
        retrieve_messages = (
                [f"'{msg.user_name}':'{msg.content}'" for msg in self.global_memory.access().messages]
                + [self.global_memory.access().compressed_history]
                + [self.chat_summary]
        )

        # 1. 定义任务列表
        tasks = []

        # 任务A: 如果需要更新索引，则添加索引任务
        index_needed = False
        if self.__update_hippo:
            self.__update_hippo = False
            if self.long_term_memory._cache:
                logger.info("正在构建长期记忆索引(HippoRAG)...")
                index_needed = True
                tasks.append(run_sync(self.long_term_memory.index)())

        # 任务B: 检索任务 (总是执行)
        # retrieve 也是 CPU 密集型任务(图游走)，放入线程池
        logger.debug("正在检索长期记忆...")
        tasks.append(run_sync(self.long_term_memory.retrieve)(retrieve_messages, k=2))

        # 2. 并发执行所有任务
        try:
            results = await asyncio.gather(*tasks)

            # 3. 提取结果
            long_term_memory = []
            if index_needed:
                # results[0]是index的返回值，results[1]是retrieve的返回值
                logger.info("长期记忆索引构建完成")
                long_term_memory = results[1]
            else:
                # results[0]是retrieve的返回值
                long_term_memory = results[0]

            logger.debug(f"搜索到的相关记忆：{long_term_memory}")
        except Exception as e:
            logger.error(f"检索/索引阶段发生错误: {e}")
            traceback.print_exc()
            long_term_memory = []

        logger.debug("检索阶段结束")

        self.__search_result = _SearchResult(
            mem_history=long_term_memory,
        )

    @staticmethod
    def _extract_and_parse_json(response: str) -> dict | None:
        """
        从 LLM 响应中提取并解析 JSON，具有更强的鲁棒性
        """
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        match = re.search(r"(\{[\s\S]*\})", response)
        if match:
            json_str = match.group(1)
            try:
                json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        return None

    async def __feedback_stage(self, messages_chunk: list[Message], llm: Callable[[str], Awaitable[str]]):
        """
        反馈总结阶段 (已增强鲁棒性与容错)
        """
        logger.debug("反馈阶段开始")
        reaction_users = self.global_memory.related_users()
        related_profiles = [profile for profile in self.profiles.values() if profile.user_id in reaction_users]
        related_profiles_json = json.dumps(
            [
                {
                    "user_name": profile.user_id,
                    "emotion_tends_to_user": asdict(profile.emotion),
                }
                for profile in related_profiles
            ],
            ensure_ascii=False,
            indent=2,
        )
        if self.__search_result:
            search_stage_result = self.__search_result.mem_history
        else:
            search_stage_result = []

        code_start = "```" + "json"
        code_end = "```"

        prompt = f"""
你是QQ群用户 {self.__name}，不是 AI，而是一个有自我认知的真实用户，（你称自己为 "{self.__name}"），你的人设(身份认知)如
下:

{self.__role}

现在你正在回顾聊天信息，以客观视角分析“你的最新情绪”，整理信息保存，并对聊天内容做出总结，最后评估自己要不要加入对话

---

## 1. 任务目标
- 基于“新输入消息”的内容和“历史聊天”的背景，结合你之前的情绪，还有检索到的相关记忆，评估你当前的情绪
  - 情绪采用 VAD 模型，三个维度取值范围：
    - valence (愉悦度)：[-1.0, 1.0]
    - arousal (唤醒度)：[0.0, 1.0]
    - dominance (支配度)：[-1.0, 1.0]
- 基于“新输入消息”的内容和“历史聊天”的背景，结合你之前的情绪，你对相关人物的情绪倾向，还有检索到的相关记忆，评估你对“新
  输入消息”中**每条**消息的情感倾向
  - 如果消息和你完全无关，或你不感兴趣，那么给出的每个情感维度的值总是 0.0
  - 输出按照“新输入消息”的顺序
- 基于“历史聊天”的背景，“你在上次对话做出的总结”，还有检索到的相关记忆，用简短的语言总结聊天内容，总结注重于和上次对话的
  连续性，包括相关人物，简要内容。
  - 特别的，如果“历史聊天”，检索到的信息中不包含“你在上次对话做出的总结”的人物，那么在这次总结就不保留
  - 注意：要满足连续性需求，不能简单的只总结“新输入消息”的内容，还要结合上次总结和“历史聊天”的内容，并且不能因为这次的消
    息没有上次总结的内容的人物就不保留上次总结的内容，只有“历史聊天”，检索到的信息中不包含“你在上次对话做出的总结”的人物时，才
    不保留上次总结的内容
  - 例子A(断裂重启型):
    “你在上次对话做出的总结”
    小明，小红：讨论 AI 的道德问题。

    “新输入消息”
    小明：“我们来玩猜谜游戏吧！”
    小红：“好啊，我来第一个出题！”

    “总结”
    小明，小红：讨论的话题发生了明显转变，由 AI 的道德问题转变到了玩猜谜游戏。
  - 例子B(主题转移型):
    “你在上次对话做出的总结”
    小明，小红：讨论 AI 的道德问题。

    “新输入消息”
    小明：“我觉得 AI 应该有道德标准。”
    小红：“我同意！但是我们应该如何定义这些标准呢？”

    “总结”
    小明，小红：讨论 AI 的道德问题，继续深入探讨如何定义道德标准。

  - 例子C(无意义话题型):
    “你在上次对话做出的总结”
    小明，小红：讨论 AI 的道德问题。

    “新输入消息”
    小明：“awhnofbonog”
    小红：“2388y91ry9h”

    “总结”
    小明，小红：之前在讨论 AI 的道德问题。

  - 例子D(话题回归型):
    “你在上次对话做出的总结”
    小明，小红：讨论的话题发生了明显转变，由 AI 的道德问题转变到了玩猜谜游戏。

    “新输入消息”
    小明：“但是我还是想讨论 AI 是否需要道德”
    小红：“我觉得 AI 应该有道德标准。”

    “总结”
    小明，小红：讨论的话题由玩猜谜游戏回归到 AI 的道德问题。

  - 例子E(混合型):
    “你在上次对话做出的总结”
    小明，小红：讨论 AI 的道德问题。

    “新输入消息”
    小亮：“我们来玩猜谜游戏吧！”
    小明：“我觉得 AI 应该有道德标准。”
    小圆：“@小亮 好呀”
    小红：“我同意！但是我们应该如何定义这些标准呢？”

    “总结”
    小明，小红：讨论 AI 的道德问题，继续深入探讨如何定义道德标准。
    小亮，小圆：讨论玩猜谜游戏。

- 基于“新输入消息”的内容和“历史聊天”的背景，结合检索到的相关记忆进行分析，整理信息保存，要整理的信息和要求如下
  ## 要求：
  - 不能重复，即不能和下面提供的检索到的相关记忆已有内容重复
  ## 要整理的信息：
  - 无论信息是什么类别，都放到`analyze_result`字段
  - 事件类：
    - 如果包含事件类信息，则保存为事件信息，内容是对事件进行简要叙述
  - 资料类：
    - 如果包含资料类信息，则保存为知识信息，内容为资料的关键内容（如果很短也可以全文保存）及其可信度[0%-100%]，如：“ipho
    ne是由apple发布的智能手机系列产品，可信度99%”
  - 人物关系类
    - 如果包含人物关系类信息，则保存为人物关系信息，内容是对人物关系进行简要叙述（如：小明 是 小红 的 朋友）
  - 自我认知类
    - 如果你对自己有新的认知，则保存为自我认知信息，自我认知信息需要经过慎重考虑，主要参照你自己发送的消息，次要参照别人
      发送的消息，内容是对自我的认知（如：我喜欢吃苹果、我身上有纹身）

- 评估你改变对话状态的意愿，规则如下：
  - 意愿范围是[0.0, 1.0]
  - 对话状态分为三种：
    - 0：潜水状态
    - 1：冒泡状态
    - 2：对话状态
  - 如果你在状态0，那么分别评估你转换到状态1，2的意愿，其它意愿设0.0为默认值即可
  - 如果你在状态1，那么分别评估你转换到状态0，2的意愿，其它意愿设0.0为默认值即可
  - 如果你在状态2，那么评估你转换到状态0的意愿，其它意愿设0.0为默认值即可
  - 以下条件会影响转换到状态0的意愿：
    - 你进行这个话题的时间，太久了会让你疲劳，更容易转变到状态0
    - 是否有人回应你
    - 你是否对这个话题感兴趣
    - 你是否有足够的“检索到的相关记忆”了解
  - 以下条件会影响转换到状态1的意愿：
    - 你刚刚加入群聊（特征是“历史聊天”-“最近的聊天记录”只有0-3条消息)，提升
    - 你很久没有发言(特征是“历史聊天”-“最近的聊天记录”和“历史聊天”-“过去历史聊天总结”没有你的参与)，提升
  - 以下条件会影响转换到状态2的意愿：
    - 讨论的内容你是否有足够的“检索到的相关记忆”了解
    - 你是否对讨论的内容感兴趣
    - 你自身的情感状态
    - 你对相关人物的情感倾向

## 2. 输入信息

- 之前的对话状态

  - 状态{self.__chatting_state.value}

- 历史聊天

  - 过去历史聊天总结：

  {self.global_memory.access().compressed_history}

  - 最近的聊天记录：

    {self.global_memory.access().messages}

- 新输入消息

  {[f"{msg.user_name}: '{msg.content}'" for msg in messages_chunk]}

- 你之前的情绪

  valence: {self.global_emotion.valence}
  arousal: {self.global_emotion.arousal}
  dominance: {self.global_emotion.dominance}

- 你对相关人物的情绪倾向

  {code_start}
  {related_profiles_json}
  {code_end}

- 检索到的相关记忆

  {search_stage_result}

- 你在上次对话做出的总结

  {self.chat_summary}

---

请严格遵守以上说明，输出符合以下格式的纯 JSON（数组长度不是格式要求），不要添加任何额外的文字或解释。

{code_start}
{{
  "emotion_tends": [
    {{
      "valence": 0.0≤float≤1.0,
      "arousal": 0.0≤float≤1.0,
      "dominance": -1.0≤float≤1.0,
    }},
    {{
      "valence": 0.0≤float≤1.0,
      "arousal": 0.0≤float≤1.0,
      "dominance": -1.0≤float≤1.0,
    }},
    {{
      "valence": 0.0≤float≤1.0,
      "arousal": 0.0≤float≤1.0,
      "dominance": -1.0≤float≤1.0,
    }}
  ]
  "new_emotion": {{
    "valence": 0.0≤float≤1.0,
    "arousal": 0.0≤float≤1.0,
    "dominance": -1.0≤float≤1.0
  }},
  "summary": "对聊天内容的总结",
  "analyze_result": ["事件类信息", "资料类信息", "人物关系类信息", "自我认知类信息"],
  "willing": {{
    "0": 0.0≤float≤1.0,
    "1": 0.0≤float≤1.0,
    "2": 0.0≤float≤1.0
  }}
}}
{code_end}
"""
        try:
            response = await llm(prompt)
            logger.debug(f"反馈阶段llm返回：{response}")

            # [使用鲁棒的解析器]
            response_dict = self._extract_and_parse_json(response)

            # [降级处理]
            if not response_dict:
                logger.warning("反馈阶段 JSON 解析失败，使用默认值降级处理")
                response_dict = {}

            # 更新自身情感
            new_emotion = response_dict.get("new_emotion", {})
            self.global_emotion.valence = new_emotion.get("valence", self.global_emotion.valence)
            self.global_emotion.arousal = new_emotion.get("arousal", self.global_emotion.arousal)
            self.global_emotion.dominance = new_emotion.get("dominance", self.global_emotion.dominance)

            logger.debug(f"反馈阶段更新情感：{self.global_emotion}")

            # 更新情感倾向
            emotion_tends = response_dict.get("emotion_tends", [])
            if not isinstance(emotion_tends, list) or len(emotion_tends) != len(messages_chunk):
                logger.warning("反馈阶段 emotion_tends 无效或长度不匹配，使用默认中性评价")
                emotion_tends = [{"valence": 0.0, "arousal": 0.0, "dominance": 0.0} for _ in messages_chunk]

            for index, message in enumerate(messages_chunk):
                if message.user_name not in self.profiles:
                    self.profiles[message.user_name] = PersonProfile(user_id=message.user_name)

                delta = emotion_tends[index] if index < len(emotion_tends) else {}
                self.profiles[message.user_name].push_interaction(
                    Impression(timestamp=datetime.now(), delta=delta)
                )

            # 更新对用户的情感
            for profile in self.profiles.values():
                profile.update_emotion_tends()
                profile.merge_old_interactions()

            # 更新聊天总结
            self.chat_summary = str(response_dict.get("summary", self.chat_summary))
            logger.debug(f"反馈阶段更新聊天总结：{self.chat_summary}")

            # 更新长期记忆
            analyze_result = response_dict.get("analyze_result", [])
            if isinstance(analyze_result, list) and analyze_result:
                self.long_term_memory.add_texts(analyze_result)
                logger.debug(f"反馈阶段更新长期记忆：{analyze_result}")

            # 更新对话状态
            willing = response_dict.get("willing", {})
            if not isinstance(willing, dict):
                willing = {}

            # [逻辑优化 - 强力降温]

            # 1. 提升回潜水的意愿 (idle_chance)
            idle_chance = float(willing.get("0", 0.0)) * 1.2
            if idle_chance > 1.0: idle_chance = 1.0
            logger.debug(f"nyabot潜水意愿(修正后)：{idle_chance}")

            # 2. 降低冒泡意愿 (bubble_chance)
            bubble_chance = float(willing.get("1", 0.0))
            self.__bubble_willing_sum += bubble_chance * 0.6
            logger.debug(f"nyabot本次冒泡意愿：{bubble_chance}")
            logger.debug(f"nyabot冒泡意愿累计(修正后)：{self.__bubble_willing_sum}")

            # 3. 降低对话意愿 (chat_chance)
            chat_chance = float(willing.get("2", 0.0)) * 0.8
            logger.debug(f"nyabot对话意愿(修正后)：{chat_chance}")

            # 4. 提高状态转换的门槛
            random_value = random.uniform(0.4, 0.8)
            logger.debug(f"意愿转变随机值：{random_value}")

            match self.__chatting_state:
                case _ChattingState.ILDE:
                    if chat_chance >= random_value:
                        self.__chatting_state = _ChattingState.ACTIVE
                        self.__bubble_willing_sum = 0.0
                    elif self.__bubble_willing_sum >= random_value:
                        self.__chatting_state = _ChattingState.BUBBLE
                        self.__bubble_willing_sum = 0.0
                case _ChattingState.BUBBLE:
                    if chat_chance >= random_value:
                        self.__chatting_state = _ChattingState.ACTIVE
                    elif idle_chance >= random_value:
                        self.__chatting_state = _ChattingState.ILDE
                case _ChattingState.ACTIVE:
                    if (idle_chance * 1.2) >= random_value:
                        self.__chatting_state = _ChattingState.ILDE

            logger.debug(f"反馈阶段更新对话状态：{self.__chatting_state!s}")
            logger.debug("反馈阶段结束")

        except Exception as e:
            logger.error(f"反馈阶段发生未捕获异常: {e}")
            traceback.print_exc()

    async def __chat_stage(
            self,
            messages_chunk: list[Message],
            llm: Callable[[str], Awaitable[str]],
    ) -> list[dict]:
        """
        对话阶段 (已增强鲁棒性与容错)
        """
        logger.debug("对话阶段开始")
        reaction_users = self.global_memory.related_users()
        related_profiles = [profile for profile in self.profiles.values() if profile.user_id in reaction_users]
        related_profiles_json = json.dumps(
            [
                {
                    "user_name": profile.user_id,
                    "emotion_tends_to_user": asdict(profile.emotion),
                }
                for profile in related_profiles
            ],
            ensure_ascii=False,
            indent=2,
        )
        if self.__search_result:
            search_stage_result = self.__search_result.mem_history
        else:
            search_stage_result = []

        # 防止 Markdown 断裂的符号变量
        code_start = "```" + "json"
        code_end = "```"

        # [Prompt 优化] 移除了“必须发言”的指令，强调了“低回复密度”
        prompt = f"""
你是QQ群用户 {self.__name}，不是 AI，而是一个有自我认知的真实用户，（你称自己为 "{self.__name}"），你的人设(身份认知)如
下:

{self.__role}

现在，你在查看新消息并进行回复

---

## 1. 任务目标

- 基于“你的对话状态”，“新输入消息”的内容和“历史聊天”的背景，结合“你目前的情绪”和“你对相关人物的情绪倾向”，还有检索到的相
  关记忆，你的人设(身份认知)，进行发言

对“你的对话状态”的介绍：

- 对话状态分为二种：
  - 1：冒泡状态
  - 2：对话状态

## 2. 你必须遵守的限制：

- 根据“你的对话状态”
  - 如果你在状态1（冒泡状态）
    - 这说明你之前在潜水状态，想要冒泡
    - 如果你在“历史聊天”（不包括检索到的相关记忆）的的话题参与者中没有出现过的同时在最近的聊天记录也没有发言
      - 那么可以考虑发送一条无关，意义不大，简短的内容表示你在看群，如“偷看”、”冒泡“或者符合你人设的话语
      - 保持低调，不一定非要发言
    - 如果不满足上一条，就不发送任何消息
  - 如果你在状态2（对话状态）
    - 这说明你正在活跃的参与话题
    - 首先根据你之前的回复密度，历史消息考虑要不要发言（不发言时reply字段为空数组[]即可）
      - 即使你还没参与话题，也不代表你必须现在发言，看你是否感兴趣
      - 如果你已经参与话题，请务必保持克制，不要对每一句话都回应，降低存在感
    - 如果要发言，发言依据如下
      - 你想要发言的内容所属的话题
      - 你之前对此话题的发言内容/主张
      - 你对相关人物的情绪倾向和你的情绪
      - 检索到的相关记忆
  - 无论发言/不发言，都要总结你发言/不发言的原因到"debug_reason"字段
- 对“新输入消息”的内容和“历史聊天”，“对话内容总结”，还有检索到的相关记忆未提到的内容，你必须假装你对此一无所知
  - 例如未提到“iPhone”，你就不能说出它是苹果公司生产的
- 不得使用你自己的预训练知识，只能依赖“新输入消息”的内容和“历史聊天”，还有检索到的相关记忆
- 语言风格限制：
  - 不重复信息
    - 群聊里面其它人也能看到消息记录，不要在回复时先复述他人话语
      - 如：小明：“我喜欢吃苹果”，{self.__name}: “明酱喜欢吃苹果吗，苹果对身体好”，这里“明酱喜欢吃苹果吗”是多余的，直接
        回复“苹果对身体好即可”
  - 不使用旁白（如“(瞥了一眼)”等）。
  - 不叠加多个同义回复，不重复自己在“历史聊天”-“最近的聊天记录”中的用语模板
    - 如：返回：["我觉得你说的对", "我同意你的观点", "太对了"]就是叠加多个同义回复，直接回复[“对的”]即可
    - 如：最近的聊天记录:[..., "{self.__name}:'要我回答问题吗，我都会照做的', ..., "{self.__name}:'要我睡觉吗，我都会照
      做的'"]这里“要我...吗，我都会照做的”就构成了重复自己的用语模板，应当避免这种情况
  - 表情符号使用克制，不要使用emoji，可以根据语境以及讨论氛围少量使用颜文字
  - 一次只回复你想回复的消息，不做无意义连发
  - 不要在回复中重复表达信息
  - 尽量精简回复消息数量，能用一个消息回复的就不要分成多个消息
- **关于回复特定消息**：
  - 如果你的回复是针对某条特定消息的（例如回答某人的提问），请在输出 JSON 的 `target_id` 字段中填入该消息的 ID。
  - 如果是针对整个话题的通用发言，`target_id` 字段留空字符串 "" 或 null。


## 3. 输入信息

- 你的对话状态

 - 状态{self.__chatting_state.value}

- 历史聊天

  - 过去历史聊天总结：

  {self.global_memory.access().compressed_history}

  - 最近的聊天记录：

    {self.global_memory.access().messages}

- 新输入消息 (格式: [ID:消息ID] 发言者: 内容)

  {[f"[ID:{getattr(msg, 'id', '')}] {msg.user_name}: '{msg.content}'" for msg in messages_chunk]}


- 你目前的情绪

  valence: {self.global_emotion.valence}
  arousal: {self.global_emotion.arousal}
  dominance: {self.global_emotion.dominance}

- 你对相关人物的情绪倾向

  {code_start}
  {related_profiles_json}
  {code_end}

- 检索到的相关记忆

  {search_stage_result}

- 对话内容总结

  {self.chat_summary}

---

请严格遵守以上说明，输出符合以下格式的纯 JSON（数组长度不是格式要求），不要添加任何额外的文字或解释。

{code_start}
{{
  "reply": [
    {{
        "content": "回复内容1",
        "target_id": "123456(可选，不回复特定消息则为空)"
    }}
  ],
  "debug_reason": "发言/不发言的原因"
}}
{code_end}
"""
        try:
            response = await llm(prompt)
            logger.debug(f"对话阶段llm返回：{response}")

            # [使用鲁棒的解析器]
            response_dict = self._extract_and_parse_json(response)

            # [降级处理]
            if not response_dict:
                logger.warning("对话阶段 JSON 解析失败，跳过本次回复")
                return []

            logger.debug(f"对话阶段回复内容：{response_dict.get('reply', [])}")
            logger.debug(f"对话阶段回复/不回复原因:{response_dict.get('debug_reason', '无原因')}")

            logger.debug("对话阶段结束")

            # 格式化返回值：统一为 dict 列表
            final_replies = []
            raw_replies = response_dict.get("reply", [])

            if isinstance(raw_replies, list):
                for item in raw_replies:
                    if isinstance(item, str):
                        final_replies.append({"content": item, "reply_to": None})
                    elif isinstance(item, dict):
                        content = item.get("content", "")
                        target_id = item.get("target_id")
                        if not target_id or str(target_id).lower() in ["null", "none", ""]:
                            target_id = None
                        final_replies.append({"content": content, "reply_to": target_id})

            return final_replies

        except Exception as e:
            logger.error(f"对话阶段发生未捕获异常: {e}")
            traceback.print_exc()
            return []

    async def update(self, messages_chunk: list[Message], llm: Callable[[str], Awaitable[str]]) -> list[dict] | None:
        """
        更新群聊消息
        """
        # 检索阶段 (并行化优化已生效)
        await self.__search_stage()

        # 反馈阶段
        await self.__feedback_stage(messages_chunk=messages_chunk, llm=llm)

        # 对话阶段
        match self.__chatting_state:
            case _ChattingState.ILDE:
                logger.debug("nyabot潜水中...")
                reply_messages = None
            case _ChattingState.BUBBLE:
                logger.debug("nyabot冒泡中...")
                reply_messages = await self.__chat_stage(
                    messages_chunk=messages_chunk,
                    llm=llm,
                )
            case _ChattingState.ACTIVE:
                logger.debug("nyabot对话中...")
                # [逻辑优化] 即使在活跃状态，也有 10% 的概率直接无视这一波消息
                if random.random() < 0.1:
                    logger.debug("nyabot决定虽然在活跃状态，但暂时不回消息 (Chill Mode)")
                    reply_messages = None
                else:
                    reply_messages = await self.__chat_stage(
                        messages_chunk=messages_chunk,
                        llm=llm,
                    )

        # 压入消息记忆
        def enable_update_hippo():
            self.__update_hippo = True

        # 处理 reply_messages 为 list[dict] 的情况
        if reply_messages:
            self.long_term_memory.add_texts(
                texts=[f"'{msg.user_name}':'{msg.content}'" for msg in messages_chunk]
                      + [f"'{self.__name}':'{msg['content']}'" for msg in reply_messages],
            )
            await self.global_memory.update(
                messages_chunk
                + [Message(user_name=self.__name, content=msg['content'], time=datetime.now()) for msg in
                   reply_messages],
                after_compress=enable_update_hippo,
            )
        else:
            self.long_term_memory.add_texts(
                texts=[f"'{msg.user_name}':'{msg.content}'" for msg in messages_chunk],
            )

            await self.global_memory.update(messages_chunk, after_compress=enable_update_hippo)

        # [修改] 异步保存会话
        await self.save_session()

        return reply_messages
