# nyaturingtest-enhanced

## 项目简介

本项目是基于 [nonebot-plugin-nyaturingtest](https://github.com/shadow3aaa/nonebot-plugin-nyaturingtest) 的改造版本。

核心目标是将原项目从“重本地算力”转向“重云端API”，在保留核心拟人化交互体验的同时，极大地降低部署门槛和资源消耗。

## ✨ 核心特性

### 1. 轻量化架构 (Lite Architecture)

- **去本地化模型**：移除了原项目中庞大的 `HippoRAG` (图神经网络) 和本地 `BGE-M3` Embedding 模型。
- **云端算力替代**：
  - **Embedding**: 对接 SiliconFlow (硅基流动) 的 Embedding API，速度更快，效果更强。
  - **VLM**: 图片理解模块对接 Qwen-VL (via SiliconFlow)，无需本地显存即可实现高精度识图。
  - **Rerank (重排序)**: 引入 Reranker 模型优化 RAG 检索链路，大幅提升长期记忆召回的准确性。
- **资源占用骤降**：运行时内存占用从 4GB+ 降低至 **200MB 左右**，启动速度提升至秒级。

### 2. 增强的交互逻辑

- **自主意识循环**：采用 `Producer-Consumer` 模型的后台思考循环 (`spawn_state`)，Bot 会根据群聊上下文自主决定是否发言，而非传统的“一问一答”。
- **强制响应机制**：解决了“叫不答应”的问题。当检测到 `@Bot` 或 `回复Bot消息` 时，会自动打破“潜水”状态，强制触发响应逻辑。
- **拟人化状态机**：维护 `潜水` / `冒泡` / `活跃` 三种状态，根据群聊热度动态调整插话频率。
- **多模态感知**: 能够通过 VLM 模型“看见”群聊图片，并将图片内容转化为文本纳入对话上下文，实现真正的“看图说话”。

### 3. 稳定可靠的数据持久化

- **数据库迁移**：从不稳定的 JSON 文件读写全面迁移至 **SQLite + Tortoise-ORM**。
- **实时记忆**：
  - **长期记忆**: 使用 **ChromaDB** 存储向量化记忆，支持实时写入，意外断电不丢数据。
  - **短时记忆**: 采用滑动窗口 + LLM 实时摘要 (`Summary`) 机制，自动压缩历史对话。
- **配置持久化**: 群组的 Autochat 开启/关闭状态现已存入数据库，重启后不再丢失。

### 4. 运维与监控

- **Token 智能追踪**: 内置了精确的 Token 消耗统计系统。
  - **全链路覆盖**: 包含 Chat (对话)、VLM (识图)、Feedback (反思) 三大核心环节。
  - **多维度报表**: 支持查看 24小时、7天以及历史总计的消耗情况，并按模型细分。
- **并发控制**：引入 `asyncio.Semaphore` 限制图片处理并发数，防止瞬间大量图片导致内存溢出。
- **网络优化**：API 请求强制直连 (`trust_env=False`)，并在底层配置了 `httpx` 连接池。

## 🛠️ 配置指南

在使用前，请确保你的 `.env` 文件中包含以下配置项：

```dotenv
# --- LLM 对话配置 ---
nyaturingtest_chat_openai_api_key=sk-xxxxxxxx
nyaturingtest_chat_openai_model=Qwen/Qwen3-32B
nyaturingtest_chat_openai_base_url=https://api.siliconflow.cn/v1

# --- 记忆与反馈配置 ---
# 用于记忆压缩和自我反馈的小模型 (建议使用廉价模型)
nyaturingtest_feedback_openai_model=Qwen/Qwen2.5-7B-Instruct

# --- RAG (检索增强生成) 配置 ---
# 重排序模型，用于提升记忆检索的相关性
nyaturingtest_rerank_model=BAAI/bge-reranker-v2-m3
# 重排序阈值，低于此分数的记忆片段将被过滤 (0-1)
nyaturingtest_rerank_threshold=0.05

# --- 硅基流动服务配置 ---
# 用于 Embedding, Rerank 和 VLM (视觉理解)
# 如果未单独设置，默认复用 chat_openai_api_key
nyaturingtest_siliconflow_api_key=sk-xxxxxx

# --- 启用范围 ---
# 填写允许 Bot 初始运行的群组 QQ 号列表 (后续可通过指令动态增删)
nyaturingtest_enabled_groups=[123456789, 987654321]
```

### 指令、架构与说明

---

## 🎮 指令列表

所有指令仅支持 **SUPERUSER** 使用。

### 群聊指令

| 指令                         | 别名                 | 说明                                                      |
| :--------------------------- | :------------------- | :-------------------------------------------------------- |
| `/help`                      | 帮助                 | 显示帮助信息                                              |
| `/autochat <enable/disable>` | -                    | **[新增]** 在本群启用或禁用 AI 自动插话 (持久化保存)      |
| `/token统计`                 | token使用 / 额度查询 | **[新增]** 查看本群及全局的 Token 消耗统计 (24h/7d/Total) |
| `/查询记忆 <@某人/空>`       | memory / 印象        | **[新增]** 回溯 AI 对特定群友的长期记忆与印象评价         |
| `/status`                    | 状态                 | 查看当前群组的 Bot 状态（记忆数、情感值等）               |
| `/role`                      | 当前角色             | 查看当前加载的角色设定                                    |
| `/set_role <角色名> <设定>`  | 设置角色             | 动态修改 Bot 的人设（支持实时生效）                       |
| `/presets`                   | preset               | 查看所有可用的预设文件                                    |
| `/set_preset <文件名>`       | 加载预设             | 从文件加载角色预设                                        |
| `/calm`                      | 冷静                 | 强制降低 Bot 的活跃度（进入贤者时间）                     |
| `/reset`                     | 重置                 | 清空当前群组的短期记忆和会话状态                          |

### 私聊指令

私聊指令通常需要指定群号，格式为：`指令 <群号> [参数]`。
例如：`status 123456789`

## 📂 项目架构

为了方便二次开发，简要说明各模块职责：

### 1. 核心调度层

- **`matchers.py`**: 消息接收入口。处理 OneBot 事件，过滤群组，将消息推送到后台缓冲区。
- **`state_manager.py`**: 全局状态容器。管理所有群组的 `GroupState`，负责资源（数据库、HTTP客户端）的生命周期管理。
- **`logic.py`**: 业务大脑。包含 `spawn_state` 无限循环，负责从队列取消息 -> 组装 Prompt -> 调用 LLM -> 发送消息。

### 2. 记忆系统

- **`mem.py`**: 短时记忆管理。实现了一个带有“压缩缓冲区”的滑动窗口，当积压消息达到阈值时，自动调用小模型生成摘要。
- **`vector_mem.py`**: 长期记忆管理。封装 ChromaDB 操作，实现“Embedding初筛 -> Rerank精排”的两阶段检索流程。
- **`memory_query.py`**: 记忆查询模块。负责处理用户查询记忆的指令，综合情感数据、长期记忆和短期交互生成印象报告。

### 3. 感知与表达

- **`vlm.py`**: 视觉中心。封装 OpenAI 格式的 Vision API，支持 GIF 流式读取和图片描述缓存。
- **`client.py`**: LLM 客户端。封装了带有重试机制（Retrying）的 `AsyncOpenAI` 调用。
- **`emotion.py` / `profile.py`**: 情感模型。维护基于 VAD（愉悦-唤醒-支配）理论的群友印象系统。
- **`repository.py`**: 数据仓库。统一处理 Token 统计日志、会话数据、群组开关状态的 SQLite 存取。

## ⚠️ 注意事项

1. **Token 消耗**：
   - 虽然通过“记忆摘要”和“长期记忆检索”优化了 Prompt 长度，但由于 Bot 需要不断读取群聊上下文进行“自我思考”（Feedback 阶段），Token 消耗依然可观。
   - **建议**：使用 `/token统计` 指令定期监控消耗情况。

2. **数据库文件**：
   - 运行时会在插件数据目录生成 `nyabot.sqlite`。请定期备份此文件以防数据丢失。
   - 长期记忆向量库由 ChromaDB 管理，通常位于本地文件夹。

## Special Thanks

- **原作者**: [shadow3aaa](https://github.com/shadow3aaa/) 提供的前沿架构思路。
- [**G指导**](https://gemini.google.com/app): 协助完成了代码的重构、Bug 修复以及文档编写。
