# nyaturingtest-enhanced

## 项目简介

本项目基于 [nonebot-plugin-nyaturingtest](https://github.com/shadow3aaa/nonebot-plugin-nyaturingtest) 进行修改。

## 配置指南

如果你想使用本插件，建议先从[原项目](https://github.com/shadow3aaa/nonebot-plugin-nyaturingtest)安装所需的依赖。  
如果你是Windows系统，可能会出现运行报错，可以带着你的错误日志在[这里](https://gemini.google.com)或是与之类似的地方求助。  
本插件的配置参数与原项目一致，详细说明请查阅[原项目文档](https://github.com/shadow3aaa/nonebot-plugin-nyaturingtest/blob/master/README.md)。

## 功能变更

与原项目相比，本项目主要做出了以下重大更改：

1. **架构重构（Lite化）**：
   - **移除重型依赖**：移除了本地运行的 `HippoRAG` (图神经网络) 和 `BGE-M3` 模型，重构为 **ChromaDB** + **SiliconFlow Embedding API**。
   - **性能飞跃**：插件启动速度从 10秒+ 提升至 **秒级**；运行时内存占用从 4GB+ 暴降至 **200MB 左右**。
   - **实时记忆**：长期记忆不再需要退出时构建索引，而是随对话实时写入向量库，意外断电也不会丢失数据。

2. **交互逻辑增强**：
   - **强制响应机制**：新增了对 `@Bot` 和回复消息的检测逻辑。当被用户点名时，Bot 会强制打破“潜水”或“贤者时间”状态进行回应，彻底解决了“叫不答应”的问题。
   - **意愿算法调优**：优化了拟人化状态机（潜水/冒泡/活跃）的转换逻辑，大幅降低了无关话题下的插嘴频率，但在被召唤时响应更积极。

3. **稳定性与性能优化**：
   - **并发控制**：图片处理引入了信号量锁（Semaphore）和 GIF 流式读取机制，防止因接收大量图片或大尺寸动图导致内存溢出 (OOM) 或 API 频率限制 (429)。
   - **网络增强**：API 请求配置为强制直连（`trust_env=False`），有效解决了国内环境下因系统代理格式问题导致的请求失败。

4. **数据持久化**：
   - 长期记忆使用 ChromaDB (SQLite) 存储，会话状态使用 Tortoise-ORM (SQLite) 管理，替代了不稳定的 JSON 文件读写，数据更安全。

## 项目架构与模块说明

本项目采用了模块化设计，各文件职责如下：

### 1. 核心入口与调度
- **`__init__.py`**: 插件的入口文件。
    - 负责生命周期管理（启动时连接数据库，关闭时保存数据）。
    - 注册 NoneBot 的 Matchers。
- **`state_manager.py` (状态大管家)**:
    - 管理全局的 `GroupState`（群组状态）。
    - 负责资源的初始化（`ensure_group_state`）和统一清理（`cleanup_global_resources`）。
    - **关键作用**: 解决了 Session 和 Bot 实例的循环引用问题。
- **`matchers.py` (指令接收)**:
    - 定义所有的 `on_command` (如 `/reset`, `/role`) 和 `on_message`。
    - 负责接收 OneBot 事件，并将其通过 `logic` 层传递给后台。

### 2. 业务逻辑层
- **`logic.py` (业务大脑)**:
    - **`spawn_state`**: 后台无限循环任务，负责定时从 Session 获取回复并发送。包含抗风控的随机延迟逻辑。
    - **`message2BotMessage`**: 将复杂的 QQ 消息（图片、表情包、@、回复）转译为 LLM 能读懂的纯文本格式。
    - **并发控制**: 对图片下载和识别加入了信号量锁，防止瞬间高并发。
- **`session.py` (会话核心)**:
    - 维护单个群聊的完整上下文（Session 类）。
    - **核心调度**: 串联感知 (`feedback`)、记忆检索 (`search`) 和表达 (`chat`) 三个阶段。
    - **状态机**: 维护 潜水/冒泡/活跃 三种状态，并包含针对被 @ 的强制状态修正逻辑。

### 3. 记忆与数据层
- **`vector_mem.py` (长期记忆)**:
    - 封装 `ChromaDB` 和 `SiliconFlowEmbeddingFunction`。
    - 实现了基于 MD5 的内容去重（Upsert）和 API 调用的容错处理。
- **`mem.py` (短时记忆)**:
    - 管理最近的聊天记录窗口（滑动窗口机制）。
    - 负责调用小模型对过往历史进行压缩摘要。
- **`profile.py` & `impression.py` (用户画像)**:
    - 维护 Bot 对每个群友的 VAD 情感模型（愉悦度、唤醒度、支配度）。
    - 包含情感随时间衰减的数学逻辑。
- **`models.py`**: 定义 `Tortoise-ORM` 的数据库表结构（SQLite），确保持久化存储。

### 4. 基础设施与工具
- **`client.py`**: 封装 OpenAI 格式的 API 调用，支持超时重试。
- **`vlm.py` & `image_manager.py`**: 
    - 处理图片理解。`image_manager` 负责缓存图片描述（避免重复消耗 Token），`vlm.py` 负责调用硅基流动的视觉大模型。
    - 优化了 GIF 处理逻辑，改为流式读取，防止内存溢出 (OOM)。
- **`prompts.py`**: 集中管理所有复杂的 System Prompt，包含 JSON 格式约束和 Few-Shot 示例。
- **`utils.py`**: 提供全局 HTTP 客户端、智能断句、JSON 提取等通用工具函数。
- **`config.py`**: 管理从 `.env` 读取的配置项。

## 注意事项

~~由于该插件的运作逻辑，导致token消耗会**非常非常大**（在一个一天发2k条消息的群聊中使用，一天总共约消耗30Mtoken），使用请注意。~~  
~~也许在之后会优化一下token使用。~~  
更改了feedback逻辑，以轻微降低智能的代价大幅降低了token使用。  
~~更改了长期记忆索引的逻辑，大幅降低了token使用。~~  
~~目前的token用量大约在先前的20%左右，即处理大概2k条消息总共消耗6-7Mtoken。~~  
更改了长期记忆的逻辑，极大程度地降低了token使用。现在token的使用主要在对话阶段以及反馈阶段，记忆阶段几乎不消耗。  
作为参考，目前处理2k条消息，大概消耗在1Mtoken左右。  
建议：.env文件中配置的对话用大模型用优质一点的（作为参考，我用的是DeepSeek-V3.2），会更加智能。  

## Special Thanks
[G指导](https://gemini.google.com)：帮我完成了代码的修改以及这篇readme的撰写。
