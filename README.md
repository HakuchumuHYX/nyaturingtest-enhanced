# nyaturingtest

面向 QQ 群聊的 NoneBot2 / OneBot V11 自动聊天插件，基于 [shadow3aaa](https://github.com/shadow3aaa/) 的
[`nonebot-plugin-nyaturingtest`](https://github.com/shadow3aaa/nonebot-plugin-nyaturingtest) 二次开发，
模型推理、Embedding 与 Rerank 全部走云端 API。

插件以**群**为单位维护独立会话：每个启用群有自己的短时上下文、长期向量记忆、角色设定、用户画像、
情绪状态、Token 统计与后台任务。

## 功能

- **自动群聊插话**：启用群后监听群消息，按静默窗口批量取消息，自行判断要不要说话。
- **拟人化状态**：潜水 / 冒泡 / 对话三态，由意愿值与发言冷却驱动；意愿随真实流逝时间衰减，
  被点名或命中别名时强制拉高。
- **被动固化**：长时间不参与对话时仍会周期性沉淀记忆（情绪、画像、摘要、长期记忆），但不产生回复。
- **双阶段 LLM**：Feedback（观察者）负责情绪、画像、摘要与记忆提取；Chat（角色）负责生成最终回复。
- **长期记忆**：每群独立的 ChromaDB 向量库，Embedding 召回 + Rerank 重排 + 时间衰减排序，
  支持相似度去重、事实更新（supersede）与失败写入的 WAL 重放。
- **原生多模态**：图片下载压缩后作为 `image_url` 随请求附带，Feedback 额外返回一句话图片观察写回消息文本。
- **Token 统计**：记录 prompt / completion / reasoning / DeepSeek cache hit-miss，按模型归并后渲染成卡片。
- **SQLite 持久化与每日备份**：会话、消息、画像、启用的群与 Token 明细入库；数据目录每天自动打包。

## 运行环境

插件依赖宿主 NoneBot 项目提供运行环境。本项目根目录 `pyproject.toml` 声明的关键依赖：

- Python `>=3.10`
- `nonebot2[fastapi]`、`nonebot-adapter-onebot`、`nonebot-plugin-apscheduler`
- `tortoise-orm`、`openai`、`httpx`、`json-repair`
- `chromadb`、`pillow`、`chinese-calendar`

Token 统计卡片复用同级插件目录的公共绘图库 `plugins/utils/draw/plot.py`，
运行时把 `plugins/` 插入 `sys.path` 后导入，不在依赖中声明。

## 目录结构

```text
plugins/nyaturingtest/
├── __init__.py          生命周期入口：连库 → 建表 → 载入启用群 → 注册定时任务
├── config.py            config.json 加载、AppSettings、工作区路径常量
├── handlers.py          全部 16 个 matcher：管理命令、自动消息入口、/查询记忆、/rag_debug
├── models.py            7 个 Tortoise ORM 模型
├── db.py                全部数据库读写（会话/消息/画像/交互/群开关/Token）
├── domain.py            EmotionState / Impression / PersonProfile（纯领域对象）
├── token_stats.py       Token 聚合与模型名归并 + 统计卡片 PNG 渲染
├── backup.py            备份、保留期清理、定时任务注册
├── core/
│   ├── state_manager.py 每群 GroupState、worker 守护、资源清理
│   ├── logic.py         消息入队、防抖取批、OneBot 消息转换、回复切分与发送
│   ├── orchestrator.py  一轮对话的编排：search / feedback / consolidate / chat 四个 stage
│   ├── session.py       SessionState + SessionRuntime、持久化协调、记忆读写
│   ├── engagement.py    意愿值衰减与增长、参与态滞回、冷却、相关性/兴趣打分
│   ├── llm.py           LLMClient（重试/熔断）、全局 HTTP 池、system prompt、JSON 解析
│   ├── prompts.py       Feedback / Chat 提示词模板、PromptBudget、角色预设加载、时间描述
│   ├── metrics.py       结构化事件日志、运行时计数、Token 落库任务
│   └── memory_query.py  /查询记忆：动态 k、冷却与单飞、印象生成
└── memory/
    ├── short_term.py    短时消息窗口与摘要载体（含增量落库标记）
    ├── vector.py        VectorMemory（ChromaDB）、Embedding/Rerank 客户端、RAG 检索
    ├── image.py         图片下载/缓存/压缩 → 原生多模态输入
    └── validation.py    长期记忆候选的确定性校验
```

依赖方向基本单向：`handlers.py` 在顶端（只被 `__init__.py` 导入）→ `core/` → `memory/` → `models.py` / `config.py`；
`db.py` 与 `token_stats.py` 同层，前者复用后者的 `TOKEN_FIELDS` 与模型名归并函数。
两处刻意的例外：

- `memory/vector.py` 持有全局 `BACKUP_IO_LOCK`（`threading.RLock`）。备份要打包整个向量目录，
  必须和向量写入用**同一把**进程级锁；锁定义在向量侧、由 `backup.py` 反向导入，避免循环依赖。
- `core/orchestrator.py` 从 `core/session.py` 单向导入 `ChattingState` / `FeedbackOutcome`；
  `core/state_manager.py` 内部局部导入 `logic.spawn_state`（`logic` 反向引用 `GroupState`）。

## 一轮对话怎么走

```text
群消息 → handlers.handle_auto_chat（on_message, priority=99, block=False）
       → 队列 state.messages_chunk（logic.QUEUE_MAX_SIZE=200，满了丢低优先级）
       → logic.spawn_state 后台循环：等静默 → 防抖 2s → 整批取走
       → logic._process_inbox_batch
            ├─ 按 local self-sent id 过滤自身回显
            ├─ core.llm.build_turn_calls 生成本轮 chat / feedback 两个调用闭包
            └─ core.orchestrator.ConversationOrchestrator.process_chunk
       → logic.dispatch_replies：按句切分、最多 2 条、带拟人延迟发送
```

`process_chunk` 的顺序固定，每一步之间都有代际检查：

1. `session.record_incoming` 写入短时记忆，累计固化窗口。
2. `engagement.evaluate_engagement`：意愿值按真实流逝时间衰减 → 强关联（@Bot / 回复 Bot / 命中名字别名）
   直接把意愿拉到 `RELEVANCE_WILLINGNESS_FLOOR` → 否则按内容兴趣被动增长 → 更新参与态滞回与发言冷却。
3. 不参与且不相关时：满足固化条件（`messages_since_consolidation >= 8`，或距上次尝试 180s）
   就走 `consolidate_stage`，静默沉淀记忆，**不产生回复**。
4. `search_stage`：构造 RAG query → 检索长期记忆 → 按字符预算拼成 `prompt_lines`，
   统计写进 `rag_search` 事件日志（`RAG_DEBUG_LOG=True` 时附每条记录的分数明细）。
5. `feedback_stage`（观察者模型，temperature 0.1）：解析 JSON → 应用图片观察 → 沉淀 → 发言决策。
   - `_apply_image_observations`：把 Feedback 对图片的一句话观察写回消息文本（`[图片: …]` / `[表情包: …]`），
     让历史里保留图片线索。
   - `_apply_sediment`：更新全局 VAD 情绪（`clamp_vad_value` 限幅）→ 逐条消息更新用户印象
     （峰值保持 + 时间衰减）→ 更新话题摘要 → 把 `analyze_result` 交给后台任务写长期记忆。
   - `_apply_decision`：`need_history` 为真时按时间回溯最多 20 条更早的历史消息 → 更新意愿值
     （相关时兜底抬到 `RELEVANCE_WILLINGNESS_FLOOR`）→ 潜水/冒泡/对话三态流转。
6. 意愿仍低于 `POST_FEEDBACK_SKIP_THRESHOLD` 且不相关 → 放弃本轮回复。
7. `chat_stage`（角色模型，temperature 0.7）生成回复；有回复则意愿乘以 0.35、状态置为「对话状态」。

**代际控制**：`Session.bump_generation()` 在 `set_role` / `load_preset` / `reset` / `reset_emotion` /
`calm_down` 时自增。每轮开始时记下 `generation`，所有写入路径（短时记忆、Feedback 沉淀、长期记忆、发送）
都用 `is_generation_stale()` 检查；过期就丢弃并记一条 `stale_turn_discarded` 事件。

## 记忆体系

### 短时记忆（`memory/short_term.py`）

- `deque(maxlen=200)` 滚动缓冲，`Memory.access()` 只返回最近 `SHORT_CONTEXT_LIMIT=20` 条 + 当前摘要。
- 每条 `Message` 带 `revision`；`mark_dirty` 把它放进待落库字典，`_save_session_locked` 只同步
  新增或被图片观察改写过（revision 变化）的消息，避免高频全量写库。
- `image_inputs`（原生图片负载）只在进程内短期持有，不序列化、不落库。

### 长期记忆（`memory/vector.py`）

每个群一个 ChromaDB 持久化目录 `data/nyaturingtest/vector_index_<群号>/`，
集合固定叫 `nyabot_memory`，度量 `cosine`。群之间不共享记忆。

单条记录的 metadata：

| 字段 | 含义 |
| --- | --- |
| `source` | `memory`（长期记忆）或 `preset`（角色预设写死的设定） |
| `type` / `subtype` / `category` | 记忆类别；只允许 `event`、`preference`、`profile`、`relationship` |
| `status` | `active` / `pending_supersede` / `superseded`，检索时只认 `active` |
| `subject_user_id` / `subject_user_name` | 事实描述的对象（「B 说 A 的事」填 A） |
| `speaker_user_id` / `speaker_user_name` | 说出该事实的人（填 B） |
| `confidence` / `importance` | 影响去重、衰减与排序权重 |
| `date` / `ttl_days` | `YYYYMMDD` 整数与事件存活天数（默认 90） |
| `schema_version` | 当前为 2 |

写入路径：

- **去重**：`add_memories_with_dedup` 先按 `(内容, 去重域)` 在批内去重，再按域分组做一次
  `collection.query(n_results=5)`；相似度 `1 - distance > 0.9` 视为重复 → 跳过并「强化」已有记录
  （confidence 向 1.0 靠近 20%、更新 date、`reaffirm_count + 1`），返回
  `added / skipped_empty / skipped_dedup / reinforced / dedup_errors` 计数。
- **取代**：Feedback 返回 `supersede` 时必须命中本轮检索给出的 `memory_ref`（否则记
  `rag_action_hallucination` 事件并丢弃），且目标是 `source=memory` 且非 `bot_self`。
  新记录先以 `pending_supersede` 写入，旧记录改 `superseded`，最后把新记录翻成 `active`。
- **WAL**：写失败时把操作追加到 `pending_memories.jsonl`（`add` / `supersede` 两种），
  启动时 `replay_pending()` 重放；成功的行被丢弃，仍失败的写回文件。
  记录 id 由 `operation_id` 经 `uuid5` 推导，重复写入是幂等的。
- **清理**：每天 03:30 的 `cleanup(days_retention=90)` 删除超过保留期的 `superseded` 记录，
  以及 `type=event` 且超过 `ttl_days * (1 + importance)` 的记录；预设记忆不参与清理。

检索路径（`search_memories` → `retrieve_with_decay`，跑在线程池里）：

1. `build_chat_rag_queries` 过滤低价值 query（纯标点、`[表情包]`、长度 < 4、纯 emoji 等），
   追加话题摘要与「关于<活跃用户名>」，再去重。
2. `retrieve` 一次多 query 召回，按 `1 - distance` 取最大分融合，超过 `RAG_MERGED_CANDIDATE_CAP=64`
   截断，再用 Reranker 以第一条 query 全量重排，低于 `rerank.threshold` 的丢弃。
3. `_retrieve_active_subject_records` 额外按 `subject_user_id` 做一次结构化召回
   （最多 5 条，按 importance、date 排序），补上语义检索漏掉的「活跃主体」记忆。
4. 打分并排序：

   ```text
   adjusted_score = 原始分(rerank_score 优先，否则 retrieval_score)
                  × 时间衰减 exp(-decay_rate × days_ago)
                  × 来源/类型权重 × (0.7 + 0.3 × confidence)
                  × (1 + 0.15 × importance) × 主体作用域权重
   ```

   事件半衰期约 35 天（rate 0.02），偏好/画像/关系几乎不衰减（0.003），预设恒定。
   作用域权重：活跃主体 1.10、被提到的主体 1.08、活跃发言者 1.04、其他主体 0.5。
   缺 `date` 的记忆按 60 天前处理。

### 图片（`memory/image.py`）

下载（content-type 白名单、8MB 上限、2 次重试）→ 按 `fileid`/`file_unique` 缓存原始字节到
`cache/nyaturingtest/image_cache/raw/`（48 小时过期，每天 03:00 清理）→ 压缩到最大边 1280、
像素上限 4096²，PNG 保 PNG、GIF 动图原样透传、其余转 JPEG q90 → base64 成 `VisionInput`。
全局信号量限制 3 个并发。

### 记忆候选校验（`memory/validation.py`）

只做确定性过滤：长度 ≥ 10、不在噪声词表、类别在白名单内、confidence ≥ 0.6、主体非空。
「是不是玩笑 / 有没有注入指令」交给 Feedback 的 prompt 与 confidence 判断，代码里不写启发式规则表。

## 存储层

SQLite 表（`models.py`，启动时由 `Tortoise.generate_schemas()` 建表）：

| 表 | 内容 |
| --- | --- |
| `nyabot_sessions` | 每群一条：人设、别名、VAD 情绪、摘要、最后发言时间、固化水位、聊天状态 |
| `nyabot_user_profiles` | 群内用户画像：VAD、交互次数、首次/最近交互时间 |
| `nyabot_interactions` | 每次互动的情感增量明细 |
| `nyabot_global_messages` | 消息明细（`(session, msg_id)` 唯一，含时间与会话索引） |
| `nyabot_enabled_groups` | 启用的群号，**唯一来源** |
| `nyabot_token_usage` | Token 明细：prompt / completion / cache hit / cache miss / reasoning |
| `nyabot_daily_token_usage` | 按 `(day, session, model)` 聚合的日汇总，卡片统计只读这张表 |

`db.py` 是唯一的数据库访问层，除同步工具 `sanitize_text` 外全是模块级 async 函数。
三条失败语义上的例外值得记住：

- `log_token_usages` / `get_token_stats` 只记日志不抛异常——旁路统计失败不该打断回复。
- `log_interactions` 在会话不存在时静默返回。
- `sync_messages` / `update_user_profiles` 在会话不存在时抛 `RuntimeError`。

`domain.py` 里的 `PersonProfile` 用**峰值保持 + 时间衰减**更新印象：同向取绝对值更大者，异向相加，
valence 正向半衰期约 14 小时、负向约 5 小时，dominance 约 23 小时，arousal 以 5 小时量级的时间常数
回落到 0.3。每次 `push_interaction` 前先结算衰减，`merge_old_interactions` 丢弃 5 小时前的交互记录。
这些是内存态计算，落库只写 VAD 与交互计数。

## 配置

配置文件：`plugins/nyaturingtest/config.json`（已被 `.gitignore` 忽略），模板见 `config.example.json`。
文件不存在时使用内置默认值并在日志中告警。

| 段 | 字段 |
| --- | --- |
| `chat` | `api_key` `base_url` `model` `reasoning_effort` `max_tokens` `timeout` |
| `feedback` | 同上（默认 `reasoning_effort` 为空、`max_tokens` 2048） |
| `siliconflow_api_key` | 长期记忆的 Embedding / Rerank 共用 |
| `embedding` | `model`（默认 `BAAI/bge-m3`）`base_url` `timeout` |
| `rerank` | `model` `base_url` `timeout` `threshold`（低于该分的结果丢弃） |

两个环境变量：

- `NYATURINGTEST_CONFIG_FILE`：指定其它配置文件路径。
- `NYATURINGTEST_DATA_DIR`：运行数据目录（独立部署用）；相对路径按工作区根解析。

**改配置需要重启**：插件不监听文件变更，`AppSettings` 与 LLM / Embedding / Rerank 客户端都在启动期构造。
`/autochat enable|disable` 直接写数据库并立即生效，不需要重启。

运行时策略参数（意愿阈值、RAG 预算、保留期等）不是配置项，见「策略常量在哪」。

## 命令

前缀取决于宿主 NoneBot 的 `command_start` 配置。除 `/查询记忆` 外，群聊命令均需 SUPERUSER。

| 命令 | 别名 | 说明 |
| --- | --- | --- |
| `/help` | `/帮助` | 查看帮助（群聊/私聊各一套文案） |
| `/autochat enable` | - | 在本群启用，写库并立即初始化状态 |
| `/autochat disable` | - | 在本群禁用，取消后台任务并释放资源 |
| `/status` | `/状态` | 会话状态 + reasoning_effort + 队列长度 + LLM 计数 + provider 熔断/错误 + 配置加载状态 |
| `/role` | `/当前角色` | 查看当前角色 |
| `/set_role <角色名> <角色设定>` | `/设置角色` | 修改角色，设定可含空格 |
| `/presets` | `/preset` | 列出可用预设 |
| `/set_preset <文件名>` | `/set_presets` | 加载预设，可省略 `.json` |
| `/rag_debug <query>` | `/记忆诊断` | 打印检索候选数、回退原因与 top 5 记录的分数明细 |
| `/calm` | `/冷静` | 重置情绪与画像、意愿归零、回到潜水态 |
| `/reset_emotion` | `/重置情绪` | 只重置 VAD 情绪 |
| `/reset confirm` | `/重置 confirm` | 先备份，再完全重置本群 |
| `/token统计 [all]` | `/autochat token统计` | 图片卡片；`all`/`全部`/`历史` 改为统计全部历史模型 |
| `/backup_data` | `/备份数据` | 手动触发一次备份 |
| `/查询记忆 [@用户]` | `/memory` | **普通群员可用**：生成 Bot 对目标用户的印象档案 |

私聊支持 `help`、`list_groups`、`backup_data` 与 8 个通用管理命令；通用命令在私聊下要把群号
作为第一个参数（`/status 123456`、`/reset 123456 confirm` …），群号非数字时直接提示。
`/autochat`、`/token统计`、`/rag_debug`、`/查询记忆` 仅限群聊。
群聊与私聊共用同一个 matcher（`handlers._dual_command`），避免 nonebot 的「Duplicated prefix rule」告警。

重置范围：

| 操作 | 情绪 | 用户画像 | 意愿/状态 | 短时记忆 | 长期向量记忆 | 数据库行 | 人设 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `/calm` | 重置 | 内存清空（库行保留） | 重置 | 保留 | 保留 | 保留 | 保留 |
| `/reset_emotion` | 重置 | 仅重置情绪 | 保留 | 保留 | 保留 | 保留 | 保留 |
| `/reset confirm` | 重置 | 清空 | 重置 | 清空 | 清空 | 删除 | 恢复默认 |

`/reset confirm` 会先 `bump_generation` 作废进行中的 turn，在 `session_lock` 之外做备份
（打包整个数据目录可能很久），备份失败则中止重置。这个操作不可逆。

## 角色预设

预设目录：`config/nyaturingtest/nya_presets/*.json`（相对工作区根）。`/set_preset` 每次执行都会
重新扫描目录，新增或修改文件后无需重启。内置一个「喵喵」预设兜底。

字段对应 `core/prompts.RolePreset`：`name`、`role`、`aliases`（强关联检测用）、
`knowledges` / `relationships` / `events` / `bot_self`（加载时写入向量库，`source=preset`）、
`examples`（对话样本，拼进 role 文本）、`hidden`（是否从 `/presets` 隐藏）。
加载预设会先删掉本群所有 `source=preset` 的向量记录再重写。

## 数据、备份与定时任务

工作区根 = `HakuBot-autochat/`（`config.WORKSPACE_ROOT`）。默认路径：

```text
data/nyaturingtest/                     nyabot.sqlite（主库）、vector_index_<群号>/、三个渲染字体
cache/nyaturingtest/image_cache/raw/   图片原始字节缓存，48 小时过期
data/nyaturingtest_backups/            nyabot_backup_YYYYMMDD_HHMMSS.zip
config/nyaturingtest/nya_presets/      角色预设
```

定时任务（`nonebot_plugin_apscheduler`，`misfire_grace_time=3600`）：

| 时间 | 任务 | 内容 |
| --- | --- | --- |
| 03:00 | 图片缓存清理 | 删除超过 48 小时的缓存原图 |
| 03:30 | 向量记忆维护 | 对每个已加载的群执行 `cleanup(days_retention=90)` |
| 04:00 | 数据备份 | 备份 + 原始行保留期清理 |

备份行为：走 `sqlite3` 的 backup API 取一致性快照，`shutil.copy2` 复制数据目录其余内容到临时目录，
再打成一个 zip。**排除** `nyabot.sqlite` / `-wal` / `-shm`、`__pycache__` 与 `.ttf/.otf/.ttc` 字体
（每包省约 24MB，恢复后字体需在数据目录中才能渲染卡片）。备份目录是数据目录的兄弟目录，不会自我嵌套。
保留最近 `7` 个（按 mtime）。整个「复制 + 打包」过程持有 `memory.vector.BACKUP_IO_LOCK`，
与向量写入互斥。

每次成功备份后执行保留期清理：消息明细 180 天、画像交互 180 天、Token 明细 90 天。
**向量记忆与日聚合表不参与清理**，所以 `/token统计 all` 的历史在明细行被删后依然存在。

备份包含聊天内容、用户 ID、画像与长期记忆，属敏感数据：请加密保存并定期离机复制。
三个字体文件（`SourceHanSansCN-{Regular,Bold,Heavy}.ttf`）需放在数据目录下。

## 策略常量在哪

运行时策略参数写在使用它们的模块里，调整后重载插件即可：

| 模块 | 常量的作用 |
| --- | --- |
| `core/engagement.py` | 意愿衰减率（活跃 0.04 / 闲置 0.08 每分钟）、强关联下限 0.85、参与阈值 0.47、被动增长上限 0.6 与每消息 0.026、发言冷却 16s、跳过阈值 0.32 / 0.38、启用 Rerank 的意愿阈值 0.68 |
| `core/orchestrator.py` | 固化开关与条件（消息数 8、间隔 180s、最多 60 条）、历史回溯条数 20 |
| `memory/vector.py` | `RAG_FINAL_K=20`、每 query 召回 40、合并候选上限 64、注入字符预算 1500、事件 TTL 90 天、类型权重/衰减率/作用域权重表 |
| `core/prompts.py` | `PromptBudget`：摘要 1200、最近消息 1600、历史 2400、RAG 合计 1500 / 单条 500、回溯历史 1200 字 |
| `memory/short_term.py` | 上下文窗口 20 条、缓冲上限 200 条 |
| `core/logic.py` | 防抖 2s、队列上限 200、单轮最多发 2 条、拟人延迟 1.0 + 0.1×字数（封顶 5s） |
| `core/session.py` | role 4000 字 / examples 2000 字上限、后台任务排空超时 10s |
| `core/memory_query.py` | VAD 缓存 256 条 / 24 小时、动态 k 的计算规则 |
| `handlers.py` | `/查询记忆` 冷却：同用户 30s、同群 3s |
| `memory/image.py` | 8MB / 4096² 像素上限、最大边 1280、并发 3、缓存 48 小时 |
| `memory/validation.py` | 允许的记忆类别、最低置信度 0.6、最短 10 字、噪声词表 |
| `backup.py` | 备份保留 7 个、消息/交互 180 天、Token 明细 90 天 |
| `core/llm.py` | 重试 3 次、退避基数 2s、429 熔断 30s、chat/feedback 的 system prompt |

## 开发与验证

项目不写单元测试，不为测试引入依赖注入接口、mock 接缝或测试分支。验证方式是：

```bash
cd /opt/HakuBot-autochat
./.venv/bin/python -m compileall -q plugins/nyaturingtest                        # 语法
./.venv/bin/python -c "import nonebot; nonebot.init(); nonebot.load_plugin('plugins.nyaturingtest')"   # 加载与 matcher 注册
```

然后在真实群里走一轮完整对话，并检查 `/status`、`/token统计`、`/rag_debug`。
改代码后需重启 `bot.py`：运行中的实例不会自动加载新代码。

约定：

- 设计文档统一写在 `HakuBot-autochat/docs/`。
- 改存储结构时，数据变更写成一次性脚本手工执行，不挂到启动路径。
- 新字段直接改契约并同步所有调用方，不加兼容别名或迁移分支。

## 致谢

- 原项目作者 [shadow3aaa](https://github.com/shadow3aaa/)。
