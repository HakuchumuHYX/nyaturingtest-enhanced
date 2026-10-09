# nyaturingtest

面向 QQ 群聊的 NoneBot2 / OneBot V11 自动聊天插件，基于 [shadow3aaa](https://github.com/shadow3aaa/) 的
[`nonebot-plugin-nyaturingtest`](https://github.com/shadow3aaa/nonebot-plugin-nyaturingtest) 二次开发，
模型推理、Embedding 与 Rerank 全部走云端 API。

插件以**群**为单位维护独立会话：每个启用群有自己的短时上下文、长期记忆、角色设定、用户画像、
情绪状态、Token 统计与后台任务。

## 功能

- **自动群聊插话**：启用群后监听群消息，按静默窗口批量取消息，自行判断要不要说话。
- **拟人化参与**：规则意愿值随时间衰减、随群聊热度增长，决定要不要考虑说话；Feedback 的接话意愿按场景门槛
  决定说不说。被点名（名字/别名、@、回复 Bot）必回；Bot 接话后 3 分钟内，刚才的聊天对象再开口会优先接住，
  聊天不会因为说完一句就断掉；自己最近发言占比超过 20% 时不再主动插嘴，避免刷屏。
  潜水 / 冒泡 / 对话三态由意愿值与对话窗口派生，只用于展示和 prompt。
- **被动固化**：长时间不参与对话时仍会周期性沉淀记忆（情绪、画像、摘要、长期记忆），但不产生回复。
- **双阶段 LLM**：Feedback（观察者）负责情绪、画像、摘要与记忆提取；Chat（角色）负责生成最终回复。
- **长期记忆**：两层。记忆碎片（向量与元数据存在 SQLite，每群加载成 numpy 矩阵做暴力余弦召回 + Rerank 重排 +
  时间衰减排序，相似度去重，全部有 TTL）是原料；用户档案与群志由每日整理任务从碎片原地重写，永久保留。
- **原生多模态**：图片下载压缩后作为 `image_url` 随请求附带，Feedback 额外返回一句话图片观察写回消息文本。
- **Token 统计**：记录 prompt / completion / reasoning 与缓存命中（`prompt_tokens_details.cached_tokens`，
  未命中部分由 `prompt_tokens` 减去命中数推出），按模型归并后渲染成卡片。
- **SQLite 持久化与每日备份**：会话、消息、画像、启用的群与 Token 明细入库；数据目录每天自动打包。

## 运行环境

插件依赖宿主 NoneBot 项目提供运行环境。本项目根目录 `pyproject.toml` 声明的关键依赖：

- Python `>=3.10`
- `nonebot2[fastapi]`、`nonebot-adapter-onebot`、`nonebot-plugin-apscheduler`
- `tortoise-orm`、`openai`、`httpx`、`json-repair`
- `numpy`、`pillow`、`chinese-calendar`

Token 统计卡片复用同级插件目录的公共绘图库 `plugins/utils/draw/plot.py`，
运行时把 `plugins/` 插入 `sys.path` 后导入，不在依赖中声明。

## 目录结构

```text
plugins/nyaturingtest/
├── __init__.py          生命周期入口：连库 → 建表 → 载入启用群 → 注册定时任务
├── config.py            config.json 加载、AppSettings、工作区路径常量
├── handlers.py          全部 17 个 matcher：管理命令、自动消息入口、/查询记忆、/rag_debug
├── models.py            8 个 Tortoise ORM 模型
├── db.py                全部数据库读写（会话/消息/画像/交互/群开关/Token）
├── domain.py            EmotionState / PersonProfile（纯领域对象）
├── token_stats.py       Token 聚合与模型名归并 + 统计卡片 PNG 渲染
├── notes_card.py        群志卡片 PNG 渲染：按【段名】分段，称呼与梗、近期专门排版，自带标点禁则断行
├── backup.py            备份、保留期清理、定时任务注册
├── core/
│   ├── state_manager.py 每群 GroupState、worker 守护、资源清理
│   ├── logic.py         消息入队、防抖取批、OneBot 消息转换、回复切分与发送
│   ├── orchestrator.py  一轮对话的编排：search / feedback / consolidate / chat 四个 stage
│   ├── session.py       SessionState + SessionRuntime、持久化协调、记忆读写
│   ├── engagement.py    意愿值衰减与增长、对话窗口、派生聊天状态、相关性/兴趣打分
│   ├── llm.py           LLMClient（重试/熔断/指标/Token 记录）与 chat、feedback 两个全局实例、HTTP 池、JSON 解析
│   ├── prompts.py       Feedback / Chat 提示词模板、字符预算常量、角色预设加载、时间描述
│   ├── metrics.py       结构化事件日志、运行时计数、Token 落库任务
│   ├── digest.py        每日整理：记忆碎片 → 用户档案 / 群志
│   └── memory_query.py  /查询记忆：冷却、动态 k、VAD 推断、印象生成
└── memory/
    ├── short_term.py    短时消息窗口（含增量落库标记）
    ├── vector.py        VectorMemory（SQLite + numpy）、共享 Embedding/Rerank 客户端、RAG 检索、每日维护
    ├── image.py         图片下载/缓存/压缩 → 原生多模态输入
    └── validation.py    长期记忆候选的确定性校验
```

依赖方向基本单向：`handlers.py` 在顶端（只被 `__init__.py` 导入）→ `core/` → `memory/` → `models.py` / `config.py`；
`db.py` 与 `token_stats.py` 同层，前者复用后者的 `TOKEN_FIELDS` 与模型名归并函数。
一处刻意的例外：
`core/orchestrator.py` 从 `core/session.py` 单向导入 `Session` / `ChattingState`；
  `core/state_manager.py` 内部局部导入 `logic.spawn_state`（`logic` 反向引用 `GroupState`）。

提示词构造有两条硬约束，都是为了命中上游的前缀缓存（命中部分按缓存价计费）：

- **模板是唯一常量**：不允许按本轮情况分叉（图片观察要求写死在模板里，由是否有图片决定模型怎么用）。
- **动态输入字段顺序固定为「不变量 → 每轮变化」**：`bot_name` / `role` / `examples_text` / `presets`
  在最前，`related_profiles`、`search_result`、`summary` 居中，`emotion` / `recent_msgs` / `new_msgs` /
  `time_info` 在最后。相邻两轮的公共前缀因此从约 24% 提升到约 48%（chat）、38% 提升到约 59%（feedback）。
- `presets`（角色预设条目，每轮相同）与 `search_result`（每轮检索结果）分开注入，前者不占记忆字符预算。

## 一轮对话怎么走

```text
群消息 → handlers.handle_auto_chat（on_message, priority=99, block=False）
       → 队列 state.messages_chunk（deque(maxlen=200)，满了丢最旧的）
       → logic.spawn_state 后台循环：等静默 → 防抖 2s → 整批取走
       → logic._process_inbox_batch
            ├─ 按 local self-sent id 过滤自身回显
            ├─ core.llm.build_turn_calls 生成本轮 chat / feedback 两个调用闭包
            └─ core.orchestrator.ConversationOrchestrator.process_chunk
       → logic.dispatch_replies：距上次发送不足 16s 先等待 → 按句切分、最多 2 条、带拟人延迟发送 → 记 last_speak_time
```

`process_chunk` 的顺序固定，每一步之间都有代际检查：

1. `session.record_incoming` 写入短时记忆，累计固化窗口。
2. `engagement.evaluate_engagement`：意愿值按真实流逝时间衰减（0.03/分钟）→ 强关联（@Bot / 回复 Bot /
   命中名字别名）直接把意愿拉到 0.85 → 否则按内容兴趣被动增长（每条 0.05 × 兴趣系数，最多涨到 0.7）
   → 判断「聊天对象在接话」：Bot 3 分钟内说过话，且这批消息里有上次接话那批人（`conversation_partners`）。
   强关联、聊天对象在接话、或（意愿 ≥ `ENGAGE_THRESHOLD`(0.45) 且最近 20 条里自己发言占比 < 20%）才算参与。
   @Bot 与回复 Bot 由 `handlers._addresses_bot` 判断，不用 `event.to_me`
   （`.env` 的 `NICKNAME=[""]` 会让适配器把几乎所有消息都判成 to_me）。
3. 不参与时：满足固化条件（`messages_since_consolidation >= 8`，或距上次尝试 180s）
   就走 `consolidate_stage`，静默沉淀记忆，**不产生回复**。
4. `search_stage`：构造 RAG query → 检索长期记忆 → 预设条目与记忆行分别按字符预算整理成
   `preset_lines` / `memory_lines`，统计写进 `rag_search` 事件日志。
5. `feedback_stage`（观察者模型，temperature 0.1）：解析 JSON → 应用图片观察 → 沉淀 → 发言决策。
   - `_apply_image_observations`：把 Feedback 对图片的一句话观察写回消息文本（`[图片: …]` / `[表情包: …]`），
     让历史里保留图片线索。
   - `_apply_sediment`：更新全局 VAD 情绪（`parse_feedback` 已校验限幅）→ 逐条消息更新用户印象
     （峰值保持 + 时间衰减）→ 更新话题摘要 → 把 `analyze_result` 交给后台任务写长期记忆。
   - `_apply_decision`：`need_history` 为真时按时间回溯最多 20 条更早的历史消息 → 规则意愿与模型的
     `willing` 返回；规则意愿向它靠拢一半，相关时兜底到 0.85。
6. 说不说看 `willing` 与场景门槛：被点名必回；聊天对象接话 ≥ 0.45；主动插嘴 ≥ 0.6。
   每次判断记一条 `speak_decision` 事件（含 llm_willing、门槛、自己发言占比），调参看这个。
7. `chat_stage`（角色模型，temperature 0.8）生成回复；动态输入额外带上
   `my_recent_replies`（Bot 最近 6 句），prompt 要求不重复自己的句式、不编造群友说过的话。
   有回复则意愿乘以 0.7，并把这批消息的发言人记为 `conversation_partners`。
   发言冷却不再跳过整轮，而是在发送前补足 16s 间隔，对方秒回时对话不会被掐断。

**代际控制**：`Session.bump_generation()` 在 `set_role` / `load_preset` / `reset` / `reset_emotion` /
`calm_down` 时自增。每轮开始时记下 `generation`，阶段边界和写入点（短时记忆、Feedback 沉淀、长期记忆、发送）
都用 `session.stale(generation, stage)` 检查；过期就丢弃并记一条 `stale_turn_discarded` 事件。
长期记忆写入在 SQLite 事务内再确认一次代际：`reset` 先自增 generation 再删库，所以放行的写入一定早于删除。

## 记忆体系

### 短时记忆（`memory/short_term.py`）

- `deque(maxlen=200)` 滚动缓冲，`Memory.access()` 只返回最近 `SHORT_CONTEXT_LIMIT=20` 条；话题摘要只存在 `SessionState.chat_summary`。
- 每条 `Message` 带 `revision`；`mark_dirty` 把它放进待落库字典，`_save_session_locked` 只同步
  新增或被图片观察改写过（revision 变化）的消息，避免高频全量写库。
- `image_inputs`（原生图片负载）只在进程内短期持有，不序列化、不落库。
- `mentions`（被 @、被回复的人：QQ 号 → 群名片）同样只在进程内，写长期记忆时用来给没发言的人挂号。

### 长期记忆（`memory/vector.py`）

记忆存在 `nyabot.sqlite` 的 `nyabot_memories` 表，向量是归一化后的 float32 字节（`embedding` 列）。
每个 `VectorMemory` 首次使用时把本群有效向量读成一个 `(n, dim)` 矩阵，检索就是一次矩阵乘法；
万级数据只要十几毫秒，删除是真删除。之前用 ChromaDB 时 HNSW 删除只打标记、从不回收，
TTL 清理会让索引无限膨胀（大群一度 3.5 倍于存活条数），这是换掉它的原因。

| 字段 | 含义 |
| --- | --- |
| `category` | 记忆类别；只允许 `event`、`preference`、`profile`、`relationship` |
| `subject_user_id` / `subject_user_name` | 事实描述的对象（「B 说 A 的事」填 A） |
| `speaker_user_id` / `speaker_user_name` | 说出该事实的人（填 B）；由代码按 `source` 指向的第一条消息定，不用模型自报 |
| `source_msg_ids` | 依据的新消息 msg_id（空格分隔，对应 `nyabot_global_messages.msg_id`），事后据此找回原话；历史行为空 |
| `is_correction` | 更正条：替换掉被纠正的旧记忆，整理档案/群志时据此删改旧说法 |
| `confidence` / `importance` | 影响去重、衰减与排序权重 |
| `date` | `YYYYMMDD` 整数，时间衰减按它算 |
| `expires_at` | `date` 之后 `基础 TTL × (1 + importance)` 天的次日；基础 TTL event 90 天、其余类别 180 天 |
| `embedding_model` | 生成向量的模型；加载时与配置不一致直接报错，换模型必须先重嵌入 |

主体与说话人（`orchestrator._resolve_user`）：「A 说/对 B 怎样」记成主体 B、说话人 A，名字一律取上下文里该号的当前群名片。
模型常把说话人 A 的 id 抄进主体、名字却写 B：这时以名字为准把主体落到 B（说话人已记 A）；其余以 id 为准；
上下文里找不到的人只留名字、不挂 id；「上下文」包括近期发言人和被 @、被回复的人（`Message.mentions`）。
整理档案时的称呼取消息记录里的最新群名片（`db.get_latest_user_names`）。
每条候选必须带 `source`（依据的 new_msgs 下标），缺失就以 `missing_source` 拒绝：说话人只认来源消息的发送者，
没有来源的记忆事后无法核对——历史数据正是因为缺这个才修不动。
代码只认群名片原文，外号、谐音对不对得上人全靠抽取提示词：确认不了就只留原称呼、不挂 id，
曾出现模型因字音相近把外号认成另一个群友并写进正文。回复引用的发送者因此也取群名片而不是 QQ 昵称。

写入路径（`add_memories_with_dedup`）：

- 一批候选只调一次 embedding，去重与写入共用这次结果。
- **去重**：先按 `(内容, 主体, 类别)` 批内去重，再在同 `(主体, 类别)` 的已有记忆里算余弦；
  `> 0.9` 视为重复 → 跳过并「强化」已有记录（confidence 向 1.0 靠近 20%、更新 date 与 expires_at、
  `reaffirm_count + 1`）。
- embedding 调用失败时照样落库（`embedding` 为空、不做去重），每日维护补算。
- **更正**（`correct` 动作）：检索行带本轮编号（`RetrievalResult.refs`：`m3` → 记忆 id），Feedback 发现新消息
  明确纠正了某条记忆、档案、群志或角色刚说错的话时，写出更正后的事实并用 `target` 指认旧条。更正条不参与去重
  （它和旧条往往高度相似），同一事务里删除旧行、写入新行（`is_correction=1`），日志记一行「更正替换」，
  然后整体丢掉矩阵缓存。指认不到旧条时只写更正条。触发条件写在提示词里：本人否认或更新自己的事、
  有人指出角色说错并给出正确说法、有 @/回复为据的明确纠正；玩笑、反讽、起哄、没依据的定论都不算。
  错的和对的不会同时留在库里，否则下次检索两条都出来，模型继续判错或含糊。
- 写入后增量追加到内存矩阵；加载与写入共用一把 `asyncio.Lock`，避免加载期间提交的行两头落空。
- **清理**：每天 03:30 先整理档案与群志（见下节），再由 `maintain_memories()` 一条 `DELETE WHERE expires_at < now`
  清掉所有群的过期记忆、补算缺失的向量，然后让已加载的群下次重新读矩阵。先整理后删除，
  所以碎片到期前一定已经被归纳进档案。

检索路径（`retrieve_with_decay`）：

1. `build_chat_rag_queries` 过滤低价值 query（纯标点、`[表情包]`、长度 < 4、纯 emoji 等），
   追加话题摘要，再去重。「关于当前说话人」由档案每轮直接注入，不再靠检索。
2. 多 query 一次 embedding、一次矩阵乘法，每条 query 取 top-k，按最高分融合，超过
   `RAG_MERGED_CANDIDATE_CAP=64` 截断，再用 Reranker 以第一条 query 全量重排，低于 `rerank.threshold` 的丢弃。
3. 注入 prompt 的行格式是 `【m3|主体:名字|d:20261002】正文`，更正条开头为 `【更正|m3|…】`：
   模型能看出一条记忆挂在谁身上，也能用编号指认要更正的那条；提示词要求更正优先于档案与群志。
4. 打分并排序：

   ```text
   adjusted_score = 原始分(rerank_score 优先，否则 retrieval_score)
                  × 时间衰减 exp(-decay_rate × days_ago)
                  × 类型权重 × (0.7 + 0.3 × confidence)
                  × (1 + 0.15 × importance) × 主体作用域权重
   ```

   事件半衰期约 35 天（rate 0.02），偏好/画像/关系几乎不衰减（0.003）。
   作用域权重：活跃主体 1.10、被提到的主体 1.08、活跃发言者 1.04、其他主体 0.5。

### 用户档案与群志（`core/digest.py`）

碎片只是原料、到期就删；对一个人、一个群的长期认知由这两段文字承载，条数等于人数，不会无限增长。

| | 存放 | 整理时机 | 篇幅（提示词要求，不硬截断） |
| --- | --- | --- | --- |
| 用户档案 | `nyabot_user_profiles.summary` | 新碎片 ≥ 10 条，或有新碎片且有更正 / 从没整理过 / 水位已满 7 天 | 300 字左右 |
| 群志 | `nyabot_sessions.group_notes` | 有新碎片且有更正 / 从没整理过 / 水位已满 7 天（即每周，有更正时当晚） | 800 字左右 |

- **档案的材料**：主体是此人的碎片，加上此人说到别人的碎片（提示词里标「说到别人」，只取能体现此人态度、关系、习惯的部分），
  所以「A 说 B」会同时进 B 和 A 的档案（`digest.profile_rows_by_user`）。
- **水位**：`summarized_until` / `notes_summarized_until` 是已整理到的最新碎片 `created_at`，每条碎片只整理一次。
- **整理**：旧文本 + 新碎片交给 Feedback 模型重写出完整新版本；冲突以新碎片为准。
  更正条在材料里标「|更正」（分段要点里写作「更正：…」），提示词要求按它删改旧档案/群志里冲突的内容；
  碎片层面错的旧条已被替换，这一步把错误从档案和群志里也清掉。
- **分块**：新碎片按 `created_at` 升序切块（档案约 8000 字、群志约 12000 字）逐块重写直到追平，不截断、不丢弃。
  同一批写入的碎片 `created_at` 相同，切口只落在时间变化处，否则水位停在半批中间，剩下半批永远整理不到。
- **失败**：某块调用失败就停在该块，水位不动，下一晚重试。代际变化（reset 等）后立即放弃写入。
- **读路径**：本轮发言人的档案随 `related_profiles[].summary` 注入，群志作为 `group_notes` 跟在预设后面
  （每天只变一次，利于前缀缓存）。两者在 `SessionState` 里是只读副本，**不随 `save_session` 回写**，
  只由整理任务写库并同步已加载群的内存；`reset` 一并清空。
- `/查询记忆` 把档案作为最高优先级资料生成印象，但不展示档案原文：群里所有人都看得到输出。
- 整理提示词要求完整重写（不追加）、归纳不罗列，并排除隐私（真实姓名、健康、住址、家人、金额、账号等），
  因为档案每轮注入，`/查询记忆` 也会展示。

### 图片（`memory/image.py`）

下载（content-type 白名单、8MB 上限、2 次重试）→ 按 `fileid`/`file_unique` 缓存原始字节到
`cache/nyaturingtest/image_cache/raw/`（48 小时过期，每天 03:00 清理）→ 压缩到最大边 1280、
像素上限 4096²，PNG 保 PNG、动图只取第一帧（上游不接受 GIF）、其余转 JPEG q90 → base64 成 `VisionInput`。
全局信号量限制 3 个并发。

### 记忆候选校验（`memory/validation.py`）

只做确定性过滤：长度 ≥ 10（「好的」「哈哈哈」这类噪声都更短）、类别在白名单内、confidence ≥ 0.6、主体非空。
「是不是玩笑 / 有没有注入指令」交给 Feedback 的 prompt 与 confidence 判断，代码里不写启发式规则表。

## 存储层

SQLite 表（`models.py`，启动时由 `Tortoise.generate_schemas()` 建表）：

| 表 | 内容 |
| --- | --- |
| `nyabot_sessions` | 每群一条：人设、别名、预设条目、VAD 情绪、摘要、最后发言时间、固化水位、聊天状态、群志及其水位 |
| `nyabot_user_profiles` | 群内用户画像：VAD、交互次数、首次/最近交互时间、长期档案及其水位 |
| `nyabot_interactions` | 每次互动的情感增量明细 |
| `nyabot_global_messages` | 消息明细（`(session, msg_id)` 唯一，含时间与会话索引） |
| `nyabot_memories` | 长期记忆：正文、向量、主体/说话人、置信度/重要度、日期与过期时间 |
| `nyabot_enabled_groups` | 启用的群号，**唯一来源** |
| `nyabot_token_usage` | Token 明细：prompt / completion / cache hit / cache miss / reasoning，其中 cache hit 取 `prompt_tokens_details.cached_tokens` |
| `nyabot_daily_token_usage` | 按 `(day, session, model)` 聚合的日汇总，卡片统计只读这张表 |

`db.py` 是唯一的数据库访问层，除同步工具 `sanitize_text` 外全是模块级 async 函数。
三条失败语义上的例外值得记住：

- `log_token_usages` / `get_token_stats` 只记日志不抛异常——旁路统计失败不该打断回复。
- `log_interactions` 在会话不存在时静默返回。
- `sync_messages` / `update_user_profiles` 在会话不存在时抛 `RuntimeError`。

`domain.py` 里的 `PersonProfile` 用**峰值保持 + 时间衰减**更新印象：同向取绝对值更大者，异向相加，
valence 正向半衰期约 14 小时、负向约 5 小时，dominance 约 23 小时，arousal 以 5 小时量级的时间常数
回落到 0.3。每次 `push_interaction` 前先结算衰减。
这些是内存态计算，落库只写 VAD 与交互计数。

## 配置

配置文件：`plugins/nyaturingtest/config.json`（已被 `.gitignore` 忽略），模板见 `config.example.json`。
文件必须存在；缺失字段用内置默认值补齐，`chat` / `feedback` 的 `base_url` 与 `model` 必填，缺了启动即报错。

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

**改配置需要重启**：插件不监听文件变更，`AppSettings` 在启动期构造，Embedding / Rerank 客户端首次使用时按它创建。
`/autochat enable|disable` 直接写数据库并立即生效，不需要重启。

运行时策略参数（意愿阈值、RAG 预算、保留期等）不是配置项，见「策略常量在哪」。

## 命令

前缀取决于宿主 NoneBot 的 `command_start` 配置。除 `/查询记忆` 外，群聊命令均需 SUPERUSER。

| 命令 | 别名 | 说明 |
| --- | --- | --- |
| `/help` | `/帮助` | 查看帮助（群聊/私聊各一套文案） |
| `/autochat enable` | - | 在本群启用，写库并立即初始化状态 |
| `/autochat disable` | - | 在本群禁用，取消后台任务并释放资源 |
| `/status` | `/状态` | 会话状态 + reasoning_effort + 队列长度 + LLM 计数 + provider 熔断/错误 |
| `/role` | `/当前角色` | 查看当前角色 |
| `/set_role <角色名> <角色设定>` | `/设置角色` | 修改角色，设定可含空格 |
| `/presets` | `/preset` | 列出可用预设 |
| `/set_preset <文件名>` | `/set_presets` | 加载预设，可省略 `.json` |
| `/rag_debug <query>` | `/记忆诊断` | 打印检索候选数、回退原因与 top 5 记录的分数明细 |
| `/group_notes` | `/群志` `/查看群志` | 把本群群志渲染成卡片图片发出（渲染失败退回纯文字）；私聊用 `group_notes <群号>` |
| `/calm` | `/冷静` | 重置情绪与画像、意愿归零、退出对话窗口 |
| `/reset_emotion` | `/重置情绪` | 只重置 VAD 情绪 |
| `/reset confirm` | `/重置 confirm` | 先备份，再完全重置本群 |
| `/token统计 [all]` | `/autochat token统计` | 图片卡片；`all`/`全部`/`历史` 改为统计全部历史模型 |
| `/backup_data` | `/备份数据` | 手动触发一次备份 |
| `/查询记忆 [@用户]` | `/memory` | **普通群员可用**：生成 Bot 对目标用户的印象档案 |

私聊支持 `help`、`list_groups`、`backup_data` 与 9 个通用管理命令；通用命令在私聊下要把群号
作为第一个参数（`/status 123456`、`/reset 123456 confirm` …），群号非数字时直接提示。
`/autochat`、`/token统计`、`/rag_debug`、`/查询记忆` 仅限群聊。
群聊与私聊共用同一个 matcher（`handlers._dual_command`），避免 nonebot 的「Duplicated prefix rule」告警。

重置范围：

| 操作 | 情绪 | 用户画像 | 意愿/状态 | 短时记忆 | 长期记忆 | 数据库行 | 人设 |
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
`knowledges` / `relationships` / `events` / `bot_self`（加载时整理成排序固定的 `【设定/类别】` 行，
存进 `nyabot_sessions.preset_lines`，每轮原样注入、不参与检索）、
`examples`（对话样本，拼进 role 文本）、`hidden`（是否从 `/presets` 隐藏）。

## 数据、备份与定时任务

工作区根 = `HakuBot-autochat/`（`config.WORKSPACE_ROOT`）。默认路径：

```text
data/nyaturingtest/                     nyabot.sqlite（主库，含长期记忆）、三个渲染字体
cache/nyaturingtest/image_cache/raw/   图片原始字节缓存，48 小时过期
data/nyaturingtest_backups/            nyabot_backup_YYYYMMDD_HHMMSS.zip
config/nyaturingtest/nya_presets/      角色预设
```

定时任务（`nonebot_plugin_apscheduler`，`misfire_grace_time=3600`）：

| 时间 | 任务 | 内容 |
| --- | --- | --- |
| 03:00 | 图片缓存清理 | 删除超过 48 小时的缓存原图 |
| 03:30 | 长期记忆维护 | 整理到期的用户档案与群志，再删除所有群的过期记忆、补算缺失的向量 |
| 04:00 | 数据备份 | 备份 + 原始行保留期清理 |

备份行为：走 `sqlite3` 的 backup API 取一致性快照，`shutil.copy2` 复制数据目录其余内容到临时目录，
再打成一个 zip。**排除** `nyabot.sqlite` / `-wal` / `-shm`、`__pycache__` 与 `.ttf/.otf/.ttc` 字体
（每包省约 24MB，恢复后字体需在数据目录中才能渲染卡片）。备份目录是数据目录的兄弟目录，不会自我嵌套。
保留最近 `7` 个（按 mtime）。长期记忆在 sqlite 里，随 backup API 快照一起拿到一致版本，不需要额外加锁。

每次成功备份后执行保留期清理：消息明细 180 天、画像交互 180 天、Token 明细 90 天。
**长期记忆与日聚合表不参与清理**，所以 `/token统计 all` 的历史在明细行被删后依然存在。

备份包含聊天内容、用户 ID、画像与长期记忆，属敏感数据：请加密保存并定期离机复制。
三个字体文件（`SourceHanSansCN-{Regular,Bold,Heavy}.ttf`）需放在数据目录下。

## 策略常量在哪

运行时策略参数写在使用它们的模块里，调整后重载插件即可：

| 模块 | 常量的作用 |
| --- | --- |
| `core/engagement.py` | 衰减 0.03/分钟、每消息被动增长 0.05 × 兴趣（0.3~1.8）且上限 0.7、参与阈值 0.45、发言占比上限 20%、接话门槛（聊天对象 0.45 / 插嘴 0.6）、强关联下限 0.85、对话窗口 180s、说话后保留 0.7、发送间隔 16s、重启初值 0.3、Rerank 阈值 0.68 |
| `core/orchestrator.py` | 固化条件（消息数 8、间隔 180s、最多 60 条）、历史回溯条数 20 |
| `memory/vector.py` | `RAG_FINAL_K=20`、每 query 召回 40、合并候选上限 64、注入字符预算 1500 / 单条 500、基础 TTL（event 90 天、其余 180 天）、类型权重/衰减率/作用域权重表 |
| `core/prompts.py` | 字符预算：摘要 1200、最近消息 1600、历史 2400、回溯历史 1200 字 |
| `memory/short_term.py` | 上下文窗口 20 条、缓冲上限 200 条 |
| `core/logic.py` | 防抖 2s、单轮最多发 2 条、拟人延迟 1.0 + 0.1×字数（封顶 5s） |
| `core/state_manager.py` | 消息队列上限 200 |
| `core/session.py` | role 4000 字 / examples 2000 字上限、后台任务排空超时 10s、保存去抖 50ms |
| `core/memory_query.py` | `/查询记忆` 冷却（同用户 30s、同群 3s）、动态 k 的计算规则 |
| `core/digest.py` | 档案触发阈值（新碎片 10 条 / 7 天）、分块字数（档案 8000、群志 12000）、整理温度 0.2 |
| `memory/image.py` | 8MB / 4096² 像素上限、最大边 1280、并发 3、缓存 48 小时 |
| `memory/validation.py` | 允许的记忆类别、最低置信度 0.6、最短 10 字 |
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
