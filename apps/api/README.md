# Agent Layer · Supervisor + DAG Planner + 六专业子 Agent

基于 **LangGraph** 的科研助手 Multi-Agent 内核。**零 API Key 可直接运行**（内置 `MockLLM`），
配置了任意 OpenAI 兼容接口即可切换到真实模型，业务代码一行不改。

---

## 1. 快速开始

```bash
cd apps/api
python -m venv .venv && ./.venv/Scripts/activate      # Windows PowerShell
pip install -r requirements.txt

cp .env.example .env                                   # 默认 LLM_PROVIDER=mock，先不改
python examples/run_demo.py                            # 跑全部示例（离线）
python examples/run_demo.py -q "对比 Transformer 与 BERT 并生成综述"
```

切换真实模型：编辑 `.env`（**密钥只放环境变量或本地 `.env`，后者已被 `.gitignore` 忽略；`.env.example` 保持留空，切勿提交真实 key**）

```ini
LLM_PROVIDER=openai_compatible
LLM_API_KEY=sk-xxxxxx
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus
```

启动 HTTP 服务（含内置前端）：

```bash
pip install -r requirements.txt
python -m app.server        # 打开 http://localhost:8000/（API 文档在 /docs）
```

前端是 `app/static/` 下的零构建单页应用：学术翻译、通用大模型、Agent 对话/HITL、系统状态。
同源托管，无需 Node、构建步骤或 CORS 配置。

学术翻译走 `POST /api/translate`：

```json
{"text": "...", "direction": "zh2en", "glossary": {"中文": "English"}, "max_chars": 3000, "max_terms": 30}
```

- `direction=zh2en`（默认）：`app/zh2en.py`，先抽术语再逐段译为学术英文；
- `direction=en2zh`：`app/translate.py`，先保护公式/引用，再按句译为中文；术语表写作 `{"English": "中文"}`；
- 都返回 `{translation, glossary, pairs, issues, stats, markdown, direction, model, latency_ms, history_id}`，
  公式、变量、代码、URL、引用原样保留。

每次翻译自动写入本地 SQLite（`SQLITE_PATH`，默认 `apps/api/data/research_assistant.db`）。
写库失败不影响译文，只在 `issues` 里提示，不会假装已保存。

表结构（`python -m app.history --init` 一次建全，Navicat 直接打开 `.db` 即可）：

- `translations`：一次翻译一行（方向、原文、译文、模型、耗时、统计、校验提示）；
- `translation_terms`：术语表按行存储，外键指向 `translations.id`；
- `chat_threads` / `chat_messages`：会话与消息（标题取首条提问，滚动摘要也落库）；
- `agent_runs`：一轮 Agent 执行的完整结果（路由、计划、答案、校验、证据、耗时、模型、状态）；
- `agent_tasks`：该轮 DAG 的子任务（状态、输入、结果、耗时、错误）；
- `generated_images`：文生图记录（提示词、实际提示词、模型、落盘图片、任务 ID）。

Agent 的会话记忆、轮次计数、任务中间态在进程重启后从上述表恢复；写库失败只告警，不影响回答。

- `GET /api/history/translations?limit=20&offset=0&direction=en2zh`：翻译历史摘要；
- `GET /api/history/translations/{id}`：单条翻译全文 + 术语；
- `GET /api/history/threads?limit=20&offset=0`：会话列表（消息数、执行轮数、最后一条预览）；
- `GET /api/history/threads/{thread_id}`：会话详情（消息 + 每轮执行与子任务）；
- `DELETE /thread/{thread_id}`：删除会话（级联删除消息与执行记录）。

通用大模型是独立模块（`app/general.py`），与 Agent 完全分离，不路由、不检索、不要求引用：

- `POST /api/general/chat`：`{messages:[{role,content}]}` → `{answer, model, usage, search?}`；天气/新闻/行情等实时问题会自动走百炼联网搜索（命中关键词才调用，普通问答不产生搜索费用）；搜索失败时降级为普通回答并在 `warnings` 标注“不含实时数据”；
- `POST /api/general/image`：multipart `file` → 图片 OCR 文本（Tesseract，≤ 25 MB）；
- `POST /api/general/image/chat`：multipart `file` + `question` → 多模态模型看图回答（默认 `qwen3-vl-plus`，≤ 10 MB）；
- `POST /api/general/image/generate`：`{prompt, size, n, negative_prompt}` → 通义万相异步生成图片；
- 语音识别 / 朗读使用浏览器原生 Web Speech API（Chrome / Edge），后端不接收音频。

图片生成相关配置（`.env`）：

```ini
VISION_MODEL=qwen3-vl-plus
IMAGE_MODEL=wanx2.1-t2i-turbo
IMAGE_API_BASE=https://dashscope.aliyuncs.com/api/v1
IMAGE_TIMEOUT=180
```

生成结果下载到 `app/static/generated/`，通过 `/static/generated/<file>` 访问；
自动清理 7 天前或超过 200 个的旧文件。`size` 形如 `1024*1024`（512–1440，总像素 ≤ 2.1M），`n` 为 1–4。

对话输入框的「附件」按钮走 `POST /api/parse`（multipart `file`）：
支持 PDF、DOCX、PNG/JPG/TIFF/BMP/WEBP、TXT/MD，单文件 ≤ 25 MB、PDF ≤ 100 页；
返回 `{filename, kind, text, chars, truncated, meta, warnings}`。
附件文本通过与 `query` 分离的 `attachment` 字段送入 Agent，不参与路由匹配。

### 科研工作流模块（文献库 · 阅读器 · 图谱 · 推荐 · 写作）

前端五个视图与后端 `app/library.py` / `app/studio.py` 对应，数据全部落在本地：

- `data/library/<paper_id>/`：`original.pdf`、`docir.json`（解析块）、`reader.json`（目录/引用/bbox）、
  `chunks.json`（检索块）、`translations.json`（段落译文缓存）；
- `data/recommend/`：推荐历史、反馈 JSONL 与 `latest.json`。

| 能力 | 端点 |
|---|---|
| 文献库 | `GET /api/library/stats`、`GET /api/papers`、`POST /api/papers/upload`（一次 ≤20 个 PDF）、`GET/PATCH/DELETE /api/papers/{id}`、`POST /api/papers/{id}/retry`、`POST /api/papers/{id}/replace`、`GET /api/papers/{id}/compare/{other}`（版本差异） |
| 阅读器 | `GET /api/papers/{id}/file`、`asset/{name}`、`reader`、`markdown`；`POST /api/papers/{id}/translate`、`translate-selection` |
| 文献问答 | `POST /api/qa`（单篇/多篇，返回 `[E1]` 级证据与 bbox/页码） |
| 下载 | `POST /api/papers/download`（arXiv ID 或标题检索）、`POST /api/papers/{id}/references/download` |
| 引用图谱 | `POST /api/graph`、`/api/graph/survey`、`/api/graph/future-work` |
| 推荐 | `GET /api/recommend`、`POST /api/recommend/refresh`、`POST /api/recommend/feedback` |
| 写作 | `POST /api/write/framework`、`/diagram`、`/chart` |

解析在后台线程池执行（同时 2 篇），状态 `queued → parsing → ready/failed`，前端轮询进度。
`RECOMMEND_INTERVAL_HOURS>0` 时启动守护线程定期刷新 arXiv 推荐。
参考文献只下载标题相似度 ≥0.82 的条目，低价值条目跳过并计数，不伪造。

---

## 2. Agent 架构

### 2.1 Supervisor 路由（三层漏斗）

| 层级 | 手段 | 延迟/成本 | 覆盖场景 |
|---|---|---|---|
| ① 显式指令 | `/parse` `/search` `/translate` `/graph` `/survey` | ~0ms / 0 token | 用户明确指定，100% 确定 |
| ② 规则打分 | 领域正则 → route + confidence ≥ 0.7 直接命中 | ~1ms / 0 token | 70%+ 高频查询，可 CI 断言 |
| ③ LLM 分类 | 输出 `{route, complexity, confidence, reason}` | ~300ms | 长尾模糊意图 |

**复合意图判定**（→ 交给 Planner 拆 DAG）：一句中出现 ≥2 个领域动词，或命中
`(对比|分析).*(并|且|然后).*(综述|生成)` 等模式。示例：

- `Transformer 的注意力复杂度是多少？` → `rag`（单专家）
- `对比 Transformer 与 BERT 并生成综述` → 命中 rag + writing → `planner`（DAG）
- `你好` → `clarify`（不调工具、不烧 token）

### 2.2 Task Planning：意图 → 子任务 DAG

```
对比 Transformer 与 BERT 并生成综述
        │
        ▼
    t1 检索双方核心文献 [retrieval]
       ╱            ╲
t2 多维度对比 [rag]   t3 构建引用图 [graph]      ← 同层可并行（生产用 Send 扇出）
       ╲            ╱
    t4 撰写综述正文 [writing]
        │
    t5 导出 LaTeX [writing]
```

`Planner` 输出后由 `validate_and_repair()` 兜底，保证「模型产出不可信也能执行」：

| 异常 | 修复动作 |
|---|---|
| id 缺失/重复 | 自动编号 `t{i}` |
| agent 非法 | 回落 `rag` |
| `depends_on` 指向不存在 id | 丢弃该依赖 |
| 依赖成环 | DFS 检测并断开回溯边 |
| 整体不可用 | 回落启发式模板计划 |

`Executor` 按拓扑分批，同批内的产出**立刻对后序子任务可见**，
保证 writing 阶段一定能吃到上游检索/对比的证据。

### 2.3 六个专业子 Agent

| Agent | 授权 scope | 主要工具 | 交付物 |
|---|---|---|---|
| `parse` | m1 | `parse_document` `extract_scanned_ocr` `semantic_dedup` | DocIR、去重报告、版本链 |
| `retrieval` | m4 + shared | `hybrid_search` `fetch_paper` `semantic_dedup` | 候选论文、元数据 |
| `rag` | m2 + shared | `hybrid_search` `compare_claims` | 带引用锚点的结论 |
| `translation` | m3 + shared | `translate_block` | block 级译文 + bbox 锚点 |
| `graph` | m5 + shared | `build_citation_graph` `fetch_paper` | 节点/边/社区/主路径 |
| `writing` | m6 + shared | `write_section` `render_chart` `export_latex` | Markdown / Vega-Lite / LaTeX |

工具授权即 `ToolRegistry.specs(scopes=...)`：每个 Agent 只看到自己的工具，
既降误调用，也压缩 prompt。

### 2.4 抗幻觉闭环

`Verifier` 四道闸，任一不过 → `Reflector` → 重规划/重执行（上限 `AGENT_MAX_REVISION=2`）：

1. **锚点真实性**：`[[ev:doc#block]]` 必须命中本次任务真实取到的证据池；
2. **引文字面存在**：`corpus.find_quote()` 校验 quote 原文命中，拦住"看起来很真"的编造；
3. **覆盖度**：每 2 条非引用论断至少 1 个独立锚点；
4. **模型语义复核**：抓改写、张冠李戴。

---

## 3. Function Calling 规范

**单一信封**——所有工具一律返回同一结构，错误也走信封（让 LLM 能自纠，而不是把异常抛炸链路）：

```json
{
  "ok": true,
  "tool": "hybrid_search",
  "data": {"hits": [...]},
  "cited": [{"doc_id":"doc_transformer","block_id":"doc_transformer#b003",
             "quote":"自注意力的计算复杂度为 O(n^2 · d)","page":5,"bbox":[72,150,523,240],"score":0.87}],
  "metrics": {"latency_ms": 3, "strategy": "bm25+dense+rerank"},
  "error": null, "hint": null
}
```

**Pydantic 即 Schema**——入参模型自动生成 OpenAI `tools` 格式，杜绝手写 JSON Schema 漂移：

```python
class SearchArgs(BaseModel):
    query: str = Field(..., description="检索 Query（中英均可）")
    top_k: int = Field(6, description="返回条数")

registry.register(Tool("hybrid_search", "...", SearchArgs, hybrid_search, scopes=("m2","m4","shared")))
```

工具清单见 `app/agents/tools/builtin.py`（11 个，覆盖六模块最小闭环）。

---

## 4. 状态管理

四类记忆职责边界见 `app/memory/store.py` 顶部注释：

| 记忆 | 内容 | 作用域 |
|---|---|---|
| 会话记忆 | 提问/结论，超窗（>16 条）自动压缩为滚动摘要 | `thread_id` |
| 文献库上下文 | 文献清单、指纹、术语表规模 | `user_id` |
| 任务中间态 | SubTask DAG 与每步产物 | `thread_id` |
| 长期资产 | 术语表 / 翻译记忆 / 用户画像 | `user_id`，跨会话累积 |

LangGraph 的 `MemorySaver` 负责图状态 checkpoint（刷新页面不丢上下文）；
把 `default_checkpointer()` 换成 `PostgresSaver` 即可生产化，
配合 `interrupt_before=["executor"]` 实现 HITL（见 `runner.resume()`）。

---

## 5. 目录结构

```
apps/api/
├── app/
│   ├── config.py                 环境变量 → Settings（含自动降级判定）
│   ├── utils.py                  JSON 抽取、日志、计时
│   ├── schemas/
│   │   ├── agent.py              EvidenceCard / ToolEnvelope / SubTask / PlanDAG / VerifyResult
│   │   └── state.py              GraphState（含 reducer 语义）
│   ├── providers/llm.py          OpenAI 兼容 Adapter + MockLLM + chat_json
│   ├── memory/store.py           四类记忆（可换 Redis）
│   ├── agents/
│       ├── dag.py                DAG 校验修复 / 成环断开 / 拓扑分批
│       ├── toolloop.py           Function Calling 执行循环（ReAct 微循环）
│       ├── graphs.py             LangGraph 主图装配 + checkpointer
│       ├── runner.py             run / resume / get_task_state
│       ├── nodes/
│       │   ├── supervisor.py     路由三层漏斗
│       │   ├── planner.py        DAG 规划（含 replan）
│       │   ├── executor.py       拓扑执行 + 失败重试
│       │   ├── specialists.py    六个专业子 Agent
│       │   └── gates.py          校验 / 反思 / 收口
│       └── tools/
│           ├── registry.py       工具注册 + 作用域 + 统一信封
│           ├── builtin.py        11 个内置工具
│           └── corpus.py         离线语料与混合检索内核（DocIR 替身）
│   ├── rag/                      541 页教材入库与分级加载（§13）
│   ├── retrieval/                学术 RAG 混合检索：Dense+BM25→RRF→Rerank（§14）
│   ├── general.py                通用大模型（问答 + 图片理解/OCR + 文生图），独立于 Agent
│   ├── history.py                历史 SQLite：翻译/会话/Agent 执行/生图（Navicat 可打开）
│   ├── translate.py              英译中（占位符保护 + 长句拆分 + 双语对齐）
│   └── zh2en.py                  中译英（术语抽取 + 学术润色）
├── examples/run_demo.py
├── app/static/                   零构建前端（index.html / app.js / styles.css）
└── app/server.py                 FastAPI 入口（/chat、/api/parse、/api/general/*、/api/translate）
```

---

## 6. 示例输出（mock 模型实跑）

```
查询     : 对比 Transformer 与 BERT 并生成综述
路由     : planner  (complexity=composite, confidence=0.93)
任务 DAG : 5 个子任务
   t1 [done] retrieval  deps=-       检索 Transformer 与 BERT 的原始论文与关键变体
   t2 [done] rag        deps=t1      多维度对比分析
   t3 [done] graph      deps=t1      构建引用关系图
   t4 [done] writing    deps=t2,t3   撰写综述正文
   t5 [done] writing    deps=t4      导出 LaTeX/BibTeX
证据池   : 6 条（均含 doc_id#block_id / page / bbox）
抗幻觉校验: passed=True coverage=1.0 quote_faithful=True
```

---

## 7. 生产化替换清单

| 位置 | 当前 | 生产替换 |
|---|---|---|
| LLM | `MockLLM` / OpenAI 兼容端 | DeepSeek / Qwen / GLM，加降级链 + 预算熔断 |
| 检索 | 内存 TF-IDF 近似 | pgvector + pg_trgm + BGE-M3 + bge-reranker-v2-m3 |
| 语料 | `tools/corpus.py` | PostgreSQL(pgvector) + MinIO + Docling/MinerU 解析产物 |
| 外部检索 | `PAPER_DB` | OpenAlex + Crossref + Semantic Scholar 三源互补 |
| Checkpoint | `MemorySaver` | `PostgresSaver`（断点续跑 / HITL） |
| 并发 | 按层串行 | LangGraph `Send` 扇出 + Celery 承载重工具 |
| 可观测 | 日志 + trace 列表 | Langfuse / OTel，`thread_id` 贯穿前后端 |

---

## 8. 意图分类器（12 类）

`app/agents/intent_classifier.py` —— 独立的意图识别模块，可直接替换 Supervisor 的第一层。

```bash
python app/agents/intent_classifier.py --selftest
python app/agents/intent_classifier.py "帮我解析 data/scan_541.pdf 并对 1706.03762v1 和 v7 去重"
```

```json
{
  "intent": "document_parse",
  "confidence": 0.92,
  "entities": {"paper_ids": ["arXiv:1706.03762v1", "arXiv:1706.03762v7"],
               "query": "解析 并对 1706.03762v1 和 v7 去重", "file_path": "data/scan_541.pdf"},
  "need_clarify": false,
  "clarify_question": ""
}
```

设计要点：

| 要点 | 实现 |
|---|---|
| JSON 模式 | `response_format={"type":"json_object"}`，`strategy=auto` 自动降级 function calling |
| 解析容错 | 整体 JSON → 代码块 → 子串三级抽取，失败转规则分类，**永不抛异常给上游** |
| 实体兜底 | 模型漏抽时用正则补 `file_path`（`.pdf/.docx/...`）、arXiv ID、DOI、`doc_id`、版本链 `vN` |
| 澄清触发 | `confidence < 0.6`，或意图所需实体缺失（`FILE_INTENTS` 缺文件且无 paper_id、`PAPER_INTENTS` 缺论文标识与 query） |
| 离线可用 | 无 API Key 时走 `mock_classify()`，13 条自测全通过 |

意图 → Agent 桥接见模块内 `INTENT_TO_AGENT`（如 `dedup_check→parse`、`cross_paper_reasoning→rag`、`survey_generate→writing`）。

---

## 9. 已验证的运行结果与 Windows 注意事项

实跑 `python examples/run_demo.py`（mock 模型）结果：

| 输入 | 路由 | 计划 | 证据 | 校验 |
|---|---|---|---|---|
| `Transformer 的注意力复杂度是多少？` | `rag` (0.75) | — | 4 条 | passed=True coverage=1.0 |
| `对比 Transformer 与 BERT 并生成综述` | `planner` (0.90) | 5 子任务 DAG 全 done | 4 条 | passed=True coverage=1.0 |
| `帮我解析 sample.pdf 并对 arxiv:1706.03762v1 与 v7 去重` | `parse` (0.90) | — | — | passed=True（结构化意图豁免锚点） |
| `/graph ...` | `graph` (1.0) | — | — | passed=True |
| `你好` | `clarify` (0.90) | — | — | passed=True |

Windows 控制台默认 GBK，建议先执行 `$env:PYTHONIOENCODING="utf-8"`（代码内已避免使用 emoji）。

两个易踩的坑（已在代码里处理）：

1. **复用 `thread_id` 会让证据池跨轮污染** —— checkpointer 对带 `operator.add` reducer 的字段是追加语义，因此 `run()` 每轮开一个新的图线程 `{thread_id}#t{n}`，`resume()` 用返回的 `graph_thread`。
2. **`block_id` 必须是文档内局部 ID** —— 全局拼 `doc#block` 会得到 `doc#doc#b000`；跨文档定位用 `(doc_id, block_id)` 二元组。

---

## 10. 任务规划器（目标 + 工具 → 可执行 DAG）

`app/agents/task_planner.py`

```bash
python app/agents/task_planner.py --selftest                       # 8/8 通过（离线）
python app/agents/task_planner.py -g "对比 Transformer 与 BERT 并生成综述" --json
python app/agents/task_planner.py -g "扫描件 OCR 后去重" -t parse_document,extract_scanned_ocr,semantic_dedup
```

输出：

```json
{"goal": "对比 Transformer 与 BERT 并生成综述",
 "tasks": [
  {"id":"t0","action":"本地文献库检索：...","tool":"hybrid_search","depends_on":[],"output":"库内证据卡（优先复用已有文献）"},
  {"id":"t1","action":"外部补充最新进展（本地不足时）","tool":"fetch_paper","depends_on":["t0"],"output":"外部候选论文，标注补充外部证据"},
  {"id":"t2","action":"汇总对比并梳理脉络","tool":"compare_claims","depends_on":["t0","t1"],"output":"对比矩阵 + 引用锚点"},
  {"id":"t3","action":"按大纲撰写综述各章节","tool":"write_section","depends_on":["t2"],"output":"带 [[ev:doc#block]] 的 Markdown"},
  {"id":"t4","action":"导出 LaTeX 与 BibTeX","tool":"export_latex","depends_on":["t3"],"output":"可编译 .tex + .bib"}]}
```

两条硬规则**提示词与代码双重强制**：

| 规则 | 提示词 | 代码兜底（`validate_and_repair`） |
|---|---|---|
| R1 优先复用文献库 | 分析/写作前必须有本地检索任务 | 无本地任务则插根节点 `t0`，原根节点挂到它下面 |
| R2 外部检索本地优先 | 外部任务须 `depends_on` 本地任务 | 检查每个外部任务的依赖，缺则补最早本地任务 id |

其余修复：id 去重与重编号、悬空/自依赖丢弃、DFS 断环、工具名别名归一
（`arxiv_search → fetch_paper` 等 20+ 条）、action 关键词回填、任务数截断至 7、拓扑重排输出。

降级链：`json_object` → `function_call` → 启发式模板（8 类范式：综述/对比/解析/OCR/去重/翻译/图谱/可视化/问答），
无 Key 或模型异常都返回合法 JSON。

与执行器对接：`to_subtasks(plan)` 直接产出 `app.schemas.SubTask` 兼容字典（工具 → 六 Agent 映射见 `TOOL_TO_AGENT`），
可喂给 `executor_node` 的 DAG 执行。

---

## 11. M1 文档解析：PDF → DocIR

`app/parsing/`（PyMuPDF + XY-Cut + PaddleOCR 兜底），**零外部素材即可自测**：

```bash
pip install -r requirements.txt                 # 含 pymupdf
python -m app.parsing.cli --selftest            # 现场合成双栏论文 + 扫描页，12/12 通过
python -m app.parsing.cli paper.pdf -o docir.json --assets assets/
```

### 选型理由

| 候选 | 角色 | 结论 |
|---|---|---|
| **PyMuPDF (pymupdf)** | 内核：版式块、span 字号/字体/粗细、**bbox**、表格线框、栅格化 | ✅ 主选。C 实现、541 页吞吐高、坐标精确、无模型依赖，本题 5 项要求里 4 项（版式/坐标/表格/图注）直接靠它 |
| **PyMuPDF4LLM** | PDF→Markdown/LLM 友好文本 | ⚠️ 辅助。表格与代码块保留较好，但它**只输出文本流、丢 bbox**，无法支撑"标记每个块页码与坐标供前端定位"，故只作兜底/对照 |
| **PaddleOCR PP-StructureV2/V3** | 扫描件版面分析 + 表格结构 + 公式检测 | ✅ OCR 首选。中文论文版面（双栏/页眉页脚/图注）召回好，且**返回区域 bbox**，能直接映射回 PDF 坐标；有 CPU 可跑 |
| pytesseract | 轻量 OCR 兜底 | ✅ 二级降级。只做 `image_to_data` 行级聚合，低置信（<30）词丢弃 |
| Nougat / MathPix | 公式 LaTeX | ⚠️ 可选钩子。PDF 文本层**没有 LaTeX**——LaTeX 只能来自专用模型或人工标注；本题默认走"字形符号表转写"，并显式标注 `latex_approx` |
| Docling / MinerU | 端到端解析 | ⚠️ 备选。效果好但重依赖 + 推理慢，适合离线批处理，不适合本系统在线链路 |

### 流水线

```
PDF → 页面几何(span/字号/字体/bbox)
    → 分栏判定(x 轴投影找空白带) + XY-Cut 阅读顺序
    → 标题层级树(编号 > 字号倍数 > 字重，保留章节编号)
    → 块分类(标题/正文/列表/公式/表格/图/图注/参考文献/页眉页脚)
    → 扫描页判定 → PaddleOCR → tesseract → 裁图保留(type=unknown)
    → ParseResult(JSON)
```

### 输出契约

```json
{"layout":"double",
 "layout_detail":{"page_columns":[1,2],"gutters":[...],"scanned_pages":[]},
 "title_tree":[{"text":"Method","level":1,"number":"2","page":1,"bbox":[...],
                "children":[{"text":"Scaled Dot-Product Attention","level":2,"number":"2.1","page":1,"bbox":[...]}]}],
 "blocks":[{"id":"p1b7","type":"formula","text":"softmax(QK^T / √d_k)V",
            "latex":"softmax(QK^{T} / \\sqrt{d_{k}})V","latex_approx":true,
            "page":1,"bbox":[60,480,300,498],"confidence":0.95,"source":"pymupdf+heuristic"},
           {"id":"p2b3","type":"table","markdown":"| Model | EN-DE | EN-FR |","page":2,"bbox":[...],"caption":"Table 1: ..."},
           {"id":"p2b9","type":"reference","text":"[1] Vaswani et al. ...","meta":{"ref_id":"1","arxiv":"arXiv:1706.03762"}}],
 "stats":{...}, "warnings":[...]}
```

### 反臆造设计（对应"不得臆造内容"约束）

| 场景 | 处理 |
|---|---|
| 公式 | 只把**真实抽取到的字形**按符号表转写，`latex_approx=true`；覆盖率 < 0.6 → `type="unknown"` + 裁图保留原区域 |
| 表格 | 用 `find_tables()` 的**单元格 bbox × words 空间包含**填格（避免 `extract()` 把 `EN-DE` 拼成 `ENDE\n-`），拿不到才用对齐启发式并标 `source` |
| 图/未知区 | 保留 `bbox` + `image_path`（`assets/` 下 PNG），前端可直接展示原图 |
| 参考文献 | 只切条目 + 抽 ID/DOI/arXiv/URL（正则可验证），**不解析作者/年份/期刊** |
| 无 OCR 后端 | 不做猜测，整页裁图为 `unknown`，`source="no_ocr_backend"` |

### 与 Agent 工具集成

`parse_document` 工具：文件真实存在 → 走 `app.parsing.pipeline.parse_pdf`，正文块提升为 `EvidenceCard`（带 page/bbox，供 RAG 引用与前端高亮）；文件不存在 → 回落内置语料（离线演示）。

```python
env = registry.invoke("parse_document", {"file_path": "paper.pdf"}, RunContext(thread_id="t"))
# ok=True layout=double blocks=29 cited=10 engine=pymupdf+xycut
```

### 已修的三个真实坑

1. `Rect.intersect()` **原地修改**矩形 → 覆盖率分母被覆盖成 0 并抛异常（被外层 `except` 吞掉，表格静默消失）：先取面积再求交。
2. 自己替换产生的 `{}` 被算成"未映射符号" → 转写越成功覆盖率越低（`QK^{T}` 只有 0.33）：花括号不计入分母。
3. `row.cells` 在新版 PyMuPDF 是 **bbox 元组**而非对象：两种形态都要兼容。

---

## 12. 文献版本管理：语义去重 + 版本判定 + 逐段差异

`app/agents/dedup.py`（OpenAI 兼容 embedding + LLM；无 Key 自动降级为离线哈希向量 + difflib）

```bash
python app/agents/dedup.py --selftest                      # 15/15 通过（离线）
python app/agents/dedup.py --demo                          # arxiv:1706.03762 v1 vs v7
python app/agents/dedup.py --a a.json --b b.json --json    # 自定义两篇
python app/agents/dedup.py --a doc_transformer --b doc_bert --no-llm
```

输出（四个主键固定，其余为可追溯细节）：

```json
{"verdict": "duplicate",
 "similarity": 0.8964,
 "version_advice": "新上传版本更新（库中 arxiv:1706.03762v1 → 新上传 arxiv:1706.03762v7）：建议覆盖库中旧版本，并将旧版本存档以便回溯。",
 "diff_summary": "共 1 段比对：未变 0 / 修订 1 / 新增 0 / 删除 0。\n- [修订] Abstract：字数 42 → 58；新增表述：，并补充了消融实验与更多基线对比",
 "evidence": [{"field":"abstract","score":0.8274,
               "a_quote":"提出仅用注意力的序列转换模型 Transformer，在机器翻译上取得新 SOTA。",
               "b_quote":"提出仅用注意力的序列转换模型 Transformer，在机器翻译上取得新 SOTA，并补充了消融实验与更多基线对比。"}],
 "field_scores": {"title":1.0,"abstract":0.8274,"body":0.0},
 "weights": {"title":0.4,"abstract":0.6,"body":0.0},
 "diff_detail": [...], "versions": {"a":"...v1","b":"...v7","cmp":1,"same_base":true},
 "engine": {"embedding":"openai-compatible|local-hash","diff":"llm|difflib"},
 "warnings": [...]}
```

### 判定逻辑

| 环节 | 做法 |
|---|---|
| 相似度 | `title 0.25 + abstract 0.35 + body 0.40` 加权余弦；缺摘要 → 权重重分配为 `title 0.3 / body 0.7`，缺正文 → `title 0.4 / abstract 0.6`（**实际权重回传在 `weights`**，避免"分数对不上权重"） |
| 正文相似度 | **双向最佳匹配均值**：单向匹配会被"新版新增段落"拖低，v1→v7 加了消融实验时单向分明显偏低 |
| 元数据优先 | 同一主干（`1706.03762v1`/`v7`）直接判 duplicate，避免同一篇被计成两篇——这是两类错误里更贵的那一类 |
| 阈值 | `≥0.82 duplicate` / `≥0.55 related` / 否则 `distinct`（`--dup-th/--rel-th` 可调） |
| 证据 | 每个字段给出**两侧原文字段**（`a_quote`/`b_quote` + `a_ref`/`b_ref`），而不是只丢一个相似度数字 |
| 版本比较 | 解析 `vN` / `v1.2.3` / 日期 → 可比较元组；`1=B 更新, -1=B 更旧, 0=相同, None=不可比较` |
| 覆盖建议 | B 更新→"建议覆盖并将旧版存档"；B 更旧→"库中已有更新版本建议保留…是否覆盖？"；相同→"无需覆盖"；不可比较→"建议人工确认" |
| 逐段差异 | 段落贪心对齐（阈值 0.45）→ `unchanged(≥0.97) / modified / added / removed`；modified 用 LLM 总结，无 Key 用 `difflib` + 数值正则 |
| 反评价 | diff 提示词显式禁用"更好/更清晰/改进/优化"等词；自测里断言这些词不出现 |

### 接入 Agent 工具

`semantic_dedup` 工具在**恰好两篇且都能取到内容**时走真实比对，其余情况回落到原有的"版本链聚类"：

```python
registry.invoke("semantic_dedup", {"doc_ids": ["arxiv:1706.03762v1", "arxiv:1706.03762v7"]}, ctx)
# ok=True verdict=duplicate sim=0.8964 canonical=arxiv:1706.03762v7 deduped=1
# cited=[arxiv:1706.03762v1#abstract]
```

两条与抗幻觉闸口的边界（都是实跑踩出来的）：

1. **只把字面可校验的片段提升为证据卡**。标题不在语料块里，若一并做成 `EvidenceCard`，
   `find_quote` 必然失败 → 整条链路被判"引文不忠实"。不可引用的片段记录在
   `metrics.uncitable_evidence`，供人工查看而**不进入证据池**。
2. **版本链也必须注册成语料块**（已在 `corpus.py` 里把 `VERSION_ABSTRACTS` 注册为
   `{doc_id: 'arxiv:...vN', block_id: 'abstract'}`），否则 v1/v7 的引文查无此句。

---

## 13. RAG 入库：541 页教材的分层切块与长上下文加载

`app/rag/`（`python -m app.rag.cli --selftest` 14/14 通过；541 页实跑见下）

### 13.1 流水线

```
PDF ──①流式解析(页批 24)──► 轻量块 + 字号直方图 + 标题候选 ──► spill 落盘 ──释放
     ──②由直方图定正文字号 → 候选里定标题 → 建标题树
     ──③回放 spill(一次一块) → 按"节"边界切 L0/L1/L2 → 章节 lead / 章摘要(有界缓冲)
     ──► chunks_{T0,L0,L1,L2}.jsonl + index.json(元信息) + report.json(内存报告)
检索：T0 摘要层(只用元信息打分) → L0 章节层 → L1 段落层 → L2 细节层（逐层才读盘）
```

**为什么是两阶段**：标题树需要全文的正文字号与所有标题，而"攒够全文再切"会让峰值随文档线性增长。
所以阶段一**只留直方图与短块候选**（内存与页数无关），阶段二**流式回放 spill**，
峰值只与"页批大小 + 单块大小"同阶。

### 13.2 切块参数

| 层级 | 单位 | 目标 tok | 重叠 tok | 上限 tok | 用途 |
|---|---|---|---|---|---|
| T0 摘要层 | 章（各节 lead 首句） | 256 | — | 256 | 541 页级全局导航 |
| L0 章节层 | 整节（lead 截断） | 512 | — | 512 | 章节整体语义、定位候选章 |
| L1 段落级 | 段落块 | 512 | 64 | 768 | 主力检索粒度 |
| L2 句子级 | 句子 | 128 | 32 | 192 | 精确引用与细节展开 |

取值理由：

| 参数 | 理由 |
|---|---|
| L2 128/32 | 中文一句约 20-40 token；128 覆盖 3-5 句，32 的重叠保住一个完整句子 + 主语，跨块指代不断链 |
| L1 512/64 | top-k=6 时约 3k token，留出答案空间；64 ≈ 1-2 句，足以衔接上一块末尾的论点 |
| L0 lead 512 | 章节层**不拼全文**（单章可达数万 token），只取开头 lead 承载章节主题 |
| 不跨节切分 | 跨节拼接会让 `title_path` 失真——溯源元数据是这套 RAG 的一等公民 |

可调：`--para-target/--para-overlap/--sent-target/--sent-overlap`；token 估算优先用 `tiktoken(cl100k)`，
未安装时用内置估算（CJK 1 字≈1 token、拉丁≈3.6 字符/token）。

### 13.3 每个 chunk 的溯源元数据

```json
{"id":"textbook_541:L1:00412", "level":"L1", "doc_id":"textbook_541",
 "section_id":"textbook_541:L0:00052",          // 所属小节
 "section_number":"5.2", "parent_id":"textbook_541:L0:00052",
 "page_start":8, "page_end":8, "n_tokens":530,
 "block_ids":["p8b3","p8b4","p8b5"],            // 回指 DocIR 块，供前端高亮
 "bbox":[72.0,132.5,523.0,298.1],
 "store":"L1", "offset":184223, "length":1604}  // 正文地址：seek 读取，不常驻
```

标题路径做**字典编码**（`index.paths[section_id] -> ["第 5 章 反向传播", "5.2 链式法则"]`），
同一小节的几十个 chunk 共享一份字符串——不做这一步，465 块时索引 222KB；做完后 163KB。

### 13.4 分级加载（摘要层 → 章节层 → 细节层）

```python
loader = TieredLoader(index=ChunkIndex.load("idx/index.json"), base_dir=Path("idx"))
res = loader.load("反向传播 链式法则", token_budget=3000)
# used=2998/3000  tiers={'T0': 2, 'L0': 4, 'L1': 5, 'L2': 2}
# [T0] p8-9   259tok  第 5 章 反向传播
# [L0] p8-8   521tok  第 5 章 反向传播 > 5.2 链式法则
# [L1] p8-8   530tok  第 5 章 反向传播 > 5.2 链式法则
# IO: 加载 28 块 / 15.5 KB（占正文 0.2%）
```

**不是"先加载全部再排序"**：第 1 层只用常驻内存的元信息（标题路径）打分，**零正文 I/O**；
之后每层的读盘量与上一层选出的数量成正比，而不是与 541 页成正比。
预算用尽即停止并记入 `dropped_by_budget`。

### 13.5 内存/显存控制

| 手段 | 实现 |
|---|---|
| 流式解析 | `iter_page_batches(doc, 24)` 按页批处理，批完即弃，不构建全量页/块列表 |
| 磁盘换内存 | 阶段一结果立刻 spill 到 `_blocks_spill.jsonl`，用完即删 |
| 偏移按需加载 | 正文落盘，索引只存 `(offset, length)`，取正文用 `seek` + 定长读 |
| 缓存淘汰 | `LRUCache(max_bytes)` 按**字节预算**淘汰最久未读，支持 `shrink_to()` 主动缩容 |
| 预算守卫 | `--memory-limit 512` 超限告警；`BudgetGuard` 可挂回调（如清缓存） |
| 三层监控 | `tracemalloc`（堆，必有）+ `psutil`（RSS）+ `torch.cuda`（显存，有 GPU 才有） |

内存监控用法（`app/rag/memory.py`）：

```python
mon = MemoryMonitor()
with mon.stage("phase1_parse_spill"):
    ...                       # 自动记录起止 RSS / 堆分配 / 增量 / 峰值
ok, msg = mon.check(512)      # 限额判定，不抛异常
print(mon.report())           # 阶段表格
print(mon.as_dict())          # 进日志/接口
```

### 13.6 541 页实跑数据

```
页数 542  正文块 11725  标题 1345  小节 1345
chunk 10219（L2 6641 / L1 1896 / L0 1345 / T0 337）  总 token 2,306,057
索引常驻 3810.7 KB   正文落盘 7654.5 KB
阶段                  耗时ms   RSS MB   增量    堆分配 MB
phase1_parse_spill     4518   144.9   +8.6      1.7
phase2_headings          20   147.5   +2.3      3.2
phase2_chunk_emit      3898   170.1  +22.6     12.4
峰值 RSS 185.8 MB | 峰值堆分配 19.4 MB | 限额 512MB 通过
```

内存归因（诚实拆解，别把基线算进优化成果）：裸进程 21.9MB → 导入子系统（含 PyMuPDF）136.3MB →
打开 541 页文档 137.9MB。所以**入库工作集约 48MB**，其中阶段二索引占约 23MB，
解析与切块阶段各自只涨 8-9MB——这正是"页批 + spill"的效果。

### 13.7 命令

```bash
python -m app.rag.cli --selftest                                  # 14/14（离线）
python -m app.rag.cli --make-sample book.pdf --pages 541          # 合成 541 页教材
python -m app.rag.cli --ingest book.pdf --out idx/ --page-batch 24 --memory-limit 512
python -m app.rag.cli --query "反向传播 链式法则" --index idx/ --budget 3000
```

### 13.8 实跑踩出的四个坑

1. **合成中文 PDF 必须用 `insert_font` 注册别名**（`F0/F1`），
   直接 `insert_text(fontfile=...)` 会被 base14 的 WinAnsi 编码表吃掉汉字，
   抽出全 `·`，标题识别 100% 失效。
2. **页眉判定只看位置会误杀页首标题**——第 5 章标题排在页首 63pt 处被当页眉丢弃，
   **整章在索引里消失**。必须加"字号不超过本页正文中位数"的护栏（已同步修 `app/parsing/pipeline.py`）。
3. **块内混排字号必须再切开**——PyMuPDF 会把"正文末行 + 紧随的大号标题"合成一个块，
   标题文本被正文污染成 `…应用部分。 第 1 章 线性代数基础`。
4. **LRU 里 `next(iter(OrderedDict))` 拿到的是键**，写成 `k, _ = ...` 会把字符串拆成字符。

### 一个必须知道的限制

无 `EMBEDDING_API_KEY` 时用的是**确定性哈希向量**（字符 bigram 签名），只能反映用词重叠，
**无法捕捉"同主题不同论文"的语义相关性**：离线跑 `doc_transformer` vs `doc_bert` 得 0.16（distinct），
换成真实 embedding 后通常落在 `related` 区间。结果里会以 `engine.embedding=local-hash` + `warnings` 明确标注，
线上请配置 `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL`。

---

## 14. 学术 RAG 混合检索：Dense + BM25 → RRF → Rerank

`app/retrieval/`（`python -m app.retrieval.cli --selftest` 7/7 通过；**零密钥可跑**）

### 14.1 链路

```
query
  │ plan_query()            意图 / 查询改写（≤4 条）/ 元数据过滤
  ├── dense  : 每条改写 → embedding → 向量库 ANN（带 filter）
  └── sparse : 每条改写 → BM25（先筛后打分）
        │ 每条改写 × 每一路 = 一个 run
     RRF 融合  score = Σ w / (k + rank)
        │ 过滤降级：严格过滤召回 < min_hits → 退到小节软过滤，绝不返回空
     Top-N → Rerank（cross-encoder；无 Key 用启发式特征）
        │
     Top-K（doc_id / section_path / 页码 / block_ids，可直接做引用卡）
```

为什么是"每条改写各算一个 run"而不是把改写拼成一条长 query：后者会让长尾改写淹没主查询信号。
为什么用 RRF 而不是分数加权：余弦 0~1 与 BM25 0~30 量纲不可比，归一化又对 top-k 截断敏感。

### 14.2 参数（`RetrievalConfig`）

| 参数 | 默认 | 理由 |
|---|---|---|
| `dense_top_k` / `sparse_top_k` | 50 / 50 | 召回要宽；RRF 后只留前几条，召回窄了 rerank 救不回来 |
| `fusion_top_k` | 40 | 送进 cross-encoder 的条数，延迟/效果平衡点 |
| `rerank_top_k` / `final_top_k` | 8 / 5 | 重排保留 8，进 LLM 上下文 5（约 2.5k token） |
| `rrf_k` | 60 | Cormack 2009 推荐；k∈[20,100] 不敏感，**换 k 不如换 top_k** |
| `min_hits` | 3 | 过滤后低于此值自动降级 |
| `weights` | dense 1.0 / sparse 1.0 | 型号/数值/术语密集的领域可提到 1.0 : 1.2 |

调参顺序：先扩 top_k 看召回够不够 → 再调 rrf_k → 最后才动 weights。

### 14.3 "细粒度超参 / 硬件环境"问题为什么需要改写 + 过滤

| 失败模式 | 现象 | 对策 |
|---|---|---|
| 问法与原文措辞不同 | 问"用了几张卡"，原文写 `8×A100 (80GB)` | 查询改写补规范说法（experimental setup / GPU memory …） |
| 答案藏在特定小节 | 超参/硬件多在 Experimental Setup、Appendix，向量上离问题句更远 | `section_type` 元数据过滤把候选压进正确小节 |
| 干扰项更像问题 | Related Work 复述别人的超参，向量距离反而更近 | 小节过滤 + 重排的实体/数值硬信号 |

实测（内置对抗语料，`llm:related` 里写满 `batch size 256 / 学习率 5e-4 / V100`）：
问"LLM 微调用了多少张 A100，学习率和 batch size 是多少"，
纯 dense 会把 Related Work 排进前列，加了小节过滤 + 重排后 `llm:setup` 稳定第 1，
且 trace 里如实标出 `filter_relaxed=true`（严格过滤只剩 1 条，自动降级）。

写入向量库时应带这几个契约字段（`types.py`）：`section_type`、`hyperparams`、`hardware`、
`n_gpus`、`has_table`、`year`。过滤条件 `Filter` 与后端无关，三种库各自翻译。

### 14.4 换向量库（业务代码一行不改）

```python
from app.retrieval.stores import build_store
from app.retrieval.pipeline import AcademicRetrievalPipeline

pipe = AcademicRetrievalPipeline(
    chunks,
    store=build_store("qdrant", url="http://localhost:6333"),   # 或 chroma / pgvector / memory
).build(recreate=False)          # 接生产库必须 False：True 会清空同名集合
res = pipe.search("用了多少张 A100？")
```

| 后端 | 适用 | 元数据过滤 |
|---|---|---|
| `memory` | 单测 / ≤5 万块 | 全功能，零依赖 |
| `qdrant` | 生产首选 | 最强；**过滤字段必须建 payload index**，`n_gpus`/`year` 用 integer、`has_table` 用 bool，否则 range 过滤静默失效 |
| `pgvector` | 已有 Postgres | 强，可直接 JOIN 文献表；HNSW + jsonb |
| `chroma` | 原型 | 弱：数组字段会 join 成 `"A100\|80GB"`；None 元数据必须剔除 |

### 14.5 命令

```bash
cd apps/api
python -m app.retrieval.cli --selftest                          # 7 项断言（零依赖）
python -m app.retrieval.cli                                     # 4 个示例问题
python -m app.retrieval.cli --query "ViT 在 CIFAR-100 上的准确率？" --top-k 3
python -m app.retrieval.cli --params                            # 参数建议表
python -m app.retrieval.cli --corpus rag-index --index idx/     # 接 §13 的入库产物
python -m app.retrieval.cli --store qdrant --no-recreate        # 接真实向量库
python -m app.retrieval.cli --query "..." --json                # 给上游 Agent 消费
```

`--json` 的输出含 `plan / engines / stages / hits`，`stages.filter_relaxed` 与
`hits[].ranks`（每条改写、每一路的名次）可直接进前端 trace 面板。

### 14.6 实跑时修掉的七个坑

1. **样例语料根本构建不起来**：4 个 `_mk()` 调用漏了必填的 `stype=`，`TypeError` 直接中断。
2. **重排的两个主特征永远是 0**：pipeline 里 run 名是 `dense:0 / sparse:1`（每条改写各一路），
   重排却按 `scores["dense"]` 取值 → 永远取不到。改为按前缀聚合（`rerank.run_scores`）。
3. **`A100` 被当成数值 `100`**：`extract_numbers` 无边界匹配，导致所有含 `V100/10000` 的块
   `number` 特征全为 1.0，重排信号被稀释。加了字母/连字符边界，且命中改为整词匹配。
4. **cross-encoder 必然走兜底**：请求 `top_n=min(top_k, n)` 却要求返回 `n` 条分数，条数不等就抛异常。
   改为请求全部文档、按 `index` 回填分数。
5. **Qdrant 的 `n_gpus` 用 keyword 建索引**，`range` 过滤失效；且 `client.search` 在 1.15+ 已删除。
   改为按字段真实类型建索引，优先 `query_points`、老客户端回退 `search`。
6. **Chroma 元数据带 `None`**（缺 year 的语料）写入报错；`_flat` 直接丢弃 None。
7. **"对比"被当成结果意图**："对比 A 与 B 的方法"会加 `results` 小节过滤，把方法段全筛掉。

### 14.7 边界

- **BM25 是进程内的**：`self.chunks` 全量常驻。百万级请换 Qdrant sparse vectors / OpenSearch
  （`bm25.py` 的接口只有 `build/search`，替换成本低）。
- **改写是规则的，不调 LLM**：可复现、零成本、可控；代价是抓不到跨语言同义改写。
  要上 LLM 改写，只需替换 `query.rewrite()`，`plan.rewrites` 协议不变。
- **过滤降级是刻意的**：宁可漏过滤，也不返回空结果；trace 里 `filter_relaxed` 明确标注，
  下游可据此决定是否提示"以下结果未严格限定实验设置小节"。
