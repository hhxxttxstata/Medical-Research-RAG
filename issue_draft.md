# 项目清理审计：冗余文件/接口、求职信息残留、代码 Bug

## 背景

仓库已公开为 **Medical-Research-RAG**。公开仓库应干净、专业、可信。本次由三个并行审计（冗余文件/接口、求职内容残留、代码 Bug 审查）完成，发现三类问题：大量冗余文件与遗留接口、个人求职（秋招/面试）材料残留、若干明显代码 Bug。本 issue 作为整改清单。

---

## 一、冗余文件与接口

### 1.1 冗余入口/启动脚本

| 文件 | 判定 | 说明 |
|------|------|------|
| `app.py` | 保留（去重） | 主服务入口；底部 `__main__` 的 uvicorn.run 与 `run.py` 重复，二选一 |
| `run.py` | 保留但重复 | 仅 uvicorn 启动包装，与 app.py 自带启动段重复 |
| `main.py` | 遗留 | CLI 交互入口，功能独立但 README/启动脚本均未提及，未文档化 |
| `gradio_app.py`（远程） | 废弃 | Gradio 前端（RAG+CTPA 诊断），唯一入口 `start.bat` 与后端 `/diagnosis/predict` 均已删 |
| `start.bat`（远程） | 废弃 | 旧一键启动脚本，已被 `start-local.bat` 取代 |
| `run_eval_topk8.py` | 废弃 | 临时 hack（monkey-patch 换目录跑 top_k=8），依赖旧 Chroma 管线 `eval/run_evaluation.py` |
| `Dockerfile.frontend`（远程） | 死文件 | 依赖已删的 `gradio_app.py` + `requirements-frontend.txt` + compose 的 gradio service |
| `evaluate.py` | 保留 | 现行评测入口（但 eval/ 下有 5 个评测入口并存，建议收敛） |

### 1.2 遗留/冗余 API 接口

远程 `app.py` 仍暴露 13 条路由，其中 3 条为遗留诊断接口（本地已删，push 后消失）：

- `POST /diagnosis/predict`、`GET /diagnosis/model`、`POST /chat-with-ct` — **遗留、与定位冲突**：CTPA 影像诊断端点，README 明言"不提供任何临床诊断建议"；无任何调用方（唯一调用方 gradio_app.py 已删）。依赖的 `src/diagnosis.py` 本地已删。
- `GET /knowledge-base/collections`、`GET /knowledge-base/tags` — 存疑：前端与脚本均无调用，仅 README 文档化，保留或删除二选一。
- `POST /query`（本地新增）— 新端点，调用方仅静态演示页 `frontend/public/agent-demo.html`；主前端未接入，建议主前端接入或明确去留。

**接口契约不一致**（非冗余但值得修）：前端 `use-chat.ts` 发送 `mode`(auto/rag/agent) 字段，后端 `ChatRequest` 无此字段被静默忽略；前端期望响应含 `agent_info`/`mode`/`session_id`，后端 `ChatResponse` 不返回。

### 1.3 冗余文档

- `docs/final_step15_16.md` — 历史实验记录（自注"彻底停止实验"），含"面试话术"章节 → 归档或删，至少删求职章节
- `docs/archive/`（10 个文件）— 9 个 step 过程文档 + `final_step.md` + `review_cross_doc_gold.html`(117KB) → 建议整体移出公开仓库（可留私有备份）
- `DOCKER_DEPLOY.md` — 与 README 快速开始部分重复（但含排障表），保留或并入 README
- `rebuild_log.txt`、`rebuild_log2.txt` — 构建日志（未提交），删除
- `models/README.md`（远程）— 随诊断模型栈废弃

### 1.4 远程残留文件（本地已删/归档，公开仓库仍在）

- 废弃代码：`gradio_app.py`、`start.bat`、`requirements-frontend.txt`、`Dockerfile.frontend`、`src/diagnosis.py`
- CTPA 训练栈：`teacher/` ×3（config_attention / resnet25d_attention / train_attention.py）
- 旧数据文档：`data/04~17` 等 10 个旧文档（系统部署手册/企业数据安全/员工办公规定/用户手册/RAG 架构/环境配置/AI 伦理/开发规范/部署说明/API 设计）
- 已归档但远程旧位置仍在：根目录 `final_step.md`、`step13.5-14.md`；`docs/` 下 7 个 step 文档；`scripts/` 下 27 个实验脚本（bad_case_*、chunk_ablation、step4~step15 等）
- `src/agentic_rag_v1_backup.py` — 备份冗余（仅被 holdout 脚本引用），随 holdout 归档或删

### 1.5 大文件/重复文件/构建产物入库

- **7 个 PDF 字节级重复**（约 31.6 MB）：`data/*.pdf` 与 `data/pe_literature/*.pdf` 完全一致（2108.09987v1、35tmi05-shin、978-3-658-41657-7、s00330-022-09071-0、s00330-024-10872-8、s12880-022-00763-z、s41598-021-95249-3）→ 删根目录副本
- `lucene_bm25_index/`（约 9 MB）— Whoosh 索引二进制被跟踪（.gitignore 声明忽略但已跟踪文件不受影响）→ `git rm --cached`
- `data-hold/`（4 个 PDF 约 25 MB）、`reference/` — .gitignore 声明"个人资料目录，不入库"但被跟踪（da890f2 引入）→ `git rm --cached`（reference 内含求职材料，见第二节）
- 仓库体积 92.4 MB，大头是 PDF 与索引二进制；`eval_results/` 9 个大 JSON 被跟踪（可接受，但建议定期清理）
- `src/index_builder.py` — 死代码（全仓无 import，且 import 即实例化），含"面试价值"docstring → 删除
- `eval/` 旧 Chroma 管线（run_evaluation / judge / report / visualize）— 仅被 run_eval_topk8.py 引用 → 随其归档
- `requirements.txt` vs `requirements-docker.txt` — 双清单漂移（缺 pymupdf/含 ragas）→ 统一为一份

---

## 二、秋招/求职信息残留（应删除）

> ⚠️ **隐私提示**：`scripts/_verify_resume_pdf.py` 硬编码了真实姓名与个人简历路径，绝不能提交。

### 2.1 整文件属于求职材料

| 文件 | 状态 | 内容 |
|------|------|------|
| `docs/interview_talk.md` | 已公开 | 开头即"用途：秋招面试"，90 秒简历版 + 8-10 分钟深挖版 + 数字速查表 + 措辞纪律 |
| `reference/industrial-rag-agent-cheatsheet.html` | 已公开（违反 .gitignore） | 含"面试要点"（"先讲优点……避免：过度批评自己的代码"），RAG 面试速查表 |
| `disadvantage.md` | 未提交 | "本项目的不足之处（诚实清单，面试前必读）"，通篇"面试官"话术 |
| `scripts/_verify_resume_pdf.py` | 未提交 | 硬编码 `C:\Users\tata\Desktop\latex简历\...\谢子拓-个人简历-AI应用开发.pdf`（**真实姓名**）→ 从暂存区移除并删除 |
| `scripts/demo_decision_loop.py` | 已提交 | 无 LLM/索引的决策路径演示 stub（面试演示素材） |

### 2.2 局部含求职话术（已公开，建议删除对应章节/注释）

- `docs/final_step15_16.md` — "面试话术"、"面试可讲"、"面试讲稿"章节
- `docs/archive/` 7 个文档（final_step / step10_agentic_rag / step11_ablation / step12_benchmark / step13_v2 / step13.5-14 / step105_policy_qualification）— "面试表达/面试亮点（秋招表达）/回答面试官"章节；`final_step.md` 甚至出现生造词 "Interviewization"
- `src/` 11 个模块 docstring："面试价值/面试亮点/面试讲解点"（auth.py 含"秋招阶段"、lucene_bm25.py 含"为什么不用 Elasticsearch——面试演示项目"、cost_aware_agentic_rag.py 含"面试话术对应"等）
- `eval/metrics.py`（"面试可引用"）、`eval/report.py`（生成报告模板含"四、面试中可以讲的技术亮点"）、`eval/visualize.py`（"秋招项目展示"）
- `scripts/archive/demo_cases.py`（"面试展示素材"）、`scripts/archive/step15_grounded_eval.py`（"面试可讲"）

### 2.3 其他

- 求职色彩提交 3 条（已 push，改写需 `git filter-repo` + force push）：`039044e`、`aac628a`（"chore: 秋招前整改"）、`1120b1c`（"面试讲法"）——若不动历史，后续提交不再出现求职字样
- `.claude/CLAUDE.md` — 通篇"面向秋招求职展示"，未被跟踪（.gitignore 已忽略）→ 删除文件或保持忽略
- `.gitignore` 已声明忽略 `reference/`、`data-hold/`，但 5 个文件仍被跟踪 → `git rm --cached`
- 无害命中（无需处理）：`review_cross_doc_gold.html` 的"岗位"（知识库样本正文）、`DOCKER_DEPLOY.md` 的"内推"（"国内推荐"误命中）、CV/CVAT（专业缩写）、"包装"(wrapper)

---

## 三、RAG 系统明显 Bug

> 来源：app.py、src/ 全模块、eval/ 评测脚本只读代码审查。共报告 4 高、8 中、8 低。

### 🔴 高危

1. **`app.py:302-319` 上传接口路径穿越（任意文件写/删 + 无大小限制）** — filename 未做 basename 净化直接 `os.path.join`，`..\`/绝对路径可任意写文件；解析失败时 `os.remove` 沿遍历路径删文件；无大小限制 → 内存 DoS。
2. **`src/milvus_store.py:278-290` delete_collection 后 `_loaded_once` 未复位** — `_ensure_loaded`(150-164) 依赖该标志；Milvus Standalone（docker-compose 默认）重建知识库后新集合**永不 load**，检索静默返回 `[]`，全量拒答直到重启。
3. **`src/prompt_injection.py` 中文注入完全绕过** — 全部 15 条规则均为英文正则，中文注入（如"忽略之前所有指令"）无法拦截；且 `re.compile(r"override")` 语义过宽会误杀正常英文问题。
4. **`src/retriever.py:333-357` 实例属性 `_out_of_domain` 跨请求污染** — `rag_pipeline.py:233` 跨请求读取；4 线程池并发（`enable_rewrite=True`）下 A 请求 OOD 状态污染 B 请求 → 随机误拒答（默认不触发，条件竞态）。

### 🟡 中危

5. **`app.py:431` `/chat/stream` 无认证**（无 `Depends(verify_api_key)`），未授权可无限消耗 LLM；`auth.py:48-50` API_KEY 未配置时全站无认证。
6. **`src/document_processor.py:1048` parent_id 错配** — `chunk_id//3` 与实际 `_make_parent_chunks` 从 1 递增的编号错配（small 1-2 映射到不存在的 parent_0）→ Small-to-Big 静默失效。
7. **`src/retriever.py:276-279` pop 掉 `_retriever` 字段** → `generator.py:75` `has_bm25_support` 服务链路恒 False，"BM25 双重确认放行"逻辑永不生效。
8. **`src/generator.py:706-714` `_parse_json_response` 未校验 JSON 顶层类型** — LLM 输出数组时 `data.get` 抛 AttributeError 冒泡 → 整条查询报"系统错误"而非重试。
9. **`eval/run_evaluation.py:100-105` 传 `RAGPipeline` 不存在的 `persist_dir` 参数** → TypeError 启动即崩（残留 ChromaDB API）。
10. **`eval/run_full_pipeline_eval.py:52-56` expected_hit 子串匹配恒 False**（expected_doc 带 `.md` vs filename 无扩展名）→ Hit Rate 系统性偏低，与 MRR/NDCG 口径自相矛盾。
11. **`src/logger.py:71-138` 多线程无锁改 `_stats` + 每次整文件写 stats.json** → 计数丢失、文件并发写损坏。
12. **`src/cache.py:387-391` invalidate_all 只清内存不清 Redis** → 重建知识库后 30-60 分钟内返回陈旧检索/回答。

### 🟢 低危

- `generator.py:884-910` 死代码；`generator.py:622-625` 缺字段 JSON 被置 valid
- `knowledge_base.py:118/163` 调用 MilvusStore 不存在的方法；`text_splitter.py:1024/1032` 尾部碎块静默丢弃
- `milvus_store.py:303` limit=10000 截断 BM25 语料；`lucene_bm25.py:184-210` searcher 异常不 close
- `watcher.py:125-155` 用旧版 split_document 与初始索引口径不一致；`cache.py:79-84` LRUCache 锁懒初始化竞态

> **总体结论**：系统有明显 bug，风险集中在 ①安全（路径穿越 + 中文注入无效 + 流式接口无认证）、②数据可用性（Milvus rebuild 后检索静默失效 + Redis 缓存不失效）、③并发正确性（retriever OOD 标志、logger 统计、LRU 锁）。建议优先修 #1/#2/#3/#4。

---

## 四、建议处理优先级

- **P0（公开前必修）**：`scripts/_verify_resume_pdf.py`（隐私，真实姓名）；全部求职材料（interview_talk.md、disadvantage.md、reference/cheatsheet、src/eval 内"面试价值"话术）；远程遗留诊断栈（/diagnosis/*、/chat-with-ct、src/diagnosis.py、gradio_app.py、teacher/）；高危 bug #1 路径穿越
- **P1**：bug #2/#3/#4；7 个重复 PDF（31.6 MB）；`lucene_bm25_index`/`data-hold`/`reference` `git rm --cached`；run_eval_topk8 + 旧 Chroma 管线；rebuild_log×2；docs/archive 与 scripts/archive 移出公开仓库
- **P2**：入口收敛（app.py/run.py/main.py 三选一或文档化）；eval 入口收敛（5 个并存）；requirements 统一；/knowledge-base/*、/query 去留定论；接口契约对齐（mode/agent_info/session_id）；CI 卡住排查（CI workflow 在最近提交上长期 in_progress）
