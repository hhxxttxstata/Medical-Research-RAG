"""
大模型生成模块
基于检索内容生成带有引用的回答
支持结构化输出、拒答机制、引用溯源、结果验证
"""

import json
import os
import random
import re
import sys
import time
from typing import Any

from dotenv import load_dotenv

from .embeddings import is_mostly_english

# Windows GBK 编码兼容
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

load_dotenv()


# ──────────────────────────────────────────────
#  一、多因子相关性判断
# ──────────────────────────────────────────────


def compute_relevance(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    top1_threshold: float = 0.35,
    avg_threshold: float = 0.25,
    overlap_threshold: float = 0.03,
) -> dict[str, Any]:
    """
    多因子相关性评分，判断检索内容是否与用户问题相关。

    中英文自适应：
      - 中文 query → 字符级 2-gram 重叠
      - 英文 query → 词级 word 重叠

    返回:
        {
            "is_relevant": bool,       # 是否相关
            "top1_score": float,       # 最高分
            "avg_score": float,        # 平均分
            "overlap": float,          # 重叠率
            "reason": str              # 判断理由
        }
    """
    if not retrieved_chunks:
        return {
            "is_relevant": False,
            "top1_score": 0.0,
            "avg_score": 0.0,
            "overlap": 0.0,
            "reason": "检索结果为空",
        }

    # ── 1. 语义分数：向量相似度（与现有阈值适配） ──
    top1_score = retrieved_chunks[0].get("_vector_score")
    if top1_score is None:
        top1_score = retrieved_chunks[0]["score"]
    scores: list[float] = []
    for c in retrieved_chunks:
        vs = c.get("_vector_score")
        scores.append(vs if vs is not None else c["score"])
    avg_score = sum(scores) / len(retrieved_chunks)

    # ── 2. BM25 辅助信号：前 3 个 chunk 是否有 hybrid 检索的双重确认 ──
    has_bm25_support = any(c.get("_retriever") == "hybrid" for c in retrieved_chunks[:3])

    # ── 3. 根据 query 语言选择不同的重叠算法 ──
    query_is_en = is_mostly_english(query)

    if query_is_en:
        # 英文：字符 3-gram（比 word-level 更鲁棒，不受词形变化和停用词影响）
        q = query.lower()
        query_ngrams = set(q[i : i + 3] for i in range(len(q) - 2))
        overlap_scores = []
        for chunk in retrieved_chunks:
            chunk_text = chunk["text"][:500].lower()
            chunk_ngrams = set(chunk_text[i : i + 3] for i in range(len(chunk_text) - 2))
            if query_ngrams:
                intersection = query_ngrams & chunk_ngrams
                overlap = len(intersection) / len(query_ngrams) if query_ngrams else 0
                overlap_scores.append(overlap)
        overlap_required = 0.01  # 英文 3-gram 更稀疏，适当降低门槛
    else:
        # 中文：字符级 2-gram 重叠（不变）
        query_bigrams = set(query[i : i + 2] for i in range(len(query) - 1))
        overlap_scores = []
        for chunk in retrieved_chunks:
            chunk_text = chunk["text"][:500]
            chunk_bigrams = set(chunk_text[i : i + 2] for i in range(len(chunk_text) - 1))
            if query_bigrams:
                intersection = query_bigrams & chunk_bigrams
                overlap = len(intersection) / len(query_bigrams) if query_bigrams else 0
                overlap_scores.append(overlap)
        overlap_required = 0.03

    avg_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0

    # ── 4. 综合判断 ──
    # 语义分权重 0.6，文本重叠权重 0.4
    combined_score = top1_score * 0.6 + avg_overlap * 0.4
    combined_threshold = 0.25
    # BM25 双重确认 = 两种独立检索器都认为相关 → 加分
    bm25_bonus = 0.05 if has_bm25_support else 0.0

    if top1_score < 0.10 and avg_overlap < 0.02:
        is_relevant = False
        reason = (
            f"语义分过低(top1={top1_score:.3f})且文本几乎无重叠(overlap={avg_overlap:.3f})，综合分={combined_score:.3f}"
        )
    elif avg_overlap < overlap_required and not has_bm25_support:
        # 文本重叠过低 → 不相关（但如果有 BM25 双重确认，则放行）
        is_relevant = False
        reason = (
            f"文本重叠不足(overlap={avg_overlap:.3f}<{overlap_required})，"
            f"语义分top1={top1_score:.3f} 综合分={combined_score:.3f}"
        )
    elif combined_score + bm25_bonus >= combined_threshold:
        is_relevant = True
        reason = f"综合分达标(={combined_score:.3f})，语义分top1={top1_score:.3f}+重叠{avg_overlap:.3f}"
        if has_bm25_support:
            reason += "+BM25确认"
    elif top1_score > 0.60 and avg_overlap > 0.05:
        is_relevant = True
        reason = f"语义分较高(top1={top1_score:.3f})且有一定重叠({avg_overlap:.3f})"
    else:
        is_relevant = False
        reason = (
            f"综合分未达标(={combined_score:.3f}<{combined_threshold})，"
            f"语义分top1={top1_score:.3f} 重叠={avg_overlap:.3f}"
        )
        if has_bm25_support:
            reason += "（有BM25支撑但综合分仍不足）"

    return {
        "is_relevant": is_relevant,
        "top1_score": round(top1_score, 4),
        "avg_score": round(avg_score, 4),
        "overlap": round(avg_overlap, 4),
        "reason": reason,
    }


# ──────────────────────────────────────────────
#  二、Prompt 构建
# ──────────────────────────────────────────────


def _cacheable_system_message(is_stream: bool = False) -> dict[str, str]:
    """系统消息——纯静态，包含结构化工单 + Few-shot"""
    return {
        "role": "system",
        "content": (
            "你是一个专业的医学AI知识助手，擅长结合检索到的参考文档回答医学问题。\n"
            "\n"
            "## 核心规则\n"
            "1. 严格基于参考文档回答，每句结论必须引用编号 [N]\n"
            "2. 严禁虚构引用——不要编造参考文档中不存在的编号\n"
            "3. 如果参考文档不包含足够信息，如实说明，不要编造\n"
            "4. 输出严格遵循下方 JSON 结构，不要添加多余字段\n"
            "\n"
            "## 输出格式\n"
            "必须返回合法 JSON，包含以下字段：\n"
            "```json\n"
            "{\n"
            '  "diagnosis": "明确的结论性判断（一句话概括）",\n'
            '  "confidence": "高/中/低",\n'
            '  "evidence": ["基于参考文档的具体依据，每一条末尾标注引用 [1][2]"],\n'
            '  "suggestion": "基于现有信息的建议（如需要进一步检查、需结合临床、或确认事项）",\n'
            '  "sources": ["引用来源文件名列表"],\n'
            '  "uncertainty": "如果信息不充分，说明缺失了什么" \n'
            "}\n"
            "```\n"
            "\n"
            "## 示例\n"
            "用户问题：CTPA检查对肺栓塞诊断有什么价值？\n"
            "```json\n"
            "{\n"
            '  "diagnosis": "CTPA（CT肺动脉造影）是诊断肺栓塞的金标准影像学检查方法",\n'
            '  "confidence": "高",\n'
            '  "evidence": [\n'
            '    "CTPA可清晰显示肺动脉主干及分支内的血栓，直接征象为血管内充盈缺损 [1]",\n'
            '    "CTPA对段以上肺动脉栓塞的敏感性达94%，特异性达96% [2]"\n'
            "  ],\n"
            '  "suggestion": "如CTPA结果阴性但临床高度怀疑PE，建议结合D-二聚体检测及下肢静脉超声进一步排除",\n'
            '  "sources": ["肺栓塞影像诊断指南", "CTPA临床应用专家共识"],\n'
            '  "uncertainty": ""\n'
            "}\n"
            "```\n"
            "\n"
            "用户问题：肺栓塞的治疗方案是什么？\n"
            "```json\n"
            "{\n"
            '  "diagnosis": "肺栓塞的急性期治疗以抗凝为基石，重症患者需溶栓治疗",\n'
            '  "confidence": "高",\n'
            '  "evidence": [\n'
            '    "血流动力学稳定的急性肺栓塞患者，推荐初始抗凝治疗，普通肝素或低分子肝素 [1]",\n'
            '    "高危PE（伴休克或低血压）患者，如无禁忌证，推荐溶栓治疗 [2]",\n'
            '    "溶栓后出血风险较高，需严格评估禁忌证 [3]"\n'
            "  ],\n"
            '  "suggestion": "建议根据sPESI评分进行危险分层后制定个体化治疗方案",\n'
            '  "sources": ["肺栓塞诊断与治疗指南", "急性肺栓塞抗栓治疗共识"],\n'
            '  "uncertainty": "具体用药剂量和疗程需结合患者体重、肾功能及出血风险评估" \n'
            "}\n"
            "```\n"
            "\n"
            "## 回答要求\n"
            '- uncertainty 字段不得为空——至少写"无"\n'
            "- 每个 evidence 条目必须有至少一个 [N] 引用标记\n"
            "- 不要复述无关背景，只输出与问题直接相关的信息"
        ),
    }


def build_rag_prompt(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    relevance: dict[str, Any] | None = None,
    use_prefix_cache: bool = True,
    diagnosis_result: dict | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    """
    构建 RAG Prompt（Prompt Caching 优化版）

    结构：
      messages = [
        system  ← 纯静态，被所有请求共享前缀
        user    ← 稳定前缀（参考文档说明） + 动态段（chunks + query）
      ]

    Prompt Caching 命中场景：
      - 相同 system prompt → 跨 session 共享 cache
      - 相同 user prefix → 同一会话中连续请求共享

    返回:
        (messages_list, source_map, relevance_info)
        messages_list 可直接传给 OpenAI chat API
    """
    if relevance is None:
        relevance = compute_relevance(query, retrieved_chunks)

    # ── 动态部分放在 user message 末尾 ──
    context_parts = []
    source_map: dict[str, dict[str, Any]] = {}

    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk["metadata"]
        filename = meta.get("filename", "未知")
        section_title = meta.get("section_title", meta.get("heading", ""))
        parent_content = meta.get("parent_content", "")

        ref_parts = [f"[{i}] 文件: {filename}"]
        if section_title:
            ref_parts.append(f"章节: {section_title}")

        source_tag = " | ".join(ref_parts)

        chunk_content = chunk["text"]
        if parent_content and len(parent_content) > len(chunk_content) * 1.5:
            chunk_content = f"{chunk_content}\n\n> 📖 **完整上下文（本节）**\n{parent_content}"

        context_parts.append(f"{source_tag}\n{chunk_content}\n")

        source_map[str(i)] = {
            "filename": filename,
            "page": str(meta.get("page", "")),
            "section": section_title,
            "score": round(chunk["score"], 3),
            "text_preview": chunk["text"][:100],
        }

    context = "\n".join(context_parts)

    # ── 注入 CTPA 诊断结果（当用户上传了影像时） ──
    diagnosis_block = ""
    if diagnosis_result and diagnosis_result.get("success"):
        prob = diagnosis_result["probability"]
        pred = "阳性" if diagnosis_result["prediction"] else "阴性"
        risk_level = "高风险" if prob >= 0.9 else "中风险" if prob >= 0.7 else "低风险" if prob >= 0.5 else "阴性"
        num_slabs = diagnosis_result.get("num_slabs", 0)
        inference_time = diagnosis_result.get("inference_time", 0)

        diagnosis_block = (
            "\n## CTPA 影像辅助诊断结果\n"
            f"- 肺栓塞概率: {prob:.2%}\n"
            f"- 分类: {pred}（{risk_level}）\n"
            f"- 分析切片数: {num_slabs}\n"
            f"- 模型推理耗时: {inference_time}s\n"
        )
        attn = diagnosis_result.get("attention_weights", [])
        if attn:
            max_attn = max(attn)
            diagnosis_block += f"- 最大注意力权重: {max_attn:.4f}\n"

        # 如果存在可视化 base64，不塞进 prompt（太长），只记录
        if diagnosis_result.get("visualization"):
            diagnosis_block += "- 可视化图像已独立返回，请参考可视化数据\n"

        diagnosis_block += "\n【注意】以上结果来自 AI 辅助诊断模型，仅供参考，需结合临床判断。\n"

    # 稳定前缀：跨请求不变（可命中 Prompt Cache）
    stable_prefix = "请根据以下参考文档回答用户问题。\n## 参考文档\n"

    # 动态后缀：随查询变化
    dynamic_suffix = f"{context if context else '（无参考文档）'}\n{diagnosis_block}\n## 用户问题\n{query}\n\n## 回答\n"

    user_content = stable_prefix + dynamic_suffix if use_prefix_cache else f"{stable_prefix}{dynamic_suffix}"

    messages = [
        _cacheable_system_message(),
        {"role": "user", "content": user_content},
    ]

    return messages, source_map, relevance


def build_quick_prompt(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    relevance: dict[str, Any] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    """构建快速回答 Prompt——只输出结论，≤150字，同样 cache 友好"""
    if relevance is None:
        relevance = compute_relevance(query, retrieved_chunks)

    context_parts = []
    source_map: dict[str, dict[str, Any]] = {}
    for i, chunk in enumerate(retrieved_chunks[:2], 1):
        meta = chunk["metadata"]
        source_tag = f"[{i}] 文件: {meta.get('filename', '未知')}"
        if meta.get("section_title"):
            source_tag += f" | 章节: {meta['section_title']}"
        context_parts.append(f"{source_tag}\n{chunk['text']}\n")
        source_map[str(i)] = {"filename": meta.get("filename", ""), "text_preview": chunk["text"][:100]}

    context = "\n".join(context_parts)

    stable_prefix = "请用最简洁的方式回答下面的医学问题。\n## 参考文档\n"
    dynamic_suffix = f"{context if context else '（无）'}\n\n## 问题\n{query}\n\n## 快速回答\n"

    user_content = stable_prefix + dynamic_suffix

    messages = [
        _cacheable_system_message(),
        {"role": "user", "content": user_content},
    ]

    return messages, source_map, relevance


def build_verbose_prompt(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    relevance: dict[str, Any] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    """构建详细回答 Prompt——复用 build_rag_prompt"""
    return build_rag_prompt(query, retrieved_chunks, relevance)


# ──────────────────────────────────────────────
#  三、引用验证（后处理）
# ──────────────────────────────────────────────


def validate_citations(answer: str, source_map: dict[str, Any]) -> dict[str, Any]:
    """
    验证回答中的引用编号是否都在 source_map 范围内。
    返回验证结果和修复建议。
    """
    # 提取所有 [N] 格式的引用
    cited = set(re.findall(r"\[(\d+)\]", answer))
    available = set(source_map.keys())
    cited_valid = cited & available
    cited_invalid = cited - available
    missing = available - cited_valid  # 可用但未引用的来源

    has_invalid = len(cited_invalid) > 0
    has_unused = len(missing) > 0 and len(available) > 0

    return {
        "cited_valid": sorted(cited_valid, key=int),
        "cited_invalid": sorted(cited_invalid, key=int),
        "unused": sorted(missing, key=int),
        "has_invalid_citations": has_invalid,
        "has_unused_sources": has_unused,
        "total_cited": len(cited_valid),
        "total_available": len(available),
    }


# ──────────────────────────────────────────────
#  四、生成器实现
# ──────────────────────────────────────────────


class LLMGenerator:
    """大模型生成器

    内置能力：
      - 指数退避重试（可配置重试次数 + 基础延迟）
      - Circuit Breaker（连续 N 次失败后熔断 T 秒）
      - 多 API 提供者自动切换
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,  # 降低生成冗余，减少幻觉
        max_tokens: int = 2048,
        # 指数退避重试
        retry_max_attempts: int = 3,
        retry_base_delay: float = 1.0,
        # Circuit Breaker
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 30.0,
    ):
        # 按优先级读取 API 配置：deepseek-chat
        self.api_key = (
            api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("SILICON_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        )
        self.base_url = (
            base_url
            or os.getenv("DEEPSEEK_BASE_URL")
            or os.getenv("SILICON_BASE_URL")
            or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        self.model = (
            model
            or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            or os.getenv("SILICON_MODEL")
            or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        )
        self.temperature = temperature
        self.max_tokens = max_tokens if max_tokens else 4096  # 默认4096，适应混合回答
        # 默认回答模式：简洁 400-600 tokens
        self.default_max_tokens = 600
        self.verbose_max_tokens = 1200

        # 指数退避重试配置
        self.retry_max_attempts = retry_max_attempts
        self.retry_base_delay = retry_base_delay

        # Circuit Breaker 状态
        self.cb_threshold = circuit_breaker_threshold
        self.cb_timeout = circuit_breaker_timeout
        self._cb_state = "closed"  # "closed" | "open" | "half_open"
        self._cb_failures = 0
        self._cb_open_until = 0.0

        # Step 14 Observability：按调用类型分类计数（生成/评分/拆解/策略）
        self._calls = {"generation": 0, "grader": 0, "decompose": 0, "policy": 0, "total": 0}

    def call_counts(self) -> dict[str, int]:
        """返回按类型统计的 LLM 调用次数（Step 14 成本观测）"""
        return dict(self._calls)

    def _track_call(self, call_type: str) -> None:
        if call_type in self._calls:
            self._calls[call_type] += 1
            self._calls["total"] += 1

    def _is_valid_api_key(self, key: str | None) -> bool:
        """检查 API Key 是否有效（排除空值和占位符）"""
        if not key or key == "":
            return False
        placeholders = ["your-api-key-here", "sk-your-", "your_"]
        return not any(p in key.lower() for p in placeholders)

    def _detect_api_type(self) -> str:
        """检测使用的 API 类型"""
        base_url = self.base_url or ""
        if not self._is_valid_api_key(self.api_key):
            return "ollama"
        if "deepseek" in base_url.lower():
            return "deepseek"
        if "silicon" in base_url.lower():
            return "siliconflow"
        if "openai" in base_url.lower():
            return "openai"
        return "openai_compatible"

    # ── Circuit Breaker ──────────────────────────────────

    def _check_circuit_breaker(self) -> bool:
        """检查熔断器状态。True = 允许请求，False = 熔断中"""
        if self._cb_state == "open":
            if time.monotonic() >= self._cb_open_until:
                self._cb_state = "half_open"
                print("  🔌 Circuit Breaker: OPEN → HALF_OPEN （试探放行）")
                return True
            print("  🔌 Circuit Breaker: OPEN 中，快速拒绝")
            return False
        return True

    def _record_success(self):
        """调用成功 → 关闭熔断器"""
        self._cb_failures = 0
        if self._cb_state == "half_open":
            self._cb_state = "closed"
            print("  🔌 Circuit Breaker: HALF_OPEN → CLOSED （恢复）")

    def _record_failure(self):
        """调用失败 → 计数，达到阈值则熔断"""
        self._cb_failures += 1
        if self._cb_failures >= self.cb_threshold:
            self._cb_state = "open"
            self._cb_open_until = time.monotonic() + self.cb_timeout
            print(f"  🔌 Circuit Breaker: CLOSED → OPEN （连续{self._cb_failures}次失败，熔断{self.cb_timeout}s）")

    # ── 指数退避重试 ──────────────────────────────────

    def _with_retry(self, fn, *args, **kwargs):
        """带指数退避重试的执行调用

        对网络/超时/5xx 类错误重试。4xx 不重试（请求本身有问题）。
        """
        last_exception = None
        for attempt in range(self.retry_max_attempts):
            try:
                result = fn(*args, **kwargs)
                self._record_success()
                return result
            except Exception as e:
                last_exception = e
                # 4xx 不重试
                err_str = str(e).lower()
                if any(code in err_str for code in ["401", "403", "404", "400", "422"]):
                    if "401" in err_str or "unauthorized" in err_str or "api key" in err_str:
                        print("  ⚠️ API Key 无效，不重试")
                    self._record_failure()
                    raise

                self._record_failure()

                if attempt < self.retry_max_attempts - 1:
                    # 指数退避 + jitter
                    delay = self.retry_base_delay * (2**attempt) * (0.8 + 0.4 * random.random())
                    print(f"  🔄 重试 #{attempt + 1}/{self.retry_max_attempts} ({delay:.1f}s)...")
                    time.sleep(delay)

        raise last_exception

    def generate_stream(
        self,
        prompt_and_source: tuple,
        validate: bool = True,
    ) -> str:
        """
        调用大模型生成回答

        参数:
            prompt_and_source: (messages, source_map, relevance_info) 元组
                messages = [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
                (兼容旧版 (str, dict, dict) 格式)
            validate: 是否进行引用验证后处理

        返回:
            生成的回答文本
        """
        if isinstance(prompt_and_source[0], str):
            # 旧版格式 (prompt_str, source_map, relevance) → 兼容
            prompt, source_map, relevance = prompt_and_source
            messages = [
                {"role": "system", "content": "你是一个专业的 AI 知识助手。"},
                {"role": "user", "content": prompt},
            ]
        else:
            messages, source_map, relevance = prompt_and_source

        has_valid_key = self._is_valid_api_key(self.api_key)

        try:
            if has_valid_key:
                answer = self._call_openai_with_messages(messages)
            else:
                try:
                    answer = self._call_ollama_with_messages(messages)
                except Exception:
                    answer = self._fallback_structured_response2(source_map, messages)

            if validate and source_map:
                citation_result = validate_citations(answer, source_map)
                if citation_result["has_invalid_citations"]:
                    invalid = citation_result["cited_invalid"]
                    warning = (
                        f"\n\n⚠️ **引用验证提示**：回答中引用了不存在的来源编号 {invalid}，"
                        f"请以实际提供的【参考文档】编号为准。"
                    )
                    answer += warning

            return answer
        except Exception as e:
            raise

    def generate(
        self,
        prompt_and_source: tuple,
        validate: bool = True,
    ) -> str:
        """
        调用大模型生成回答（混合模式）

        参数:
            prompt_and_source: (prompt_text, source_map, relevance_info) 元组
            validate: 是否进行引用验证后处理

        返回:
            生成的回答文本
        """
        return self.generate_stream(prompt_and_source, validate)

    # ── 结构化生成入口（替代旧 generate，加 Self-reflection + Citation enforcement） ─────

    def generate_structured(
        self,
        prompt_and_source: tuple,
        self_reflect: bool = False,
    ) -> dict:
        """生成结构化 JSON 回答

        流程：
          1. 生成 JSON          2. 验证引用
          3. （可选）Self-reflection：补全缺失字段
          4. 引用不合法最多重试 1 次

        Returns {"structured": dict, "raw": str, "valid": bool}
        """
        self._track_call("generation")
        messages, source_map, relevance = prompt_and_source

        # 第一次生成
        raw = self.generate_stream((messages, source_map, relevance))
        structured, valid = self._parse_json_response(raw, source_map)

        # Citation enforcement：无效引用时重试一次
        if not valid:
            fix_prompt = (
                "警告：上轮输出中包含无效引用编号或缺失引用。\n"
                "请重新输出 JSON，确保每个 evidence 条目末尾有 [N] 标记，"
                "且 N 来自下方参考文档中的编号。\n\n"
                "可用编号: " + ", ".join(source_map.keys())
            )
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": fix_prompt},
            ]
            raw = self.generate_stream((messages, source_map, relevance))
            structured, valid = self._parse_json_response(raw, source_map)

        # Self-reflection：补全 completeness
        if self_reflect and valid:
            missing = [k for k in ["diagnosis", "evidence", "suggestion", "uncertainty"] if not structured.get(k)]
            if missing or len(structured.get("evidence", [])) < 2:
                reflect_prompt = (
                    "请基于参考文档补充以下缺失内容，只输出补充的 JSON 字段，不要重复已有内容。\n"
                    "缺失字段: " + ", ".join(missing)
                    if missing
                    else "evidence 不足 2 条，请补充一条。\n"
                )
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": reflect_prompt},
                ]
                raw2 = self.generate_stream((messages, source_map, relevance))
                try:
                    extra = json.loads(raw2.strip().removeprefix("```json").removesuffix("```").strip())
                    structured.update(extra)
                except Exception:
                    pass

        return {"structured": structured, "raw": raw, "valid": valid}

    @staticmethod
    def _parse_json_response(raw: str, source_map: dict) -> tuple[dict, bool]:
        """解析 LLM 输出为 JSON，验证引用"""
        try:
            text = raw.strip()
            # 去掉 markdown code fence（可能在开头或中间）
            fence = "```"
            if text.startswith(fence):
                # remove leading ``` line
                text = text.split("\n", 1)[1] if "\n" in text else text[len(fence) :]
                # remove trailing ``` and anything after it
                if fence in text:
                    text = text[: text.index(fence)].strip()
            elif fence in text:
                # 可能 AI 回答了文字再输出 JSON
                idx = text.index(fence)
                text = text[idx + 3 :].strip()
                end = text.index(fence) if fence in text else len(text)
                text = text[:end].strip()
            if text.startswith("json"):
                text = text[4:].strip()
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}, False

        # 验证必备字段
        for key in ["diagnosis", "evidence", "confidence"]:
            if key not in data:
                return data, False

        # 验证引用：每个 evidence 必须有 [N]
        valid_range = set(source_map.keys())
        for ev in data.get("evidence", []):
            cited = set(re.findall(r"\[(\d+)\]", ev))
            if cited and not cited.issubset(valid_range):
                return data, False
        return data, True

    # ── Chat 接口（带 Circuit Breaker + 重试） ──────────

    def _get_chat_fn(self):
        """返回适合当前配置的纯文本聊天函数"""
        if self._is_valid_api_key(self.api_key):
            return self._call_openai_chat
        return self._call_ollama_chat

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int = 2048,
        call_type: str | None = None,
    ) -> str:
        """直接传入 messages 列表，适用于多轮对话场景

        call_type: 调用分类（generation/grader/decompose/policy），用于 Step 14 成本观测。
        """
        if call_type:
            self._track_call(call_type)
        start = time.monotonic()

        # Circuit Breaker 检查
        if not self._check_circuit_breaker():
            return self._fallback_text("API 熔断中")

        try:
            fn = self._get_chat_fn()
            result = self._with_retry(fn, messages, temperature, max_tokens)
        except Exception:
            result = self._fallback_text("API 暂时不可用")

        return result

    def _call_openai_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int = 2048,
    ) -> str:
        """通用 OpenAI 兼容 API 调用（接受任意 messages 列表）"""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=30)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens,
            "timeout": 30,
        }
        # DeepSeek 非思考模式保险：禁用 reasoning 输出（deepseek-chat 天然非思考，此参数仅兼容 v4 系列）
        if "deepseek" in (self.base_url or "").lower():
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def _call_openai_chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int = 2048,
    ):
        """流式调用 OpenAI 兼容 API，逐 chunk yield token"""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=30)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "timeout": 30,
        }
        if "deepseek" in (self.base_url or "").lower():
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        response = client.chat.completions.create(**kwargs)
        for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    def _call_ollama_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> str:
        """通用 Ollama 调用（接受任意 messages 列表）"""
        import requests

        # Ollama chat API 格式：{"model": "...", "messages": [...], "stream": false}
        response = requests.post(
            url=f"{os.getenv('OLLAMA_URL', 'http://localhost:11434')}/api/chat",
            json={
                "model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature if temperature is not None else self.temperature,
                },
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def _fallback_text(self, reason: str) -> str:
        """当 LLM 不可用时的兜底输出"""
        return f"[LLM 不可用: {reason}]"

    def _call_openai_with_messages(self, messages: list[dict[str, str]]) -> str:
        """调用 OpenAI 兼容 API（messages 列表格式，不重新包装）

        Prompt Caching 兼容——直接透传 build_rag_prompt 的 messages 结构，
        system message 作为 prefix cache 的 key。
        """
        wants_verbose = any(
            kw in (messages[-1]["content"][:200] if messages else "") for kw in ["详细", "详细解释", "展开", "具体说明"]
        )
        mt = self.verbose_max_tokens if wants_verbose else self.default_max_tokens
        return self._call_openai_chat(messages=messages, max_tokens=mt)

    def _call_ollama_with_messages(self, messages: list[dict[str, str]]) -> str:
        """调用本地 Ollama（messages 列表格式）"""
        import requests

        response = requests.post(
            url=f"{os.getenv('OLLAMA_URL', 'http://localhost:11434')}/api/chat",
            json={
                "model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
                "messages": messages,
                "stream": False,
                "options": {"temperature": self.temperature},
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def _fallback_structured_response2(
        self, source_map: dict[str, dict[str, Any]], messages: list | None = None
    ) -> str:
        """无 API key 时的兜底回答（新版 messages 兼容）"""
        if not source_map:
            return self._build_refusal_response(_extract_query_from_messages(messages) if messages else "")

        source_refs = []
        for idx, info in source_map.items():
            parts = [f"[{idx}] 文件: {info['filename']}"]
            if info.get("page"):
                parts.append(f"页码: {info['page']}")
            source_refs.append(" | ".join(parts))

        sources_section = "\n".join(f"> {s}" for s in source_refs)

        return f"""**结论：**
根据知识库中检索到的相关文档，该问题已有覆盖信息。

**引用来源：**
{sources_section}

**不确定信息：**
未配置大模型 API，当前为检索模式，无法生成 AI 推理回答。"""

    def _call_openai(self, prompt: str) -> str:
        """旧版兼容——将字符串 prompt 包装为 messages"""
        wants_verbose = any(kw in prompt[:200] for kw in ["详细", "详细解释", "展开", "具体说明"])
        mt = self.verbose_max_tokens if wants_verbose else self.default_max_tokens
        return self._call_openai_chat(
            messages=[
                {"role": "system", "content": "你是一个专业的 AI 知识助手。"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=mt,
        )

        """调用本地 Ollama 服务"""
        import requests

        response = requests.post(
            url=os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"),
            json={
                "model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature},
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["response"]

    def _build_refusal_response(self, query: str, reason: str = "") -> str:
        """构造标准拒答回复"""
        parts = [
            "**结论：**",
            f'知识库中未找到与"{query}"相关的依据。',
            "",
            "**依据：**",
            "无",
            "",
            "**引用来源：**",
            "无",
            "",
            "**不确定信息：**",
            "-",
            "",
            "**建议下一步：**",
            "请尝试换一种表述方式提问，或联系管理员添加相关文档到知识库。",
        ]
        if reason:
            parts.extend(["", f"> 📊 拒答原因: {reason}"])
        return "\n".join(parts)

    def _fallback_structured_response(self, prompt: str, source_map: dict[str, dict[str, Any]]) -> str:
        """
        当没有 API 时的兜底结构化回答
        从检索内容中提取信息，组装成标准格式
        """
        query = _extract_query_from_prompt(prompt)

        if not source_map:
            return self._build_refusal_response(query, "检索结果为空")

        # 组装结构化回答
        source_refs = []
        for idx, info in source_map.items():
            parts = [f"[{idx}] 文件: {info['filename']}"]
            if info.get("page"):
                parts.append(f"页码: {info['page']}")
            if info.get("paragraph"):
                parts.append(f"段落: {info['paragraph']}")
            source_refs.append(" | ".join(parts))

        sources_section = "\n".join(f"> {s}" for s in source_refs)

        return f"""**结论：**
根据知识库中检索到的相关文档，该问题已有覆盖信息。以下为检索到的相关文档摘要。

**依据：**
1. 检索到 {len(source_map)} 篇相关文档，内容与问题高度相关[1]
2. 具体信息请参考下方引用来源中的文档内容

**引用来源：**
{sources_section}

**不确定信息：**
- 未配置大模型 API，当前为检索模式，无法生成 AI 推理回答
- 实际内容请直接查阅上方原始文档片段

**建议下一步：**
1. 配置 LLM API（DeepSeek / SiliconFlow / Ollama）获取 AI 生成的精准回答
2. 直接阅读上方【参考文档】中的原始内容"""


def _extract_query_from_prompt(prompt: str) -> str:
    """从 Prompt 中提取用户问题（旧版字符串格式）"""
    lines = prompt.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("## 用户问题"):
            rest = line.replace("## 用户问题", "").strip()
            if rest:
                return rest
            if i + 1 < len(lines):
                return lines[i + 1].strip()
    return ""


def _extract_query_from_messages(messages: list[dict[str, str]] | None) -> str:
    """从 messages 列表中提取用户问题（新版格式）"""
    if not messages:
        return ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # 从 "## 用户问题\n..." 中提取
            if "## 用户问题" in content:
                return content.split("## 用户问题")[-1].split("##")[0].strip()
            if "## 问题" in content:
                return content.split("## 问题")[-1].split("##")[0].strip()
            # 直接当问题（简短的 user message）
            if len(content) < 500:
                return content[:200]
    return ""


def create_generator() -> LLMGenerator:
    """创建生成器实例（主回答用，从环境变量读取）"""
    return LLMGenerator()


def create_rewrite_generator() -> LLMGenerator | None:
    """创建 Query Rewriting 专用生成器

    读取 REWRITE_API_KEY / REWRITE_BASE_URL / REWRITE_MODEL 环境变量。
    未配置时返回 None（Retriever 会降级回退到主 generator）。
    """
    api_key = os.getenv("REWRITE_API_KEY", "")
    base_url = os.getenv("REWRITE_BASE_URL", "")
    model = os.getenv("REWRITE_MODEL", "")
    if not api_key or not base_url or not model:
        return None
    return LLMGenerator(api_key=api_key, base_url=base_url, model=model)
