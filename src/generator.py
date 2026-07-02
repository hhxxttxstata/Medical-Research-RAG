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
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

# Windows GBK 编码兼容
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

from opentelemetry import trace as otel_trace

from .embeddings import is_mostly_english
from .monitoring.metrics import record_llm_latency

# ── 可观测性 ────────────────────────────────────────
from .monitoring.tracing import get_tracer

_tracer = None  # lazy init


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
    avg_score = sum(
        c.get("_vector_score") if c.get("_vector_score") is not None else c["score"] for c in retrieved_chunks
    ) / len(retrieved_chunks)

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


def build_rag_prompt(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    relevance: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """
    构建 RAG Prompt（混合模式）

    返回:
        (prompt_text, source_map, relevance_info)
    """
    if relevance is None:
        relevance = compute_relevance(query, retrieved_chunks)

    # ── 构建带引用的上下文（支持 Small-to-Big parent context） ──
    context_parts = []
    source_map: dict[str, dict[str, Any]] = {}

    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk["metadata"]
        filename = meta.get("filename", "未知")
        page = meta.get("page", "")
        para_start = meta.get("paragraph_start", "")
        para_end = meta.get("paragraph_end", "")
        section_title = meta.get("section_title", meta.get("heading", ""))
        parent_content = meta.get("parent_content", "")

        # 构建精确引用标记
        ref_parts = [f"[{i}]"]
        ref_parts.append(f"文件: {filename}")
        if section_title:
            ref_parts.append(f"章节: {section_title}")
        if page:
            ref_parts.append(f"页码: {page}")

        source_tag = " | ".join(ref_parts)

        # Small-to-Big: 如果存在 parent context，把更完整的上下文也提供
        chunk_content = chunk["text"]
        if parent_content and len(parent_content) > len(chunk_content) * 1.5:
            chunk_content = f"{chunk_content}\n\n> 📖 **完整上下文（本节）**\n{parent_content}"

        context_parts.append(f"{source_tag}\n{chunk_content}\n")

        source_map[str(i)] = {
            "filename": filename,
            "page": str(page) if page else "",
            "section": section_title,
            "score": round(chunk["score"], 3),
            "text_preview": chunk["text"][:100],
        }

    context = "\n".join(context_parts)
    num_sources = len(retrieved_chunks)
    has_relevant = relevance["is_relevant"]

    # ── 自身知识补充规则 ──
    knowledge_rule = f"""
## 自身知识补充规则
1. **优先使用【参考文档】中的信息**，引用时标注编号 [1]、[2] 等
2. {"" if has_relevant else "**【参考文档】与问题相关性较低**，请主要依靠你的自身知识来回答。"}
3. 如果你要使用自身知识（而非参考文档）中的内容回答，必须用 **【自身知识】** 标注，例如：
   - "根据参考文档[1]，肺栓塞CT表现为...（【自身知识】此外，急性PE的CTPA典型征象还包括...）"
   - 或者单独段落：**（【自身知识】）** 后面跟你的补充内容
4. 如果你对自身知识没有把握，请在回答末尾注明"**建议进一步查阅权威文献确认**"
5. 严禁虚构引用——不要编造参考文档中不存在的 [编号]
6. 即使参考文档为空或不相关，你也可以基于自身知识回答，但必须清晰标注"""

    # ── 完整 Prompt ──
    prompt = f"""你是一个专业的 AI 知识助手，擅长结合检索到的资料和自身知识给出高质量回答。

## 核心原则
1. **优先使用参考文档**：回答中首先使用【参考文档】提供的材料
2. **自身知识补充**：当参考文档不足时，允许使用你的自身知识扩展回答，但必须明确标注
3. **结构化输出**：按五段式结构输出
4. **精确引用**：引用参考文档时标注 [编号]

## 输出格式（必须严格遵守）

**结论：**
（1-2句话总结核心答案，包括参考文档中的关键信息和你自身知识的补充）

**依据：**
（分点列出，每条注明来源）
（格式：N. 论述内容 [引用编号或【自身知识】]）

**引用来源：**
（列出实际引用的参考文档来源）
> [1] 文件: xxx | 页码: x
> [2] 文件: xxx | 页码: x

**不确定信息：**
（列出基于自身知识但不太确定的内容，如果没有则填"无"）

**建议下一步：**
（基于回答内容给出可操作建议）
{knowledge_rule}

## 参考文档
{context if context else "（无参考文档）"}

## 用户问题
{query}

## 回答
"""

    return prompt, source_map, relevance


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


@dataclass
class ChatWithToolsResult:
    """带工具调用的聊天响应

    content:        LLM 的文本回复（无 tool_calls 时的回答）
    tool_calls:     结构化工具调用列表 [{id, type, function: {name, arguments}}]
    finish_reason:  终止原因：stop | tool_calls | length
    is_degraded:    True 表示该结果是降级而非 LLM 原始响应
                    （Function Calling 失败 → 纯文本兜底）
    """

    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    is_degraded: bool = False


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
        temperature: float = 0.1,  # 降低温度，减少幻觉
        max_tokens: int = 2048,
        # 指数退避重试
        retry_max_attempts: int = 3,
        retry_base_delay: float = 1.0,
        # Circuit Breaker
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 30.0,
    ):
        # 按优先级读取 API 配置：DeepSeek > SiliconFlow > OpenAI
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
            or os.getenv("DEEPSEEK_MODEL")
            or os.getenv("SILICON_MODEL")
            or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        )
        self.temperature = temperature
        self.max_tokens = max_tokens if max_tokens else 4096  # 默认4096，适应混合回答

        # 指数退避重试配置
        self.retry_max_attempts = retry_max_attempts
        self.retry_base_delay = retry_base_delay

        # Circuit Breaker 状态
        self.cb_threshold = circuit_breaker_threshold
        self.cb_timeout = circuit_breaker_timeout
        self._cb_state = "closed"  # "closed" | "open" | "half_open"
        self._cb_failures = 0
        self._cb_open_until = 0.0

    def _is_valid_api_key(self, key: str) -> bool:
        """检查 API Key 是否有效（排除空值和占位符）"""
        if not key or key == "":
            return False
        placeholders = ["your-api-key-here", "sk-your-", "your_"]
        return not any(p in key.lower() for p in placeholders)

    def _detect_api_type(self) -> str:
        """检测使用的 API 类型"""
        if not self._is_valid_api_key(self.api_key):
            return "ollama"
        if "deepseek" in self.base_url.lower():
            return "deepseek"
        if "silicon" in self.base_url.lower():
            return "siliconflow"
        if "openai" in self.base_url.lower():
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
        prompt, source_map, relevance = prompt_and_source

        # ── 追踪 span ──
        global _tracer
        if _tracer is None:
            _tracer = get_tracer()
        _span_ctx = _tracer.start_as_current_span("generator.generate")
        _span = _span_ctx.__enter__()
        _span.set_attribute("model_name", self.model)
        has_valid_key = self._is_valid_api_key(self.api_key)
        _span.set_attribute("has_api_key", has_valid_key)

        start = time.monotonic()
        try:
            # ── 检查 API Key 并调用模型 ──
            if has_valid_key:
                try:
                    answer = self._call_openai(prompt)
                except Exception as e:
                    print(f"  ⚠️ API 调用失败: {e}")
                    print("  🔄 使用本地模式...")
                    answer = self._fallback_structured_response(prompt, source_map)
            else:
                # 本地 Ollama
                try:
                    answer = self._call_ollama(prompt)
                except Exception:
                    answer = self._fallback_structured_response(prompt, source_map)

            # ── 后处理：验证引用有效性 ──
            if validate and source_map:
                citation_result = validate_citations(answer, source_map)
                if citation_result["has_invalid_citations"]:
                    invalid = citation_result["cited_invalid"]
                    warning = (
                        f"\n\n⚠️ **引用验证提示**：回答中引用了不存在的来源编号 {invalid}，"
                        f"请以实际提供的【参考文档】编号为准。"
                    )
                    answer += warning

            elapsed = time.monotonic() - start
            _span.set_attribute("answer_length", len(answer))
            _span.set_attribute("duration_ms", round(elapsed * 1000, 1))
            record_llm_latency(elapsed, model=self.model, api_type=self._detect_api_type())

            return answer
        except Exception as e:
            elapsed = time.monotonic() - start
            _span.record_exception(e)
            _span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, str(e)))
            record_llm_latency(elapsed, model=self.model, api_type="error")
            raise
        finally:
            _span_ctx.__exit__(None, None, None)

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
    ) -> str:
        """面向对话 / ReAct 循环的聊天接口

        直接传入 messages 列表，适用于多轮对话场景（如 Agent 的 ReAct 循环）。
        内置 Circuit Breaker + 指数退避重试。
        """
        global _tracer
        if _tracer is None:
            _tracer = get_tracer()
        start = time.monotonic()
        with _tracer.start_as_current_span("generator.chat") as span:
            span.set_attribute("model_name", self.model)

            # Circuit Breaker 检查
            if not self._check_circuit_breaker():
                elapsed = time.monotonic() - start
                span.set_attribute("duration_ms", round(elapsed * 1000, 1))
                record_llm_latency(elapsed, model=self.model, api_type=self._detect_api_type())
                return self._fallback_text("API 熔断中")

            try:
                fn = self._get_chat_fn()
                result = self._with_retry(fn, messages, temperature, max_tokens)
            except Exception:
                result = self._fallback_text("API 暂时不可用")

            elapsed = time.monotonic() - start
            span.set_attribute("duration_ms", round(elapsed * 1000, 1))
            record_llm_latency(elapsed, model=self.model, api_type=self._detect_api_type())
            return result

    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> ChatWithToolsResult:
        """带原生 Function Calling 支持的聊天接口

        当 tools 参数提供且后端 API 支持时，返回结构化的 tool_calls。
        否则降级到纯文本聊天。

        Args:
            messages:          对话消息列表
            tools:             OpenAI 格式的工具定义列表
            tool_choice:       "auto" | "none" | "required"
            temperature:       温度（可选，覆盖默认值）
            max_tokens:        最大 token 数
            parallel_tool_calls: 是否允许并行工具调用（默认 True）

        Returns:
            ChatWithToolsResult: 含 content 和/或 tool_calls
        """
        global _tracer
        if _tracer is None:
            _tracer = get_tracer()
        start = time.monotonic()
        with _tracer.start_as_current_span("generator.chat_with_tools") as span:
            span.set_attribute("model_name", self.model)
            has_valid_key = self._is_valid_api_key(self.api_key)
            span.set_attribute("has_api_key", has_valid_key)
            has_tools = tools is not None and len(tools) > 0
            span.set_attribute("has_tools", has_tools)

            # ── 有 API Key → OpenAI/DeepSeek 兼容路径 ──
            if has_valid_key and tools:
                try:
                    result = self._call_openai_chat_with_tools(
                        messages=messages,
                        tools=tools,
                        tool_choice=tool_choice or "auto",
                        temperature=temperature,
                        max_tokens=max_tokens,
                        parallel_tool_calls=parallel_tool_calls,
                    )
                    elapsed = time.monotonic() - start
                    span.set_attribute("duration_ms", round(elapsed * 1000, 1))
                    span.set_attribute("tool_call_count", len(result.tool_calls))
                    record_llm_latency(elapsed, model=self.model, api_type=self._detect_api_type())
                    return result
                except Exception:
                    # Function Calling API 调用失败，降级到纯文本
                    pass

            # ── 无 API Key（Ollama 路径） ──
            if not has_valid_key and tools:
                try:
                    result = self._call_ollama_chat_with_tools(
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                    )
                    elapsed = time.monotonic() - start
                    span.set_attribute("duration_ms", round(elapsed * 1000, 1))
                    span.set_attribute("tool_call_count", len(result.tool_calls))
                    record_llm_latency(elapsed, model=self.model, api_type="ollama")
                    return result
                except Exception:
                    pass

            # ── 降级到纯文本 ──
            text = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
            elapsed = time.monotonic() - start
            span.set_attribute("duration_ms", round(elapsed * 1000, 1))
            record_llm_latency(elapsed, model=self.model, api_type=self._detect_api_type())
            return ChatWithToolsResult(content=text, tool_calls=[], finish_reason="stop", is_degraded=True)

    def _call_openai_chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float | None = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> ChatWithToolsResult:
        """调用 OpenAI 兼容 API 并传入 tools 参数

        返回结构化的 ChatWithToolsResult，包含 content 和/或 tool_calls。
        """
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=30)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens,
            "timeout": 30,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
            kwargs["parallel_tool_calls"] = parallel_tool_calls

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        # 提取 tool_calls（如果有）
        tool_calls_data: list[dict[str, Any]] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_data.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )

        return ChatWithToolsResult(
            content=msg.content,
            tool_calls=tool_calls_data,
            finish_reason=choice.finish_reason or "stop",
        )

    def _call_ollama_chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        temperature: float | None = None,
    ) -> ChatWithToolsResult:
        """调用 Ollama 的 chat API 并传入 tools 参数

        Ollama 0.3+ 开始支持 tools（实验性），失败时降级到纯文本。
        """
        import requests

        try:
            payload: dict[str, Any] = {
                "model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature if temperature is not None else self.temperature,
                },
            }
            if tools:
                payload["tools"] = tools

            response = requests.post(
                url=f"{os.getenv('OLLAMA_URL', 'http://localhost:11434')}/api/chat",
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            msg = data.get("message", {})

            tool_calls_data: list[dict[str, Any]] = []
            for tc in msg.get("tool_calls", []):
                tool_calls_data.append(
                    {
                        "id": tc.get("id", "call_ollama"),
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": json.dumps(tc["function"]["arguments"]),
                        },
                    }
                )

            return ChatWithToolsResult(
                content=msg.get("content"),
                tool_calls=tool_calls_data,
                finish_reason=data.get("done_reason", "stop"),
            )
        except Exception:
            # Fallback: text-only
            text = self._call_ollama_chat(messages, temperature)
            return ChatWithToolsResult(content=text, tool_calls=[], finish_reason="stop")

    def _call_openai_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int = 2048,
    ) -> str:
        """通用 OpenAI 兼容 API 调用（接受任意 messages 列表）"""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=30)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens,
            timeout=30,
        )
        return response.choices[0].message.content

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

    def _call_openai(self, prompt: str) -> str:
        """调用 OpenAI 兼容 API（30s 超时）"""
        return self._call_openai_chat(
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的 AI 知识助手。"
                    "你擅长结合提供的参考文档和自身知识给出高质量回答。"
                    "使用参考文档时必须标注编号，使用自身知识时必须标注【自身知识】。"
                    "严禁虚构引用，但允许基于自身知识合理补充。",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
        )

    def _call_ollama(self, prompt: str) -> str:
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
    """从 Prompt 中提取用户问题"""
    lines = prompt.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("## 用户问题"):
            # 取 "## 用户问题" 后面的内容，或者下一行
            rest = line.replace("## 用户问题", "").strip()
            if rest:
                return rest
            if i + 1 < len(lines):
                return lines[i + 1].strip()
    return ""


def create_generator() -> LLMGenerator:
    """创建生成器实例"""
    return LLMGenerator()
