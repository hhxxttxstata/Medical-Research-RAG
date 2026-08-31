"""
failure_taxonomy.py — 失败根因枚举词表（P1-2）

把四类失败面（retrieval / policy / generation / refusal）细化为一组
可统计、可聚合的根因码。bad_cases.json 条目的可选 taxonomy 字段
必须取值于此词表（scripts/audit_bad_cases.py 校验）；
scripts/locate_failure.py 定位失败面后给出建议码（SUGGESTED_BY_SURFACE）。

约定:
  - 码为大写下划线字符串，稳定命名——新增可以，改名/复用语义不行
  - 一条 case 可挂多个码（taxonomy: [..]），按主要根因在前
"""

TAXONOMY = {
    # ── 检索面 ──
    "RETRIEVAL_MISS": "gold 在库内但未被召回（候选池不可达）",
    "MISSING_HOP": "多跳题缺失部分 hop 的证据",
    "RERANK_DROP": "候选池含 gold，重排/截断后掉出最终证据集",
    "GOLD_ANNOTATION_ERROR": "gold_chunk_ids / 期望答案标注错位（非系统缺陷）",
    # ── 策略面 ──
    "PREMATURE_ACCEPT": "证据不足即 ACCEPT",
    "UNSUPPORTED_ACCEPT": "答案不被证据支持仍 ACCEPT（Relevance ≠ Support）",
    "FALSE_ABSTAIN": "可答题被误拒",
    "OOD_MISS": "应拒答的题未拒答",
    "POLICY_MISMATCH": "实际 route 与 expected_route 不符",
    "DECOMP_FAILURE": "该拆解的题未拆解或拆解执行失败",
    # ── 生成面 ──
    "CITATION_MISMATCH": "引用与答案内容不符",
    "NUMERIC_MISMATCH": "数值与证据不符",
    "DIRECTION_FLIP": "方向翻转（升高/降低、优效/劣效颠倒）",
    "SIGNIFICANCE_FLIP": "显著/不显著翻转",
    "ENDPOINT_SWAP": "终点互换（把 A 指标答成 B 指标）",
    "PICO_EXPANSION": "结论外推超出研究人群/干预（PICO 扩张）",
    "CORRELATION_CAUSATION": "把相关性表述为因果性",
    "MALFORMED_OUTPUT": "结构化输出损坏/不可解析",
    # ── 运行时 ──
    "TIMEOUT": "LLM/检索超时",
    "OPERATIONAL_ERROR": "API/网络等运行时故障",
}

# 失败面 → 建议码（locate_failure.py 输出建议时使用，人工确认后固化进 taxonomy）
SUGGESTED_BY_SURFACE = {
    "retrieval": ["RETRIEVAL_MISS", "MISSING_HOP", "RERANK_DROP", "GOLD_ANNOTATION_ERROR"],
    "policy": ["POLICY_MISMATCH", "DECOMP_FAILURE", "PREMATURE_ACCEPT", "UNSUPPORTED_ACCEPT"],
    "refusal": ["FALSE_ABSTAIN", "OOD_MISS", "UNSUPPORTED_ACCEPT"],
    "generation": [
        "NUMERIC_MISMATCH",
        "CITATION_MISMATCH",
        "MALFORMED_OUTPUT",
        "DIRECTION_FLIP",
        "SIGNIFICANCE_FLIP",
        "ENDPOINT_SWAP",
        "PICO_EXPANSION",
        "CORRELATION_CAUSATION",
    ],
}


def is_valid(code: str) -> bool:
    return code in TAXONOMY


def suggest_for_surface(surface: str) -> list[str]:
    return SUGGESTED_BY_SURFACE.get(surface, [])
