"""
评估指标计算模块
提供 RAG 系统检索和拒答的量化指标

指标说明:
  - Hit Rate:       预期文档出现在检索结果中的比例（检索覆盖面）
  - MRR:            第一个正确答案的倒数排名的均值（排序质量）
  - NDCG@K:         归一化折损累积增益（排序头部质量）
  - Refusal Accuracy: 对知识库外问题正确拒答 + 库内问题不拒答的综合准确率
  - Semantic Score:  外部高精度 embedding 模型评估 query-doc 语义相似度（embedding 质量）
  - Passage Diversity: 检索结果来自多少个不同源文档（检索多样性）
  - Answer Token Efficiency: 回答 token 数 / 检索文档 token 数（信息压缩比）
  - False Positive Rate: 对 OOD 问题的误判率
"""

import math
from collections import Counter
from typing import Any

# ══════════════════════════════════════════════════
#  一、检索指标（原有）
# ══════════════════════════════════════════════════


def hit_rate(results: list[dict[str, Any]]) -> float:
    """命中率 = 预期文档出现在检索结果中的查询 / 总查询数

    只计算 category 为 exact_match 的查询。
    每道题如果 expected_doc 在任意一条检索结果的 filename 中，
    就算命中。
    """
    exact = [r for r in results if r.get("category") == "exact_match"]
    if not exact:
        return 0.0
    hits = sum(1 for r in exact if r.get("expected_hit", False))
    return hits / len(exact)


def _is_target_match(filename: str, target: str) -> bool:
    """filename 是否命中 expected_doc（兼容带/不带扩展名两种形式）"""
    if not filename:
        return False
    return filename == target or filename == target.rsplit(".", 1)[0]


def doc_hit(expected_doc: str, sources: list[dict[str, Any]]) -> bool:
    """expected_doc（可能带扩展名）是否命中任一检索来源的 filename

    与 _hit_rank 同一语义：比较完整名与去扩展名两种形式。
    """
    if not expected_doc:
        return False
    return any(_is_target_match((s.get("metadata") or {}).get("filename", ""), expected_doc) for s in sources)


def _hit_rank(r: dict[str, Any]) -> int | None:
    """expected_doc 在检索结果中的首次出现位置（从 1 开始），未命中返回 None

    expected_doc 带扩展名（.md/.txt），sources 的 filename 不带，
    故匹配时同时比较「完整名」与「去扩展名」两种形式。
    """
    target = r.get("expected_doc")
    if not target:
        return None
    for i, s in enumerate(r.get("sources", []), start=1):
        if _is_target_match((s.get("metadata") or {}).get("filename", ""), target):
            return i
    return None


def mrr(results: list[dict[str, Any]]) -> float:
    """Mean Reciprocal Rank（平均倒数排名）

    对每道 exact_match 题，用 expected_doc 在检索结果中的真实 rank
    计算 1/rank，再对全部题目取平均；未命中则贡献 0。
    """
    exact = [r for r in results if r.get("category") == "exact_match" and r.get("expected_doc")]
    if not exact:
        return 0.0

    total_rr = 0.0
    for r in exact:
        rank = _hit_rank(r)
        if rank is not None:
            total_rr += 1.0 / rank
    return total_rr / len(exact)


def ndcg_at_k(results: list[dict[str, Any]], k: int = 5) -> float:
    """NDCG@K（归一化折损累积增益）

    每题只有一个相关文档（expected_doc），理想排序下它排在首位，
    故 IDCG@K = 1.0；DCG@K = Σ gain_i / log2(i+1)。
    """
    exact = [r for r in results if r.get("category") == "exact_match" and r.get("expected_doc")]
    if not exact:
        return 0.0

    total_dcg = 0.0
    for r in exact:
        target = r["expected_doc"]
        dcg = 0.0
        hit = False
        for i, s in enumerate(r.get("sources", [])[:k], start=1):
            if not hit and _is_target_match((s.get("metadata") or {}).get("filename", ""), target):
                hit = True
                dcg += 1.0 / math.log2(i + 1)  # 仅首次出现计 gain（rank1 时 log2(2)=1，无折扣）
        total_dcg += dcg  # IDCG@K = 1.0（唯一相关文档）
    return total_dcg / len(exact)


def average_precision(results: list[dict[str, Any]]) -> float:
    """平均准确率（AP）

    对每道 exact_match 题，AP = Σ(P@k × rel_k) / num_relevant。
    每题只有一个相关文档 → 命中时 AP = 1/rank，未命中为 0。
    """
    exact = [r for r in results if r.get("category") == "exact_match" and r.get("expected_doc")]
    if not exact:
        return 0.0

    total = 0.0
    for r in exact:
        rank = _hit_rank(r)
        if rank is not None:
            total += 1.0 / rank
    return total / len(exact)


# ══════════════════════════════════════════════════
#  二、拒答指标（原有）
# ══════════════════════════════════════════════════


def refusal_accuracy(results: list[dict[str, Any]]) -> float:
    """拒答综合准确率

    对 out_of_knowledge 类：期望系统拒答（correct_refusal=True）
    对 exact_match/cross_doc 类：期望系统不拒答（correct_refusal=True）
    """
    if not results:
        return 0.0
    correct = sum(1 for r in results if r.get("correct_refusal", False))
    return correct / len(results)


def refusal_precision_and_recall(results: list[dict[str, Any]]) -> dict[str, float]:
    """拒答的精确率和召回率

    - precision: 拒答的回答中，有多少是正确的（确实应该拒答）
    - recall: 应该拒答的问题中，有多少被正确拒答了
    """
    ood = [r for r in results if r.get("category") == "out_of_knowledge"]
    if not ood:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = sum(1 for r in ood if r.get("correct_refusal", False))
    fn = len(ood) - tp

    non_ood = [r for r in results if r.get("category") != "out_of_knowledge"]
    fp = sum(1 for r in non_ood if not r.get("correct_refusal", False) and r.get("is_refusal_in_answer", False))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# ══════════════════════════════════════════════════
#  三、新增：语义质量指标
# ══════════════════════════════════════════════════


def fake_positive_rate(results: list[dict[str, Any]]) -> float:
    """OOD 问题的误判率

    对 out_of_knowledge 类，统计系统给了错误回答（即没有拒答）的比例。
    值越低越好（0% = 所有 OOD 问题都被正确拒答）。
    """
    ood = [r for r in results if r.get("category") == "out_of_knowledge"]
    if not ood:
        return 0.0
    fakes = sum(1 for r in ood if not r.get("correct_refusal", False))
    return fakes / len(ood)


def passage_diversity(results: list[dict[str, Any]]) -> float:
    """检索结果多样性

    对每个 query，统计检索到的 top-K 片段来自多少个不同的源文档。
    返回所有 query 的平均值。值越高表示检索覆盖面越好。
    """
    queries_with_sources = [r for r in results if r.get("num_retrieved", 0) > 0]
    if not queries_with_sources:
        return 0.0

    total_diversity = 0.0
    for r in queries_with_sources:
        sources = r.get("sources", [])
        if sources:
            filenames = set()
            for s in sources:
                fn = s.get("metadata", {}).get("filename", "")
                if fn:
                    filenames.add(fn)
            total_diversity += len(filenames)
        else:
            total_diversity += 0

    return round(total_diversity / len(queries_with_sources), 2)


def semantic_score_metric(results: list[dict[str, Any]]) -> float:
    """语义相似度评分

    计算所有检索结果中 query 和 top-1 chunk 之间的平均语义相似度。
    从 evaluation record 中读取已有的 top_score 字段。

    值说明：
      - e5-small (384d): 正常 ~0.3-0.6，当前系统 ~0.03 ⇒ 有问题
      - text-embedding-3-small (1536d): 正常 ~0.4-0.7
    """
    scored = [r for r in results if r.get("top_score", 0) > 0]
    if not scored:
        return 0.0
    return round(sum(r["top_score"] for r in scored) / len(scored), 4)


def answer_token_efficiency(results: list[dict[str, Any]]) -> float:
    """回答 token 效率

    回答长度 / 检索文档总长度。
    > 1.0: 回答比检索到的文档还长，可能超出了检索信息
    ~0.3-0.8: 正常的信息压缩和摘要
    < 0.1: 回答太短，可能忽略了检索信息
    """
    counted = 0
    total_ratio = 0.0
    for r in results:
        answer = r.get("answer_text", "") or r.get("answer", "")
        sources = r.get("sources", [])
        if not answer or not sources:
            continue
        doc_len = sum(len(s.get("text", "")) for s in sources)
        if doc_len == 0:
            continue
        total_ratio += len(answer) / doc_len
        counted += 1

    if counted == 0:
        return 0.0
    return round(total_ratio / counted, 4)


def response_time_breakdown(results: list[dict[str, Any]]) -> dict[str, float]:
    """响应时间分解

    统计检索耗时和总耗时的平均值。
    """
    if not results:
        return {"avg_total_time": 0.0, "avg_retrieval_time": 0.0, "generation_ratio": 0.0}

    total = sum(r.get("time_seconds", 0) for r in results)
    retrieval = sum(r.get("retrieval_time", 0) for r in results)
    count = len(results)

    avg_total = round(total / count, 2)
    avg_retrieval = round(retrieval / count, 2)

    return {
        "avg_total_time": avg_total,
        "avg_retrieval_time": avg_retrieval,
        "generation_ratio": round((avg_total - avg_retrieval) / avg_total, 4) if avg_total > 0 else 0,
    }


# ══════════════════════════════════════════════════
#  四、按难度分层指标
# ══════════════════════════════════════════════════


def metrics_by_difficulty(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """按难度层级计算指标

    Returns:
        {
            "easy": {"hit_rate": ..., "count": N},
            "medium": {...},
            "hard": {...},
        }
    """
    from collections import defaultdict

    by_diff = defaultdict(list)
    for r in results:
        diff = r.get("difficulty", "unknown")
        by_diff[diff].append(r)

    result = {}
    for diff, items in by_diff.items():
        result[diff] = {
            "count": len(items),
            "hit_rate": round(hit_rate(items), 4),
            "refusal_accuracy": round(refusal_accuracy(items), 4),
            "avg_semantic_score": round(sum(i.get("top_score", 0) for i in items) / len(items), 4),
        }
    return result


# ══════════════════════════════════════════════════
#  六、Chunk-level Gold 指标（Step 8）
# ══════════════════════════════════════════════════


def gold_answerability_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    """按 answerability 统计问题数（Step 8 重标注）"""
    from collections import Counter

    return dict(Counter(r.get("gold_answerability", "unlabeled") for r in results if r.get("expected_doc")))


def chunk_hit_rate(results: list[dict[str, Any]]) -> float:
    """Chunk-level Hit Rate：answer-bearing chunk 出现在检索结果中的比例

    与 hit_rate（document-level）的区别：
      - document-level: expected_doc 文件出现即命中
      - chunk-level:    gold_evidence 中的 answer_bearing_chunk_ids 出现即命中
    对 24 个原 C 类问题（文档对但 chunk 错），只有 chunk-level 能正确评价。
    """
    labeled = [r for r in results if r.get("gold_chunk_ids")]
    if not labeled:
        return 0.0
    hits = sum(1 for r in labeled if r.get("expected_hit"))
    return hits / len(labeled)


def unsupported_refusal_rate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """对标注为 unsupported 的问题，统计系统是否正确拒答/低置信

    注：当前 40 个 exact_match 重标注后无 unsupported 问题，
    该指标为 0 分母保护，供未来扩展。
    """
    unsupported = [r for r in results if r.get("gold_answerability") == "unsupported"]
    if not unsupported:
        return {"count": 0, "refused": 0, "rate": 0.0}
    refused = sum(1 for r in unsupported if r.get("correct_refusal", False))
    return {"count": len(unsupported), "refused": refused, "rate": round(refused / len(unsupported), 4)}


# ══════════════════════════════════════════════════
#  七、汇总
# ══════════════════════════════════════════════════


def compute_all_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """计算所有评估指标

    Returns:
        {
            "overall": { ... 核心指标 ... },
            "by_category": { "exact_match": {...}, ... },
            "by_difficulty": { "easy": {...}, ... },
            "refusal_detail": { "precision": ..., "recall": ..., "f1": ... },
            "semantic_score": 0.03,     # 新增
            "passage_diversity": 2.5,   # 新增
            "answer_efficiency": 0.45,  # 新增
            "false_positive_rate": 0.0, # 新增
            "time_breakdown": { ... },  # 新增
        }
    """
    cat_counts = Counter(r.get("category", "unknown") for r in results)

    metrics = {
        "overall": {
            "hit_rate": round(hit_rate(results), 4),
            "mrr": round(mrr(results), 4),
            "ndcg_at_5": round(ndcg_at_k(results, k=5), 4),
            "refusal_accuracy": round(refusal_accuracy(results), 4),
            "average_precision": round(average_precision(results), 4),
            "total_queries": len(results),
        },
        "by_category": {},
        "by_difficulty": metrics_by_difficulty(results),
        "refusal_detail": {k: round(v, 4) for k, v in refusal_precision_and_recall(results).items()},
        "semantic_score": semantic_score_metric(results),
        "passage_diversity": passage_diversity(results),
        "answer_efficiency": answer_token_efficiency(results),
        "false_positive_rate": round(fake_positive_rate(results), 4),
        "time_breakdown": response_time_breakdown(results),
        "chunk_level": {
            "labeled_count": len([r for r in results if r.get("gold_chunk_ids")]),
            "chunk_hit_rate": round(chunk_hit_rate(results), 4),
            "answerability": gold_answerability_counts(results),
            "unsupported": unsupported_refusal_rate(results),
        },
    }

    # 按类别细分
    for cat in cat_counts:
        cat_results = [r for r in results if r.get("category") == cat]
        metrics["by_category"][cat] = {
            "count": len(cat_results),
            "hit_rate": round(hit_rate(cat_results), 4),
            "refusal_accuracy": round(refusal_accuracy(cat_results), 4),
            "avg_top_score": round(sum(r.get("top_score", 0) for r in cat_results) / len(cat_results), 4),
        }

    return metrics


def print_metrics_report(metrics: dict[str, Any]) -> None:
    """格式化打印指标报告"""
    overall = metrics["overall"]
    print("\n" + "=" * 70)
    print("  📊 RAG 系统评估报告")
    print("=" * 70)
    print("\n  🔍 检索质量")
    print(f"     Hit Rate:       {overall['hit_rate']:.2%}")
    print(f"     MRR:            {overall['mrr']:.4f}")
    print(f"     NDCG@5:         {overall['ndcg_at_5']:.4f}")
    print(f"     Average Prec.:  {overall['average_precision']:.4f}")
    print("\n  📐 语义与多样性")
    print(f"     Semantic Score: {metrics.get('semantic_score', 0):.4f}")
    print(f"     Passage Div.:   {metrics.get('passage_diversity', 0):.2f} docs/query")
    print(f"     Answer Effic.:  {metrics.get('answer_efficiency', 0):.4f}")
    print("\n  🚫 拒答质量")
    print(f"     Refusal Acc.:   {overall['refusal_accuracy']:.2%}")
    print(f"     False Pos. Rate:{metrics.get('false_positive_rate', 0):.2%}")
    ref = metrics.get("refusal_detail", {})
    print(f"     Precision:      {ref.get('precision', 0):.2%}")
    print(f"     Recall:         {ref.get('recall', 0):.2%}")
    print(f"     F1:             {ref.get('f1', 0):.2%}")

    # 按难度
    by_diff = metrics.get("by_difficulty", {})
    if by_diff:
        print("\n  🎯 按难度")
        for diff in ["easy", "medium", "hard"]:
            dm = by_diff.get(diff)
            if dm:
                print(
                    f"     {diff:<8} {dm['count']}题  "
                    f"Hit Rate={dm['hit_rate']:.0%}  "
                    f"Refusal Acc={dm['refusal_accuracy']:.0%}  "
                    f"Semantic={dm['avg_semantic_score']:.4f}"
                )

    print("\n  📋 按类别")
    for cat, cat_m in metrics.get("by_category", {}).items():
        print(
            f"     {cat:<20}  {cat_m['count']}题  "
            f"Hit Rate={cat_m['hit_rate']:.0%}  "
            f"Refusal Acc={cat_m['refusal_accuracy']:.0%}"
        )

    tb = metrics.get("time_breakdown", {})
    print("\n  ⏱️  响应时间")
    print(f"     平均总耗时:  {tb.get('avg_total_time', 0):.2f}s")
    print(f"     平均检索耗时:{tb.get('avg_retrieval_time', 0):.2f}s")
    print(f"     生成占比:    {tb.get('generation_ratio', 0):.1%}")

    cl = metrics.get("chunk_level", {})
    if cl.get("labeled_count"):
        print("\n  🎯 Chunk-level Gold（Step 8 重标注）")
        print(f"     标注题数:    {cl['labeled_count']}")
        print(f"     Chunk Hit:  {cl['chunk_hit_rate']:.0%}")
        print(f"     Answerability: {cl.get('answerability', {})}")

    print(f"\n  📦 总查询数: {overall['total_queries']}")
    print("=" * 70 + "\n")
