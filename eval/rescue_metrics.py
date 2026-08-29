"""
Step 9: Rescue/Harm 统一指标定义（Rewrite 实验冻结版）

把 Step 1–7 各脚本里散落的指标计算收敛为单一权威实现，
消除三类定义漂移：

  1. FinalHit ≠ Rescue
     - FinalHit:   变体最终 Top-K 命中（无论 baseline 是否命中）
     - Rescue:     baseline miss → 变体 hit（"救回" 才叫 Rescue）
     - Harm:       baseline hit → 变体 miss（"打掉" 才叫 Harm）
     FinalHit − Rescue = baseline 本身就命中的题，不能算作 Rescue 的成绩。

  2. NetUtility = Rescue − Harm
     不是 FinalHit − Harm。用 NetUtility 做预算/门控决策。

  3. baseline comparator 必须显式声明
     所有对比都相对同一 baseline（V0 = Original query 检索 + rerank Top-K）。
     候选池/Candidate 层指标与最终层指标分开统计。

指标族：
  - hit_at_k(results, gold_ids, k)        通用 Top-K 命中
  - final_hit(ranks, k)                   变体最终命中（FinalHit）
  - rescue_harm(v0_hit, v1_hit)           单题 Rescue/Harm 判定
  - candidate_rescue(v0_pool, v1_pool)    Candidate 层 Rescue（池里有 Gold）
  - rerank_rescue(v0_topk, v1_topk)       Rerank 层 Rescue（最终进 TopK）
  - net_utility(rescues, harms)           NetUtility = Rescue − Harm

用法:
    from eval.rescue_metrics import compute_rescue_metrics
"""

import json
import re
from typing import Any

TOP_K = 5  # 生产输出深度（冻结）


# ══════════════════════════════════════════════════
#  基础判定
# ══════════════════════════════════════════════════


def gold_chunk_ids(question: dict) -> set[str]:
    """从 Step 8 标注中取 answer-bearing chunk ids（chunk-level Gold）"""
    return set(question.get("gold_evidence", {}).get("answer_bearing_chunk_ids", []))


def gold_filename(question: dict) -> str:
    return question.get("expected_doc", "")


def hit_at_k(sources: list[dict], gold_ids: set[str], expected_doc: str = "", k: int = TOP_K) -> bool:
    """Top-K 命中判定

    优先 chunk-level（gold_ids），无则退化为 document-level（expected_doc）。
    """
    if gold_ids:
        return any(s["id"] in gold_ids for s in sources[:k])
    if expected_doc:
        base = expected_doc.rsplit(".", 1)[0]
        return any(
            expected_doc == s["metadata"].get("filename", "") or s["metadata"].get("filename", "") == base
            for s in sources[:k]
        )
    return False


def gold_rank(sources: list[dict], gold_ids: set[str], expected_doc: str = "") -> int | None:
    """Gold 在结果中的 rank（1-based），不在则 None"""
    if gold_ids:
        for i, s in enumerate(sources):
            if s["id"] in gold_ids:
                return i + 1
        return None
    if expected_doc:
        base = expected_doc.rsplit(".", 1)[0]
        for i, s in enumerate(sources):
            fn = s["metadata"].get("filename", "")
            if expected_doc == fn or fn == base:
                return i + 1
        return None
    return None


# ══════════════════════════════════════════════════
#  Rescue / Harm / NetUtility
# ══════════════════════════════════════════════════


def classify_pair(v0_hit: bool, v1_hit: bool) -> str:
    """单题 Rescue/Harm 判定

    Returns:
        "rescue" — baseline miss → variant hit
        "harm"   — baseline hit → variant miss
        "both"   — 都命中
        "none"   — 都 miss
    """
    if not v0_hit and v1_hit:
        return "rescue"
    if v0_hit and not v1_hit:
        return "harm"
    return "both" if v0_hit and v1_hit else "none"


def net_utility(rescues: int, harms: int) -> int:
    """NetUtility = Rescue − Harm"""
    return rescues - harms


# ══════════════════════════════════════════════════
#  双层 Rescue 拆解（Step 5C 冻结版）
# ══════════════════════════════════════════════════


def candidate_rescue(v0_pool: list[dict], v1_pool: list[dict], gold_ids: set[str], expected_doc: str = "") -> bool:
    """Candidate 层 Rescue：V0 候选池无 Gold → V1 候选池有 Gold"""
    return not hit_at_k(v0_pool, gold_ids, expected_doc, k=len(v0_pool)) and hit_at_k(
        v1_pool, gold_ids, expected_doc, k=len(v1_pool)
    )


def rerank_rescue(v1_topk: list[dict], gold_ids: set[str], expected_doc: str = "", k: int = TOP_K) -> bool:
    """Rerank 层 Rescue：候选池有 Gold 且最终进 TopK"""
    return hit_at_k(v1_topk, gold_ids, expected_doc, k=k)


def rerank_failure(
    v1_pool: list[dict], v1_topk: list[dict], gold_ids: set[str], expected_doc: str = "", k: int = TOP_K
) -> bool:
    """Rerank Failure：候选池有 Gold，但 rerank 后仍在 TopK 之外"""
    return hit_at_k(v1_pool, gold_ids, expected_doc, k=len(v1_pool)) and not hit_at_k(
        v1_topk, gold_ids, expected_doc, k=k
    )


# ══════════════════════════════════════════════════
#  Step 12: Multi-hop / Agent Capability 指标族
# ══════════════════════════════════════════════════


def hop_gold_ids(question: dict) -> list[set[str]]:
    """从 Step 12 benchmark 取 hop-level gold（每 hop 的 answer-bearing chunk ids）"""
    return [set(h.get("gold_chunk_ids", [])) for h in question.get("hops", [])]


def evidence_recall_at_k(sources: list[dict], gold_ids: set[str], k: int = TOP_K) -> float:
    """Evidence Recall@K：gold chunks 中被检索命中的比例（0-1）"""
    if not gold_ids:
        return 0.0
    hit = sum(1 for g in gold_ids if any(s["id"] == g for s in sources[:k]))
    return hit / len(gold_ids)


def hop_recall_at_k(sources: list[dict], question: dict, k: int = TOP_K) -> float:
    """Hop Recall@K：每个 hop 是否有至少一个 gold 命中（0-1）"""
    hop_golds = hop_gold_ids(question)
    if not hop_golds:
        return 0.0
    hit_hops = sum(1 for hg in hop_golds if hg and any(s["id"] in hg for s in sources[:k]))
    return hit_hops / len([hg for hg in hop_golds if hg])


def evidence_completeness(sources: list[dict], question: dict, k: int = TOP_K) -> float:
    """Evidence Completeness：最终证据集对 hop 需求的覆盖比例

    与 hop_recall 的区别：hop_recall 只看"每个 hop 至少 1 个 gold 命中"，
    completeness 计算"所有 hop gold 中被命中的比例"（更严格）。
    """
    all_gold = set()
    for hg in hop_gold_ids(question):
        all_gold |= hg
    return evidence_recall_at_k(sources, all_gold, k)


def unnecessary_action_rate(route: list[str], question_type: str) -> bool:
    """Unnecessary Action：易题过度思考判定

    对 easy_single_hop 类：如果 Agent 的循环内动作（RETRIEVE/DECOMPOSE）超过 0 次
    且最终 ACCEPT（答对了但绕路）→ 算一次不必要动作。
    返回 True = 存在不必要动作。
    """
    if question_type != "easy_single_hop":
        return False
    loop_actions = [a for a in route[1:-1] if a in ("RETRIEVE", "DECOMPOSE")]
    return len(loop_actions) > 0


def final_answer_accuracy(answer: str, expected: str) -> bool:
    """Final Answer Accuracy：答案是否包含预期关键数值（宽松匹配）

    2026-08-17 修正（holdout30 实测 12/12 假阴性）：
      - 生成答案可能是结构化 JSON（{"diagnosis": ..., "evidence": [...]}）或
        markdown 代码块，期望串被包在 JSON 里无法子串命中 → 先解包
        diagnosis / evidence / 去代码围栏
      - 空白差异（"0.981" vs "0.981 "）归一化后匹配
    语义不变：仍是"关键信息出现在答案（含引用证据）中"的宽松判定。
    """
    if not expected:
        return False
    expected = str(expected).strip()
    if not expected:
        return False
    text = str(answer)
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            parts = []
            if data.get("diagnosis"):
                parts.append(str(data["diagnosis"]))
            for ev in data.get("evidence", []) or []:
                if isinstance(ev, str):
                    parts.append(ev)
            if parts:
                text = " ".join(parts)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    def norm(s: str) -> str:
        return re.sub(r"\s+", "", s)

    return norm(expected) in norm(text)


def policy_action_accuracy(route: list[str], expected_route: list[str]) -> bool:
    """Policy Action Accuracy：实际 route 与预期 route 的宽松匹配

    规则：终局动作一致（ACCEPT/ABSTAIN），且对 multi-hop/comparison/partial
    类要求循环内出现过 DECOMPOSE 或 RETRIEVE。

    B2 预检兼容：v2.1 对可靠多问号在位置 0 直进 DECOMPOSE（跳过全问题初始
    检索）——该预检动作等价于循环内拆解（hop 定向检索在 decompose 分支内
    执行），计入"有动作"；否则 B2 优化会被误判为"没有拆解"。
    """
    if not expected_route:
        return route[-1] == "ABSTAIN"
    if route[-1] != expected_route[-1]:
        return False
    # ABSTAIN 类：终局一致即可
    if route[-1] == "ABSTAIN":
        return True
    # easy/constraint 类：循环内不应过度动作
    if len(expected_route) == 2:
        loop_actions = [a for a in route[1:-1] if a in ("RETRIEVE", "DECOMPOSE")]
        return not loop_actions
    # multi-hop/comparison/partial 类：循环内必须有动作
    loop_actions = [a for a in route[1:-1] if a in ("RETRIEVE", "DECOMPOSE")]
    precheck_decompose = bool(route) and route[0] == "DECOMPOSE" and len(route) >= 2
    return bool(loop_actions) or precheck_decompose


def retry_recovery(route: list[str], hit: bool) -> bool:
    """Retry Recovery：循环内 RETRIEVE 后最终 ACCEPT 且命中"""
    loop_retrieve = "RETRIEVE" in route[1:-1]
    return loop_retrieve and route[-1] == "ACCEPT" and hit


def decomposition_success(route: list[str], hit: bool) -> bool:
    """Decomposition Success：触发 DECOMPOSE 且最终 ACCEPT 且命中"""
    return "DECOMPOSE" in route and route[-1] == "ACCEPT" and hit


def compute_agent_capability_metrics(cases: list[dict], k: int = TOP_K) -> dict[str, Any]:
    """Step 12 完整指标族汇总

    每题 case 需含:
        {
          "question": dict,        # benchmark 条目（含 hops/type）
          "v0_sources": [...],     # Fixed RAG 最终 TopK
          "v1_sources": [...],     # Agent 最终 evidence
          "v1_route": [...],       # Agent route（含循环内动作）
          "v1_answer": str,        # Agent 生成答案
          "v1_abstained": bool,
        }

    Returns: 全部指标 + 逐题 details（供 Failure Anatomy）。
    """
    n = len(cases)
    agg = {
        "n": n,
        "final_answer_accuracy": 0,
        "evidence_recall": 0.0,
        "hop_recall": 0.0,
        "completeness": 0.0,
        "final_rescue": 0,
        "harm": 0,
        "ood_reject_ok": 0,
        "ood_total": 0,
        "false_abstain": 0,
        "answerable_total": 0,
        "policy_action_accuracy": 0,
        "decomposition_success": 0,
        "retry_recovery": 0,
        "unnecessary_actions": 0,
        "avg_iterations": 0.0,
        "by_type": {},
    }
    details = []
    for c in cases:
        q = c["question"]
        qtype = q.get("type", "?")
        all_gold = set()
        for hg in hop_gold_ids(q):
            all_gold |= hg
        expected = q.get("final_answer", "")

        v0_sources = c.get("v0_sources", [])
        v1_sources = c.get("v1_sources", [])
        route = c.get("v1_route", [])
        abstained = c.get("v1_abstained", False)
        answer = c.get("v1_answer", "")

        # 命中（chunk-level，有 gold 才判）
        has_gold = bool(all_gold)
        v0_hit = evidence_recall_at_k(v0_sources, all_gold, k) > 0 if has_gold else False
        v1_hit = evidence_recall_at_k(v1_sources, all_gold, k) > 0 if has_gold else False

        # Final Answer
        fa = final_answer_accuracy(answer, expected)
        agg["final_answer_accuracy"] += fa

        # Evidence 指标
        er = evidence_recall_at_k(v1_sources, all_gold, k)
        hr = hop_recall_at_k(v1_sources, q, k)
        comp = evidence_completeness(v1_sources, q, k)
        agg["evidence_recall"] += er
        agg["hop_recall"] += hr
        agg["completeness"] += comp

        # Rescue / Harm（相对 V0，只有有 gold 的题有意义）
        if has_gold:
            if not v0_hit and v1_hit:
                agg["final_rescue"] += 1
            if v0_hit and not v1_hit:
                agg["harm"] += 1

        # OOD / 误拒
        if qtype == "unsupported_ood":
            agg["ood_total"] += 1
            agg["ood_reject_ok"] += abstained
        else:
            agg["answerable_total"] += 1
            if abstained:
                agg["false_abstain"] += 1

        # Policy 行为
        agg["policy_action_accuracy"] += policy_action_accuracy(route, q.get("expected_route", []))
        agg["decomposition_success"] += decomposition_success(route, v1_hit)
        agg["retry_recovery"] += retry_recovery(route, v1_hit)
        agg["unnecessary_actions"] += unnecessary_action_rate(route, qtype)
        agg["avg_iterations"] += len(route) - 1

        # 分类型统计
        agg["by_type"].setdefault(qtype, {"n": 0, "rescue": 0, "hit": 0, "false_abstain": 0})
        agg["by_type"][qtype]["n"] += 1
        agg["by_type"][qtype]["rescue"] += (not v0_hit and v1_hit) if has_gold else 0
        agg["by_type"][qtype]["hit"] += v1_hit
        agg["by_type"][qtype]["false_abstain"] += abstained

        details.append(
            {
                "id": q.get("id", ""),
                "type": qtype,
                "question": q.get("question", ""),
                "v0_hit": v0_hit,
                "v1_hit": v1_hit,
                "v0_evidence_recall": round(evidence_recall_at_k(v0_sources, all_gold, k), 3),
                "v1_evidence_recall": round(er, 3),
                "hop_recall": round(hr, 3),
                "completeness": round(comp, 3),
                "final_answer_accuracy": fa,
                "route": route,
                "abstained": abstained,
                "class": classify_pair(v0_hit, v1_hit)
                if has_gold
                else ("ood" if qtype == "unsupported_ood" else "no_gold"),
            }
        )

    n_ans = max(agg["answerable_total"], 1)
    n_ood = max(agg["ood_total"], 1)
    return {
        "n": n,
        "final_answer_accuracy": f"{agg['final_answer_accuracy']}/{n}",
        "evidence_recall": round(agg["evidence_recall"] / n, 3),
        "hop_recall": round(agg["hop_recall"] / n, 3),
        "completeness": round(agg["completeness"] / n, 3),
        "final_rescue": agg["final_rescue"],
        "harm": agg["harm"],
        "net_utility": agg["final_rescue"] - agg["harm"],
        "ood_reject": f"{agg['ood_reject_ok']}/{agg['ood_total']}",
        "false_abstain": f"{agg['false_abstain']}/{agg['answerable_total']}",
        "policy_action_accuracy": f"{agg['policy_action_accuracy']}/{n}",
        "decomposition_success": agg["decomposition_success"],
        "retry_recovery": agg["retry_recovery"],
        "unnecessary_action_rate": f"{agg['unnecessary_actions']}/{agg['answerable_total']}",
        "avg_iterations": round(agg["avg_iterations"] / n, 2),
        "by_type": agg["by_type"],
        "details": details,
    }


# ══════════════════════════════════════════════════
#  汇总入口
# ══════════════════════════════════════════════════


def compute_rescue_metrics(
    cases: list[dict],
    k: int = TOP_K,
) -> dict[str, Any]:
    """对逐题 case 列表计算 Rescue/Harm/NetUtility 汇总

    每题的 case 需含:
        {
          "question": str,
          "gold_ids": set[str] (或留空退化 document-level),
          "expected_doc": str,
          "v0_sources": [...],   # baseline 检索结果（rerank 后 TopK）
          "v1_sources": [...],   # 变体检索结果（rerank 后 TopK）
        }

    Returns:
        {
          "n": int,
          "final_hit_v0": int,       # baseline FinalHit
          "final_hit_v1": int,       # 变体 FinalHit
          "rescue": int,             # baseline miss → variant hit
          "harm": int,               # baseline hit → variant miss
          "net_utility": int,        # rescue − harm
          "details": [...],          # 逐题分类
        }
    """
    details = []
    for c in cases:
        gold_ids = c.get("gold_ids") or set()
        expected_doc = c.get("expected_doc", "")
        v0_hit = hit_at_k(c.get("v0_sources", []), gold_ids, expected_doc, k=k)
        v1_hit = hit_at_k(c.get("v1_sources", []), gold_ids, expected_doc, k=k)
        details.append(
            {
                "question": c.get("question", ""),
                "class": classify_pair(v0_hit, v1_hit),
                "v0_hit": v0_hit,
                "v1_hit": v1_hit,
                "v0_rank": gold_rank(c.get("v0_sources", []), gold_ids, expected_doc),
                "v1_rank": gold_rank(c.get("v1_sources", []), gold_ids, expected_doc),
            }
        )

    final_hit_v0 = sum(1 for d in details if d["v0_hit"])
    final_hit_v1 = sum(1 for d in details if d["v1_hit"])
    rescues = sum(1 for d in details if d["class"] == "rescue")
    harms = sum(1 for d in details if d["class"] == "harm")

    return {
        "n": len(cases),
        "final_hit_v0": final_hit_v0,
        "final_hit_v1": final_hit_v1,
        "rescue": rescues,
        "harm": harms,
        "net_utility": net_utility(rescues, harms),
        "details": details,
    }
