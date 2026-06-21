"""
日志记录模块
记录用户问题、召回文档、模型回答、耗时、错误信息、相关性判断
"""

import json
import os
from datetime import datetime
from typing import Any


class RAGLogger:
    """RAG 系统日志记录器"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        # 按日期生成日志文件
        self.date_str = datetime.now().strftime("%Y-%m-%d")
        self.log_file = os.path.join(self.log_dir, f"rag_{self.date_str}.jsonl")
        self.stats_file = os.path.join(self.log_dir, f"stats_{self.date_str}.json")

        # 当日统计
        self._stats = {
            "date": self.date_str,
            "total_queries": 0,
            "success_count": 0,
            "error_count": 0,
            "refusal_count": 0,
            "avg_response_time": 0.0,
            "total_response_time": 0.0,
            "avg_retrieval_scores": [],  # 新增：检索平均分追踪
            "avg_overlap_rates": [],  # 新增：文本重叠率追踪
        }
        self._load_stats()
        # 确保加载后新字段存在（兼容旧版 stats.json）
        self._stats.setdefault("avg_retrieval_scores", [])
        self._stats.setdefault("avg_overlap_rates", [])

    def _load_stats(self):
        """加载当日已有统计"""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, encoding="utf-8") as f:
                    self._stats = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass

    def _save_stats(self):
        """保存统计信息"""
        with open(self.stats_file, "w", encoding="utf-8") as f:
            json.dump(self._stats, f, ensure_ascii=False, indent=2)

    def log_query(
        self,
        question: str,
        retrieved_chunks: list[dict[str, Any]],
        answer: str,
        elapsed: float,
        error: str | None = None,
        is_refusal: bool = False,
        chunk_min_chars: int | None = None,
        chunk_max_chars: int | None = None,
        top_k: int | None = None,
        relevance: dict[str, Any] | None = None,
        trace_id: str = "",
        span_id: str = "",
    ):
        """记录一次查询日志（增强版）"""
        self._stats["total_queries"] += 1
        self._stats["total_response_time"] += elapsed

        if error:
            self._stats["error_count"] += 1
        else:
            self._stats["success_count"] += 1

        if is_refusal:
            self._stats["refusal_count"] += 1

        # 更新平均耗时
        if self._stats["total_queries"] > 0:
            self._stats["avg_response_time"] = round(
                self._stats["total_response_time"] / self._stats["total_queries"], 2
            )

        # 收集检索质量数据
        if retrieved_chunks:
            scores = [c["score"] for c in retrieved_chunks]
            avg_score = sum(scores) / len(scores)
            self._stats["avg_retrieval_scores"].append(avg_score)
            # 保留最近 100 条
            if len(self._stats["avg_retrieval_scores"]) > 100:
                self._stats["avg_retrieval_scores"] = self._stats["avg_retrieval_scores"][-100:]
        if relevance and "overlap" in relevance:
            self._stats["avg_overlap_rates"].append(relevance["overlap"])
            if len(self._stats["avg_overlap_rates"]) > 100:
                self._stats["avg_overlap_rates"] = self._stats["avg_overlap_rates"][-100:]

        # 构建日志记录（增强版）
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "retrieved_chunks": [
                {
                    "id": c["id"],
                    "filename": c["metadata"].get("filename", "未知"),
                    "page": c["metadata"].get("page"),
                    "paragraph_start": c["metadata"].get("paragraph_start"),
                    "paragraph_end": c["metadata"].get("paragraph_end"),
                    "score": round(c["score"], 4),
                    "text_preview": c["text"][:200],
                }
                for c in retrieved_chunks
            ],
            "answer": answer,
            "elapsed_seconds": round(elapsed, 2),
            "error": error,
            "is_refusal": is_refusal,
            "chunk_min_chars": chunk_min_chars,
            "chunk_max_chars": chunk_max_chars,
            "top_k": top_k,
            "relevance": relevance,
            "num_retrieved": len(retrieved_chunks),
        }

        # 附加 trace context（如果存在）
        if trace_id:
            log_entry["trace_id"] = trace_id
        if span_id:
            log_entry["span_id"] = span_id

        # 写入日志文件（JSONL 格式）
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        self._save_stats()

    def get_today_stats(self) -> dict[str, Any]:
        """获取当日统计"""
        stats = dict(self._stats)
        # 计算检索质量汇总
        if stats.get("avg_retrieval_scores"):
            scores = stats["avg_retrieval_scores"]
            stats["avg_retrieval_score_mean"] = round(sum(scores) / len(scores), 4)
        if stats.get("avg_overlap_rates"):
            rates = stats["avg_overlap_rates"]
            stats["avg_overlap_rate_mean"] = round(sum(rates) / len(rates), 4)
            stats["refusal_rate"] = (
                round(stats["refusal_count"] / stats["total_queries"] * 100, 1) if stats["total_queries"] > 0 else 0
            )
        return stats

    def get_recent_queries(self, n: int = 10) -> list[dict[str, Any]]:
        """获取最近 N 条查询记录"""
        if not os.path.exists(self.log_file):
            return []

        records = []
        with open(self.log_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        return records[-n:]

    def print_summary(self):
        """打印日志摘要"""
        stats = self.get_today_stats()
        print("\n" + "=" * 60)
        print("  📊 RAG 运行统计")
        print("=" * 60)
        print(f"  日期: {stats['date']}")
        print(f"  总查询: {stats['total_queries']}")
        print(f"  成功: {stats['success_count']}")
        print(f"  失败: {stats['error_count']}")
        print(f"  拒答: {stats['refusal_count']} ({stats.get('refusal_rate', 0):.1f}%)")
        print(f"  平均耗时: {stats['avg_response_time']:.2f} 秒")
        print(f"  总耗时: {stats['total_response_time']:.2f} 秒")
        if "avg_retrieval_score_mean" in stats:
            print(f"  平均检索分: {stats['avg_retrieval_score_mean']:.4f}")
        if "avg_overlap_rate_mean" in stats:
            print(f"  平均文本重叠率: {stats['avg_overlap_rate_mean']:.4f}")
        print(f"  日志文件: {self.log_file}")
        print("=" * 60 + "\n")


# 全局单例
_logger: RAGLogger | None = None


def get_logger(log_dir: str = "logs") -> RAGLogger:
    """获取日志记录器（单例）"""
    global _logger
    if _logger is None:
        _logger = RAGLogger(log_dir)
    return _logger
