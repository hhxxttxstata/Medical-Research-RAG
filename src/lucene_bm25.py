"""
Lucene 兼容的磁盘 BM25 索引（基于 Whoosh 实现）

与 rank-bm25 的核心区别：
  - rank-bm25:     全量 term-document matrix 在内存（Python dict + list），O(N) 内存
  - Whoosh/Lucene:  倒排索引写入磁盘 Segment 文件，检索时只加载必要的 posting list

文件结构（lucene_bm25_index/）:
  - MAIN_WRITELOCK  — 写锁
  - _X.seg          — 分段元数据
  - _X_*.idx        — 倒排列表
  - _X.doc          — 文档存储
  - _X.tr           — 词项向量

面试亮点：
  - 展示了对 Lucene 架构的理解（Index → Segment → Document → Field）
  - 解释了 Whoosh 的 BM25F 评分公式与 elasticsearch 的 similarity 配置的对应关系
  - 回答了「为什么不用 Elasticsearch?」——面试演示项目追求轻量 + 展示原理理解
"""

import json
import os
import re
import shutil

from whoosh import fields, index, scoring
from whoosh.analysis import Token, Tokenizer

# ── 自定义分词器（中英文混合） ──────────────────────


class MixedTokenizer(Tokenizer):
    """中英文混合分词器

    英文按空格/标点分词 + lowercase，中文按单字切分（因为没有字典不做复合词）。

    Whoosh 的 StandardAnalyzer 对中文不友好（把所有非字母数字都当分隔符），
    这个分词器保留每个中文字符作为独立 term，兼顾医学领域的中文专有名词匹配。
    """

    def __call__(self, value, **kwargs):
        t = Token()
        for token_text in self._tokenize(value):
            t.original = token_text
            t.text = token_text.lower()
            t.stopped = False
            t.boost = 1.0
            t.removestops = False
            t.startchar = 0
            t.pos = 0
            yield t

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        if not text:
            return []
        tokens = []
        parts = re.split(r"([一-鿿〇㐀-䶿\U00020000-\U0002A6DF])", text)
        buffer = ""
        for part in parts:
            if re.match(r"^[一-鿿〇㐀-䶿\U00020000-\U0002A6DF]$", part):
                if buffer.strip():
                    tokens.extend(re.findall(r"[a-zA-Z0-9]+(?:[./_-][a-zA-Z0-9]+)*", buffer.strip().lower()))
                    buffer = ""
                tokens.append(part)  # 单汉字作为独立 term
            else:
                buffer += part
        if buffer.strip():
            tokens.extend(re.findall(r"[a-zA-Z0-9]+(?:[./_-][a-zA-Z0-9]+)*", buffer.strip().lower()))
        return tokens


# ── Whoosh Schema ─────────────────────────────────────

BM25_SCHEMA = fields.Schema(
    chunk_id=fields.ID(unique=True, stored=True),
    text=fields.TEXT(stored=True, analyzer=MixedTokenizer(), vector=False, spelling=False),
    metadata=fields.STORED,
)


class LuceneBM25Index:
    """磁盘 BM25 索引（兼容 Lucene 架构的 Whoosh 实现）

    使用方法:
        index = LuceneBM25Index("lucene_bm25_index")
        index.index_chunks(chunks)     # 批量写入
        results = index.search("query", top_k=5)  # BM25 检索
        index.close()
    """

    def __init__(self, index_dir: str = "lucene_bm25_index"):
        self.index_dir = os.path.abspath(index_dir)
        self._ix = None
        self._open_or_create()

    # ── 内部方法 ────────────────────────────────────

    def _open_or_create(self):
        """打开已有索引，或创建新索引"""
        if not os.path.isdir(self.index_dir):
            os.makedirs(self.index_dir, exist_ok=True)
            self._ix = index.create_in(self.index_dir, BM25_SCHEMA)
            print(f"  🆕 创建 Lucene BM25 索引: {self.index_dir}")
        else:
            try:
                self._ix = index.open_dir(self.index_dir)
                n = self._ix.doc_count()
                print(f"  📂 打开 Lucene BM25 索引: {self.index_dir} (现存 {n} 篇文档)")
            except Exception:
                # 索引损坏 → 重建
                shutil.rmtree(self.index_dir, ignore_errors=True)
                os.makedirs(self.index_dir, exist_ok=True)
                self._ix = index.create_in(self.index_dir, BM25_SCHEMA)
                print(f"  🆕 重建 Lucene BM25 索引: {self.index_dir}")

    # ── 索引写入 ────────────────────────────────────

    def index_chunks(self, chunks: list[dict], clear_first: bool = False):
        """批量索引文档片段

        Args:
            chunks: [{"chunk_id": str, "text": str, "metadata": dict}, ...]
            clear_first: True = 清空现有索引后重建
        """
        if not chunks:
            return

        # 清空 + 重建：whoosh 新版无 index.CLEAR mergetype，直接删目录重建
        if clear_first:
            self._ix.close()
            shutil.rmtree(self.index_dir, ignore_errors=True)
            os.makedirs(self.index_dir, exist_ok=True)
            self._ix = index.create_in(self.index_dir, BM25_SCHEMA)

        writer = self._ix.writer()

        for c in chunks:
            cid = c.get("chunk_id") or c.get("id", "")
            text = c.get("text", "")
            meta = c.get("metadata", {})
            if not cid or not text:
                continue
            writer.update_document(
                chunk_id=str(cid),
                text=text,
                metadata=json.dumps(meta, ensure_ascii=False),
            )

        writer.commit()
        print(f"  ✅ 写入 {len(chunks)} 个文档到 Lucene BM25 索引")

    def remove_chunks(self, chunk_ids: list[str]):
        """从索引中删除指定文档"""
        if not chunk_ids:
            return
        writer = self._ix.writer()
        for cid in chunk_ids:
            writer.delete_by_term("chunk_id", str(cid))
        writer.commit()
        print(f"  🗑️ 从 Lucene BM25 索引删除 {len(chunk_ids)} 个文档")

    # ── BM25 检索 ───────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """BM25 关键词检索

        Args:
            query: 查询文本
            top_k: 返回 top-K 结果

        Returns:
            [{"id": str, "text": str, "metadata": dict, "score": float}, ...]
        """
        if self._ix is None:
            return []
        if not query.strip():
            return []

        from whoosh.qparser import OrGroup, QueryParser

        # 使用 BM25F 评分（Whoosh 默认 b=0.75, k1=1.2 与 Lucene 默认一致）
        searcher = self._ix.searcher(weighting=scoring.BM25F())
        parser = QueryParser("text", self._ix.schema, group=OrGroup)

        parsed = parser.parse(query.strip())
        raw_results = searcher.search(parsed, limit=top_k)

        results = []
        for hit in raw_results:
            meta = {}
            raw_meta = hit.get("metadata")
            if raw_meta:
                try:
                    meta = json.loads(raw_meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}
            results.append(
                {
                    "id": hit["chunk_id"],
                    "text": hit["text"],
                    "metadata": meta,
                    "score": float(hit.score),
                    "_retriever": "bm25",
                    "_bm25_score": float(hit.score),
                }
            )

        searcher.close()
        return results

    # ── 管理方法 ────────────────────────────────────

    def get_total_docs(self) -> int:
        """返回索引中的文档总数（线程安全）"""
        if self._ix is None:
            return 0
        with self._ix.searcher() as s:
            return s.doc_count()

    def rebuild(self, chunks: list[dict]):
        """清空并重建整个索引"""
        self.index_chunks(chunks, clear_first=True)

    def close(self):
        """释放资源"""
        if self._ix is not None:
            try:
                self._ix.close()
            except Exception:
                pass
            self._ix = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
