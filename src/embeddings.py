"""
Embedding 模块
将文本 Chunk 转换为向量表示
支持 Sentence-Transformers 和 OpenAI Embedding API

默认模型: BAAI/bge-multilingual-gemma2
- Google Gemma2 基座 + BGE 多语言微调，支持 100+ 语言
- 索引时自动加 "passage: " 前缀，查询时加 "query: " 前缀
- 维度 2048，精度远超 multilingual-e5-small (384d)
"""

import os
import re

from dotenv import load_dotenv

load_dotenv()

# ── 语言检测辅助（供其他模块导入） ──────────────

_ENGLISH_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
# 预编译 regex，避免每调用一次编译一次
_WORD_RE = re.compile(r"[a-zA-Z]+")
_CJK_RE = re.compile(r"[一-鿿]")


def is_mostly_english(text: str) -> bool:
    """检测文本是否以英文为主

    规则：
      1) 英文字母占总字符数 > 40% → 英文（快速路径）
      2) 英文字母 25-40% → 比较英文单词数与中文字数，英:中 > 2:1 则判定为英文
         （解决医学文献含大量数字/希腊字母/符号时拉丁字母比例偏低的问题）
      3) 其他 → 中文

    用于路由到不同的 chunk 策略 / relevance 算法 / embedding 前缀。
    """
    if not text.strip():
        return False

    total = len(text)
    en_count = sum(1 for c in text if c in _ENGLISH_CHARS)

    # 规则 1：高英文字母比例 → 直接判定为英文
    if en_count > total * 0.40:
        return True

    # 规则 2：英文字母在 25-40% 之间 → 用英文词/中文字数比验证
    if en_count > total * 0.25:
        en_words = len(_WORD_RE.findall(text))
        cjk_chars = len(_CJK_RE.findall(text))
        # 英文单词数超过中文字数 2 倍 → 判定为英文
        if en_words > cjk_chars * 2:
            return True

    return False


# ── Embedding 模型名称常量 ─────────────────────

DEFAULT_MODEL = "intfloat/multilingual-e5-base"
"""默认多语言模型：E5 中型版本，768 维，中英文兼具，适合医学文献检索。"""


class EmbeddingProvider:
    """Embedding 提供者基类"""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class SentenceTransformerEmbedding(EmbeddingProvider):
    """基于 Sentence-Transformers 的本地 Embedding 模型

    支持可选的前缀策略（e5 系列模型需要 "query:" / "passage:" 前缀）。
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None
        self._is_e5 = "e5" in model_name.lower()
        self._cache = None  # EmbeddingCache 实例，由外部设置
        # ModelScope 本地缓存路径
        self._ms_local_path = os.path.join(
            os.path.expanduser("~/.cache/modelscope/hub"),
            "AI-ModelScope",
            model_name.replace("/", "--"),
        )
        if not os.path.isdir(self._ms_local_path):
            base_dir = os.path.dirname(self._ms_local_path)
            if os.path.isdir(base_dir):
                for d in os.listdir(base_dir):
                    if model_name.split("/")[-1] in d:
                        self._ms_local_path = os.path.join(base_dir, d)
                        break

    def _resolve_model_path(self) -> str:
        """确定模型路径：优先本地缓存，其次 ModelScope 缓存"""
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(self.model_name, "config.json")
            if cached and os.path.isfile(cached):
                return self.model_name
        except Exception:
            pass

        local_path = os.path.abspath(self.model_name)
        if os.path.isdir(local_path):
            return local_path

        if os.path.isdir(self._ms_local_path):
            return self._ms_local_path

        return self.model_name

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            model_path = self._resolve_model_path()
            print(f"  📦 加载 Embedding 模型: {model_path}")
            is_local = os.path.isdir(model_path) or os.path.isfile(
                os.path.join(
                    os.path.dirname(model_path) if os.path.isfile(model_path) else model_path,
                    "config.json",
                )
            )
            if is_local:
                try:
                    self._model = SentenceTransformer(model_path, local_files_only=True)
                except Exception:
                    self._model = SentenceTransformer(model_path)
            else:
                self._model = SentenceTransformer(model_path)
            print(f"  ✅ 模型加载完成 (维度: {self._model.get_sentence_embedding_dimension()})")

    def warmup(self) -> None:
        """预热：加载模型并跑一次小推理，避免首次请求卡顿"""
        self._load_model()
        _ = self._model.encode(["warmup"], show_progress_bar=False, normalize_embeddings=True)

    def embed(
        self,
        texts: list[str],
        prefix: str | None = None,
    ) -> list[list[float]]:
        """生成文本向量（带 EmbeddingCache）"""
        self._load_model()

        if prefix:
            texts = [f"{prefix}{t}" for t in texts]

        # 单文本走缓存，批量不走（批量通常是首次入库）
        if len(texts) == 1 and self._cache is not None:
            cached = self._cache.get(texts[0])
            if cached:
                return [cached]
            embedding = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            result = embedding.tolist()
            self._cache.set(texts[0], result[0])
            return result

        embeddings = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()


class OpenAIEmbedding(EmbeddingProvider):
    """基于 OpenAI API 的 Embedding"""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def embed(self, texts: list[str], prefix: str | None = None) -> list[list[float]]:
        from openai import OpenAI

        if prefix:
            texts = [f"{prefix}{t}" for t in texts]

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]


def get_embedding_provider(provider: str = "local", model_name: str | None = None) -> EmbeddingProvider:
    """获取 Embedding 提供者"""
    if provider == "openai":
        return OpenAIEmbedding(model=model_name or "text-embedding-3-small")
    else:
        return SentenceTransformerEmbedding(model_name=model_name or DEFAULT_MODEL)
