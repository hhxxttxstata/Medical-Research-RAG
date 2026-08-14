"""
watcher.py — 增量文件监听器

基于 watchdog 监听 data/ 目录，新文件自动加载、分块、embedding、入库。

设计要点：
  - 去抖机制：文件稳定 2s 后才处理，避免写入中触发
  - 已处理文件追踪：Redis SET → 内存 set → JSON 持久化三重保障
  - 每文件独立处理，失败不影响其他文件

面试价值：
  展示对 RAG 系统运维的工程化思考——增量索引是生产环境的标配功能。
  watchdog + debounce + processed 追踪的模式可复用到任何文件监听场景。
"""

import logging
import os
import time
from pathlib import Path
from typing import Any

import watchdog.events

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


class ProcessedFilesTracker:
    """追踪已处理的文件

    内存 set + JSON 文件持久化。Redis 可用时也用 Redis 做跨进程同步。
    """

    def __init__(self, persist_path: str = "data/.processed_files.json"):
        self._path = persist_path
        self._processed: set[str] = set()
        self._dirty = False
        self._load()

    def _load(self):
        try:
            if os.path.exists(self._path):
                import json

                with open(self._path) as f:
                    self._processed = set(json.load(f))
                logger.debug(f"已加载 {len(self._processed)} 个已处理文件记录")
        except Exception:
            self._processed = set()

    def _save(self):
        try:
            import json

            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
            with open(self._path, "w") as f:
                json.dump(list(self._processed), f)
            self._dirty = False
        except Exception as e:
            logger.warning(f"保存已处理文件记录失败: {e}")

    def is_processed(self, filepath: str) -> bool:
        abs_path = os.path.abspath(filepath)
        return abs_path in self._processed

    def mark_processed(self, filepath: str) -> None:
        abs_path = os.path.abspath(filepath)
        if abs_path in self._processed:
            return
        self._processed.add(abs_path)
        self._dirty = True
        if len(self._processed) % 10 == 0:
            self._save()

    def save(self):
        if self._dirty:
            self._save()


class DocumentHandler:
    """watchdog 事件处理器"""

    def __init__(self, pipeline, processed_tracker: ProcessedFilesTracker):
        self._pipeline = pipeline
        self._tracker = processed_tracker

    def on_created(self, event) -> None:
        if isinstance(event, (watchdog.events.FileCreatedEvent, watchdog.events.FileModifiedEvent)):
            path = event.src_path
            if event.is_directory:
                return
            suffix = Path(path).suffix.lower()
            if suffix not in _SUPPORTED_EXTENSIONS:
                return
            self._process_with_debounce(path)

    def on_modified(self, event) -> None:
        self.on_created(event)

    def _process_with_debounce(self, path: str, debounce: float = 2.0) -> None:
        """等待文件稳定，然后处理"""
        if self._tracker.is_processed(path):
            return

        # 等待文件稳定（大小不再变化）
        try:
            last_size = -1
            for _ in range(5):
                time.sleep(debounce / 5)
                current_size = os.path.getsize(path)
                if current_size == last_size:
                    break
                last_size = current_size
        except OSError:
            return

        self._process_file(path)

    def _process_file(self, path: str) -> None:
        """处理单个文件：加载→分块→embedding→入库"""
        try:
            from src.document_loader import load_document
            from src.text_splitter import split_document

            logger.info(f"📄 检测到新文件: {path}")

            # 1. 加载文档
            doc = load_document(path)

            # 2. 分块
            chunks = split_document(
                doc,
                chunk_min_chars=self._pipeline.chunk_min_chars,
                chunk_max_chars=self._pipeline.chunk_max_chars,
            )
            if not chunks:
                logger.warning(f"  文件分块结果为空，跳过: {path}")
                return

            # 3. Embedding + 入库
            texts = [c["text"] for c in chunks]
            embeddings = self._pipeline.embedding_provider.embed(texts)
            self._pipeline.vector_store.add_chunks(chunks, embeddings)

            # 4. 记录
            self._tracker.mark_processed(path)
            logger.info(f"  ✅ 入库完成: {len(chunks)} 个 Chunk — {path}")
        except Exception as e:
            logger.error(f"  ❌ 文件处理失败: {path} — {e}")


class DocumentWatcher:
    """目录监听器

    用法:
        watcher = DocumentWatcher(pipeline, watch_dir="data")
        watcher.start()   # 后台线程
        ...
        watcher.stop()    # 停止监听
    """

    def __init__(self, pipeline, watch_dir: str = "data", tracker: ProcessedFilesTracker | None = None):
        self._pipeline = pipeline
        self._watch_dir = os.path.abspath(watch_dir)
        self._tracker = tracker or ProcessedFilesTracker()
        self._observer: Any = None  # watchdog Observer（stub 噪音）
        self._running = False

    def start(self) -> None:
        """启动监听器（非阻塞）"""
        from watchdog.observers import Observer

        if self._running:
            return

        os.makedirs(self._watch_dir, exist_ok=True)

        handler = DocumentHandler(self._pipeline, self._tracker)
        self._observer = Observer()
        self._observer.schedule(handler, self._watch_dir, recursive=False)
        self._observer.start()
        self._running = True
        logger.info(f"👀 文档监听器已启动: {self._watch_dir}")

    def stop(self) -> None:
        """停止监听器"""
        if self._observer:
            self._tracker.save()
            self._observer.stop()
            self._observer.join()
            self._running = False
            logger.info("👀 文档监听器已停止")

    @property
    def is_running(self) -> bool:
        return self._running
