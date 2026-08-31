"""
config_snapshot.py — 评测运行的配置指纹（baseline 版本化的基础）

每次评测报告写入 config_snapshot，使数字可追溯到：
    git SHA + 代码指纹（含 prompt/策略）+ 模型名 + 检索参数 + 数据集文件 hash

regression.py 对比基线前先校验 snapshot 关键字段，不一致则显式警告——
禁止"数字对不上却不知道为什么"的静默对比（handoff v2 §16 的实证教训：
原基线数字在任何一份磁盘报告里都找不到）。

用法:
    from eval.config_snapshot import build_config_snapshot
    snap = build_config_snapshot(dataset_files=["tests/benchmark_holdout.json"], fetch_k=20, top_k=5)
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 允许 `python eval/config_snapshot.py` 独立运行（脚本目录不含仓库根）
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 与 src/reranker.py CrossEncoderReranker.MODEL_NAME 一致（导入该模块会拖起
# sentence_transformers，故此处以常量声明 + 导入时覆盖的方式取值）
_RERANKER_FALLBACK = "BAAI/bge-reranker-v2-m3"


def git_sha() -> tuple[str, bool]:
    """返回 (git SHA, 工作区是否 dirty)。无 git 环境/失败 → ("unknown", True)"""
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        sha = r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        sha = "unknown"
    try:
        d = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        dirty = d.returncode != 0 or bool(d.stdout.strip())
    except Exception:
        dirty = True
    return sha, dirty


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return f"sha256:{h.hexdigest()[:16]}"


def code_fingerprint() -> str:
    """src/ 全部 .py 排序后联合 hash——prompt 模板、策略、生成逻辑的任何改动都会改变指纹。

    比单记 "prompt 版本号" 更可靠：不依赖人工记得改版本号。
    """
    h = hashlib.sha256()
    for f in sorted((ROOT / "src").glob("*.py")):
        h.update(f.name.encode("utf-8"))
        h.update(f.read_bytes())
    return f"sha256:{h.hexdigest()[:16]}"


def _embedding_model() -> str:
    env = os.getenv("EMBEDDING_MODEL")
    if env:
        return env
    try:
        from src.embeddings import DEFAULT_MODEL

        return DEFAULT_MODEL
    except Exception:
        return "unknown"


def _reranker_model() -> str:
    return os.getenv("RERANKER_MODEL", _RERANKER_FALLBACK)


def _generator_model() -> str:
    return os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def dataset_hashes(dataset_files: list[str] | None = None) -> dict[str, str]:
    """数据集文件 → sha256（前 16 位）。文件缺失时记录 "missing" 而非静默跳过。"""
    out: dict[str, str] = {}
    for rel in dataset_files or []:
        p = ROOT / rel
        out[rel] = _sha256_file(p) if p.exists() else "missing"
    return out


def build_config_snapshot(
    dataset_files: list[str] | None = None,
    top_k: int | None = None,
    fetch_k: int | None = None,
    bench: str | None = None,
    extra: dict | None = None,
) -> dict:
    """构建当前评测运行的配置快照。

    Args:
        dataset_files: 参与本次评测的数据集相对路径（计算 hash）
        top_k / fetch_k: 检索参数（holdout_eval 冻结 5 / 20）
        bench: 题集文件路径（单独记录，与 dataset_files 可重叠）
        extra: 调用方附加字段（如 agents 列表）
    """
    sha, dirty = git_sha()
    snap = {
        "git_sha": sha,
        "git_dirty": dirty,
        "code_fingerprint": code_fingerprint(),
        "models": {
            "embedding": _embedding_model(),
            "reranker": _reranker_model(),
            "generator": _generator_model(),
            "grader": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        },
        "retrieval": {"top_k": top_k, "fetch_k": fetch_k},
        "bench": bench,
        "datasets": dataset_hashes(dataset_files or []),
        "python": sys.version.split()[0],
    }
    if extra:
        snap.update(extra)
    return snap


def snapshot_values(snap: dict, key_paths: list[str]) -> dict:
    """按 "a.b.c" 点路径取 snapshot 字段（gates.json regression_snapshot_keys 的消费端）"""
    out = {}
    for kp in key_paths:
        cur: object = snap
        for part in kp.split("."):
            if isinstance(cur, dict):
                cur = cur.get(part)
            else:
                cur = None
                break
        out[kp] = cur
    return out


def diff_snapshots(current: dict, baseline: dict, key_paths: list[str]) -> list[str]:
    """返回关键口径不一致的描述列表（空列表 = 口径一致）。None 视为未知，不告警。"""
    diffs = []
    cur_vals = snapshot_values(current, key_paths)
    base_vals = snapshot_values(baseline, key_paths)
    for kp in key_paths:
        c, b = cur_vals[kp], base_vals[kp]
        if c is None or b is None:
            continue
        if c != b:
            diffs.append(f"{kp}: 基线={b} → 当前={c}")
    return diffs


if __name__ == "__main__":
    snap = build_config_snapshot(
        dataset_files=["tests/benchmark_holdout.json", "tests/benchmark_multi_hop.json"],
        top_k=5,
        fetch_k=20,
    )
    print(json.dumps(snap, ensure_ascii=False, indent=2))
