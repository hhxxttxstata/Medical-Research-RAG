"""
RAG 系统评估入口（兼容旧版）
已迁移到 eval/ 目录下的评估 Pipeline

用法:
    python -m eval.run_evaluation --help
    # 或保持旧版用法：
    python evaluate.py --quick
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval.run_evaluation import main

if __name__ == "__main__":
    main()
