import os

# 改用不同目录，避免文件锁冲突
os.environ["_EVAL_CHROMA_DIR"] = "chroma_db_eval_tmp"

# 打补丁：让 Evaluator 用新目录
import eval.run_evaluation as m  # noqa: E402

orig_init = m.Evaluator.__init__


def patched_init(self, *a, **kw):
    orig_init(self, *a, **kw)
    self.chroma_dir = os.path.abspath(os.environ.get("_EVAL_CHROMA_DIR", "chroma_db_eval_tmp"))
    # 不要删旧的


m.Evaluator.__init__ = patched_init

# 跳过 _clean_chroma
m.Evaluator._clean_chroma = lambda self: None

m.main()
