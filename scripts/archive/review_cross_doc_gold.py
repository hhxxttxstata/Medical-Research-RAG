"""
Cross-doc gold 标注复核工具

用途：人工复核 tests/cross_doc_gold.json 的 25 题标注。
生成 HTML 复核清单（浏览器打开逐题核对），同时控制台打印摘要。

每个 gold doc 显示：
  - 实际入库的 chunk 内容摘要（读 milvus parquet，非源文件——复核应以
    "检索实际看到的文本"为准）
  - 题目关键词 × 文档文本命中率（低命中提示标注可能存疑）

用法:
    python scripts/review_cross_doc_gold.py            # 生成 HTML + 打印摘要
    python scripts/review_cross_doc_gold.py --out review_cross_doc_gold.html
"""

import argparse
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

from src.agentic_rag import _GENERIC_TERMS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(
    ROOT,
    "milvus_db",
    "milvus.db",
    "collections",
    "rag_docs_c300_500",
    "partitions",
    "_default",
    "data",
    "data_000001_000887.parquet",
)

_STOP = _GENERIC_TERMS | {
    "的",
    "了",
    "是",
    "在",
    "和",
    "与",
    "或",
    "及",
    "等",
    "中",
    "上",
    "下",
    "设置",
    "关系",
    "完整",
    "流程",
    "部署",
    "训练",
    "使用",
    "综合",
    "选择",
    "处理",
    "步骤",
    "过程",
    "分别",
    "什么",
    "多少",
    "如何",
    "为什么",
    "怎样",
    "哪些",
    "是否",
    "以及",
    "进行",
    "可以",
    "需要",
    "通过",
    "情况",
    "目的",
    "对象",
    "方式",
    "方面",
    "内容",
    "类型",
    "指标",
    "作用",
    "影响",
    "分析",
    "应用",
    "相关",
    "特点",
    "优势",
    "区别",
    "对比",
    "评估",
    "结果",
    "风险",
    "标准",
    "功能",
    "问题",
    "之间",
    "不同",
    "长期",
    "包括",
    "用于",
    "主要",
}


def load_doc_texts() -> dict[str, str]:
    """从 Milvus parquet 读取每个文档的全部 chunk 文本（拼接）"""
    import pandas as pd

    df = pd.read_parquet(PARQUET)
    docs: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        meta = row["metadata"]
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        fn = meta.get("filename", "")
        if not fn:
            continue
        docs.setdefault(fn, []).append(row["text"])
    return {fn: "\n".join(chunks) for fn, chunks in docs.items()}


def keyword_hit_rate(question: str, doc_text: str) -> tuple[float, list[str]]:
    """题目区分性词在文档文本中的命中比例 + 命中词列表

    词表 = 英文词（≥3 字符）+ 2-4 字中文片段（排除停用词）。
    返回 (命中比例, 命中词列表)——列表供复核页直接展示，
    比单一比例直观（如"窗宽/窗位/HU"命中即可判断文档相关）。
    """
    q = question.lower()
    terms: set[str] = set(re.findall(r"[a-z][a-z0-9\-]{2,}", q))
    for n in (4, 3, 2):
        for i in range(len(question) - n + 1):
            frag = question[i : i + n]
            if not re.fullmatch(r"[\u4e00-\u9fff]+", frag):
                continue
            if frag in _STOP:
                continue
            terms.add(frag)
    if not terms:
        return 1.0, []
    dt = doc_text.lower()
    hit_terms = sorted(t for t in terms if t in dt)
    return len(hit_terms) / len(terms), hit_terms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="review_cross_doc_gold.html")
    args = ap.parse_args()

    with open(os.path.join(ROOT, "tests", "test_questions.json"), encoding="utf-8") as f:
        questions = json.load(f)
    with open(os.path.join(ROOT, "tests", "cross_doc_gold.json"), encoding="utf-8") as f:
        gold = json.load(f)
    gold = {k: v for k, v in gold.items() if not k.startswith("_")}

    docs = load_doc_texts()
    cd = [q for q in questions if q.get("category") == "cross_doc"]
    print(f"cross_doc 题: {len(cd)} | gold 标注: {len(gold)} | 索引文档: {len(docs)}\n")

    cards = []
    for q in cd:
        qid = q["id"]
        gold_docs = gold.get(qid, [])
        items = []
        for gd in gold_docs:
            text = docs.get(gd, "")
            hit, hit_terms = keyword_hit_rate(q["question"], text) if text else (0.0, [])
            items.append({"doc": gd, "hit": hit, "hit_terms": hit_terms, "snippet": text[:500]})
            flag = " ⚠️" if hit < 0.25 else ""
            print(f"  {qid}  {q['question'][:42]}")
            print(f"      └─ {gd[:44]}  命中={hit:.2f} {hit_terms[:8]}{flag}")
        cards.append({"id": qid, "question": q["question"], "difficulty": q.get("difficulty", ""), "items": items})

    # ── 生成 HTML ──
    def esc(s):
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    body = []
    for c in cards:
        items_html = []
        for it in c["items"]:
            terms_txt = "、".join(it["hit_terms"][:10]) if it["hit_terms"] else "（无命中词）"
            warn = (
                f'<span style="color:#d97706;font-size:12px">命中率 {it["hit"]:.0%} ⚠️ 命中词少，标注存疑</span>'
                if it["hit"] < 0.25
                else f'<span style="color:#4b9e6f;font-size:12px">命中率 {it["hit"]:.0%} ✅ 命中词：{esc(terms_txt)}</span>'
            )
            items_html.append(
                f'<div style="border:1px solid #333;border-radius:8px;padding:10px;margin:8px 0;background:#151820">'
                f'<div style="font-weight:600;color:#7fb0ff">{esc(it["doc"])}</div>'
                f"<div>{warn}</div>"
                f'<div style="font-size:12px;color:#9aa3b2;margin-top:6px;max-height:120px;overflow:auto">{esc(it["snippet"])}</div>'
                f"</div>"
            )
        body.append(
            f'<div style="border:1px solid #2a2f3a;border-radius:10px;padding:14px;margin-bottom:14px;background:#1b1f2a">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<h3 style="margin:0;font-size:15px">{c["id"]} <span style="color:#8b93a3;font-weight:400">({c["difficulty"]})</span></h3>'
            f'<div><label style="margin-right:12px;color:#4b9e6f"><input type="radio" name="{c["id"]}" value="pass"> 通过</label>'
            f'<label style="color:#d97706"><input type="radio" name="{c["id"]}" value="fix"> 需修改</label></div></div>'
            f'<div style="font-size:14px;margin:8px 0;line-height:1.6">{esc(c["question"])}</div>'
            f"{''.join(items_html)}"
            f'<textarea placeholder="修改意见（可选）" data-note="{c["id"]}" style="width:100%;height:44px;background:#0f1117;color:#e6e8ee;border:1px solid #333;border-radius:6px;padding:6px;font-size:12px"></textarea>'
            f"</div>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="UTF-8"><title>Cross-doc Gold 标注复核</title>
<style>body{{background:#0f1115;color:#e6e8ee;font-family:system-ui;max-width:900px;margin:24px auto;padding:0 16px}}
h1{{font-size:19px}} .hint{{color:#8b93a3;font-size:13px;margin-bottom:18px}}
button{{background:#4f8cff;color:#fff;border:none;border-radius:6px;padding:8px 18px;cursor:pointer;margin-bottom:16px}}</style>
</head><body>
<h1>🩺 Cross-doc Gold 标注复核（{len(cards)} 题）</h1>
<div class="hint">题源：tests/test_questions.json（81 题 Golden 数据集，cross_doc 25 题）｜标注：tests/cross_doc_gold.json<br>
逐题核对：题目答案是否确实需要标注的 2-3 个文档（看文档内容摘要与题目相关性）。勾选结果保存在浏览器本地（localStorage）。</div>
<button onclick="saveAll()">💾 保存本次复核结果</button>
<button onclick="exportResult()" style="background:#3fb96b">📋 导出修改建议（复制到剪贴板）</button>
{"".join(body)}
<script>
function saveAll(){{
  const r={{passed:[],fix:[]}};
  document.querySelectorAll('input[type=radio]:checked').forEach(x=>{{
    const id=x.name, val=x.value;
    const note=document.querySelector(`textarea[data-note="${{id}}"]`).value;
    (val==='pass'?r.passed:r.fix).push({{id,note}});
  }});
  localStorage.setItem('crossdoc_review', JSON.stringify(r));
  alert(`已保存：通过 ${{r.passed.length}} 题，需修改 ${{r.fix.length}} 题`);
}}
function exportResult(){{
  const saved=JSON.parse(localStorage.getItem('crossdoc_review')||'{{}}');
  const fix=(saved.fix||[]).map(x=>`- ${{x.id}}: ${{x.note||'(未填意见)'}}`).join('\\n');
  navigator.clipboard.writeText(fix||'(无待修改项)').then(()=>alert('已复制修改意见到剪贴板'));
}}
</script></body></html>"""
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n📄 复核页已生成: {os.path.abspath(args.out)}  （浏览器打开）")
    print(
        f"   题源核对: tests/test_questions.json 共 {len(questions)} 题 "
        f"(exact {sum(1 for q in questions if q.get('category') == 'exact_match')} / "
        f"cross {len(cd)} / ood {sum(1 for q in questions if q.get('category') == 'out_of_knowledge')})"
    )


if __name__ == "__main__":
    main()
