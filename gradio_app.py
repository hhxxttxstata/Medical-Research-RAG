"""
Gradio 前端 — RAG 问答 + CTPA 诊断
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import gradio as gr

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_CHAT = f"{API_BASE}/chat"
API_UPLOAD = f"{API_BASE}/documents/upload"
API_HEALTH = f"{API_BASE}/health"
API_DIAGNOSIS = f"{API_BASE}/diagnosis/predict"

TITLE = "🩺 肺栓塞智能问诊系统"


def _api_json(url: str, data: dict) -> dict:
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        try:
            return json.loads(detail)
        except json.JSONDecodeError:
            return {"success": False, "answer": f"HTTP {e.code}: {detail[:200]}"}
    except Exception as e:
        return {"success": False, "answer": f"请求失败: {str(e)}"}


def _multipart_post(url: str, file_path: str, filename: str) -> dict:
    boundary = f"----Boundary{int(time.time() * 1000)}"
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        + file_bytes
        + f"\r\n--{boundary}--\r\n".encode()
    )

    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        try:
            return json.loads(detail)
        except json.JSONDecodeError:
            return {"success": False, "error": f"HTTP {e.code}: {detail[:300]}"}
    except Exception as e:
        return {"success": False, "error": f"请求失败: {str(e)}"}


def _build_diagnosis_markdown(result: dict, filename: str) -> str:
    if not result.get("success"):
        return f"❌ **诊断失败**: {result.get('error', '未知错误')}"

    prob = result.get("probability", 0.0)
    pred = result.get("prediction", 0)
    risk = result.get("risk_level", "未知")
    if risk == "未知" or not risk:
        risk = "高风险" if prob >= 0.9 else "中风险" if prob >= 0.7 else "低风险" if prob >= 0.5 else "阴性"

    risk_icon = {"高风险": "🔴", "中风险": "🟡", "低风险": "🟢", "阴性": "✅"}.get(risk, "⚪")

    lines = [
        "## 🩺 肺栓塞诊断报告",
        "",
        "| 项目 | 结果 |",
        "|------|------|",
        f"| 📂 影像文件 | `{filename}` |",
        f"| {risk_icon} 诊断结果 | **{risk}** ({'阳性' if pred else '阴性'}) |",
        f"| 📊 肺栓塞概率 | **{prob:.4f}** ({prob * 100:.2f}%) |",
        f"| ⏱️  耗时 | {result.get('total_time', 0):.3f}s |",
        "",
        "> ⚠️ **免责声明:** AI 辅助诊断建议，仅供参考。",
    ]
    return "\n".join(lines)


def smart_entry(question: str, file: Any | None, history: list) -> Any:
    has_file = file is not None
    has_text = bool(question and question.strip())

    if not has_text and not has_file:
        gr.Warning("请输入问题或上传 CTPA 影像")
        return history if history else []

    if history is None:
        history = []

    filepath = None
    filename = None
    if has_file:
        filepath = file.name if hasattr(file, "name") else file
        filename = Path(filepath).name

    # ── 有文件 → 诊断 ──
    if has_file:
        diag_result = _multipart_post(API_DIAGNOSIS, filepath, filename)
        diagnosis_report = _build_diagnosis_markdown(diag_result, filename)

        user_msg = f"{question}\n\n📁 **上传文件**: `{filename}`" if has_text else f"📁 **上传**: `{filename}`"
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": diagnosis_report})
        return history

    # ── 纯文本 → RAG 问答 ──
    result = _api_json(API_CHAT, {"question": question.strip()})
    answer = result.get("answer", "（无回答）")
    parts = [answer, ""]

    sources = result.get("sources", [])
    if sources:
        parts.append("---\n### 📚 引用来源")
        for i, s in enumerate(sources, 1):
            parts.append(f"- [{i}] `{s.get('filename', '未知')}` 相似度: {s.get('score', 0):.3f}")
        parts.append("")

    meta = []
    if result.get("elapsed"):
        meta.append(f"⏱️ **{result['elapsed']:.2f}s**")
    if result.get("is_refusal"):
        meta.append("🚫 拒答")
    if meta:
        parts.append(f"> {' | '.join(meta)}")

    history.append({"role": "user", "content": question.strip()})
    history.append({"role": "assistant", "content": "\n".join(parts)})
    return history


_CSS = """
footer {display:none !important}
.gradio-container {max-width:min(92vw, 900px) !important; margin:0 auto !important; padding:16px 8px !important}
"""

with gr.Blocks(title=TITLE, css=_CSS, theme="soft") as demo:
    gr.HTML(
        f"""<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
            <h1 style="margin:0;font-size:1.4em">{TITLE}</h1>
        </div>"""
    )

    chatbot = gr.Chatbot(height=520, render_markdown=True)

    chat_input = gr.MultimodalTextbox(
        placeholder="输入问题，或上传 CTPA 影像自动诊断...",
        file_types=[".nii", ".gz", ".pdf", ".md", ".txt"],
        file_count="single",
        sources=["upload"],
        submit_btn=True,
    )

    def on_submit(msg: dict, history: list):
        text = msg.get("text", "")
        files = msg.get("files", [])
        file_path = files[0] if files else None
        return smart_entry(text, file_path, history)

    chat_input.submit(fn=on_submit, inputs=[chat_input, chatbot], outputs=[chatbot]).then(
        fn=lambda: {"text": "", "files": []}, outputs=[chat_input]
    )

    gr.HTML(
        f"""<div style="display:flex;justify-content:space-between;font-size:0.75em;color:#999;padding:6px 2px">
            <span>
                <a href="{API_BASE}/docs" target="_blank" style="color:#999">📖 API</a>
                <span style="margin:0 6px">·</span>
                🩺 肺栓塞 RAG 问答系统
            </span>
        </div>"""
    )

if __name__ == "__main__":
    port = int(os.getenv("GRADIO_PORT", "7860"))
    print(f"\n🌐 {TITLE}")
    print(f"   http://127.0.0.1:{port}\n")
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True, theme="soft")
