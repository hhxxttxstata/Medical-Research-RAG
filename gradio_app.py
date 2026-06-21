"""
Gradio 前端页面 — DeepSeek 风格简洁界面

一个聊天框 + 一个上传入口，自动判断是问问题还是看片子。
支持拖拽 CTPA 影像到输入区。
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Windows GBK 兼容
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import gradio as gr

# ── 配置 ──────────────────────────────────────────────

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_CHAT = f"{API_BASE}/chat"
API_UPLOAD = f"{API_BASE}/documents/upload"
API_HEALTH = f"{API_BASE}/health"
API_DIAGNOSIS = f"{API_BASE}/diagnosis/predict"

TITLE = "🩺 肺栓塞智能问诊系统"

# ══════════════════════════════════════════════════════════════════
#  后端辅助函数（保持不变）
# ══════════════════════════════════════════════════════════════════


def _api_json(url: str, data: dict) -> dict:
    """POST JSON 到 API"""
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
    """通用 multipart 文件上传"""
    boundary = f"----Boundary{int(time.time() * 1000)}"
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    body_before = (
        "\r\n".join(
            [
                f"--{boundary}",
                f'Content-Disposition: form-data; name="file"; filename="{filename}"',
                "Content-Type: application/octet-stream",
                "",
            ]
        )
        + "\r\n"
    ).encode("utf-8")
    body_after = f"\r\n--{boundary}--\r\n".encode()
    body = body_before + file_bytes + body_after

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


def _format_process_log(log: list) -> str:
    if not log:
        return ""
    lines = []
    for entry in log:
        step = entry.get("step", "")
        detail = entry.get("detail", "")
        status = entry.get("status", "")
        icon = {"ok": "✅", "running": "⏳", "error": "❌"}.get(status, "➡️")
        lines.append(f"- {icon} **{step}**：{detail}")
    return "\n".join(lines)


def _format_sources(sources: list) -> str:
    if not sources:
        return ""
    lines = ["| # | 文件 | 相似度 | 内容预览 |", "|---|------|--------|----------|"]
    for i, s in enumerate(sources, 1):
        fname = s.get("filename", "未知")
        score = s.get("score", 0)
        text = s.get("text", "")[:80].replace("\n", " ")
        lines.append(f"| {i} | {fname} | {score:.3f} | {text}... |")
    return "\n".join(lines)


def _build_diagnosis_markdown(result: dict, filename: str) -> str:
    """诊断结果 → Markdown 报告"""
    if not result.get("success"):
        return f"❌ **诊断失败**: {result.get('error', '未知错误')}"

    prob = result.get("probability", 0.0)
    pred = result.get("prediction", 0)
    risk = result.get("risk_level", "未知")
    if not risk or risk == "未知":
        if prob >= 0.9:
            risk = "高风险"
        elif prob >= 0.7:
            risk = "中风险"
        elif prob >= 0.5:
            risk = "低风险"
        else:
            risk = "阴性"

    risk_icon = {"高风险": "🔴", "中风险": "🟡", "低风险": "🟢", "阴性": "✅"}.get(risk, "⚪")

    lines = [
        "## 🩺 肺栓塞诊断报告",
        "",
        "| 项目 | 结果 |",
        "|------|------|",
        f"| 📂 影像文件 | `{filename}` |",
        f"| {risk_icon} 诊断结果 | **{risk}** ({'阳性' if pred else '阴性'}) |",
        f"| 📊 肺栓塞概率 | **{prob:.4f}** ({prob * 100:.2f}%) |",
        f"| ⚙️  阈值 | {result.get('threshold', 0.5)} |",
        f"| 🧩 栓塞区占比 | {result.get('mask_positive_ratio', result.get('positive_voxel_ratio', 0)):.4%} |",
        f"| ⏱️  预处理 | {result.get('preprocess_time', 0):.3f}s |",
        f"| ⏱️  推理 | {result.get('inference_time', 0):.3f}s |",
        f"| ⏱️  总计 | {result.get('total_time', 0):.3f}s |",
        "",
    ]

    advice = {
        "高风险": [
            "### ⚠️ 临床建议",
            "1. 建议立即请放射科医师复核影像",
            "2. 建议结合临床症状（呼吸困难、胸痛、咯血）综合判断",
            "3. 建议检查 D-二聚体、血气分析等实验室指标",
            "4. 视情况启动抗凝治疗评估",
        ],
        "中风险": [
            "### 📋 临床建议",
            "1. 建议结合临床评分（如 Wells 评分、sPESI 评分）评估",
            "2. 必要时请放射科医师复核",
            "3. 建议短期随访复查",
        ],
        "低风险": [
            "### 📋 临床建议",
            "1. 概率较低但仍需结合临床判断",
            "2. 如临床高度怀疑，建议进一步检查",
        ],
    }.get(
        risk,
        [
            "### 📋 临床建议",
            "当前影像未检出肺栓塞阳性征象。",
            "如临床高度怀疑，请结合其他检查综合判断。",
        ],
    )
    lines.extend(advice)
    lines.extend(["", "> ⚠️ **免责声明:** 本结果为 AI 辅助诊断建议，仅供参考，最终诊断需由临床医师确认。"])

    return "\n".join(lines)


def _save_viz_images(result: dict) -> list:
    """从诊断结果中提取可视化 base64 → 临时文件"""
    vis = result.get("visualization", {})
    if not vis:
        return []

    import base64
    import tempfile

    saved = []

    overview_b64 = vis.get("slice_overview", "")
    if overview_b64:
        p = os.path.join(tempfile.gettempdir(), f"pe_overview_{int(time.time())}.png")
        with open(p, "wb") as f:
            f.write(base64.b64decode(overview_b64))
        saved.append((p, "📊 轴向风险分布概览图"))

    for sl in vis.get("top_slices", []):
        img_b64 = sl.get("image_base64", "")
        if not img_b64:
            continue
        rank = sl.get("rank", "?")
        z = sl.get("slice_index", "?")
        prob_s = sl.get("probability", 0)
        p = os.path.join(tempfile.gettempdir(), f"pe_slice_{rank}_z{z}_{int(time.time())}.png")
        with open(p, "wb") as f:
            f.write(base64.b64decode(img_b64))
        saved.append((p, f"🔍 #{rank} 高风险切片 (Z={z}, Prob={prob_s:.3f})"))

    return saved


def _needs_knowledge_analysis(question: str) -> bool:
    """判断是否需要知识库联动分析"""
    if not question or not question.strip():
        return False
    pure_cmds = [
        "诊断",
        "看片子",
        "看影像",
        "读片",
        "分析影像",
        "分析ct",
        "分析ctpa",
        "诊断一下",
        "检测一下",
        "预测一下",
    ]
    q = question.strip()
    if len(q) < 15 and any(q.startswith(cmd) or q == cmd for cmd in pure_cmds):
        return False
    return True


# ══════════════════════════════════════════════════════════════════
#  核心处理逻辑
# ══════════════════════════════════════════════════════════════════


def smart_entry(
    question: str,
    file: Any | None,
    top_k: int,
    history: list,
) -> Any:
    """统一入口：自动判断走诊断还是问答

    返回 (history, gallery_list)
    """
    has_file = file is not None
    has_text = bool(question and question.strip())

    if not has_text and not has_file:
        gr.Warning("请输入问题或上传 CTPA 影像文件")
        return (history if history else [], None)

    if history is None:
        history = []

    filepath = None
    filename = None
    if has_file:
        filepath = file.name if hasattr(file, "name") else file
        filename = Path(filepath).name

    # ── 有文件 → 诊断 ──
    if has_file:
        question_text = question.strip() if has_text else f"请诊断这个CTPA影像：{filename}"
        return _handle_diagnosis_with_chat(question_text, filepath, filename, top_k, history)

    # ── 纯文本 → 问答 ──
    return (_handle_chat_only(question.strip(), top_k, history), None)


def _handle_chat_only(question: str, top_k: int, history: list) -> list:
    """纯文本问答"""
    result = _api_json(API_CHAT, {"question": question, "mode": "auto", "top_k": top_k})
    parts = _format_chat_response_parts(result)
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": "\n".join(parts)})
    return history


def _handle_diagnosis_with_chat(
    question: str,
    filepath: str,
    filename: str,
    top_k: int,
    history: list,
) -> tuple:
    """影像诊断 + 可选知识联动，返回 (history, gallery_list)"""
    diag_result = _multipart_post(API_DIAGNOSIS, filepath, filename)
    diagnosis_report = _build_diagnosis_markdown(diag_result, filename)
    saved_images = _save_viz_images(diag_result)

    # 用户消息
    user_content = f"{question}\n\n📁 **上传文件**: `{filename}`"
    history.append({"role": "user", "content": user_content})

    # 组装回答
    answer_parts = [diagnosis_report]

    if saved_images:
        gallery_lines = ["\n\n### 🖼️ 影像分析"]
        for img_path, caption in saved_images:
            gallery_lines.append(f"  - {caption}")
        answer_parts.append("\n".join(gallery_lines))

    # 知识库联动
    if _needs_knowledge_analysis(question):
        rag_question = f"{question}\n\n【诊断结果】{diagnosis_report}\n\n请结合以上诊断结果和知识库，给出综合分析。"
        rag_result = _api_json(API_CHAT, {"question": rag_question, "mode": "auto", "top_k": top_k})
        rag_answer = rag_result.get("answer", "")
        if rag_answer:
            answer_parts.extend(
                [
                    "\n\n---\n### 📚 知识库联动分析",
                    rag_answer,
                    "",
                ]
            )
            log = rag_result.get("process_log", [])
            if log:
                answer_parts.extend(["#### 🧠 处理过程", _format_process_log(log)])
            sources = rag_result.get("sources", [])
            if sources:
                answer_parts.extend(["#### 📚 引用来源", _format_sources(sources)])

    history.append({"role": "assistant", "content": "\n".join(answer_parts)})
    return (history, saved_images if saved_images else None)


def _format_chat_response_parts(result: dict) -> list:
    """API /chat 响应 → Markdown 段落"""
    parts = [result.get("answer", "（无回答）"), ""]

    log = result.get("process_log", [])
    if log:
        parts.extend(["---\n### 🧠 处理过程", _format_process_log(log), ""])

    sources = result.get("sources", [])
    if sources:
        parts.extend(["---\n### 📚 引用来源", _format_sources(sources), ""])

    meta = [f"⏱️ 耗时: **{result.get('elapsed', 0):.2f}s**"]
    if result.get("is_refusal"):
        meta.append("🚫 **拒答**")
    agent_info = result.get("agent_info")
    if agent_info:
        if agent_info.get("intent"):
            meta.append(f"🎯 `{agent_info['intent']}`")
        if agent_info.get("tool"):
            meta.append(f"🔧 `{agent_info['tool']}`")
    parts.append(f"> {' | '.join(meta)}")
    return parts


# ══════════════════════════════════════════════════════════════════
#  构建 Gradio 界面（DeepSeek 风格 — 极简）
# ══════════════════════════════════════════════════════════════════

_CSS = """
footer {display:none !important}
.gradio-container {max-width:min(92vw, 900px) !important; margin:0 auto !important; padding:16px 8px 0 8px !important}
"""


def _on_multimodal_submit(msg: dict, history: list):
    """MultimodalTextbox 提交处理"""
    text = msg.get("text", "")
    files = msg.get("files", [])
    file_path = files[0] if files else None
    result_history, gallery = smart_entry(text, file_path, 5, history)
    # 更新 gallery 可见性
    has_gallery = gallery is not None and len(gallery) > 0
    return result_history, gr.update(value=gallery, visible=has_gallery)


def _on_refresh_status() -> str:
    """检查系统状态"""
    try:
        with urllib.request.urlopen(API_HEALTH, timeout=5) as resp:
            data = json.loads(resp.read())
            kb = data.get("knowledge_base", {})
            return (
                f"**✅ 服务正常** &nbsp;|&nbsp; "
                f"🧩 `{kb.get('chunk_count', '?')}` Chunks &nbsp;|&nbsp; "
                f"🔧 `{kb.get('embedding', '?')}` &nbsp;|&nbsp; "
                f"🎯 Top-`{kb.get('top_k', '?')}`"
            )
    except Exception as e:
        return f"❌ 无法连接: {e}"


with gr.Blocks(title=TITLE, css=_CSS, theme="soft") as demo:
    # ── 顶部标题 ──
    gr.HTML(
        f"""<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
            <h1 style="margin:0;font-size:1.4em">{TITLE}</h1>
            <span id="status-badge" style="font-size:0.75em;color:#888;cursor:pointer"
                  onclick="document.querySelector('#refresh-btn button').click()"
                  title="点击刷新状态">● 已连接</span>
        </div>"""
    )

    # 隐藏的状态刷新按钮
    refresh_btn = gr.Button("刷新", elem_id="refresh-btn", visible=False, size="sm")
    status_display = gr.Markdown(visible=False)
    refresh_btn.click(fn=_on_refresh_status, outputs=status_display)

    # ── 聊天区域 ──
    chatbot = gr.Chatbot(
        height=520,
        render_markdown=True,
    )

    # ── 诊断可视化画廊 ──
    viz_gallery = gr.Gallery(
        height="auto",
        columns=2,
        object_fit="contain",
        show_label=False,
        visible=False,
        container=True,
    )

    # ── 输入区（MultimodalTextbox 自带文件上传 + 拖拽） ──
    chat_input = gr.MultimodalTextbox(
        placeholder="输入问题，或上传 CTPA 影像自动诊断...",
        file_types=[".nii", ".gz", ".pdf", ".md", ".txt"],
        file_count="single",
        sources=["upload"],
        submit_btn=True,
        scale=1,
    )

    # ── 事件绑定 ──
    chat_input.submit(
        fn=_on_multimodal_submit,
        inputs=[chat_input, chatbot],
        outputs=[chatbot, viz_gallery],
    ).then(
        fn=lambda: {"text": "", "files": []},
        outputs=[chat_input],
    )

    # ── 底部状态栏 ──
    gr.HTML(
        f"""<div style="display:flex;justify-content:space-between;font-size:0.75em;color:#999;padding:6px 2px">
            <span>
                <a href="{API_BASE}/docs" target="_blank" style="color:#999;text-decoration:none">📖 API</a>
                <span style="margin:0 6px">·</span>
                <a onclick="document.querySelector('#refresh-btn button').click()" style="color:#999;cursor:pointer;text-decoration:none">🔄 刷新</a>
            </span>
            <span>API: {API_BASE}</span>
        </div>"""
    )


# ── 启动 ──────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("GRADIO_PORT", "7860"))
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"

    print(f"\n🌐 {TITLE}")
    print(f"   http://127.0.0.1:{port}")
    print(f"   API: {API_BASE}\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        show_error=True,
        theme="soft",
    )
