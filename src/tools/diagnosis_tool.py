"""
肺栓塞诊断工具 (diagnose_pulmonary_embolism)

调用已训练的 CTPA 肺栓塞诊断模型，对上传的影像进行风险预测。
支持 NIfTI 格式的 CTPA 影像输入，输出肺栓塞阳性的概率评分和风险等级。
"""

import os
import sys
from typing import Any

from .base import Tool, ToolPolicy

# 延迟导入（避免循环导入）
# 在需要的地方 from ..diagnosis import CTPADiagnosisModel, create_diagnosis_model

# Windows GBK 兼容
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _compute_risk_level(prob: float) -> str:
    """根据概率计算风险等级"""
    if prob >= 0.9:
        return "高风险"
    elif prob >= 0.7:
        return "中风险"
    elif prob >= 0.5:
        return "低风险"
    return "阴性"


def _cls(risk: str) -> str:
    """风险等级颜色图标"""
    return {"高风险": "🔴", "中风险": "🟡", "低风险": "🟢", "阴性": "✅"}.get(risk, "⚪")


def _format_result(result: dict[str, Any], filename: str) -> str:
    """将推理结果格式化为人类可读的文本"""
    if not result.get("success"):
        return f"❌ 诊断失败: {result.get('error', '未知错误')}"

    prob = result["probability"]
    pred = result["prediction"]
    risk = _compute_risk_level(prob)

    lines = [
        "=" * 50,
        "  🩺 肺栓塞诊断报告",
        "=" * 50,
        f"  📂 影像文件: {os.path.basename(filename)}",
        f"  {_cls(risk)} 诊断结果: **{risk}** ({'阳性' if pred else '阴性'})",
        f"  📊 肺栓塞概率: **{prob:.4f}** ({prob * 100:.2f}%)",
        f"  ⚙️  阈值: {result['threshold']}",
        "",
    ]

    if result.get("mask_positive_ratio") is not None:
        pr = result["mask_positive_ratio"]
        lines.append(f"  🧩 栓塞区占比: {pr:.4%} (阳性体素/总体素)")
        lines.append(
            f"     阳性体素: {result.get('mask_positive_voxels', '?')} / 总计 {result.get('mask_total_voxels', '?')}"
        )

    lines.extend(
        [
            f"  ⏱️  预处理: {result.get('preprocess_time', 0):.3f}s | "
            f"推理: {result.get('inference_time', 0):.3f}s | "
            f"总计: {result.get('total_time', 0):.3f}s",
            "=" * 50,
        ]
    )

    if risk == "高风险":
        lines.append("")
        lines.append("  ⚠️ **临床建议:**")
        lines.append("  1. 建议立即请放射科医师复核影像")
        lines.append("  2. 建议结合临床症状（呼吸困难、胸痛、咯血）综合判断")
        lines.append("  3. 建议检查 D-二聚体、血气分析等实验室指标")
        lines.append("  4. 视情况启动抗凝治疗评估")
        lines.append("")
        lines.append("  ⚠️ **免责声明:** 本结果为 AI 辅助诊断建议，")
        lines.append("  仅供参考，最终诊断需由临床医师确认。")
    elif risk == "中风险":
        lines.append("")
        lines.append("  📋 **临床建议:**")
        lines.append("  1. 建议结合临床评分（如 Wells 评分、sPESI 评分）评估")
        lines.append("  2. 必要时请放射科医师复核")
        lines.append("  3. 建议短期随访复查")
        lines.append("")
        lines.append("  ⚠️ 本结果为 AI 辅助诊断建议，最终诊断需由临床医师确认。")
    elif risk == "低风险":
        lines.append("")
        lines.append("  📋 **临床建议:**")
        lines.append("  1. 概率较低但仍需结合临床判断")
        lines.append("  2. 如临床高度怀疑，建议进一步检查（如 CTPA 复查）")
        lines.append("")
        lines.append("  ⚠️ 本结果为 AI 辅助诊断建议，最终诊断需由临床医师确认。")
    else:
        lines.append("")
        lines.append("  📋 **临床建议:**")
        lines.append("  当前影像未检出肺栓塞阳性征象。")
        lines.append("  如临床高度怀疑，请结合其他检查综合判断。")
        lines.append("")
        lines.append("  ⚠️ 本结果为 AI 辅助诊断建议，最终诊断需由临床医师确认。")

    return "\n".join(lines)


class DiagnosisTool(Tool):
    """肺栓塞诊断工具

    调用已训练的 CTPA 肺栓塞诊断模型，对上传的 NIfTI 格式影像进行推理，
    输出肺栓塞概率评分和风险等级。
    """

    name = "diagnose_pulmonary_embolism"
    description = "调用肺栓塞 AI 诊断模型，对 CTPA 影像进行风险预测。输入为 NIfTI 格式 (.nii/.nii.gz) 的 CT 肺动脉造影影像，输出肺栓塞概率和风险等级"
    policy = ToolPolicy(
        access_level="confirm",
        rate_limit=10,
        require_reason=True,
    )

    def __init__(self, model=None):
        super().__init__()
        self._model = model
        self._auto_init = model is None

    # ── 模型初始化 ──────────────────────────────────────

    def _ensure_model(self) -> bool:
        """确保模型已加载，如未加载则自动初始化"""
        if self._model and hasattr(self._model, "is_loaded") and self._model.is_loaded:
            return True

        if self._auto_init:
            print("\n📦 正在初始化肺栓塞诊断模型...")
            from ..diagnosis import create_diagnosis_model  # 延迟导入

            self._model = create_diagnosis_model()
            if self._model and hasattr(self._model, "is_loaded") and self._model.is_loaded:
                print("  ✅ 肺栓塞诊断模型就绪\n")
                return True
            else:
                err = getattr(self._model, "load_error", "") if self._model else "创建失败"
                print(f"  ⚠️  {err}")
                return False

        return False

    # ── 接口方法 ────────────────────────────────────────

    def predict(self, nifti_path: str, **kwargs) -> dict[str, Any]:
        """对 CTPA 影像进行诊断

        Args:
            nifti_path: NIfTI 文件路径 (.nii / .nii.gz)

        Returns:
            诊断结果字典
        """
        if not self._ensure_model():
            return {
                "success": False,
                "error": "诊断模型未加载，请先配置 PE_MODEL_PATH 环境变量指向模型权重文件",
                "probability": 0.0,
                "prediction": 0,
            }
        return self._model.predict(nifti_path, **kwargs)

    def run(self, **kwargs) -> dict[str, Any]:
        """Agent 调用入口（兼容 Tool 基类接口）

        支持参数:
            file_path: NIfTI 文件路径（必填）
            return_mask: 是否返回分割掩膜（可选，默认 True）
        """
        nifti_path = kwargs.get("file_path", "")
        if not nifti_path:
            return {
                "success": False,
                "error": "缺少必填参数 'file_path'（NIfTI 影像文件路径）",
            }

        if not os.path.isfile(nifti_path):
            return {
                "success": False,
                "error": f"文件不存在: {nifti_path}",
            }

        # 执行推理
        result = self.predict(
            nifti_path=nifti_path,
            return_mask=kwargs.get("return_mask", True),
        )

        # 格式化为可读文本（给 LLM 的最终输出）
        formatted = _format_result(result, nifti_path)
        result["formatted_report"] = formatted
        # 补充 risk_level 字段（给 API 前端使用）
        if result.get("success") and result.get("probability") is not None:
            result["risk_level"] = _compute_risk_level(result["probability"])

        return result

    def get_schema(self) -> dict[str, Any]:
        return {
            "tool_name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "CTPA 影像文件的完整路径（NIfTI 格式，.nii 或 .nii.gz）",
                    },
                    "return_mask": {
                        "type": "boolean",
                        "description": "是否返回分割掩膜数据（默认 true）",
                        "default": True,
                    },
                },
                "required": ["file_path"],
            },
        }
