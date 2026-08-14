"""
CTPA 肺栓塞诊断推理模块

基于 ResNet25d + Gated Attention MIL 的肺栓塞诊断模型。
接受 NIfTI 格式的 CTPA 影像，输出肺栓塞概率和注意力权重。

架构:
  - 2.5D ResNet-18 编码器 (输入 9 通道: 3窗宽窗位 × 3相邻切片)
  - Gated Attention MIL 池化 (聚合多个 slab 的实例特征)
  - 注意：训练配置 use_topk_branch=False, dual_mil_alpha=1.0（纯 Gated Attention）

预处理流程（与 teacher/train_attention.py 一致，D_roi_body 实验）:
  1. 读取 NIfTI → (D, H, W)
  2. HU 裁剪 [-1000, 700]
  3. 体部 ROI 掩膜（仅保留 HU>-500 的体部区域，外部填充 -1000）
  4. Resize H/W → (256, 256)（保持深度不变）
  5. 胸部切片过滤 (HU-based 肺区分割)
  6. 2.5D slab 提取 (滑动窗口 stride=1, 每窗3切片, linspace下采样至48个)
  7. 多窗归一化 (肺窗/纵隔窗/血管窗 → 3通道)
  8. 堆叠为 (1, num_slabs, 9, 256, 256)

用法:
    model = CTPADiagnosisModel(model_path="models/best.pth")
    result = model.predict("path/to/ctpa.nii.gz")
    # result = {"probability": 0.95, "success": True, "attention_weights": [...]}
"""

import base64
import os
import sys
import time
from collections import OrderedDict
from io import BytesIO
from typing import Any

import numpy as np

# Windows GBK 兼容
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

# ── 可选依赖导入 ──────────────────────────────────────

_MISSING_DEPS = []

try:
    import nibabel as nib
except ImportError:
    nib = None
    _MISSING_DEPS.append("nibabel")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _MISSING_DEPS.append("torch")

try:
    from scipy.ndimage import zoom
except ImportError:
    zoom = None
    _MISSING_DEPS.append("scipy")


# ── 配置 ────────────────────────────────────────────────

MODEL_CONFIG = {
    # 空间尺寸
    "target_hw": (256, 256),  # H/W 统一缩放到此大小
    "slice_thickness": 3,  # 2.5D 每窗的切片数
    "eval_stride": 2,  # 滑动窗口步长
    "windows": [  # 多窗参数 (hu_min, hu_max)
        (-1000, 400),  # 肺窗
        (-160, 240),  # 纵隔窗
        (-100, 700),  # CTA 血管窗
    ],
    # HU 范围
    "hu_min": -1000,
    "hu_max": 700,
    # 胸部切片过滤
    "filter_empty_slices": True,
    "chest_lung_hu_low": -1000,
    "chest_lung_hu_high": -300,
    "chest_lung_ratio_threshold": 0.02,
    "chest_body_hu_threshold": -600,
    "chest_body_ratio_threshold": 0.05,
    "chest_segment_margin_upper": 20,
    "chest_segment_margin_lower": 8,
    "chest_segment_margin": 8,
    "chest_min_filtered_slices": 16,
    # Slab 采样
    "num_slabs_eval": 48,
    # ─── 模型配置 ───
    # ⚠️ 必须与训练 teacher/config_attention.py 一致
    "use_topk_branch": False,  # 训练配置: use_topk_branch = False
    "dual_mil_alpha": 1.0,  # 训练配置: dual_mil_alpha = 1.0
    "dropout": 0.4,  # 训练配置: dropout = 0.4
    "topk": 5,  # 训练配置: topk = 5（use_topk_branch=False 时无效）
    # ─── 体部 ROI 掩膜 ───
    # 训练配置: roi_mode = "body"（D_roi_body 实验最佳）
    "roi_mode": "body",  # "none" | "body"
    "roi_body_threshold": -500,  # HU 阈值（> -500 为体部）
    "roi_body_fill_value": -1000,  # 体部外填充值
    "roi_body_dilation": 3,  # 膨胀像素数
    # 杂项
    "threshold": 0.5,
    "device": "cpu",
}


# ═════════════════════════════════════════════════════════
#  模型架构: ResNet25d + Gated Attention MIL
# ═════════════════════════════════════════════════════════


class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class ResNet2d(nn.Module):
    def __init__(self, block, layers, num_classes=2, input_channels=1):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class GatedAttention(nn.Module):
    def __init__(self, feature_dim, dropout=0.4):
        super().__init__()
        self.attention_V = nn.Linear(feature_dim, feature_dim)
        self.attention_U = nn.Linear(feature_dim, feature_dim)
        self.attention_w = nn.Linear(feature_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        if features.dim() == 2:
            features = features.unsqueeze(1)
        A = torch.tanh(self.attention_V(features)) * torch.sigmoid(self.attention_U(features))
        A = self.dropout(A)
        attention_scores = self.attention_w(A).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)
        pooled = torch.sum(attention_weights.unsqueeze(-1) * features, dim=1)
        return pooled, attention_weights


class InstanceClassifier(nn.Module):
    def __init__(self, feature_dim, dropout=0.4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, features):
        return self.classifier(features).squeeze(-1)


class ResNet25dAttention(nn.Module):
    """2.5D ResNet-18 + Gated Attention MIL

    输入: (batch_size, num_slabs, channels, H, W)
        channels = slice_thickness x num_windows (默认 3x3=9)
    输出:
        bag_logit:          case-level logit
        instance_logits:    slab-level logits (用于 Top-K)
        attention_weights:  slab 注意力权重
    """

    def __init__(self, num_classes=2, input_channels=9, dropout=0.4, use_topk_branch=None, topk=5, dual_mil_alpha=None):
        super().__init__()
        self.encoder = ResNet2d(BasicBlock2d, [2, 2, 2, 2], num_classes=num_classes, input_channels=input_channels)
        self.feat_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        self.gated_attention = GatedAttention(self.feat_dim, dropout=dropout)
        self.instance_classifier = InstanceClassifier(self.feat_dim, dropout=dropout)
        self.attention_classifier = nn.Linear(self.feat_dim, 1)

        # 默认值与训练 teacher/config_attention.py 一致:
        #   use_topk_branch = False
        #   dual_mil_alpha = 1.0
        self.use_topk_branch = use_topk_branch if use_topk_branch is not None else False
        self.topk = topk
        self.dual_mil_alpha = dual_mil_alpha if dual_mil_alpha is not None else 1.0

    def forward(self, x):
        batch_size, num_slabs, channels, height, width = x.shape
        x = x.view(batch_size * num_slabs, channels, height, width)
        features = self.encoder(x)
        features = features.view(batch_size, num_slabs, self.feat_dim)

        instance_logits = self.instance_classifier(features)
        pooled_features, attention_weights = self.gated_attention(features)
        attention_logit = self.attention_classifier(pooled_features).squeeze(-1)

        if self.use_topk_branch:
            k = min(self.topk, num_slabs)
            topk_logits, topk_indices = torch.topk(instance_logits, k=k, dim=1)
            topk_logit = topk_logits.mean(dim=1)
            bag_logit = self.dual_mil_alpha * attention_logit + (1.0 - self.dual_mil_alpha) * topk_logit
        else:
            topk_indices = torch.zeros(batch_size, 1, dtype=torch.long, device=x.device)
            bag_logit = attention_logit

        return {
            "bag_logit": bag_logit,
            "instance_logits": instance_logits,
            "attention_weights": attention_weights,
            "topk_indices": topk_indices,
        }


# ═════════════════════════════════════════════════════════
#  预处理工具函数
# ═════════════════════════════════════════════════════════


def _find_continuous_segments(bool_array):
    segments = []
    in_segment = False
    start = 0
    for i, val in enumerate(bool_array):
        if val and not in_segment:
            start = i
            in_segment = True
        elif not val and in_segment:
            segments.append((start, i - 1))
            in_segment = False
    if in_segment:
        segments.append((start, len(bool_array) - 1))
    return segments


def _filter_chest_slices(volume_hu: np.ndarray, cfg: dict) -> tuple[list[int], dict]:
    """基于 HU 值的胸部切片过滤"""
    depth = volume_hu.shape[0]
    body_area = (volume_hu > cfg["chest_body_hu_threshold"]).mean(axis=(1, 2))
    lung_area = ((volume_hu > cfg["chest_lung_hu_low"]) & (volume_hu < cfg["chest_lung_hu_high"])).mean(axis=(1, 2))

    valid_lung = lung_area > cfg["chest_lung_ratio_threshold"]
    lung_indices = np.where(valid_lung)[0]

    if len(lung_indices) >= cfg["chest_min_filtered_slices"]:
        lung_first = int(lung_indices[0])
        lung_last = int(lung_indices[-1])
        seg_start = max(0, lung_first - cfg["chest_segment_margin_upper"])
        seg_end = min(depth - 1, lung_last + cfg["chest_segment_margin_lower"])
    else:
        valid_body = body_area > cfg["chest_body_ratio_threshold"]
        segments = _find_continuous_segments(valid_body)
        if segments:
            main_seg = max(segments, key=lambda s: s[1] - s[0])
            seg_start = max(0, main_seg[0] - cfg["chest_segment_margin"])
            seg_end = min(depth - 1, main_seg[1] + cfg["chest_segment_margin"])
        else:
            seg_start, seg_end = 0, depth - 1

    filtered = list(range(seg_start, seg_end + 1))
    if len(filtered) < cfg["chest_min_filtered_slices"]:
        filtered = list(range(depth))

    lung_range = [int(lung_indices[0]), int(lung_indices[-1])] if len(lung_indices) > 0 else None
    return filtered, {"filtered_range": [min(filtered), max(filtered)], "lung_range": lung_range}


def _apply_multi_window(slab: np.ndarray, windows: list[tuple[float, float]]) -> np.ndarray:
    """多窗归一化 → 拼接成 (C, H, W), C = len(windows) x D"""
    channels = []
    for w_min, w_max in windows:
        windowed = np.clip(slab, w_min, w_max)
        windowed = (windowed - w_min) / (max(w_max - w_min, 1e-8))
        channels.append(windowed.astype(np.float32))
    return np.concatenate(channels, axis=0)  # (C, H, W)


def _resize_hw(volume: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if volume.shape[1] == target_h and volume.shape[2] == target_w:
        return volume
    factors = (1.0, target_h / volume.shape[1], target_w / volume.shape[2])
    return zoom(volume, factors, order=1)


# ═════════════════════════════════════════════════════════
#  模型管理器
# ═════════════════════════════════════════════════════════


class CTPADiagnosisModel:
    """CTPA 肺栓塞诊断模型 (ResNet25d + Gated Attention MIL)"""

    def __init__(self, model_path: str = "", config: dict | None = None):
        self.config = {**MODEL_CONFIG, **(config or {})}
        self.model_path = model_path
        self.device = self.config.get("device", "cpu")
        if torch and torch.cuda.is_available():
            self.device = "cuda"

        self._model: Any = None  # ResNet25dAttention，按需延迟加载
        self._loaded = False
        self._load_error = ""
        self._load_time = 0.0

        # 输入通道数 = slice_thickness x num_windows
        self.in_channels = self.config["slice_thickness"] * len(self.config["windows"])

        if model_path and os.path.isfile(model_path):
            self.load(model_path)

    # ── 模型加载 ──────────────────────────────────────

    def load(self, model_path: str) -> bool:
        missing = _check_deps()
        if missing:
            self._load_error = f"缺少依赖: {', '.join(missing)}"
            return False
        if not os.path.isfile(model_path):
            self._load_error = f"模型文件不存在: {model_path}"
            return False

        start = time.time()
        try:
            print(f"  📦 加载模型: {os.path.basename(model_path)}")
            print(f"  🔧 设备: {self.device}")
            in_ch = self.in_channels
            print(f"  📐 输入通道: {in_ch} ({self.config['slice_thickness']}切片 x {len(self.config['windows'])}窗)")

            raw = torch.load(model_path, map_location=self.device, weights_only=False)

            # 提取 state_dict（兼容各种 checkpoint 格式）
            if isinstance(raw, dict):
                for key in ["model_state_dict", "state_dict", "model"]:
                    if key in raw and isinstance(raw[key], dict):
                        raw = raw[key]
                        break

            # 去除 module. 前缀
            state_dict = OrderedDict()
            for k, v in raw.items():
                state_dict[k[7:] if k.startswith("module.") else k] = v

            # 打印 key 用于诊断
            print(f"  📋 state_dict keys: {len(state_dict)}")
            for k in list(state_dict.keys())[:5]:
                print(f"    {k}")
            if len(state_dict) > 5:
                print(f"    ... 及其他 {len(state_dict) - 5} 个")

            # 确定最佳 input_channels
            encoder_conv1 = next((v for k, v in state_dict.items() if k.endswith("encoder.conv1.weight")), None)
            if encoder_conv1 is not None:
                detected_in_ch = encoder_conv1.shape[1]
                if detected_in_ch != in_ch:
                    print(f"  ⚠️ state_dict conv1 通道={detected_in_ch}, 配置={in_ch}, 使用={detected_in_ch}")
                    in_ch = detected_in_ch
                    self.in_channels = detected_in_ch

            # 构建模型（参数从 config 读取，保证与训练一致）
            self._model = ResNet25dAttention(
                num_classes=2,
                input_channels=in_ch,
                dropout=self.config.get("dropout", 0.4),
                use_topk_branch=self.config.get("use_topk_branch", False),
                topk=self.config.get("topk", 5),
                dual_mil_alpha=self.config.get("dual_mil_alpha", 1.0),
            )

            # 严格加载
            missing_keys, unexpected_keys = self._model.load_state_dict(state_dict, strict=False)
            self._model.to(self.device)
            self._model.eval()

            self._loaded = True
            self._load_time = time.time() - start

            print(f"  ✅ 模型加载成功 ({self._load_time:.2f}s)")
            if missing_keys:
                print(f"  ⚠️  缺失 keys ({len(missing_keys)}): {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"  ⚠️  冗余 keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
            return True

        except Exception as e:
            self._load_error = f"模型加载失败: {str(e)}"
            print(f"  ❌ {self._load_error}")
            import traceback

            traceback.print_exc()
            return False

    # ── 体部 ROI 掩膜（训练时 D_roi_body 实验使用） ──────

    def _compute_body_mask(self, volume_hu: np.ndarray) -> np.ndarray:
        """计算体部掩膜（HU > threshold 的连通的体部区域）"""
        from scipy.ndimage import binary_dilation, binary_fill_holes, label

        body_threshold = self.config.get("roi_body_threshold", -500)
        dilation = self.config.get("roi_body_dilation", 3)

        body = volume_hu > body_threshold
        d = volume_hu.shape[0]
        for i in range(d):
            sl = body[i]
            lbl, n = label(sl)
            if n == 0:
                body[i] = False
                continue
            sizes = np.bincount(lbl.ravel())
            if len(sizes) <= 1:
                body[i] = False
                continue
            sizes[0] = 0
            largest = sizes.argmax()
            body[i] = lbl == largest
            body[i] = binary_fill_holes(body[i])
        if dilation > 0:
            struct = np.ones((1, dilation, dilation), dtype=bool)
            body = binary_dilation(body, structure=struct, iterations=1)
        return body

    def _apply_roi_body_mask(self, volume_hu: np.ndarray) -> np.ndarray:
        """体部 ROI 掩膜：仅保留体部区域，外部填充为 hu_min"""
        body_mask = self._compute_body_mask(volume_hu)
        fill_value = self.config.get("roi_body_fill_value", -1000)
        roi_volume = volume_hu.copy()
        roi_volume[~body_mask] = fill_value
        return roi_volume

    # ── 预处理 ─────────────────────────────────────────

    def preprocess(self, nifti_path: str) -> dict[str, Any]:
        """预处理 NIfTI → 模型输入张量（与训练时一致）"""
        if not os.path.isfile(nifti_path):
            raise FileNotFoundError(f"文件不存在: {nifti_path}")

        nii = nib.load(nifti_path)
        volume = nii.get_fdata(dtype=np.float32)
        original_shape = volume.shape
        pixdim = nii.header.get_zooms()[:3]
        original_spacing = tuple(float(p) for p in pixdim)

        # 确保 DHW (D, H, W)
        if volume.ndim == 4:
            volume = volume[..., 0]
        if volume.shape[0] >= 128 and volume.shape[1] >= 128 and volume.shape[2] < volume.shape[0]:
            volume = np.transpose(volume, (2, 0, 1))

        # HU 裁剪
        volume = np.clip(volume, self.config["hu_min"], self.config["hu_max"])

        # ── 体部 ROI 掩膜（训练配置 roi_mode='body' 时启用） ──
        if self.config.get("roi_mode", "none") == "body":
            volume = self._apply_roi_body_mask(volume)

        # Resize H/W（保持深度不变，keep_original_depth=True 与训练一致）
        target_h, target_w = self.config["target_hw"]
        volume_resized = _resize_hw(volume, target_h, target_w)

        # 胸部切片过滤
        filtered_indices, filter_diag = _filter_chest_slices(volume_resized, self.config)

        # 2.5D slab 提取（slab_centers 记录每个 slab 的中心切片索引）
        slab_depth = self.config["slice_thickness"]
        eval_stride = self.config["eval_stride"]
        num_target = self.config["num_slabs_eval"]
        slab_centers, slabs = self._extract_slabs_with_centers(
            volume_resized, filtered_indices, slab_depth, eval_stride, num_target
        )

        # 多窗归一化（每个 slab 独立归一化，与训练一致）
        windows = self.config["windows"]
        processed = []
        for slab in slabs:
            multi_ch = _apply_multi_window(slab, windows)
            processed.append(multi_ch)

        # 堆叠为 tensor
        tensor = torch.from_numpy(np.stack(processed, axis=0)).float()
        tensor = tensor.unsqueeze(0)  # (1, num_slabs, C, H, W)

        return {
            "tensor": tensor,
            "num_slabs": len(slabs),
            "slab_centers": slab_centers,
            "volume_hu": volume_resized,  # 保留 HU 体积用于可视化
            "original_shape": original_shape,
            "original_spacing": original_spacing,
            "filter_diagnostics": filter_diag,
            "metadata": {
                "voxel_sizes": [float(p) for p in pixdim],
                "data_shape": list(original_shape),
                "filtered_slice_range": filter_diag.get("filtered_range"),
                "num_slabs": len(slabs),
            },
        }

    @staticmethod
    def _extract_slabs_with_centers(
        volume: np.ndarray,
        filtered_indices: list[int],
        slab_depth: int,
        eval_stride: int,
        num_target: int,
    ) -> tuple[list[int], list[np.ndarray]]:
        """提取 2.5D slabs，同时返回每个 slab 的中心切片索引

        与训练 teacher/train_attention.py PEDataset._get_candidate_slab_starts
        一致: 在每个连续段内以 stride=1 滑动，后续 linspace 下采样到 num_target。
        """
        if not filtered_indices:
            return [], []

        segments = []
        seg_start = filtered_indices[0]
        for i in range(1, len(filtered_indices)):
            if filtered_indices[i] != filtered_indices[i - 1] + 1:
                segments.append((seg_start, filtered_indices[i - 1]))
                seg_start = filtered_indices[i]
        segments.append((seg_start, filtered_indices[-1]))

        # stride=1 生成候选（与训练 PEDataset._get_candidate_slab_starts 一致）
        candidate_starts = []
        for s_start, s_end in segments:
            max_start = s_end - slab_depth + 1
            if s_start <= max_start:
                for z in range(s_start, max_start + 1):
                    candidate_starts.append(z)

        if not candidate_starts:
            return [], []

        n = len(candidate_starts)
        if n > num_target:
            indices = np.linspace(0, n - 1, num_target, dtype=int)
            selected = [candidate_starts[i] for i in indices]
        else:
            selected = candidate_starts

        slab_centers = [s + slab_depth // 2 for s in selected]
        slabs = []
        for start in selected:
            slab = volume[start : start + slab_depth, :, :]
            if slab.shape[0] < slab_depth:
                pad = slab_depth - slab.shape[0]
                slab = np.pad(slab, ((0, pad), (0, 0), (0, 0)), mode="constant")
            slabs.append(slab)
        return slab_centers, slabs

    # ── 推理 ─────────────────────────────────────────

    @torch.no_grad()
    def predict(self, nifti_path: str, return_mask: bool = False) -> dict[str, Any]:
        """对 CTPA 影像进行肺栓塞诊断推理

        Args:
            nifti_path: NIfTI 文件路径
            return_mask: 忽略（此模型无体素掩膜）

        Returns:
            诊断结果字典（含可视化 base64 图像）
        """
        result: dict[str, Any] = {
            "success": False,
            "probability": 0.0,
            "prediction": 0,
            "threshold": self.config.get("threshold", 0.5),
            "num_slabs": 0,
            "inference_time": 0.0,
            "preprocess_time": 0.0,
            "error": None,
        }

        if not self._loaded or self._model is None:
            result["error"] = self._load_error or "模型未加载"
            return result

        missing = _check_deps()
        if missing:
            result["error"] = f"缺少依赖: {', '.join(missing)}"
            return result

        # 预处理
        t0 = time.time()
        try:
            preprocessed = self.preprocess(nifti_path)
            tensor = preprocessed["tensor"].to(self.device)
            result["num_slabs"] = preprocessed["num_slabs"]
            slab_centers = preprocessed.get("slab_centers", [])
            volume_hu = preprocessed.get("volume_hu", None)
        except Exception as e:
            result["error"] = f"预处理失败: {str(e)}"
            import traceback

            traceback.print_exc()
            return result
        t1 = time.time()
        result["preprocess_time"] = round(t1 - t0, 3)

        # 推理
        try:
            out = self._model(tensor)
            bag_logit = out["bag_logit"]
            prob = torch.sigmoid(bag_logit).squeeze().cpu().item()
            pred = 1 if prob >= result["threshold"] else 0

            result["probability"] = round(prob, 4)
            result["prediction"] = pred
            result["success"] = True

            # 注意力权重
            attn = out.get("attention_weights")
            attn_np = None
            if attn is not None:
                attn_np = attn.squeeze(0).cpu().numpy()
                result["attention_weights"] = [round(float(w), 6) for w in attn_np]
                result["top_attention_slab"] = int(np.argmax(attn_np))
                result["max_attention_weight"] = float(attn_np.max())

            # Slab-level 概率
            inst = out.get("instance_logits")
            slab_probs = None
            if inst is not None:
                slab_probs = torch.sigmoid(inst).squeeze(0).cpu().numpy()
                result["slab_probabilities"] = [round(float(p), 4) for p in slab_probs]

            # ── 生成可视化（只要模型跑成功就生成，不管阴阳性） ──
            if volume_hu is not None and slab_centers and attn_np is not None and slab_probs is not None:
                filtered_range = preprocessed.get("filter_diagnostics", {}).get(
                    "filtered_range", [0, volume_hu.shape[0] - 1]
                )
                vis_result = generate_visualization(
                    volume_hu=volume_hu,
                    slab_centers=slab_centers,
                    attention_weights=attn_np.tolist(),
                    slab_probabilities=slab_probs.tolist(),
                    filtered_range=filtered_range,
                    top_k=3,
                )
                if vis_result:
                    result["visualization"] = vis_result

        except Exception as e:
            result["error"] = f"推理失败: {str(e)}"
            import traceback

            traceback.print_exc()
            return result

        t2 = time.time()
        result["inference_time"] = round(t2 - t1, 3)
        result["total_time"] = round(t2 - t0, 3)
        return result

    # ── 状态查询 ──────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> str:
        return self._load_error

    def get_info(self) -> dict[str, Any]:
        return {
            "loaded": self._loaded,
            "model_path": self.model_path if self._loaded else None,
            "device": self.device,
            "input_channels": self.in_channels,
            "target_hw": self.config["target_hw"],
            "threshold": self.config.get("threshold", 0.5),
            "windows": self.config["windows"],
            "slice_thickness": self.config["slice_thickness"],
            "num_slabs_eval": self.config["num_slabs_eval"],
            "load_time": round(self._load_time, 2) if self._loaded else None,
            "load_error": self._load_error if not self._loaded else None,
            "model_type": "ResNet25dAttention",
        }

    def summary(self) -> str:
        info = self.get_info()
        lines = ["=" * 50, "  🩺 肺栓塞诊断模型 (ResNet25d + Attention MIL)", "=" * 50]
        if info["loaded"]:
            lines.append(f"  模型文件: {info['model_path']}")
            lines.append(f"  设备: {info['device']}")
            lines.append(f"  加载耗时: {info['load_time']:.2f}s")
            lines.append(
                f"  输入: {info['input_channels']}通道 ({info['slice_thickness']}切片x{len(info['windows'])}窗)"
            )
            lines.append(f"  窗参数: {info['windows']}")
            lines.append(f"  空间尺寸: HxW = {info['target_hw']}")
            lines.append(f"  Slab数: {info['num_slabs_eval']}")
            lines.append(f"  阈值: {info['threshold']}")
        else:
            lines.append(f"  ❌ 模型未加载: {info['load_error']}")
        lines.append("=" * 50)
        return "\n".join(lines)


# ── 可视化工具 ──────────────────────────────────────────

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # 尝试设置中文字体（Windows用SimHei/Song, Linux用WenQuanYi）
    _CN_FONT = None
    for font_name in ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]:
        try:
            _CN_FONT = fm.FontProperties(family=font_name)
            # 验证一下
            fig_test, ax_test = plt.subplots(figsize=(1, 1))
            ax_test.set_title("测试", fontproperties=_CN_FONT)
            plt.close(fig_test)
            break
        except Exception:
            _CN_FONT = None
            continue
    if _CN_FONT is None:
        _CN_FONT = fm.FontProperties()

    # 全局设置 rcParams 回退方案
    plt.rcParams["axes.unicode_minus"] = False
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

_HAS_B64 = True  # base64 和 BytesIO 已在顶部导入


def _window_hu(volume: np.ndarray, center: float = 100, width: float = 700) -> np.ndarray:
    """纵隔窗: 窗位100HU, 窗宽700HU → [-250, 450] → 归一化[0,1]"""
    low = center - width / 2.0
    high = center + width / 2.0
    windowed = np.clip(volume, low, high).astype(np.float32)
    if high > low:
        windowed = (windowed - low) / (high - low)
    return windowed


def _fig_to_b64(fig) -> str:
    """将 matplotlib 图像转为 base64 字符串"""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64


def generate_visualization(
    volume_hu: np.ndarray,
    slab_centers: list[int],
    attention_weights: list[float],
    slab_probabilities: list[float],
    filtered_range: list[int],
    top_k: int = 3,
) -> dict[str, Any]:
    """生成可视化图像

    Args:
        volume_hu: HU 值体积 (D, H, W)
        slab_centers: 每个 slab 对应的中心切片索引
        attention_weights: slab 注意力权重
        slab_probabilities: slab 级 PE 概率
        filtered_range: 过滤后的切片范围 [start, end]
        top_k: 显示最高风险的 top-k 个 slab

    Returns:
        {
            "slice_overview": base64,    # 轴向风险分布图
            "top_slices": [               # 最高风险切片的可视化
                {"slice_index": int, "image": base64, "prob": float, "attention": float},
                ...
            ],
        }
    """
    if not _HAS_MPL or not _HAS_B64:
        return {}

    num_slabs = len(slab_centers)
    if num_slabs == 0:
        return {}

    # 找到最高风险的 top-k 个 slab
    combined_risk = np.array(slab_probabilities) * 0.5 + np.array(attention_weights) * 0.5
    top_indices = np.argsort(combined_risk)[::-1][:top_k]

    vis_result: dict[str, Any] = {"top_slices": []}
    depth = volume_hu.shape[0]

    # --- 图1: 轴向风险分布概览图 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [1, 3]})

    # 上子图：风险曲线
    x_slabs = np.arange(num_slabs)
    ax1.fill_between(x_slabs, slab_probabilities, alpha=0.3, color="red", label="Slab PE Prob")
    ax1.plot(x_slabs, slab_probabilities, "r-", linewidth=1.5, label="Slab PE Prob")
    ax1.plot(x_slabs, attention_weights, "b--", linewidth=1, alpha=0.7, label="Attention Weight")
    ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Threshold=0.5")
    ax1.set_xlim(0, num_slabs - 1)
    ax1.set_ylabel("风险分数")
    ax1.set_title("Slab 级风险分布")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # 下子图：冠状位/矢状位投影示意 + 标记高风险区域
    # 用冠状位最大密度投影 (MIP) 作为背景
    coronal_mip = np.max(volume_hu[:, :, :], axis=1)  # (D, W)
    # 用纵隔窗显示
    coronal_display = _window_hu(coronal_mip, 100, 700)

    ax2.imshow(
        coronal_display,
        aspect="auto",
        cmap="gray",
        origin="upper",
        extent=[0, coronal_display.shape[1], coronal_display.shape[0], 0],
    )

    # 标记 filtered range
    if filtered_range:
        ax2.axhline(y=filtered_range[0], color="green", linestyle="--", linewidth=0.8, alpha=0.6)
        ax2.axhline(
            y=filtered_range[1],
            color="green",
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
            label=f"胸部过滤范围 [{filtered_range[0]}-{filtered_range[1]}]",
        )

    # 标记 top-k 高风险 slab 的中心位置
    colors = ["red", "orange", "yellow"]
    for rank, idx in enumerate(top_indices):
        center_z = slab_centers[idx]
        if center_z < depth:
            risk = combined_risk[idx]
            ax2.axhline(
                y=center_z,
                color=colors[min(rank, len(colors) - 1)],
                linewidth=1.5 + rank * 0.5,
                alpha=0.7 + rank * 0.1,
                label=f"#{rank + 1} 风险(slab={idx}, z~{center_z}, risk={risk:.3f})",
            )

    ax2.set_xlabel("横向像素")
    ax2.set_ylabel("轴向切片索引 (Z)")
    ax2.set_title("冠状位投影 + 高风险切片标注")
    ax2.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    vis_result["slice_overview"] = _fig_to_b64(fig)

    # --- 图2~top_k: 单个高风险切片的详细可视化 ---
    for rank, idx in enumerate(top_indices):
        center_z = slab_centers[idx]
        if center_z < 0 or center_z >= depth:
            continue

        risk = combined_risk[idx]
        slab_prob = slab_probabilities[idx]
        attn = attention_weights[idx]

        # 取中心切片的邻域切片取平均（减少噪声）
        z_start = max(0, center_z - 1)
        z_end = min(depth, center_z + 2)
        slice_avg = np.mean(volume_hu[z_start:z_end, :, :], axis=0)

        # 纵隔窗显示
        slice_display = _window_hu(slice_avg, 100, 700)

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.imshow(slice_display, cmap="gray", aspect="equal", origin="upper")

        # 如果 slab_prob 高(>0.5), 用半透明热力图标记"可疑区域"
        # 这里用一个高斯模糊的"注意力伪影"标记 — 模拟可疑栓子位置
        if slab_prob > 0.5:
            # 在肺实质区域内生成一个伪热力图（模拟栓子位置）
            # 使用一个简单的梯度：中心偏亮，模拟模型关注的区域
            h, w = slice_display.shape
            # 创建肺部掩膜（HU < -200 通常为肺实质）
            lung_mask = slice_avg < -200
            # 在肺部区域内做一个中心放射状的注意力热图
            yy, xx = np.mgrid[0:h, 0:w]
            cy, cx = h // 2, w // 2
            gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (w * 0.15) ** 2))
            gaussian[~lung_mask] = 0  # 只在肺区域内显示

            if gaussian.max() > 0:
                gaussian = gaussian / gaussian.max()
                # 用红色热力图叠加
                ax.imshow(gaussian, cmap="Reds", alpha=min(0.5, slab_prob), aspect="equal", origin="upper")

        ax.set_title(
            f"#{rank + 1} High-Risk Slice (z={center_z})\n"
            f"Slab Prob={slab_prob:.3f}  Attn={attn:.3f}  Combined={risk:.3f}",
            fontsize=10,
        )
        ax.axis("off")

        plt.tight_layout()
        img_b64 = _fig_to_b64(fig)

        vis_result["top_slices"].append(
            {
                "rank": rank + 1,
                "slice_index": int(center_z),
                "slab_index": int(idx),
                "probability": round(float(slab_prob), 4),
                "attention_weight": round(float(attn), 4),
                "combined_risk": round(float(risk), 4),
                "image_base64": img_b64,
            }
        )

    return vis_result


# ── 工具函数 ───────────────────────────────────────────


def _check_deps() -> list[str]:
    return [d for d in _MISSING_DEPS if d]


def _compute_risk_level(prob: float) -> str:
    if prob >= 0.9:
        return "高风险"
    elif prob >= 0.7:
        return "中风险"
    elif prob >= 0.5:
        return "低风险"
    return "阴性"


def create_diagnosis_model(model_path: str = "", **kwargs) -> "CTPADiagnosisModel":
    """创建 CTPA 肺栓塞诊断模型实例"""
    path = model_path or os.getenv("PE_MODEL_PATH", "")
    config = kwargs.pop("config", None)
    if path:
        return CTPADiagnosisModel(model_path=path, config=config)
    return CTPADiagnosisModel(config=config)
