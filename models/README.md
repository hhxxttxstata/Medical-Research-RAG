# 肺栓塞诊断模型权重

阅读本目录存放训练好的肺栓塞诊断模型权重文件。

## 支持格式
- **PyTorch**: `.pth` / `.pt` (推荐)

## 配置方式
通过环境变量 `PE_MODEL_PATH` 指定模型路径：

```bash
# Windows CMD
set PE_MODEL_PATH=models/pe_model.pth

# PowerShell
$env:PE_MODEL_PATH="models/pe_model.pth"

# 或直接在 .env 文件中添加
PE_MODEL_PATH=models/pe_model.pth
```

## 模型要求
- 输入: 3D CTPA 体积 (NIfTI 格式, `.nii` / `.nii.gz`)
- 默认输入尺寸: (128, 256, 256) — (Depth, Height, Width)
- 预处理: 窗位 100 HU, 窗宽 700 HU, 重采样至 1mm 各向同性
- 输出: 肺栓塞概率 (0~1) + 体素级分割掩膜

## 兼容性
- 加载时 `strict=False`，支持自定义模型架构
- 如骨架网络 `_PEDiagnosisNet` 与你的模型结构不匹配，可在 `src/diagnosis.py` 中修改
- 支持纯 CPU 推理，也支持 CUDA GPU 加速
