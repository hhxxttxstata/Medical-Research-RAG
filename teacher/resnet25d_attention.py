import math
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import resnet18 as tv_resnet18

    try:
        from torchvision.models import ResNet18_Weights
    except ImportError:
        ResNet18_Weights = None
except Exception:
    tv_resnet18 = None
    ResNet18_Weights = None


# 18/34
class BasicBlock_2d(nn.Module):
    expansion = 1  # 每一个conv的卷积核个数的倍数

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):  # downsample对应虚线残差结构
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channel)  # BN处理
        self.relu = nn.ReLU(inplace=True)  # 尽量使用inplace操作flag，节省显存
        self.conv2 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # 捷径上的输出值，为了保证原始输入与卷积后的输出层叠加时维度相同
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


# 50,101,152
class Bottleneck_2d(nn.Module):
    """
    :param in_channel: 输入block的之前的通道数
    :param out_channel: 在block中间处理的时候的通道数
            out_channel*self.extention:输出的维度
    :param stride:卷积步长
    :param downsample:在_make_layer函数中赋值，在resnet每层链接的第一个卷积层需要改变通道
    """

    expansion = 4  # 4倍，类变量，可通过类名修改

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            bias=False,
        )  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=stride,
            bias=False,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel * self.expansion,  # 输出*4
            kernel_size=1,
            stride=1,
            bias=False,
        )  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
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
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity  # 残差连接
        out = self.relu(out)

        return out


class ResNet2d(nn.Module):
    def __init__(self, block, layers, num_classes=2, input_channels=1):
        super().__init__()
        self.in_channels = 64
        self.input_channels = input_channels
        self.num_classes = num_classes

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
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
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

    def load_pretrained_weights(self, pretrained_path=None, strict=False, verbose=True):
        return load_resnet2d_pretrained(
            self,
            pretrained_path=pretrained_path,
            strict=strict,
            verbose=verbose,
        )


# -----------------------------
# 预训练权重加载辅助函数
# -----------------------------
def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, OrderedDict):
        return checkpoint

    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "network"]:
            value = checkpoint.get(key)
            if isinstance(value, (dict, OrderedDict)):
                return value

    raise ValueError("无法从 checkpoint 中解析 state_dict。")


def _strip_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        new_state_dict[key] = value
    return new_state_dict


def _adapt_first_conv_weight(conv1_weight, input_channels):
    pretrained_in_channels = conv1_weight.shape[1]
    if pretrained_in_channels == input_channels:
        return conv1_weight

    if input_channels == 1 and pretrained_in_channels == 3:
        return conv1_weight.mean(dim=1, keepdim=True)

    if pretrained_in_channels == 1 and input_channels > 1:
        return conv1_weight.repeat(1, input_channels, 1, 1) / float(input_channels)

    repeat_times = int(math.ceil(float(input_channels) / float(pretrained_in_channels)))
    expanded = conv1_weight.repeat(1, repeat_times, 1, 1)[:, :input_channels, :, :]
    expanded = expanded * (float(pretrained_in_channels) / float(input_channels))
    return expanded


def _get_torchvision_resnet18_state_dict():
    if tv_resnet18 is None:
        raise ImportError("未检测到 torchvision，无法自动加载 ImageNet 预训练权重。")

    try:
        if ResNet18_Weights is not None:
            model = tv_resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            model = tv_resnet18(pretrained=True)
    except TypeError:
        model = tv_resnet18(pretrained=True)

    return model.state_dict()


def load_resnet2d_pretrained(model, pretrained_path=None, strict=False, verbose=True):
    if pretrained_path is None:
        state_dict = _get_torchvision_resnet18_state_dict()
        source = "torchvision resnet18 ImageNet"
    else:
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"预训练权重文件不存在: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        state_dict = _extract_state_dict(checkpoint)
        source = pretrained_path

    state_dict = _strip_module_prefix(state_dict)

    if "conv1.weight" in state_dict:
        state_dict["conv1.weight"] = _adapt_first_conv_weight(
            state_dict["conv1.weight"],
            model.conv1.in_channels,
        )

    model_state = model.state_dict()

    # 分类头维度不一致时自动跳过，便于迁移到2分类任务
    if "fc.weight" in state_dict and state_dict["fc.weight"].shape != model_state["fc.weight"].shape:
        state_dict.pop("fc.weight")
    if "fc.bias" in state_dict and state_dict["fc.bias"].shape != model_state["fc.bias"].shape:
        state_dict.pop("fc.bias")

    loadable_state_dict = OrderedDict()
    skipped_keys = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            loadable_state_dict[key] = value
        else:
            skipped_keys.append(key)

    load_result = model.load_state_dict(loadable_state_dict, strict=strict)

    if verbose:
        print("Loaded pretrained weights from:", source)
        print(f"Loaded params: {len(loadable_state_dict)}/{len(model_state)}")
        if skipped_keys:
            print("Skipped keys:", skipped_keys)
        if load_result.missing_keys:
            print("Missing keys:", load_result.missing_keys)
        if load_result.unexpected_keys:
            print("Unexpected keys:", load_result.unexpected_keys)

    return model


# -----------------------------
# 工厂函数
# -----------------------------
def resnet2d_18(num_classes=2, input_channels=1, pretrained=False, pretrained_path=None):
    """
    2.5D ResNet-18 model

    Args:
        num_classes: 分类数
        input_channels: 输入通道数，CT 一般为 1
        pretrained: True 时自动加载 torchvision 的 ImageNet 预训练权重
        pretrained_path: 本地预训练权重路径；若提供，则优先使用该路径
    """
    model = ResNet2d(BasicBlock_2d, [2, 2, 2, 2], num_classes=num_classes, input_channels=input_channels)

    if pretrained_path is not None:
        model.load_pretrained_weights(pretrained_path=pretrained_path)
    elif pretrained:
        model.load_pretrained_weights(pretrained_path=None)

    return model


class GatedAttention(nn.Module):
    def __init__(self, feature_dim, dropout=0.4):
        super().__init__()
        self.feature_dim = feature_dim
        self.dropout = nn.Dropout(dropout)
        self.attention_V = nn.Linear(feature_dim, feature_dim)
        self.attention_U = nn.Linear(feature_dim, feature_dim)
        self.attention_w = nn.Linear(feature_dim, 1, bias=False)

    def forward(self, features):
        """
        Args:
            features: (batch_size, num_instances, feature_dim)
        Returns:
            pooled: (batch_size, feature_dim)
            attention_weights: (batch_size, num_instances)
        """
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
    def __init__(
        self,
        num_classes=2,
        input_channels=3,
        pretrained=True,
        pretrained_path=None,
        dropout=0.4,
        use_topk_branch=True,
        topk=5,
        dual_mil_alpha=0.5,
    ):
        super().__init__()
        self.encoder = resnet2d_18(
            num_classes=num_classes,
            input_channels=input_channels,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
        )
        self.feat_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        self.gated_attention = GatedAttention(self.feat_dim, dropout=dropout)
        self.instance_classifier = InstanceClassifier(self.feat_dim, dropout=dropout)
        self.attention_classifier = nn.Linear(self.feat_dim, 1)

        self.use_topk_branch = use_topk_branch
        self.topk = topk
        self.dual_mil_alpha = dual_mil_alpha

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_instances, thickness, H, W) for 2.5D
        Returns:
            dict with keys: bag_logit, instance_logits, attention_weights, topk_indices
        """
        batch_size, num_instances, thickness, height, width = x.shape

        x = x.view(batch_size * num_instances, thickness, height, width)
        features = self.encoder(x)
        features = features.view(batch_size, num_instances, self.feat_dim)

        instance_logits = self.instance_classifier(features)

        pooled_features, attention_weights = self.gated_attention(features)
        attention_logit = self.attention_classifier(pooled_features).squeeze(-1)

        if self.use_topk_branch:
            k = min(self.topk, num_instances)
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


def resnet25d_attention(
    num_classes=2,
    input_channels=3,
    pretrained=True,
    pretrained_path=None,
    dropout=0.4,
    use_middle_slice_only=True,
    use_topk_branch=True,
    topk=5,
    dual_mil_alpha=0.5,
):
    return ResNet25dAttention(
        num_classes=num_classes,
        input_channels=input_channels,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        dropout=dropout,
        use_topk_branch=use_topk_branch,
        topk=topk,
        dual_mil_alpha=dual_mil_alpha,
    )
