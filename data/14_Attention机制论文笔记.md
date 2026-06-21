# Attention Is All You Need 论文笔记

## 论文信息
- 标题：Attention Is All You Need
- 作者：Ashish Vaswani, Noam Shazeer, Niki Parmar 等
- 发表：NeurIPS 2017

## 核心贡献

该论文提出了一种完全基于注意力机制的序列到序列模型架构——Transformer，彻底摒弃了传统的循环和卷积结构。Transformer 在机器翻译任务上取得了当时最优的结果，同时训练速度显著优于基于 RNN 的模型。

## 核心概念

### Scaled Dot-Product Attention
缩放点积注意力是 Transformer 中最基础的注意力机制。给定查询矩阵 Q、键矩阵 K 和值矩阵 V，注意力输出计算如下：
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) × V

其中 d_k 是键向量的维度，除以 sqrt(d_k) 防止点积结果过大导致 softmax 梯度消失。

### Multi-Head Attention
多头注意力允许模型在不同表示子空间中联合关注信息。具体做法是将 Q、K、V 通过 h 组不同的线性投影映射到 h 个子空间，分别计算注意力后再拼接和投影。

### Positional Encoding
由于 Transformer 不包含递归和卷积结构，无法天然感知序列中 token 的位置信息。因此需要额外加入位置编码，使用不同频率的正弦和余弦函数生成位置向量。

## 架构细节

### 编码器
Transformer 编码器由 N=6 个相同的层堆叠而成，每层包含两个子层：
1. 多头自注意力子层
2. 逐位置的前馈神经网络（FFN）子层
每个子层后使用残差连接和层归一化（Layer Normalization）。

### 解码器
解码器结构与编码器类似，但每层包含三个子层：
1. 掩码多头自注意力子层（防止看到未来位置）
2. 交叉注意力子层（将编码器输出作为 K、V）
3. FFN 子层

## 实验结果

在 WMT 2014 英德翻译任务上，Transformer 的 BLEU 值达到 28.4，比之前最优结果高出 2.0 以上。在英法翻译任务上达到 41.0 BLEU，训练成本仅为其他模型的几分之一。

## 后续影响

Transformer 架构已经成为现代深度学习的基础设施，衍生出 BERT、GPT、ViT、Swin Transformer 等众多里程碑式模型，深刻影响了 NLP、CV 和多媒体领域的发展。
