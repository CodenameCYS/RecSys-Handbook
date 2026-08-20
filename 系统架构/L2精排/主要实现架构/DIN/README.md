# DIN

## Core Idea

Deep Interest Network（DIN）不把用户历史压缩成与候选无关的固定向量，而是以当前候选物品为 query，对历史行为逐条计算相关性。不同候选因此会激活不同的用户兴趣。

## Architecture

1. 将历史物品和目标物品映射到同一 Embedding 空间。
2. 对每个历史行为构造 $[e_h,e_t,e_h-e_t,e_h\odot e_t]$。
3. Activation Unit 输出候选感知的注意力权重，并通过 mask 忽略 padding。
4. 加权聚合历史兴趣，与目标物品交互后输入 MLP 预测 CTR。

当前样板聚焦 DIN 的核心注意力机制。生产实现通常还会加入用户画像、上下文、物品属性、Dice 激活、正则化、特征共享与校准。

## Files

- [model.py](./model.py)：Activation Unit、变长历史 mask 和 DIN 预测网络。
- [train.py](./train.py)：合成变长行为序列并训练点击预测。

安装 PyTorch 后执行 `python train.py`。

## DIN 与其他模型

- DeepFM 重点学习通用显式与隐式特征交互。
- DIN 重点建模“历史兴趣与当前候选”的相关性。
- DIEN 进一步使用 GRU 和辅助损失刻画兴趣演化。
- SASRec/BERT4Rec 更侧重行为顺序和长距离序列依赖。