# BPR Matrix Factorization

## 核心思路

协同过滤从用户-物品交互矩阵中学习偏好。这里选择适合隐式反馈的矩阵分解基线：每个用户和物品拥有一个隐向量，分数为 $\hat y_{ui}=p_u^\top q_i$。

## 数据构造

正例来自点击、购买或有效观看。BPR 训练样本是三元组 $(u,i^+,i^-)$：用户 $u$ 与正物品 $i^+$ 有交互，而 $i^-$ 从未观测集合或曝光未点集合采样。按时间切分数据，避免未来交互进入训练矩阵。

## 典型结构与损失

示例实现矩阵分解，并使用 BPR 损失：

$$
\mathcal{L}_{BPR}=-\log\sigma(\hat y_{ui^+}-\hat y_{ui^-})+\lambda\lVert\Theta\rVert_2^2
$$

## 文件与运行

- [model.py](./model.py)：用户/物品 Embedding 和 BPR 损失。
- [train.py](./train.py)：生成隐式反馈三元组并训练 Top-K 推荐。

安装 PyTorch 后执行 `python train.py`。

## 部署方式

可离线计算每个用户的 Top-N 并写入 KV，也可将用户/物品隐向量接入 ANN。模型目录只负责向量和打分定义，索引生命周期由共享基础设施管理。

## 局限与扩展

主要限制是冷启动、交互稀疏和流行度偏差。可继续整理加权 ALS、LightGCN、多行为图模型、自监督图学习和因果去偏。