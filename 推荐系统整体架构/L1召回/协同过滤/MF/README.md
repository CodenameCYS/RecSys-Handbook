# Matrix Factorization

本目录提供最小的隐式反馈矩阵分解实现。模型直接学习用户 embedding $p_u$ 与物品 embedding $q_i$，以内积作为偏好分数：

$$
\hat y_{ui}=p_u^\top q_i
$$

## 1. 模型

[model.py](./model.py) 包含两张可训练 embedding 表，`score` 计算用户与物品的成对内积。基础实现未加入全局偏置、用户偏置和物品偏置；需要拟合显式评分时，可以使用：

$$
\hat y_{ui}=\mu+b_u+b_i+p_u^\top q_i
$$

MF 描述的是模型参数化方式，损失函数则由任务目标决定。显式评分预测、隐式反馈分类和 Top-K 排序使用的训练目标并不相同。

## 2. 常见损失函数

### 2.1 显式反馈：MSE

评分、观看时长等连续目标常使用均方误差。设 $\Omega$ 为已观测交互集合：

$$
\mathcal L_{MSE}
=\frac{1}{|\Omega|}\sum_{(u,i)\in\Omega}
\left(r_{ui}-\hat y_{ui}\right)^2
+\lambda\lVert\Theta\rVert_2^2
$$

该目标只对观测值拟合，不应直接把所有未观测位置当成真实评分 $0$。

### 2.2 隐式反馈 pointwise 目标：BCE

点击、收藏、购买等二值目标可以使用二元交叉熵。对采样集合 $\mathcal D$ 中的正负样本，令 $y_{ui}\in\{0,1\}$：

$$
\mathcal L_{BCE}
=-\frac{1}{|\mathcal D|}\sum_{(u,i,y)\in\mathcal D}
\left[y\log\sigma(\hat y_{ui})
+(1-y)\log\left(1-\sigma(\hat y_{ui})\right)\right]
+\lambda\lVert\Theta\rVert_2^2
$$

BCE 分别约束每个 user-item 对的分数。训练结果依赖负样本数量与采样分布，线上排序通常直接使用 logit $\hat y_{ui}$。

### 2.3 隐式反馈 pairwise 目标：BPR

BPR（Bayesian Personalized Ranking）直接优化同一用户下正物品与负物品的相对次序。对三元组集合：

$$
\mathcal D_{BPR}=\{(u,i^+,i^-)\mid i^+\in I_u^+,\ i^-\notin I_u^+\}
$$

其损失为：

$$
\mathcal L_{BPR}=-\frac{1}{B}\sum_{b=1}^{B}\log\sigma
\left(p_{u_b}^\top q_{i_b^+}-p_{u_b}^\top q_{i_b^-}\right)
+\lambda\mathcal R_B
$$

当前实现中的 batch 正则项精确对应：

$$
\mathcal R_B=
\frac{1}{Bd}\sum_{b=1}^{B}
\left(
\lVert p_{u_b}\rVert_2^2
+\lVert q_{i_b^+}\rVert_2^2
+\lVert q_{i_b^-}\rVert_2^2
\right)
$$

其中 $B$ 是 batch 大小，$d$ 是 embedding 维度。由于 [model.py](./model.py) 对三个 embedding 张量分别调用 `square().mean()`，同一 ID 在 batch 中重复出现时会按出现次数参与正则化。使用求和或只对唯一 ID 正则化也属于常见实现，但需要相应调整 $\lambda$；这些缩放约定不会改变 BPR 排序项本身。

代码中的 `bpr_loss` 先计算正负分差：

$$
\Delta_{uij}=\hat y_{ui^+}-\hat y_{ui^-}
$$

再通过 `-logsigmoid(Δ).mean()` 最小化 BPR 损失。当正物品分数高于负物品分数时，损失下降。因此当前实现与上述 BPR 公式一致。

### 2.4 隐式反馈全矩阵目标：Weighted MSE

Weighted ALS 常把未观测位置作为低置信度负反馈。定义：

$$
y_{ui}=\mathbb I(r_{ui}>0),\qquad c_{ui}=1+\alpha r_{ui}
$$

优化目标为：

$$
\mathcal L_{WMSE}
=\sum_{u,i}c_{ui}\left(y_{ui}-p_u^\top q_i\right)^2
+\lambda\left(\sum_u\lVert p_u\rVert_2^2+
\sum_i\lVert q_i\rVert_2^2\right)
$$

该目标通常使用交替最小二乘求解，而不是对所有 $|U||I|$ 位置执行普通 SGD。

### 2.5 损失函数选择

| 任务 | 常用损失 | 训练样本 | 特点 |
| --- | --- | --- | --- |
| 显式评分预测 | MSE | 已观测评分 | 优化数值拟合误差 |
| 隐式点击/转化预测 | BCE | 正例与采样负例 | pointwise 概率目标 |
| 隐式 Top-K 排序 | BPR | $(u,i^+,i^-)$ 三元组 | 直接优化正负相对顺序 |
| 隐式全矩阵分解 | Weighted MSE | 全部位置及置信度 | 常与 ALS 配合 |

本目录的代码与训练示例使用 **BPR loss**，因为目标是隐式反馈下的 Top-K 推荐，而不是评分回归。

## 3. 数据与训练

[train.py](./train.py) 构造合成正例集合，并为每个正例选择一个未交互物品形成 $(u,i^+,i^-)$。真实训练通常采用在线三元组采样：每轮随机用户和正边，再按均匀、热门或难负例分布采 $i^-$，避免将全部未观测矩阵物化。

生产流程还需要：

- 用训练集建立 `user -> positive items`，负样本不得命中任何已知正例。
- 按事件时间切分，不能用验证/测试行为构造训练三元组。
- 处理用户与物品 ID 映射、OOV 和已删除实体。
- 对热门度、活跃度分桶评测 Recall@K/NDCG@K。
- 保存模型、映射表、训练窗口、采样器和随机种子版本。

## 4. 导出与服务

训练后，`user_embeddings.weight` 和 `item_embeddings.weight` 就是两侧向量。可以离线计算每个用户的 Top-N，也可以将物品向量发布到 ANN，在线读取用户向量做最大内积检索。两侧向量、ID 映射和索引必须来自同一模型版本。

纯 MF 用户塔只是 `user_id -> embedding` 查表，不能直接响应刚发生的新行为。生产中可提高刷新频率、叠加近期物品向量，或换成能编码历史序列的双塔用户编码器。

## 5. 运行

```powershell
python train.py
```

依赖 PyTorch。示例采用全 batch 合成数据，只用于展示 BPR 的梯度训练和 Top-K 检索；真实大规模训练应使用 mini-batch 数据加载和动态负采样。