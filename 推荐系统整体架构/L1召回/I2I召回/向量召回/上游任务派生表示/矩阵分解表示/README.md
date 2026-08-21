# 矩阵分解表示

矩阵分解（MF）从 user-item 交互矩阵学习用户和物品的低维隐向量。其主要目标是 user-to-item 预测，但训练得到的 item embedding 也可用于 I2I 近邻检索。

完整的 MF 原理、损失与训练实现见[协同过滤/MF](../../../../协同过滤/MF/README.md)。本章聚焦如何将物品因子转换为 I2I。

## 1. 基本形式

设交互矩阵为 $R$：

$$
R\approx UV^\top
$$

$U\in\mathbb R^{|\mathcal U|\times d}$ 是用户因子，$V\in\mathbb R^{|\mathcal I|\times d}$ 是物品因子。预测分数为：

$$
\hat r_{ui}=p_u^\top q_i
$$

训练后使用 $q_i$ 作为物品 embedding。I2I 相似度常取：

$$
sim_{cos}(i,j)=
\frac{q_i^\top q_j}{\lVert q_i\rVert_2\lVert q_j\rVert_2}
$$

若线上 user-item 检索使用内积，也可以直接使用 $q_i^\top q_j$，但向量范数可能混入热门度或置信度。余弦更强调方向相似，内积同时偏好大范数物品。

## 2. 不同目标产生不同几何

### 2.1 显式反馈 MSE

$$
\mathcal L=\sum_{(u,i)\in\Omega}
(r_{ui}-p_u^\top q_i)^2+\lambda\lVert\Theta\rVert_2^2
$$

物品向量编码评分模式。两个物品被相似用户给出相似评分时更可能接近。

### 2.2 隐式反馈 Weighted ALS

$$
\mathcal L=\sum_{u,i}c_{ui}
(p_{ui}-p_u^\top q_i)^2
+\lambda\lVert\Theta\rVert_2^2
$$

它把未观测交互作为低置信度负反馈，物品空间反映用户偏好模式，而不是原始共现次数。

### 2.3 BPR

$$
\mathcal L=-\sum_{(u,i^+,i^-)}
\log\sigma(p_u^\top q_{i^+}-p_u^\top q_{i^-})
+\lambda\lVert\Theta\rVert_2^2
$$

BPR 直接优化用户对正负物品的排序。item-item 余弦是训练后的副产物，不是目标函数直接约束的关系，因此必须单独评测近邻质量。

## 3. 与 ItemCF 的关系

ItemCF 直接在高维用户空间比较物品列向量，MF 则将这些交互模式压缩到低维空间：

$$
R^\top R
\quad\longrightarrow\quad
VV^\top
$$

ItemCF 更保留局部共现和解释性；MF 可以通过低秩共享泛化到没有直接共现但受相似用户偏好的物品。MF 仍然依赖历史行为，对无交互新品无能为力。

## 4. 去偏与后处理

MF item embedding 的范数、密度和近邻可能受到曝光频次影响。构建 I2I 时常执行：

- L2 归一化后使用 cosine ANN。
- 对热门物品设置最大入度或分桶 quota。
- 排除同一物品、失效物品和不允许的跨类目关系。
- 对极低频物品回退到内容 embedding。
- 将 MF 相似度与 ItemCF、文本分数分别校准后融合。

若模型包含 item bias $b_i$，不要把 bias 拼进向量计算余弦；它通常表达全局热门度，而非方向性相似。

## 5. 离线与在线

```text
MF checkpoint -> 导出 item factors -> L2 归一化
              -> ANN 或全量 Top-N -> item-neighbor KV
```

物品量较小时可分块矩阵乘法计算 $VV^\top$；大规模物品池使用 ANN。必须保持 checkpoint、ID 映射、归一化配置和索引版本一致。

## 6. 评测

除用户级 I2I Recall@K 外，应比较 MF-I2I 与 ItemCF 的邻居重合率、覆盖率、热门集中度和增量命中。若 MF 的 user-item 指标很好但 item-item 关系标注集较差，说明其物品空间适合个性化打分，却不适合直接解释为物品相似度。

MF-I2I 适合作为协同泛化通道，而不应默认替代 ItemCF 或内容近邻。