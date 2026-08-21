# 序列预测表示

序列模型根据有序用户行为预测下一物品或被遮蔽物品。训练后，item token embedding、输出分类器权重或内容编码器表示可以用于 I2I。与对称共现不同，它能学习“从物品 $i$ 之后更可能到物品 $j$”的转移语义。

## 1. 序列与会话

用户行为序列记为：

$$
S_u=(i_1,i_2,\ldots,i_T)
$$

应按时间排序，并根据空闲间隔、场景切换和业务边界切分 session。过长窗口会混合多个意图，过短窗口则损失长期依赖。

行为类型可加入 token、权重或 side feature：

$$
h_t=e_{item}(i_t)+e_{action}(a_t)+e_{position}(t)+e_{time}(\Delta t_t)
$$

## 2. Next-item Prediction

自回归模型根据前缀预测下一物品：

$$
P(i_{t+1}|i_{\le t})=
\operatorname{softmax}(h_t^	op V+b)
$$

损失为：

$$
\mathcal L_{next}=-\sum_t
\log P(i_{t+1}|i_{\le t})
$$

全量 softmax 在百万物品上成本很高，可使用 sampled softmax、负采样、分层 softmax 或候选内 softmax。

若只使用最后一个物品作为上下文，模型退化为有方向的 item transition。使用 Transformer、GRU 或卷积后，$h_t$ 融合整个前缀，此时它本质上更接近 user-to-item 序列召回。为了生成静态 I2I，可固定只输入单个种子，或从输出矩阵中导出 item-to-item 转移近邻。

## 3. Masked-item Modeling

随机遮蔽序列中的物品，使用双向上下文恢复：

$$
\mathcal L_{mask}=-\sum_{t\in M}
\log P(i_t|S_u^{masked})
$$

该目标类似 BERT4Rec。它能同时利用左右上下文，适合离线表示预训练；在线下一跳预测时必须确保输入中没有未来行为。

Mask 比例过高会破坏局部意图，过低则训练信号稀疏。随机替换还可能把真实相关物品当成噪声，应结合 session 长度调参。

## 4. 输出哪一种 embedding

序列模型可能包含三类物品表示：

1. **输入 item embedding**：表示物品作为历史上下文的角色。
2. **输出分类器 embedding**：表示物品作为预测目标的角色。
3. **内容 encoder embedding**：由文本、图像和属性生成，可覆盖新品。

若输入与输出矩阵共享参数，空间天然统一；不共享时可以计算：

$$
s(i,j)=(e_i^{input})^\top e_j^{output}
$$

这是一种非对称 I2I，适合下一跳关系，但不能直接用单一对称余弦索引。可分别建立 source/query 向量和 target/corpus 向量的 MIPS 索引。

## 5. 与 Item2Vec 的区别

| 维度 | Item2Vec | 序列预测模型 |
| --- | --- | --- |
| 上下文 | 固定局部窗口 | 整个前缀或双向上下文 |
| 顺序 | 常被弱化 | 显式建模位置和方向 |
| 模型 | embedding 点积 | GRU/Transformer 等上下文编码器 |
| 目标 | 窗口共现判别 | 下一物品或 Mask 恢复 |
| 成本 | 低 | 较高 |

## 6. 负采样与流行度

采样 softmax 的负例分布会影响 embedding 几何。全局随机负例过于容易；按流行度采样更贴近竞争集合，但需要修正采样概率，否则 logits 学到的是采样分布而非完整 softmax。

同 session 的其他真实物品不应无条件当作负例。可使用时间上相距较远的物品、曝光未点击或 ANN 难负例，并过滤同款、续集等已知关系。

## 7. I2I 导出与在线使用

有两种服务方式：

- **静态 I2I**：对每个 item 的 query embedding 查询 target ANN，物化 Top-N 转移近邻表。
- **实时序列召回**：在线编码完整近期序列得到 $h_T$，直接查询 item ANN。

前者延迟低、解释为“基于某个种子”；后者能表达多行为交互，但属于序列 U2I。生产系统可以同时保留，两者分别评估增量价值。

## 8. 数据泄漏与评测

训练样本必须保证输入行为早于目标行为。按时间切分用户序列，词表、热门度、负采样分布和 ANN 索引也只能使用训练时点可见数据。

静态 I2I 使用真实下一跳集合评估 Recall@K、MRR 和 NDCG@K；完整序列模型使用用户级 next-item 指标。还应按 session 长度、种子热门度、新旧物品和时间间隔分桶，并监控热门集中度与重复推荐。

序列模型适合消费顺序明显的场景；若目标只是稳定的同类替代关系，内容模型或 ItemCF 往往更直接、更易维护。