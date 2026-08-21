# SimCSE-style：Dropout 双视图

SimCSE 原本是句向量方法，并非经典 I2I 算法。本章借用其无监督表示思想：同一物品的特征两次经过共享的带 dropout encoder，随机性形成两个视图；同一物品跨视图为正例，batch 内其他物品为负例。

它与 [InfoNCE](../InfoNCE/README.md) 不是互斥方法：SimCSE-style 描述正例视图如何生成，训练 loss 仍然是 InfoNCE。

## 1. 双视图编码

对物品特征 $x_i$：

$$
z_i^{(1)}=
\operatorname{normalize}(f_\theta(x_i;\xi_i^{(1)}))
$$

$$
z_i^{(2)}=
\operatorname{normalize}(f_\theta(x_i;\xi_i^{(2)}))
$$

$\xi_i^{(1)},\xi_i^{(2)}$ 是两次独立 dropout mask。encoder 参数完全共享，差异只来自随机前向过程。

正例不是两个不同物品，而是同一物品的两个随机表示。因此它学习的是对 dropout 扰动保持稳定的实例表示。

## 2. 双向对比目标

相似度矩阵为：

$$
S_{ij}=\frac{(z_i^{(1)})^\top z_j^{(2)}}{\tau}
$$

第 $i$ 行的正例是第 $i$ 列。双向损失为：

$$
\mathcal L=
\frac12\left[
CE(S,y)+CE(S^\top,y)
\right],
\qquad y_i=i
$$

[example.py](./example.py) 精确实现这一形式，温度固定为 $0.1$，两侧均通过 `cross_entropy` 训练。

## 3. 为什么 dropout 可以形成训练信号

若两次前向完全相同，正例向量也完全相同，模型缺少学习随机不变性的约束。Dropout 随机屏蔽中间神经元，使 encoder 必须在部分内部特征缺失时仍产生一致表示。

Dropout 概率存在权衡：

- 太低时两个视图几乎一致，增强信号弱。
- 太高时语义信息被破坏，正例匹配变难。
- 较小 batch 中负例不足，模型可能只记住实例边界。
- 较大 batch 增加负例，也增加真实相似物品被当成假负例的概率。

## 4. 输入特征决定可学内容

示例输入为类目 one-hot 与属性 one-hot：

$$
x_i=[onehot(category_i);onehot(attribute_i)]
$$

encoder 是：

```text
Linear -> ReLU -> Dropout -> Linear -> L2 normalize
```

它只能学习输入特征包含的类目和属性结构。真实场景可使用：

- 标题、描述和标签的文本 encoder。
- 图像或视频 encoder。
- 类目、品牌、价格带和数值属性。
- 多模态融合表示。

若输入只有唯一 item ID，任务容易退化为实例识别：模型只需让每个 ID 与自身接近、与其他 ID 分开，不会自动把业务相关物品聚在一起。

## 5. 除 dropout 外的增强

物品内容可构造更多双视图：

- 文本 token dropout、字段 dropout 或同义改写。
- 图像随机裁剪、颜色扰动和不同关键帧。
- 属性字段遮蔽。
- 多模态视图，例如同一物品的文本与图片。
- 图上的边 dropout 或子图视图。

增强必须保持推荐语义。删除品牌、型号、作者或关键实体可能让视图改变物品身份；图像水平翻转可能改变带文字或方向性的商品含义。应按业务字段定义可变与不可变部分。

## 6. 假负例与多正例

batch 中其他物品默认都是负例，但同款 SKU、同系列内容或近重复物品可能是真正正例。处理方式包括：

- 对同 SPU、同内容指纹和已知关系构造 positive mask。
- 在分母中屏蔽重复 item ID。
- 将跨物品行为正例与自身双视图同时作为多正例。
- 对同类目高相似候选降权。
- 使用跨批队列时同步保存 item ID 和关系元数据。

只使用自身双视图会强调实例区分；加入跨物品正例后，空间才会明确形成可用于推荐的物品簇。

## 7. 防止表示退化

对比 loss 通常通过大量负例防止所有向量坍缩到同一点，但仍可能出现各向异性或 hubness：少数向量成为大量物品的最近邻。

建议监控：

- 每个维度的方差和协方差谱。
- 正例、随机负例和难负例的相似度分布。
- 最近邻入度分布与 hub 物品。
- 向量范数以及归一化前激活大小。
- 不同类目和新品的覆盖率。

也可加入 variance/covariance 正则、whitening 或更丰富的跨物品监督，但需通过检索指标验证。

## 8. 与其他自监督方法的区别

| 方法 | 正例 | 主要目标 |
| --- | --- | --- |
| SimCSE-style | 同一物品的 dropout 双视图 | 随机扰动不变性 |
| Item2Vec | 同 session 窗口中的不同物品 | 行为上下文预测 |
| 跨模态对比 | 同一物品的文本与图像 | 模态对齐 |
| 图对比学习 | 同节点的两种图增强 | 图结构不变性 |
| Masked modeling | 恢复被遮蔽字段或 token | 信息重建 |

这些目标可以组合，但每个 loss 的数据量和梯度尺度需要校准。

## 9. 训练、导出与新品

训练时 `model.train()` 保持 dropout 开启；导出 embedding 时必须调用 `model.eval()` 关闭 dropout，保证同一物品产生确定表示。随后执行 L2 归一化并建立 cosine ANN 或 Top-N 近邻表。

内容特征 encoder 可以直接编码新品，前提是新品字段完整且预处理版本一致。若某些模态缺失，应在训练时加入 modality dropout 和缺失标识，而不是在上线时临时填充任意零值。

## 10. 评测

除自身双视图检索准确率外，必须使用跨物品业务关系和用户下一次行为评测 Recall@K/NDCG@K。自身检索接近 100% 只表示实例可区分，并不证明 I2I 近邻有价值。

重点比较：

- 通用内容 encoder 与 SimCSE-style 微调后的差异。
- 仅自身正例与加入行为正例的差异。
- 新品、成熟、长尾和模态缺失分桶。
- 与文本 TF-IDF、ItemCF 和 InfoNCE pair 模型的增量命中。

## 11. 示例

[example.py](./example.py) 使用类目和属性 one-hot 特征、MLP 和 dropout 双视图训练，并在 `eval` 模式导出余弦近邻：

```powershell
python example.py
```

示例没有跨物品正例、多正例 mask 和跨批队列，仅用于说明共享 encoder 的双随机视图训练。