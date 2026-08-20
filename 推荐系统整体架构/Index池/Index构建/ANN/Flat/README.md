# Flat 精确检索

Flat 不建立用于缩小搜索范围的近似结构，而是让查询向量与 Index 池中的每个向量都计算一次距离，再返回精确 Top-K。它不是 ANN，却是理解和评测所有 ANN 方法的起点。

## 核心直觉

假设 Index 池有 $N$ 个 $d$ 维商品向量。要找与查询向量 $\boldsymbol q$ 最相似的商品，最直接的方法是逐个比较：

1. 计算 $\boldsymbol q$ 与全部商品向量的相似度或距离。
2. 从 $N$ 个分数中选出最优的 $K$ 个。
3. 按分数排序并将向量行号转换回业务商品 ID。

Flat 不会漏掉真正的近邻，因此结果可以作为 ANN 的 ground truth。代价是每次查询都要读取全部向量并完成约 $N \times d$ 次数值运算，查询复杂度为 $O(Nd)$。

## 距离与归一化

本例使用余弦相似度：

$$
s_{\cos}(\boldsymbol q, \boldsymbol x)
= \frac{\boldsymbol q^{\mathsf T}\boldsymbol x}
{\|\boldsymbol q\|_2\|\boldsymbol x\|_2}
$$

若先将查询和商品向量归一化为单位向量，那么分母都为 1，余弦相似度就变成一次内积：

$$
s_{\cos}(\hat{\boldsymbol q}, \hat{\boldsymbol x})
= \hat{\boldsymbol q}^{\mathsf T}\hat{\boldsymbol x}
$$

因此实际实现可以把全部商品向量组成矩阵 $X \in \mathbb R^{N \times d}$，一次矩阵向量乘法 $X\hat{\boldsymbol q}$ 就得到所有分数。归一化不是无关紧要的预处理：如果遗漏它，向量范数会影响内积排序，检索语义将不再是余弦相似度。

## 构建过程

Flat 的“构建”很轻量：

1. 校验业务 ID 数量与向量行数一致。
2. 将向量转换成统一的 `float32`。
3. 对每个向量做 L2 归一化。
4. 保存向量矩阵，并单独保存“矩阵行号到业务 ID”的映射。

业务 ID 不应被假设为从 0 连续递增。向量矩阵适合使用连续行号寻址，而业务 ID 可能是稀疏的长整数或字符串，两者应通过显式映射解耦。

## 查询过程

示例查询分为四步：

1. 将查询向量归一化。
2. 通过 `self._vectors @ normalized_query` 计算全部余弦分数。
3. 使用 `np.argpartition` 在线性期望时间内找出 Top-K 的无序位置，避免对全部 $N$ 个分数完整排序。
4. 只对这 $K$ 个候选排序，再映射为 `(item_id, score)`。

`argpartition` 只优化了 Top-K 选择，无法减少与全部向量算分的成本。Flat 的主要瓶颈通常是扫描向量所需的内存带宽，其次才是算术运算。

## 与示例代码的对应关系

[index.py](./index.py) 中：

- `normalize`：对一批向量执行 L2 归一化，并用极小值保护零范数。
- `FlatCosineIndex.add`：保存业务 ID 和归一化后的向量矩阵。
- `FlatCosineIndex.search`：全量算分、选取 Top-K、排序并还原业务 ID。
- `_item_ids[position]`：体现内部位置与业务 ID 的解耦。

运行示例：

```powershell
python .\index.py
```

## 适用场景

- 小规模在线 Index 池。
- 离线批量检索，尤其是能使用 GPU 矩阵乘法时。
- 为 ANN 生成精确 Top-K 真值并计算 Recall@K。
- 在上线前验证距离函数、归一化和 ID 映射是否正确。

## 常见误区

- **用 ANN 自己评测 ANN**：没有 Flat 真值，就无法知道返回结果丢失了多少真实近邻。
- **距离语义不一致**：不能用余弦 Flat 真值评价未经归一化的内积 ANN。
- **把全排序当成必要步骤**：只需要 Top-K 时应使用 partial selection。
- **只看计算量**：数据量变大后，向量矩阵能否驻留内存以及内存带宽通常更关键。

Flat 的意义不仅是“小数据时够用”，更重要的是为后续近似方法提供一个语义正确、结果精确且容易排错的参照系。