# LR 与 Wide & Deep

LR（Logistic Regression）以稀疏字段、数值特征和人工构造的交叉为输入，是 CTR/CVR 精排中最重要的可解释基线。Wide & Deep 在 LR 的 wide 分支外并联 DNN 的 deep 分支：前者记忆已经验证的共现与规则，后者从稠密 embedding 中泛化到未显式出现的字段组合。它们解决的是特征表达与字段交叉问题，不替代候选感知历史、序列依赖或多任务结构。

本页与[基础与特征交叉总览](../README.md)配套使用。字段词表、缺失、OOV、时间可用性及训练服务一致性必须先满足[特征工程与训练服务一致性](../../../数据与特征/特征工程与训练服务一致性/README.md)的约束。

## 原始文献

| 模型 | 来源 | 核心贡献 |
| --- | --- | --- |
| LR | Cox, *The Regression Analysis of Binary Sequences*, JRSS B, 1958 | 用对数几率线性建模二分类概率，构成工业排序的稳定基线。 |
| Wide & Deep | Cheng et al., *Wide & Deep Learning for Recommender Systems*, DLRS 2016 | 将记忆型 wide 特征与泛化型 deep 表征联合训练。 |

## LR：结构与能力边界

对于样本特征向量 $\boldsymbol{x}$，LR 输出点击或转化概率：

$$
p(y=1 \mid \boldsymbol{x}) = \sigma(\boldsymbol{w}^{\top}\boldsymbol{x}+b),
\qquad
\sigma(z) = \frac{1}{1+e^{-z}}.
$$

```mermaid
flowchart LR
    A[类别字段、数值字段与人工交叉] --> B[稀疏和数值特征向量 x]
    B --> C[线性加权]
    C --> D[Sigmoid]
    D --> E[CTR 或 CVR 概率]
```

类别字段通常经 one-hot、哈希或字典索引形成稀疏特征；连续值可经缺失标记、归一化或分桶后输入。为了让线性模型表达非加性关系，需要显式加入交叉，例如 $x_{\text{user\_segment}} \times x_{\text{item\_category}}$、$x_{\text{scene}} \times x_{\text{recall\_source}}$。

| 能力 | LR 的表现 | 说明 |
| --- | --- | --- |
| 一阶字段贡献 | 强 | 系数与特征语义直接对应，便于排查特征和标签问题。 |
| 已知规则交叉 | 强 | 依赖人工构造，适合业务语义稳定的组合。 |
| 未见组合泛化 | 弱 | 稀疏交叉无法从相似实体或相邻分桶迁移知识。 |
| 训练与推理成本 | 很低 | 稀疏线性计算易扩展，也适合作为降级模型。 |
| 概率校准 | 易治理 | 仍需按场景与人群检查校准，不能假定天然可靠。 |

## Wide & Deep：联合记忆与泛化

Wide & Deep 的预测 logit 可写为：

$$
z = \boldsymbol{w}_{\text{wide}}^{\top}[\boldsymbol{x}, \boldsymbol{\phi}(\boldsymbol{x})]
+ \boldsymbol{w}_{\text{deep}}^{\top}\boldsymbol{h}_L+b,
$$

其中 $\boldsymbol{\phi}(\boldsymbol{x})$ 是人工交叉，$\boldsymbol{h}_L$ 是由字段 embedding 拼接后经过多层感知机得到的表示。两分支联合最小化同一任务损失；wide 分支保留高置信规则，deep 分支通过共享 embedding 为稀疏和长尾组合提供平滑泛化。

```mermaid
flowchart LR
    A[原始字段与上下文特征] --> B[特征编码]
    B --> C[人工交叉 phi(x)]
    B --> D[字段 Embedding]
    B --> E[原始稀疏与数值特征]
    D --> F[拼接]
    F --> G[多层感知机]
    E --> H[Wide 线性分支]
    C --> H
    G --> I[Deep 分支 logit]
    H --> J[Wide 分支 logit]
    I --> K[logit 求和]
    J --> K
    K --> L[Sigmoid]
    L --> M[CTR 或 CVR 概率]
```

| 分支 | 输入与机制 | 优势 | 主要风险 |
| --- | --- | --- | --- |
| Wide | 原始稀疏特征、人工交叉、计数或规则特征 | 可解释、稳定、可精确记忆 | 交叉维护成本高，覆盖外组合无泛化。 |
| Deep | 字段 embedding、多值池化、MLP | 自动学习非线性和相似性 | 易受词表、缺失语义、特征漂移和参数预算影响。 |
| 联合输出 | 两路 logit 相加后 sigmoid | 可同时利用规则与泛化 | 任一路离在线不一致都会污染最终概率。 |

## 适用场景与选型

| 场景或症状 | 首选 | 原因 |
| --- | --- | --- |
| 新建精排链路、标签或特征质量尚待验证 | LR | 结构最简单，能更快定位 schema、样本和服务问题。 |
| 业务规则稳定，需要保留可审计交叉 | Wide & Deep | 将确定性规则放入 wide，同时保留 deep 的长尾泛化。 |
| 多离散字段的二阶交叉维护困难 | [DeepFM](../DeepFM/README.md) | FM 可自动学习二阶交互，减少人工交叉枚举。 |
| 显式高阶交叉仍有可复现增益 | [DCNv2](../DCNv2/README.md) | Cross Network 可控制交叉阶数与参数化方式。 |
| 历史与当前候选的匹配更关键 | [用户兴趣建模](../../用户兴趣建模/README.md) | 字段交叉无法替代候选相关的行为兴趣。 |

不宜仅因模型简单而忽略它：当样本很少、线上延迟严格、特征可解释性要求高，LR 或轻量 Wide & Deep 往往是长期可用的生产模型，而非一次性对照。

## 最小可运行示例

[model.py](./model.py) 实现了按字段 embedding 求和的 `LogisticRegression` 与并联 wide/deep 分支的 `WideAndDeep`；[train.py](./train.py) 用固定随机种子的用户、物品、场景及人工 `user-item` 交叉字段训练两者。

```powershell
python train.py
```

示例输入为整型离散字段，模型输出未经概率校准的 logit。第四个字段代表预先构造的人工交叉，说明 wide 分支仍依赖可在线一致重建的交叉特征。

## 训练、服务与评测检查

- 将训练与线上交叉函数、哈希种子、分桶边界、词表版本和默认值版本化；训练出现的交叉必须能在请求时刻重建。
- 对长尾类别、未登录用户、新物品、空多值字段和 OOV 设置明确回退，避免静默使用错误索引。
- Wide 与 deep 的输入字段应有单独的覆盖率、缺失率和分布监控；规则交叉不能通过离线回填未来信息。
- 固定候选、样本切分、特征集合和参数预算，再与 DeepFM/DCNv2 比较 AUC、LogLoss、ECE、分群指标及 P99。
- 将模型概率与最终业务效用分开：概率校准、融合权重和 L3 列表规则需要独立版本化与实验验证。

## 优缺点总结

| 模型 | 优点 | 局限 |
| --- | --- | --- |
| LR | 快、稳、易解释、便于诊断和降级 | 高阶关系依赖人工特征，长尾泛化较弱。 |
| Wide & Deep | 兼顾规则记忆与 embedding 泛化，可平滑演进 | 仍需维护 wide 交叉，DNN 增加参数、监控和服务复杂度。 |

## 常见误解

- **LR 只是过时的对照模型**：它常是验证特征、标签、校准和可服务性的关键基线。
- **Wide 特征越多覆盖越好**：每个交叉都应有稳定语义、足够覆盖和线上可用字段。
- **Deep 分支会自动修复错误特征**：词表漂移、时间泄漏和缺失语义错误会同时损害两条分支。
- **wide 的高权重意味着因果规则**：模型系数只反映给定训练分布下的相关性，需用实验或因果分析判断业务动作。
