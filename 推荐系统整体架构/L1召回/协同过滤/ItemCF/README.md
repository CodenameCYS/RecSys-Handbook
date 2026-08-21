# ItemCF

ItemCF（Item-based Collaborative Filtering）根据用户历史或会话中的共同交互构造物品近邻。离线产物通常是 `item -> Top-K similar items`，线上以用户近期物品为种子展开、合并并过滤候选。

经典 ItemCF 不学习 user/item embedding。它直接统计物品共现并进行归一化，因此与基于 item embedding 的 ANN I2I 在服务形态上相似，但离线算法不同。

## 1. 数据表示

### 1.1 用户-物品矩阵

设交互矩阵为：

$$
R\in\{0,1\}^{|\mathcal U|\times|\mathcal I|}
$$

第 $i$ 列 $R_{\cdot i}$ 表示哪些用户与物品 $i$ 发生过目标交互。记：

$$
U(i)=\{u\mid R_{ui}=1\}
$$

忽略归一化时：

$$
R^\top R\in\mathbb R^{|\mathcal I|\times|\mathcal I|}
$$

其中 $(R^\top R)_{ij}=|U(i)\cap U(j)|$，即物品 $i,j$ 的共同用户数。

### 1.2 会话序列

内容流、电商和音乐场景通常保留行为顺序，并按超时间隔切分 session：

```text
session = [item_1, item_2, ..., item_n]
```

只在局部窗口中统计物品对，可以降低用户长期历史中无关兴趣之间的噪声，并表达更接近“看完当前物品后继续消费什么”的局部关系。

[model.py](./model.py) 接收 `list[list[int]]`。列表外层是 session 集合，内层物品顺序由旧到新排列。

## 2. 共现统计

### 2.1 全历史共现

不考虑顺序时，物品对的基础共现量为：

$$
C(i,j)=|U(i)\cap U(j)|
$$

该定义适合“共同被同一用户消费”的对称相关性，但用户的超长历史会产生大量跨兴趣物品对。

### 2.2 会话窗口共现

设窗口半径为 $w$，session $s$ 中位置 $p$ 的物品为 $x_{s,p}$。当前实现统计：

$$
C(i,j)=\sum_s\sum_p
\mathbb I(x_{s,p}=i)
\sum_{q:\,0<|q-p|\le w}
\mathbb I(x_{s,q}=j)
$$

每个 source 位置只与左右最多 $w$ 个位置配对。当前窗口是双向的，因此在没有重复物品干扰时，$i\rightarrow j$ 和 $j\rightarrow i$ 都会得到贡献。

代码没有加入距离或时间权重，窗口内每次共现贡献均为 $1$。生产中可改为：

$$
C(i,j)=\sum_{(i,j)\text{ 共现}}
\frac{1}{1+|p_i-p_j|}
\exp\left(-\frac{|t_i-t_j|}{\tau}\right)
$$

也可以只统计后向窗口 $q>p$，得到有方向的 $C(i\rightarrow j)$。

### 2.3 单物品频次

当前实现通过 `item_counts.update(set(session))` 统计：

$$
N(i)=|\{s\mid i\in s\}|
$$

因此 $N(i)$ 是包含物品 $i$ 的 session 数，不是物品在日志中的总出现次数。同一 session 内重复出现的物品只对 $N(i)$ 贡献一次，但每个位置仍可能参与窗口共现。

## 3. 物品相似度

当前实现使用余弦式归一化：

$$
sim(i,j)=\frac{C(i,j)}{\sqrt{N(i)N(j)}}
$$

若共现定义为共同用户数且 $N(i)=|U(i)|$，该式就是二值物品列向量的余弦相似度：

$$
sim_{cos}(i,j)=
\frac{R_{\cdot i}^\top R_{\cdot j}}
{\lVert R_{\cdot i}\rVert_2\lVert R_{\cdot j}\rVert_2}
=\frac{|U(i)\cap U(j)|}{\sqrt{|U(i)|\,|U(j)|}}
$$

分母降低热门物品仅凭较大曝光规模获得高共现的问题，但不能完全消除热门偏差。

### 3.1 其他归一化方法

- **Jaccard**：$sim_J(i,j)=|U(i)\cap U(j)|/|U(i)\cup U(j)|$。
- **Lift**：比较联合出现概率与独立出现概率，能强化超出随机预期的关系。
- **Log-likelihood ratio**：适合分析共现是否显著偏离独立假设。
- **Swing**：通过共同用户对及用户重叠程度抑制热门与刷行为影响。
- **条件概率**：$P(j\mid i)=C(i,j)/N(i)$，天然具有方向性，但偏向热门目标物品。

离线指标、线上覆盖和业务场景共同决定相似度公式，不能仅根据单一离线 Recall 选择。

## 4. 离线近邻表构建

```mermaid
flowchart LR
    A[用户行为日志] --> B[按超时间隔切分 session]
    B --> C[过滤与去重]
    C --> D[窗口内生成 item pairs]
    D --> E[累计 C(i,j) 与 N(i)]
    E --> F[归一化/热门惩罚]
    F --> G[每个 item 截断 Top-K]
    G --> H[item-neighbor KV]
```

当前 `fit` 的执行过程为：

1. 遍历所有 session，统计 session 级物品频次 $N(i)$。
2. 对每个 source 位置遍历左右窗口，累计 `pair_counts[source][target]`。
3. 计算 $C(i,j)/\sqrt{N(i)N(j)}$。
4. 对每个 source 的候选按相似度降序排列。

示例代码保留每个物品的全部已共现邻居，没有在 `fit` 中执行 Top-K 截断。生产构建必须在排序或堆合并阶段只保留固定数量邻居，避免近邻表无限增长。

典型离线产物为：

```text
item_id -> [
  (neighbor_item_id, similarity, support, reason, version),
  ...
]
```

除相似度外，保留共现支持度 $C(i,j)$ 有助于过滤偶然关系和生成解释特征。

### 4.1 计算复杂度

若 session $s$ 长度为 $L_s$，窗口半径为 $w$，窗口配对量近似为：

$$
O\left(\sum_s L_s\min(L_s,2w)\right)
$$

固定小窗口时近似线性于总行为数。若对每个 session 全量两两配对，则长 session 的复杂度会退化到 $O(L_s^2)$。

生产中还应：

- 截断或降采样超长 session。
- 过滤机器人、刷量和异常重复行为。
- 对超级热门物品降采样或降低配对贡献。
- 使用分布式聚合累计 item pair，并通过局部 Top-K 降低 shuffle 与存储量。

## 5. 在线多种子召回

从用户近期历史选择种子序列：

$$
S_u=[s_1,s_2,\ldots,s_m]
$$

当前实现约定列表按时间从旧到新排列，随后逆序遍历。最新种子的 rank 为 $1$，权重为：

$$
w_r=\frac{1}{r}
$$

候选分数为：

$$
score(u,j)=\sum_{r=1}^{m}
\frac{1}{r}\,sim(s_{m-r+1},j)
$$

所有种子物品都会从候选中排除，多个种子命中同一候选时分数累加。当前示例只排除传入的 `seed_items`，不会自动过滤用户更早的完整历史；生产服务需要额外传入并过滤完整已消费集合。

更一般的种子权重可以写为：

$$
w_{ui}=w_{behavior}(u,i)\cdot
w_{position}(u,i)\cdot
\exp\left(-\frac{t-t_{ui}}{\tau}\right)
$$

```mermaid
flowchart LR
    A[用户近期行为] --> B[选择并加权种子 item]
    B --> C[批量查询 item-neighbor KV]
    C --> D[种子权重 × 相似度]
    D --> E[跨种子累加与去重]
    E --> F[已看/资格过滤]
    F --> G[Top-N ItemCF 候选]
```

生产服务通常还需要：

- 限制种子数、每个种子的近邻展开数和总候选数。
- 对点击、收藏、购买等行为设置不同种子权重。
- 过滤已消费、下架、不可见、地域或年龄不合规物品。
- 保留最佳种子、最佳单路相似度、命中种子数等解释和排序特征。
- 对近邻缺失或 KV 超时设置热门、类目或 embedding I2I 降级。

## 6. ItemCF 与其他 I2I 方法

ItemCF 是离线算法，I2I 是线上检索方向。以下方法都能提供 `item -> items` 服务，但关系来源不同：

| 方法 | 关系来源 | 是否有 embedding | 主要特点 |
| --- | --- | --- | --- |
| ItemCF | 用户/session 共现 | 否 | 透明、稳定、易解释 |
| Item2Vec | 序列上下文预测 | 是 | 能学习平滑的局部语义 |
| 双塔 item embedding | 用户-物品监督目标 | 是 | 可结合内容与多任务信号 |
| GraphCF item embedding | 二部图多层传播 | 是 | 吸收高阶协同关系 |

ItemCF 的近邻通常更强调共同消费和局部转移；embedding ANN 的近邻可能表达更平滑但较难解释的潜在关系。实际系统常并行使用并分别控制配额。

统一的多种子查询、融合和过滤架构见 [I2I 召回综述](../../I2I召回/I2I召回综述.md)。

## 7. 数据更新、评测与监控

### 7.1 数据与版本

- 按业务间隔切分 session，避免跨天或长停顿行为被错误配对。
- 明确重复点击是否去重，以及购买、收藏、点击是否混合统计。
- 对数据窗口、窗口半径、相似度公式、过滤规则和近邻表统一版本化。
- 短期近邻适合表达热点和即时转移，长期近邻适合表达稳定相关性，可使用双表融合。

### 7.2 离线评测

应按时间切分：使用预测时点之前的 session 构建近邻表，以每个测试 session 的前缀预测后续物品。常用指标包括 Recall@K、HitRate@K、MRR、NDCG@K、覆盖率和新颖度。

评测时需要避免把目标物品之后的行为用于共现统计，否则会发生时间泄漏。还应分别报告热门、长尾、新品和不同历史长度用户的结果。

### 7.3 线上监控

重点监控近邻表覆盖率、种子有效率、每请求展开量、去重率、已看过滤率、空结果率、KV 延迟、候选热门度和通道独占贡献。

## 8. 局限与适用场景

ItemCF 的优势包括物品关系较稳定、近邻表易缓存、线上延迟低、结果可解释。主要局限包括：

- 新物品没有共现，无法仅靠 ItemCF 获得可靠近邻。
- 热门物品容易支配共现统计。
- 只统计无序共现时难以表达严格的消费方向。
- 窗口过小会降低覆盖，窗口过大会引入跨兴趣噪声。
- 多种子简单求和可能让高活跃用户或重复主题过度放大。

ItemCF 特别适合“看了又看”“买了又买”“下一首”“相关推荐”等局部意图明显的场景，也是常见的基础 I2I 通道。

## 9. 代码结构与运行

- [model.py](./model.py)：`fit` 实现 session 级频次、窗口共现、余弦式归一化和邻居排序；`recommend` 实现逆序种子加权、候选累加、种子过滤与 Top-N。
- [train.py](./train.py)：构造会话数据并展示近邻和多种子推荐。

在本目录执行：

```powershell
python train.py
```

示例只依赖 Python 标准库，并使用内存字典展示算法过程。大规模系统应使用分布式 pair 聚合、Top-K 截断和版本化 KV 发布。

用户邻域方法见 [UserCF](../UserCF/README.md)，潜向量方法见 [MF](../MF/README.md)，图传播方法见 [GraphCF](../GraphCF/README.md)。