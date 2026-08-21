# 知识与关系相似度

知识与关系 I2I 使用明确的实体、属性和业务关系生成近邻。它不一定计算“相似度”，而是回答“与种子存在何种关系”，特别适合配件、兼容、续集、同作者和同系列等场景。

## 1. 关系类型

常见边包括：

```text
item --belongs_to--> category
item --made_by--> brand
item --created_by--> author
item --part_of--> series
item --compatible_with--> device
item --accessory_of--> item
item --next_episode--> item
item --substitute_for--> item
```

其中 `compatible_with` 可能对称，`accessory_of` 和 `next_episode` 通常有方向。统一使用对称余弦会丢失这些语义。

## 2. 规则路径召回

最直接的方法是定义元路径。例如：

$$
item\rightarrow brand\rightarrow item
$$

召回同品牌商品；

$$
item\rightarrow series\rightarrow item
$$

召回同系列内容。路径分数可按关系置信度和中间节点流行度加权：

$$
s(i,j)=\sum_{p:i\leadsto j}
w(p)\prod_{e\in p}c_e
$$

对连接大量物品的宽泛实体应降权，类似 IDF：

$$
w(entity)=\log\frac{|I|}{1+degree(entity)}
$$

规则路径高度可解释，适合强业务约束，但需要知识治理和冲突处理。

## 3. 图距离与随机游走

Personalized PageRank 从种子物品出发：

$$
\pi_i=(1-\alpha)e_i+\alpha P^\top\pi_i
$$

$\pi_i(j)$ 表示从种子 $i$ 出发访问 $j$ 的稳态概率。重启概率控制局部性；边类型和方向可进入转移矩阵 $P$。

PPR 能结合多条路径，但热门枢纽可能获得过高分数。可使用边类型权重、度归一化和最大路径长度限制。

## 4. 知识图 embedding

TransE 类模型将关系建模为平移：

$$
e_h+e_r\approx e_t
$$

训练目标让正三元组 $(h,r,t)$ 的距离低于负三元组。查询某种关系时应使用关系条件向量 $e_i+e_r$ 检索目标，而不是仅计算 $e_i$ 与 $e_j$ 的余弦。

双线性模型可写为：

$$
s(h,r,t)=e_h^\top W_r e_t
$$

不同 $W_r$ 表达不同方向和关系。知识图 embedding 适合补全缺失关系，但预测边不能无条件视为事实，需要置信度阈值和业务审核。

## 5. 与协同和内容信号结合

知识关系可以作为：

- 独立近邻表，与 ItemCF 和内容通道并行。
- 图模型中的边类型或节点特征。
- 对比学习和 Triplet 的正例来源。
- 候选过滤或加分规则。
- 冷启动物品与成熟物品之间的桥梁。

强事实关系通常适合规则直出；弱语义关系适合 embedding 泛化。不要为了统一技术栈而把可精确查询的兼容表改造成近似 ANN。

## 6. 数据治理

知识来源可能包括商品主数据、内容元数据、人工运营、实体链接和模型抽取。必须记录关系来源、置信度、生效时间、失效时间和审核状态。

典型风险包括实体消歧错误、关系过期、方向写反、跨区域不兼容和抽取模型幻觉。线上强关系召回应只使用达到质量阈值的数据。

## 7. 服务与评测

显式关系可存储为：

```text
(source_item, relation_type) -> [(target_item, score, evidence)]
```

这样可以保留方向和解释。embedding 关系则为每种关系建立 query 变换与 target ANN，或离线物化关系近邻。

评测按关系类型分别计算 Precision@K、Recall@K、覆盖率和事实一致性；兼容与安全关系通常更重视 Precision。用户级评测还要验证该关系通道对点击、转化和后续行为的增量价值。

异构图、PPR 和知识召回的系统级介绍见[图与知识召回](../../../其他召回方式/图与知识召回/README.md)。