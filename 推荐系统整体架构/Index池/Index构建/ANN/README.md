# ANN 方法与实现

近似最近邻检索（Approximate Nearest Neighbor, ANN）通过牺牲少量近邻精度，降低高维向量 Top-K 检索的延迟或内存成本。本目录以精确 Flat 为评测基线，并覆盖五类具有代表性的 ANN 结构。

## 目录结构

```text
ANN/
├── README.md
├── requirements.txt
├── Flat/
│   ├── README.md
│   └── index.py
├── LSH/
│   ├── README.md
│   └── index.py
├── IVF/
│   ├── README.md
│   └── index.py
├── HNSW/
│   ├── README.md
│   └── index.py
├── Annoy/
│   ├── README.md
│   └── index.py
└── IVFPQ/
    ├── README.md
    └── index.py
```

## 方法版图

| 方法 | 经验 Index 池规模 | 核心思想 | 查询调节参数 | 优点 | 主要代价 |
| --- | --- | --- | --- | --- | --- |
| [Flat](./Flat/README.md) | $10^3$–$10^5$；GPU/离线批处理可到 $10^6$ 级 | 与所有向量精确比较 | 无 | 结果精确、基线可靠、无需训练 | 查询复杂度随商品数线性增长 |
| [LSH](./LSH/README.md) | $10^5$–$10^8$ | 局部敏感哈希使相近向量更可能同桶 | 哈希表数、位数、探测桶数 | 易分布式化、增量简单 | 高召回通常需要较多表和内存 |
| [IVF](./IVF/README.md) | $10^6$–$10^9$ | 聚类后只扫描最近的若干倒排桶 | `nprobe` | 速度与召回易调节，适合大规模 | 需要训练，分桶不均会形成热点 |
| [HNSW](./HNSW/README.md) | $10^5$–$10^8$；内存充足时可更大 | 在多层小世界图上进行贪心搜索 | `ef` | 高召回、低延迟、无需量化 | 内存较高，构建和更新成本较大 |
| [Annoy](./Annoy/README.md) | $10^5$–$10^8$ | 多棵随机投影树缩小搜索空间 | `search_k` | 只读索引简单，适合 mmap 与多进程共享 | 构建后不适合频繁增删 |
| [IVF-PQ](./IVFPQ/README.md) | $10^7$–$10^{10}$；通常需要分片 | IVF 缩小范围，PQ 压缩向量并近似算距 | `nprobe`、码长 | 显著降低内存，适合超大规模 | 量化损失，训练与调参更复杂 |

表中的规模是以约 64–256 维向量、常规 CPU 在线服务、单索引或常规分片为背景的经验量级，并非算法硬上限。区间有意保留重叠：可用内存、延迟目标、并发量、更新频率、元数据过滤比例、硬件加速和可接受的 Recall@K 都可能使实际边界前后移动一个甚至多个数量级。

Flat 不是 ANN，但应始终作为真值基线。ScaNN、DiskANN、CAGRA 等系统分别面向量化与硬件优化、SSD 图检索和 GPU 图检索；它们属于同一选型空间，但本目录先用上述方法覆盖哈希、聚类、图、树和量化五种基础思想。

## 距离函数

### 欧氏距离

$$
d_{L2}(\mathbf{q}, \mathbf{x}) = \|\mathbf{q} - \mathbf{x}\|_2
$$

### 余弦相似度

$$
s_{\cos}(\mathbf{q}, \mathbf{x}) = \frac{\mathbf{q}^{\mathsf{T}}\mathbf{x}}{\|\mathbf{q}\|_2\|\mathbf{x}\|_2}
$$

### 内积

$$
s_{IP}(\mathbf{q}, \mathbf{x}) = \mathbf{q}^{\mathsf{T}}\mathbf{x}
$$

若库内向量与查询向量均做 L2 归一化，余弦相似度排序与内积排序一致。距离函数、归一化和模型训练目标必须匹配。

## 统一构建流程

1. 清洗 `(item_id, embedding)`，检查维度、NaN、重复 ID 和异常范数。
2. 固定训练集、建库集、查询集与精确 Flat Top-K 真值。
3. 训练需要学习参数的索引，例如 IVF 聚类中心和 PQ 码本。
4. 添加商品向量并持久化商品 ID 映射、参数和版本 manifest。
5. 扫描查询参数，形成 Recall@K、P95/P99、QPS 和内存曲线。
6. 在真实过滤条件下复测，并执行影子加载、预热和原子发布。

## 评测原则

对查询集合 $Q$，用 Flat 结果 $E_K(q)$ 评价 ANN 结果 $A_K(q)$：

$$
\operatorname{Recall@K} = \frac{1}{|Q|}\sum_{q \in Q}\frac{|A_K(q) \cap E_K(q)|}{K}
$$

一次可信的基准测试应：

- 使用生产向量分布、维度、数据规模和过滤比例。
- 预热索引并分别报告单查询与批量查询。
- 报告尾延迟，而不只报告平均延迟。
- 将索引文件、服务常驻内存和构建峰值内存分开记录。
- 扫描 `ef`、`nprobe`、`search_k` 等参数，而不只比较默认值。
- 对同一 Recall 水平比较性能，避免用低召回换取的延迟优势。

## 运行样例

所有示例使用固定随机种子生成小型数据，并打印查询结果。先安装依赖：

```powershell
pip install -r requirements.txt
```

然后在本目录运行任一示例：

```powershell
python .\Flat\index.py
python .\LSH\index.py
python .\IVF\index.py
python .\HNSW\index.py
python .\Annoy\index.py
python .\IVFPQ\index.py
```

这些样例用于解释 API 和核心数据结构，不代表生产参数。生产环境应使用独立基准集选型，并补充分片、过滤、持久化、增量索引、版本发布和监控。

## 快速选型建议

- 数据量较小或需要生成真值：Flat。
- 需要简单增量写入或分布式哈希路由：LSH。
- CPU 大规模检索且可接受离线训练：IVF。
- 内存充足，追求高召回与低延迟：HNSW。
- 索引只读、需要 mmap 和多进程共享：Annoy。
- 数据量极大、内存是核心瓶颈：IVF-PQ。

最终结论必须来自业务数据基准。相同算法在不同维度、向量分布、过滤模式和硬件上的排序可能完全不同。