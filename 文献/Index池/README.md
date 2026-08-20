# Index 池文献

本目录整理向量索引、近似最近邻搜索和大规模检索系统的代表文献。算法选择不能只比较理论复杂度，应在真实向量、过滤条件和硬件上联合测量 Recall@K、P95/P99、吞吐、内存、构建成本与更新能力。

| 分类 | 文献 | 研究问题 | 入口 |
| --- | --- | --- | --- |
| 局部敏感哈希 | Indyk, Motwani, *Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality*, STOC 1998 | LSH 与近似近邻理论 | [DOI](https://doi.org/10.1145/276698.276876) |
| 随机投影树 | Dasgupta, Freund, *Random Projection Trees and Low Dimensional Manifolds*, STOC 2008 | 随机投影空间划分 | [DOI](https://doi.org/10.1145/1374376.1374452) |
| 乘积量化 | Jégou et al., *Product Quantization for Nearest Neighbor Search*, 2011 | PQ 压缩与近似距离计算 | [DOI](https://doi.org/10.1109/TPAMI.2010.57) |
| 图索引 | Malkov, Yashunin, *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs*, 2020 | HNSW | [arXiv](https://arxiv.org/abs/1603.09320) |
| GPU 检索 | Johnson et al., *Billion-scale Similarity Search with GPUs*, 2019 | FAISS 与十亿级检索 | [arXiv](https://arxiv.org/abs/1702.08734) |
| 向量量化 | Guo et al., *Accelerating Large-Scale Inference with Anisotropic Vector Quantization*, ICML 2020 | ScaNN 与各向异性量化 | [arXiv](https://arxiv.org/abs/1908.10396) |

后续笔记应记录数据规模、维度、距离函数、索引参数、动态更新能力、过滤方式、硬件配置，以及召回率与延迟的完整折中曲线。