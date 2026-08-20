# L1 召回文献

本目录暂不提交 PDF，只维护可核验的论文入口。检索日期：2026-08-20。

## 协同过滤与 I2I

| 文献 | 研究问题 | 入口 |
| --- | --- | --- |
| Sarwar et al., *Item-based Collaborative Filtering Recommendation Algorithms*, WWW 2001 | Item-based CF | [DOI](https://doi.org/10.1145/371920.372071) |
| Hu et al., *Collaborative Filtering for Implicit Feedback Datasets*, ICDM 2008 | 隐式反馈加权矩阵分解 | [DOI](https://doi.org/10.1109/ICDM.2008.22) |
| Rendle et al., *BPR: Bayesian Personalized Ranking from Implicit Feedback*, UAI 2009 | 隐式反馈成对排序 | [arXiv](https://arxiv.org/abs/1205.2618) |
| Linden et al., *Amazon.com Recommendations: Item-to-Item Collaborative Filtering*, 2003 | 工业级 I2I | [DOI](https://doi.org/10.1109/MIC.2003.1167344) |
| He et al., *LightGCN*, SIGIR 2020 | 简化图协同过滤 | [arXiv](https://arxiv.org/abs/2002.02126) |

## 双塔与负采样

| 文献 | 研究问题 | 入口 |
| --- | --- | --- |
| Covington et al., *Deep Neural Networks for YouTube Recommendations*, RecSys 2016 | 大规模候选生成 | [DOI](https://doi.org/10.1145/2959100.2959190) |
| Yi et al., *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations*, RecSys 2019 | 采样偏差修正 | [DOI](https://doi.org/10.1145/3298689.3346996) |
| Li et al., *Multi-Interest Network with Dynamic Routing*, CIKM 2019 | 多兴趣召回 | [arXiv](https://arxiv.org/abs/1904.08030) |

## ANN 与检索系统

| 文献 | 研究问题 | 入口 |
| --- | --- | --- |
| Jégou et al., *Product Quantization for Nearest Neighbor Search*, 2011 | PQ 压缩 | [DOI](https://doi.org/10.1109/TPAMI.2010.57) |
| Malkov, Yashunin, *Hierarchical Navigable Small World Graphs*, 2020 | HNSW | [arXiv](https://arxiv.org/abs/1603.09320) |
| Johnson et al., *Billion-scale Similarity Search with GPUs*, 2019 | FAISS | [arXiv](https://arxiv.org/abs/1702.08734) |
| Guo et al., *Accelerating Large-Scale Inference with Anisotropic Vector Quantization*, ICML 2020 | ScaNN | [arXiv](https://arxiv.org/abs/1908.10396) |

## 生成式召回

| 文献 | 研究问题 | 入口 |
| --- | --- | --- |
| Rajput et al., *Recommender Systems with Generative Retrieval (TIGER)*, NeurIPS 2023 | 语义 ID 与自回归检索 | [arXiv](https://arxiv.org/abs/2305.05065) |
| Zhai et al., *Actions Speak Louder than Words*, ICML 2024 | HSTU 与生成式推荐 | [arXiv](https://arxiv.org/abs/2402.17152) |
| *OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment*, 2025 | 召回排序统一 | [arXiv 检索](https://arxiv.org/search/?query=OneRec%3A+Unifying+Retrieve+and+Rank&searchtype=title) |

后续逐篇笔记应记录问题、数据与负采样、结构、损失、检索/解码方式、基线、指标、复现条件和局限。