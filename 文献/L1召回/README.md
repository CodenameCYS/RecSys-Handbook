# L1 召回文献

本目录暂不提交 PDF，只维护可核验的论文入口。检索日期：2026-08-20。

## 协同过滤与 I2I

| 文献 | 研究问题 | 入口 |
| --- | --- | --- |
| Resnick et al., *GroupLens: An Open Architecture for Collaborative Filtering of Netnews*, CSCW 1994 | 基于用户邻域的协同过滤系统 | [DOI](https://doi.org/10.1145/192844.192905) |
| Sarwar et al., *Item-based Collaborative Filtering Recommendation Algorithms*, WWW 2001 | Item-based CF | [DOI](https://doi.org/10.1145/371920.372071) |
| Hu et al., *Collaborative Filtering for Implicit Feedback Datasets*, ICDM 2008 | 隐式反馈加权矩阵分解 | [DOI](https://doi.org/10.1109/ICDM.2008.22) |
| Koren et al., *Matrix Factorization Techniques for Recommender Systems*, Computer 2009 | 矩阵分解、偏置与工程实践 | [DOI](https://doi.org/10.1109/MC.2009.263) |
| Rendle et al., *BPR: Bayesian Personalized Ranking from Implicit Feedback*, UAI 2009 | 隐式反馈成对排序 | [arXiv](https://arxiv.org/abs/1205.2618) |
| Linden et al., *Amazon.com Recommendations: Item-to-Item Collaborative Filtering*, 2003 | 工业级 I2I | [DOI](https://doi.org/10.1109/MIC.2003.1167344) |
| He et al., *LightGCN*, SIGIR 2020 | 简化图协同过滤 | [arXiv](https://arxiv.org/abs/2002.02126) |

## I2I 表示与对比学习

| 文献 | 研究问题 | 入口 |
| --- | --- | --- |
| Barkan and Koenigstein, *Item2Vec: Neural Item Embedding for Collaborative Filtering*, 2016 | Skip-gram 物品表示 | [arXiv](https://arxiv.org/abs/1603.04259) |
| Oord et al., *Representation Learning with Contrastive Predictive Coding*, 2018 | InfoNCE 对比目标 | [arXiv](https://arxiv.org/abs/1807.03748) |
| Hadsell et al., *Dimensionality Reduction by Learning an Invariant Mapping*, CVPR 2006 | Pairwise contrastive margin loss | [DOI](https://doi.org/10.1109/CVPR.2006.100) |
| Khosla et al., *Supervised Contrastive Learning*, NeurIPS 2020 | 多正例监督对比学习 | [arXiv](https://arxiv.org/abs/2004.11362) |
| Gao et al., *SimCSE: Simple Contrastive Learning of Sentence Embeddings*, EMNLP 2021 | Dropout 双视图；迁移模式而非 I2I 原始算法 | [arXiv](https://arxiv.org/abs/2104.08821) |

## 双塔与负采样

| 文献 | 研究问题 | 入口 |
| --- | --- | --- |
| Huang et al., *Learning Deep Structured Semantic Models for Web Search using Clickthrough Data*, CIKM 2013 | DSSM 双编码器与语义匹配 | [Microsoft Research](https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/) |
| Mikolov et al., *Distributed Representations of Words and Phrases and their Compositionality*, NeurIPS 2013 | Negative Sampling 与 $3/4$ 次幂 unigram 分布 | [arXiv](https://arxiv.org/abs/1310.4546) |
| Covington et al., *Deep Neural Networks for YouTube Recommendations*, RecSys 2016 | 大规模候选生成 | [DOI](https://doi.org/10.1145/2959100.2959190) |
| Schnabel et al., *Recommendations as Treatments: Debiasing Learning and Evaluation*, ICML 2016 | 曝光选择偏差与倾向评分 | [arXiv](https://arxiv.org/abs/1602.05352) |
| Yi et al., *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations*, RecSys 2019 | 采样偏差修正 | [DOI](https://doi.org/10.1145/3298689.3346996) |
| Li et al., *Multi-Interest Network with Dynamic Routing*, CIKM 2019 | 多兴趣召回 | [arXiv](https://arxiv.org/abs/1904.08030) |
| Yang et al., *Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations*, 2020 | 均匀与批内等混合负采样 | [arXiv](https://arxiv.org/abs/2003.12420) |
| Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering*, EMNLP 2020 | 双编码器、批内与难负例 | [arXiv](https://arxiv.org/abs/2004.04906) |
| Xiong et al., *Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval*, ICLR 2021 | ANCE 异步 ANN 难负例挖掘 | [arXiv](https://arxiv.org/abs/2007.00808) |
| He et al., *Momentum Contrast for Unsupervised Visual Representation Learning*, CVPR 2020 | Momentum encoder 与跨批负例队列 | [arXiv](https://arxiv.org/abs/1911.05722) |

## 序列用户塔与度量学习

| 文献 | 研究问题 | 入口 |
| --- | --- | --- |
| Hidasi et al., *Session-based Recommendations with Recurrent Neural Networks*, ICLR 2016 | GRU 会话序列推荐 | [arXiv](https://arxiv.org/abs/1511.06939) |
| Kang and McAuley, *Self-Attentive Sequential Recommendation*, ICDM 2018 | SASRec 自回归序列推荐 | [arXiv](https://arxiv.org/abs/1808.09781) |
| Sun et al., *BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer*, CIKM 2019 | 双向 Transformer 序列推荐 | [arXiv](https://arxiv.org/abs/1904.06690) |
| Schroff et al., *FaceNet: A Unified Embedding for Face Recognition and Clustering*, CVPR 2015 | Triplet loss 与 semi-hard negatives | [arXiv](https://arxiv.org/abs/1503.03832) |
| Wang et al., *Cross-Batch Memory for Embedding Learning*, CVPR 2020 | 跨批 embedding memory | [arXiv](https://arxiv.org/abs/1912.06798) |

## 生成式召回

| 文献 | 研究问题 | 入口 |
| --- | --- | --- |
| Rajput et al., *Recommender Systems with Generative Retrieval (TIGER)*, NeurIPS 2023 | 语义 ID 与自回归检索 | [arXiv](https://arxiv.org/abs/2305.05065) |
| Lee et al., *Autoregressive Image Generation using Residual Quantization*, CVPR 2022 | RQ-VAE 与残差离散表示基础 | [arXiv](https://arxiv.org/abs/2203.01941) |
| Zhai et al., *Actions Speak Louder than Words*, ICML 2024 | HSTU 与生成式推荐 | [arXiv](https://arxiv.org/abs/2402.17152) |
| *OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment*, 2025 | 召回排序统一 | [arXiv 检索](https://arxiv.org/search/?query=OneRec%3A+Unifying+Retrieve+and+Rank&searchtype=title) |

## 内容、图与知识召回

| 文献 | 研究问题 | 入口 |
| --- | --- | --- |
| Robertson and Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009 | 稀疏内容检索与 BM25 | [DOI](https://doi.org/10.1561/1500000019) |
| Perozzi et al., *DeepWalk: Online Learning of Social Representations*, KDD 2014 | 随机游走图表示 | [arXiv](https://arxiv.org/abs/1403.6652) |
| Grover and Leskovec, *node2vec: Scalable Feature Learning for Networks*, KDD 2016 | 可控偏置随机游走 | [arXiv](https://arxiv.org/abs/1607.00653) |
| Bordes et al., *Translating Embeddings for Modeling Multi-relational Data*, NeurIPS 2013 | TransE 知识图表示 | [NeurIPS](https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html) |
| Page et al., *The PageRank Citation Ranking: Bringing Order to the Web*, 1999 | 随机游走与 Personalized PageRank 基础 | [Stanford](https://ilpubs.stanford.edu:8090/422/) |
| Ying et al., *Graph Convolutional Neural Networks for Web-Scale Recommender Systems*, KDD 2018 | PinSage 与工业级图召回 | [arXiv](https://arxiv.org/abs/1806.01973) |
| Wang et al., *KGAT: Knowledge Graph Attention Network for Recommendation*, KDD 2019 | 协同信号与知识图谱传播 | [arXiv](https://arxiv.org/abs/1905.07854) |
| Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021 | CLIP 跨模态对比表示 | [arXiv](https://arxiv.org/abs/2103.00020) |
| Hinton et al., *Distilling the Knowledge in a Neural Network*, 2015 | 知识蒸馏 | [arXiv](https://arxiv.org/abs/1503.02531) |

## 探索与反馈采集

| 文献 | 研究问题 | 入口 |
| --- | --- | --- |
| Li et al., *A Contextual-Bandit Approach to Personalized News Article Recommendation*, WWW 2010 | LinUCB 与新闻探索 | [arXiv](https://arxiv.org/abs/1003.0146) |
| Chapelle and Li, *An Empirical Evaluation of Thompson Sampling*, NeurIPS 2011 | Thompson Sampling 的实证比较 | [NeurIPS](https://proceedings.neurips.cc/paper/2011/hash/e53a0a2978c28872a4505bdb51db06cd-Abstract.html) |
| Dudik et al., *Doubly Robust Policy Evaluation and Learning*, ICML 2011 | 探索策略的反事实离线评价 | [PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/doublyRobust.pdf) |

向量索引和 ANN 相关论文统一见 [Index 池文献](../Index池/README.md)。后续逐篇笔记应记录问题、数据与负采样、结构、损失、检索/解码方式、基线、指标、复现条件和局限。