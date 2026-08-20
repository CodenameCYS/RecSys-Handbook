# 评测指标主要实现架构

| 模块 | 内容 | 状态 |
| --- | --- | --- |
| [Top-K Metrics](./TopKMetrics/README.md) | Precision、Recall、HitRate、MRR、NDCG | 已提供标准库实现 |
| 概率评测 | AUC、PR-AUC、LogLoss、Brier、ECE | 待补分桶与分群校准样板 |
| 在线实验 | 分流、功效、CUPED、SRM、显著性 | 待补实验分析样板 |
| 系统评测 | 时延、吞吐、资源、成本 | 待补压测报告模板 |
| 生态评测 | 覆盖、多样性、新颖性、公平 | 待补列表级实现 |

指标实现必须明确输入语义、平均方式、空真值处理和 K 的定义，并使用手算样例进行单元测试。