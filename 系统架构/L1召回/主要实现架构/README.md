# L1 召回主要实现架构

| 架构 | 说明 | 典型产物 |
| --- | --- | --- |
| [Two-Tower](./TwoTower/README.md) | 用户塔和物品塔学习可检索向量 | 用户向量、物品 ANN 索引 |
| [BPR](./BPR/README.md) | 从隐式反馈中学习隐因子 | 用户/物品向量 |
| [I2I](./I2I/README.md) | 根据会话共现建立物品近邻 | 物品近邻表 |
| [Generative Retrieval](./GenerativeRetrieval/README.md) | 生成语义 ID 或物品标识 | 候选物品序列 |

## 每种架构的文件

- `README.md`：思路、数据、结构、损失、部署和局限。
- `model.py`：核心模型或算法实现。
- `train.py`：构造小型合成数据并训练/拟合的可运行示例。

示例统一使用 PyTorch；I2I 的统计式基线仅使用 Python 标准库。生产系统还需补充数据管道、分布式训练、检查点、配置、监控和完整测试。