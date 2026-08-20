# 推荐系统手册

面向工业推荐系统的中文知识整理项目。当前按“评测指标、系统架构、冷启动”三个顶层问题域组织，其中系统架构进一步拆分为 L1 召回、L2 精排和 L3 后处理。

## 阅读入口

- [评测指标](./评测指标/README.md)：离线指标、在线实验与系统指标。
- [系统架构](./系统架构/README.md)：推荐漏斗的端到端结构与阶段接口。
- [冷启动](./冷启动/README.md)：用户、物品和场景冷启动。

## 完整目录

```text
RecSys-Handbook/
├── README.md
├── 评测指标/
│   ├── README.md
│   ├── 综述.md
│   ├── 文献/README.md
│   └── 主要实现架构/
│       ├── README.md
│       └── TopKMetrics/{README.md,metrics.py,example.py}
├── 系统架构/
│   ├── README.md
│   ├── 综述.md
│   ├── L1召回/
│   │   ├── README.md
│   │   ├── 综述.md
│   │   ├── 文献/README.md
│   │   ├── 共享基础设施/
│   │   │   ├── README.md
│   │   │   ├── 数据与采样.md
│   │   │   ├── ANN与索引.md
│   │   │   └── 在线服务与融合.md
│   │   └── 主要实现架构/
│   │       ├── README.md
│   │       ├── TwoTower/{README.md,model.py,train.py}
│   │       ├── BPR/{README.md,model.py,train.py}
│   │       ├── I2I/{README.md,model.py,train.py}
│   │       └── GenerativeRetrieval/{README.md,model.py,train.py}
│   ├── L2精排/
│   │   ├── README.md
│   │   ├── 综述.md
│   │   ├── 文献/README.md
│   │   └── 主要实现架构/
│   │       ├── README.md
│   │       ├── DeepFM/{README.md,model.py,train.py}
│   │       └── DIN/{README.md,model.py,train.py}
│   └── L3后处理/
│       ├── README.md
│       ├── 综述.md
│       ├── 文献/README.md
│       └── 主要实现架构/
│           ├── README.md
│           └── MMR/{README.md,rerank.py,example.py}
└── 冷启动/
    ├── README.md
    ├── 综述.md
    ├── 文献/README.md
    └── 主要实现架构/
        ├── README.md
        └── ContentTwoTower/{README.md,model.py,train.py}
```

模型脚本是用于理解数据流、模型结构和损失函数的最小示例，不直接等同于生产实现。I2I、MMR 和 Top-K 指标仅依赖 Python 标准库；双塔、协同过滤、生成式召回、DeepFM 和冷启动内容塔依赖 PyTorch。

## 维护约定

- 顶层 `README.md` 负责目录导航，`综述.md` 负责系统性论述，方法目录中的 `README.md` 负责单个模型。
- 单个方法只解释其特有的数据、结构、损失和推理方式；ANN、索引更新与多路融合等共性内容集中在“共享基础设施”。
- 方法说明统一采用“核心思路、数据构造、模型结构、训练目标、部署接口、评价、局限与前沿”的顺序。
- 文献目录只维护论文元数据和链接，不直接提交受版权保护的 PDF。
