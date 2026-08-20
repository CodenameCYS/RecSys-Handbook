# 推荐系统手册

面向工业推荐系统的中文知识整理项目。正文按“推荐系统整体架构”和“冷启动问题”组织，论文索引集中在顶层“文献”目录。推荐系统整体架构进一步拆分为 Index 池、L1 召回、L2 精排和 L3 后处理，评测方法则结合各模块职责分别介绍。

## 阅读入口

- [推荐系统整体架构](./推荐系统整体架构/README.md)：从 Index 池到召回、精排和后处理的端到端推荐链路。
- [冷启动问题](./冷启动问题/README.md)：新用户、新物品、新场景的先验构造、探索反馈与状态迁移。
- [文献](./文献/README.md)：按冷启动、Index 池、召回、精排、后处理和评测主题整理的论文入口。

## 完整目录

```text
RecSys-Handbook/
├── README.md
├── 推荐系统整体架构/
│   ├── README.md
│   ├── 推荐系统整体架构综述.md
│   ├── Index池/
│   │   ├── README.md
│   │   ├── Index池综述.md
│   │   ├── 选品/README.md
│   │   └── Index构建/
│   ├── L1召回/
│   │   ├── README.md
│   │   ├── 综述.md
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
│   │   └── 主要实现架构/
│   │       ├── README.md
│   │       ├── DeepFM/{README.md,model.py,train.py}
│   │       └── DIN/{README.md,model.py,train.py}
│   └── L3后处理/
│       ├── README.md
│       ├── 综述.md
│       └── 主要实现架构/
│           ├── README.md
│           └── MMR/{README.md,rerank.py,example.py}
├── 冷启动问题/
│   ├── README.md
│   ├── 冷启动问题综述.md
│   ├── InterviewAndPractice.md
│   └── 方法详解/
│       ├── NewUser.md
│       ├── NewItem.md
│       ├── TransferAndFewShot.md
│       ├── Exploration.md
│       └── CaseStudies.md
└── 文献/
    ├── README.md
    ├── 冷启动问题/冷启动文献综述.md
    ├── Index池/README.md
    ├── L1召回/README.md
    ├── L2精排/README.md
    ├── L3后处理/README.md
    └── 评测指标/README.md
```

模型脚本是用于理解数据流、模型结构和损失函数的最小示例，不直接等同于生产实现。I2I 和 MMR 仅依赖 Python 标准库；双塔、协同过滤、生成式召回、DeepFM 和 DIN 依赖 PyTorch。冷启动问题按业务问题、方法机制和案例组织，不单列某个模型作为完整解决方案。

## 维护约定

- 顶层 `README.md` 负责目录导航；各问题域的综述文档负责系统性论述；实现目录中的 `README.md` 负责单个模型；顶层文献目录负责论文索引。
- 单个方法只解释其特有的数据、结构、损失和推理方式；ANN、索引更新与多路融合等共性内容集中在“共享基础设施”。
- 方法说明统一采用“核心思路、数据构造、模型结构、训练目标、部署接口、评价、局限与前沿”的顺序。
- 评价指标按 Index、L1、L2、L3 和冷启动的职责就地说明，跨阶段实验原则由整体架构综述统一串联。
- 文献目录只维护论文元数据、公开链接和阅读笔记，不直接提交受版权保护的 PDF。
