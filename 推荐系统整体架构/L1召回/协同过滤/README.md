# 协同过滤

协同过滤利用用户与物品之间的历史交互发现群体偏好规律，在不直接理解文本、图片或类目语义的情况下生成个性化候选。它主要解决“如何从谁与什么发生过交互这一稀疏行为数据中，推断用户可能喜欢但尚未交互的物品”的问题。

本目录同时覆盖邻域方法、潜因子方法和图方法。它们共享用户-物品交互这一数据基础，但得到关系的方式、离线产物和线上服务形态不同。完整概念体系和方法关系见[协同过滤综述](./协同过滤综述.md)。

## 内容导航

- [协同过滤综述](./协同过滤综述.md)：统一解释交互矩阵、U2I/I2I 服务方向、四类方法的联系、评测口径和工程边界。
- [UserCF](./UserCF/README.md)：比较用户行为集合，寻找相似用户并聚合其代表性行为。
- [ItemCF](./ItemCF/README.md)：统计物品共现关系，构建稳定的 `item -> neighbors` 近邻表。
- [矩阵分解](./MF/README.md)：将用户偏好压缩到低维 user/item embedding，并通过 BPR 等目标学习排序关系。
- [GraphCF](./GraphCF/README.md)：把交互矩阵解释为二部图，通过 LightGCN 等方法聚合多跳协同信号。

## 组织逻辑

目录按照协同关系的表达方式递进组织：UserCF 与 ItemCF 显式计算邻居，MF 将关系隐式编码进低秩向量，GraphCF 再通过图传播显式引入多跳结构。这样既能比较不同算法如何使用同一份交互数据，也能区分算法类型与线上 U2I、I2I 检索接口，避免把 ItemCF、矩阵分解和所有 I2I 服务混为一类。

## 目录结构

```text
协同过滤/
├── README.md
├── 协同过滤综述.md
├── UserCF/
│   ├── README.md
│   ├── model.py
│   └── train.py
├── ItemCF/
│   ├── README.md
│   ├── model.py
│   └── train.py
├── MF/
│   ├── README.md
│   ├── model.py
│   └── train.py
└── GraphCF/
    ├── README.md
    ├── model.py
    └── train.py
```