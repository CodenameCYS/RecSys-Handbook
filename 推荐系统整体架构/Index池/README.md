# Index 池

Index 池负责从全量供给中形成可推荐商品池，并将商品组织为可供 L1 召回低延迟查询的键值索引和向量索引。这里的“商品”是广义概念，可以表示商品、视频、文章、音乐、广告、房源、商户或其他推荐对象。

## 目录结构

```text
Index池/
├── README.md
├── Index池综述.md
├── 选品/
│   └── README.md
└── Index构建/
	├── README.md
	├── hash_index.py
	└── ANN/
		├── README.md
		└── 各方法实现目录/
```

## 内容导航

| 内容 | 说明 |
| --- | --- |
| [Index 池综述](./Index池综述.md) | Index 池的职责边界、完整链路、数据模型、发布维护和评价指标。 |
| [选品](./选品/README.md) | 从全量商品中执行有效性、安全、质量和场景约束，形成可推荐商品子集。 |
| [Index 构建](./Index构建/README.md) | 构建 ID-to-Item 键值索引与 Embedding-to-Item 向量索引。 |
| [ANN](./Index构建/ANN/README.md) | ANN 方法版图、选型依据、评测方法和 Python 示例。 |