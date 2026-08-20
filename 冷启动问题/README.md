# 冷启动问题

本目录介绍推荐系统中的冷启动问题，即系统在新用户、新物品或新业务场景缺少历史交互数据时，如何生成初始推荐、获取有效反馈，并逐步过渡到常规个性化推荐。

## 文件架构

```text
冷启动问题/
├── README.md
├── 冷启动问题综述.md
├── InterviewAndPractice.md
├── 方法详解/
│   ├── NewUser.md
│   ├── NewItem.md
│   ├── TransferAndFewShot.md
│   ├── Exploration.md
│   └── CaseStudies.md
└── 文献/
    └── 冷启动文献综述.md
```

## 文件说明

| 文件 | 主要内容 |
| --- | --- |
| [冷启动问题综述.md](./冷启动问题综述.md) | 系统介绍冷启动的类型、困难、解决方法、工业闭环、应用场景、评测方式和方法对比。 |
| [InterviewAndPractice.md](./InterviewAndPractice.md) | 整理冷启动相关的面试问题、系统设计思路、线上问题诊断、优化顺序和上线检查项。 |
| [方法详解/NewUser.md](./方法详解/NewUser.md) | 介绍新用户冷启动中的上下文先验、主动偏好采集、会话兴趣建模和策略迁移。 |
| [方法详解/NewItem.md](./方法详解/NewItem.md) | 介绍新物品冷启动中的内容表示、多模态特征、协同融合、探索流量和生命周期管理。 |
| [方法详解/TransferAndFewShot.md](./方法详解/TransferAndFewShot.md) | 介绍跨域迁移、域适配、元学习、少样本学习以及负迁移诊断。 |
| [方法详解/Exploration.md](./方法详解/Exploration.md) | 介绍 Bandit、奖励设计、安全探索、反馈日志和反事实评价。 |
| [方法详解/CaseStudies.md](./方法详解/CaseStudies.md) | 分析短视频、电商、新闻、广告、本地生活和社区等场景中的冷启动方案。 |
| [文献/冷启动文献综述.md](./文献/冷启动文献综述.md) | 按研究主题整理代表论文、实验协议、复现注意事项和推荐阅读路线。 |