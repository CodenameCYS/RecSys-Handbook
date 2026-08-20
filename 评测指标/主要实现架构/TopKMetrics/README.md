# Top-K Metrics

输入为每个用户的有序推荐列表和真实相关物品集合。当前实现包含 Precision@K、Recall@K、HitRate@K、MRR@K 和二元相关性的 NDCG@K。

- [metrics.py](./metrics.py)：单用户指标与宏平均。
- [example.py](./example.py)：两个用户的最小示例。

执行 `python example.py`，仅依赖 Python 标准库。

评测时必须在模型生成候选之前固定时间切分和候选协议。若真实集合为空，当前实现跳过该用户；生产报告需要同时给出被跳过用户数量。