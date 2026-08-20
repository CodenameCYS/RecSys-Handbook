# MMR

MMR 在每一步选择相关性高、同时与已选集合不太相似的物品：

$$
i^*=\arg\max_i\left[\lambda r_i-(1-\lambda)\max_{j\in S}sim(i,j)\right]
$$

$\lambda$ 越大越偏向精排相关性，越小越强调多样性。相似度可来自内容标签、Embedding 或共现关系。

- [rerank.py](./rerank.py)：标准库实现的 MMR。
- [example.py](./example.py)：对带二维向量的候选进行重排。

执行 `python example.py`。生产环境应增加业务硬约束、分数校准、稳定排序、超时回退和列表级指标监控。