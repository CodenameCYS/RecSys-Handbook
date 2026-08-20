# L2 精排主要实现架构

| 方法族 | 代表模型 | 当前状态 |
| --- | --- | --- |
| [DeepFM](./DeepFM/README.md) | FM、DeepFM | 已提供模型与训练脚本 |
| [DIN](./DIN/README.md) | DIN、DIEN | 已提供候选感知注意力与变长序列样板 |
| 基础 CTR | LR、Wide & Deep | 待补线性与深层联合基线 |
| 显式特征交叉 | DCN、DCNv2、xDeepFM | 待补 Cross/CIN 样板 |
| 注意力特征交互 | AutoInt、FiBiNET | 待补 Self-Attention/SENet 样板 |
| 兴趣演化 | DIEN | 待补 GRU、辅助损失与 AUGRU 样板 |
| 序列模型 | BST、SASRec、BERT4Rec | 待补 Transformer 序列样板 |
| 多任务模型 | ESMM、MMoE、PLE | 待补 CTR/CVR 联合训练样板 |
| 多场景模型 | STAR、M2M | 待补共享与场景特定参数样板 |

方法目录使用业界通用英文模型名，并统一包含 `README.md`、`model.py` 和 `train.py`。后续会补充相同数据口径下的对比实验。