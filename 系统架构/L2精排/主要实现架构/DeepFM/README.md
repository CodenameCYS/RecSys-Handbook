# DeepFM

## 主要思路

DeepFM 同时学习一阶线性项、FM 二阶交互和深层非线性交互。示例输入用户 ID、物品 ID 和场景 ID 三个离散字段，共享 Embedding 后计算二阶 FM 项与 MLP 输出。

## 训练目标

点击预估采用二元交叉熵：

$$
\mathcal{L}=-y\log\sigma(z)-(1-y)\log(1-\sigma(z))
$$

## Files

- [model.py](./model.py)：DeepFM 的一阶、二阶和深层部分。
- [train.py](./train.py)：合成 CTR 数据训练与预测。

安装 PyTorch 后执行 `python train.py`。生产实现需增加连续特征、多值特征、缺失处理、特征交叉配置、校准和分布式 Embedding。