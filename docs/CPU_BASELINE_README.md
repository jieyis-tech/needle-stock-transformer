# Needle MASTER final-project prototype

这是一个已经可以运行的 **CPU 原型**：使用你本地已完成的 HW2 Needle，实现
MASTER 的主要结构，并新增精确分块 attention 及反向重计算。模型计算与求导
均在 Needle / NumPy 中完成，PyTorch 仅在独立正确性测试中出现。

## 已实现

- 市场状态门控、位置编码、股票内时间 attention、股票间 attention、时间加权聚合和预测头。
- 普通 Needle 计算图 attention，用作基础对照。
- 分块 attention `TensorOp`：query/key 双向分块、online softmax；反向重建概率块，计算 dQ/dK/dV。
- 稠密手写 VJP 对照：同样只支持一阶梯度，用来区分“分块收益”和“避免反向计算图开销的收益”。
- 数值梯度、独立 PyTorch 输出/梯度对照、股票排列等变性、短训练轨迹一致性检查。
- 有限步合成训练、独立子进程的内存/耗时实验。

## 目录和来源

| 路径 | 内容 |
|---|---|
| `vendor/needle/` | 本地 HW2 Needle 原样快照，19 个 Python 文件 |
| `vendor/PROVENANCE.json` | 原路径及逐文件 SHA256；快照不作为本项目新增贡献 |
| `master_needle/attention.py` | 本项目新增的分块 attention 与手写一阶反向 |
| `master_needle/model.py` | 本项目的 MASTER 模型结构 |
| `tests/test_master.py` | 独立数值验证，PyTorch 仅作对照 |
| `scripts/verify.py` | 验证快照、运行测试、保存误差记录 |
| `scripts/smoke_train.py` | 40 步合成数据拟合，验证可以训练 |
| `scripts/benchmark.py` | 三种 attention 的 CPU 基准测试 |
| `results/` | 本次实际运行结果，详见 `RESULTS.md` |

没有修改课程作业目录或 `C:\AI_STOCK`。没有使用已训练的股票模型权重。
选择 HW2 是因为其 NumPy CPU 实现已经完成；本地 HW3 仍有占位代码，
且其已有 C++/CUDA 扩展是 Linux `.so`，不能直接在 Windows Python 中加载。

## 一键复现

本次解释器为 `C:\Users\15272\.conda\envs\AI_stock_environment\python.exe`。
在项目目录的 PowerShell 中运行：

```powershell
.\run_checks.ps1
```

也可以单独运行：

```powershell
& 'C:\Users\15272\.conda\envs\AI_stock_environment\python.exe' -X utf8 scripts/verify.py
& 'C:\Users\15272\.conda\envs\AI_stock_environment\python.exe' -X utf8 scripts/smoke_train.py
& 'C:\Users\15272\.conda\envs\AI_stock_environment\python.exe' -X utf8 scripts/benchmark.py
```

`verify.py` 使用 `MKL_THREADING_LAYER=SEQUENTIAL` 避免本地 NumPy 与
PyTorch 的重复 Intel OpenMP 初始化；不使用 `KMP_DUPLICATE_LIB_OK`。
其他机器安装 `requirements.txt`；PyTorch 仅为正确性测试的可选依赖。
`run_checks.ps1` 中的解释器路径是本机路径，需要在其他机器调整。

## 最小调用

```python
import numpy as np
import master_needle  # exposes the local vendored Needle on sys.path
import needle as ndl
from master_needle import MASTER

model = MASTER(input_dim=9, market_dim=16, dim=32, block_size=64)
x = ndl.Tensor(np.random.randn(64, 12, 9).astype('float32'))
market = ndl.Tensor(np.zeros(16, dtype='float32'))
score = model(x, market)  # (64,), one score per stock
score.sum().backward()
```

## 计算与实验边界

分块 attention 的输入是 `(..., Nq, d)`、`(..., Nk, d)`、`(..., Nk, dv)`。
前导 batch 维完全相同；支持 float32/float64；支持不整除块长的尾块。
一次显式分数块至多为 `block_size × block_size`。保存 Q/K/V、输出和每行的
log-normalizer；不会为每个块保留完整 autodiff 子图。

前向使用在线 softmax：每个 query 块维护当前最大值 m、分母 l 和加权和 a；
处理新 key 块时，以新的最大值重新缩放旧的 l/a。这避免显式存储 N×N 概率矩阵。
反向按块重算 P，然后使用

```text
dV = P^T dO
dS = P * (dO V^T - rowsum(dO * O))
dQ = scale * dS K
dK = scale * dS^T Q
```

将各 key/query 块的贡献累加。精确指与稠密 softmax attention 数学等价，
不表示浮点结果逐位相同，也不表示计算复杂度从平方降为线性。

当前限制：

- CPU / NumPy backend，尚未移植到 HW3 NDArray / CUDA，不能称为完整 FlashAttention。
- attention 和 Select 的自定义反向只支持一阶梯度，不支持二阶梯度。
- dropout 固定关闭，没有 attention mask / padding 支持；输入必须有限。
- MASTER 是保留主要结构的小型实现，没有对齐现有生产配置、处理管线与预训练权重。
- 合成训练是同一批样本的可训练性检查，不是股票预测表现或样本外收益证据。
- 基准测的是单独 attention，非完整 MASTER 或整个训练流程。
- 普通 Needle 反向会建立梯度图，稠密手写 VJP 对照用于隔离这一差别。
- 峰值 RSS 包括 Python/NumPy，不能当作 GPU 显存；图数组字节数与进程峰值分别记录。

## 接下来的课程项目工作

1. 接入课程当前 NDArray/GPU backend，验证同样的一阶梯度和数值稳定性。
2. 按真实模型需要增加 mask、训练 dropout 和可复现随机数处理。
3. 用预先固定日期的小型真实面板做 MASTER 对齐；只在明确范围后重建数据缓存。
4. 完成 GPU 显存/耗时、不同规模和不同块长的消融实验。
5. 写 proposal、设计说明及 Colab 报告。现阶段不承诺相对 PyTorch 的速度优势。

## 参考资料

- Tong Li 等，2024，*MASTER: Market-Guided Stock Transformer for Stock Price Forecasting*，
  https://doi.org/10.1609/aaai.v38i1.27767 。已有选股项目中的
  `monthly_daily_hgru/probe_crossmodels.py` 提供本次结构对照；本项目无跨目录运行时依赖。
- Tri Dao 等，2022，*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*，
  https://arxiv.org/abs/2205.14135 。本次实现借鉴分块、在线归一化和反向重算思想，
  未实现论文中的完整 GPU kernel 与 IO 优化。
- 课程要求：https://dlsyscourse.org/project/ 。包含作业解答的代码应按课程要求私有提供。
