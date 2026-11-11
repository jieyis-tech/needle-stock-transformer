# CPU prototype verification — 2026-10-21

这是本机实际运行结果；当前后端为 HW2 Needle + NumPy CPU，非 CUDA。

## Correctness

- 9 个测试全部通过；包含部分尾块、不同 batch/序列长度、float32/float64、有限差分、共享输入梯度累加。
- 完整小型 MASTER 与独立 PyTorch 实现比较输出、输入/市场梯度及所有参数梯度（float64、dropout=0）。
- 普通与分块模型通过 5 次 Adam 更新的参数轨迹一致性检查；通过股票排列等变性检查。
- 原样快照的 19 个文件通过 SHA256 检查。

| 检查 | 最大绝对误差 |
|---|---:|
| attention_max_abs | 2.384e-07 |
| attention_gradient_max_abs | 2.265e-06 |
| master_max_abs | 1.288e-14 |
| master_parameter_gradient_max_abs | 9.139e-13 |

不同检查的 dtype/尺度不同，不应直接用上述误差大小互相比较。

## Synthetic training

24 个合成股票、6 个时间步、9 个输入特征，dim=16。40 次 Adam 更新，loss 1.120881 → 0.007506，耗时 0.309 秒。
同一批样本的拟合检查，不能说明真实数据泛化、RankIC 或收益。

## Attention CPU benchmark

每组在新子进程中运行 3 次，BLAS 单线程，无 warmup；下面是中位数。T=4、heads=2、head_dim=16、FP32、block=64。

| 股票数 | 实现 | 前向图数组 MiB | 进程峰值 RSS MiB | 前向＋反向秒 |
|---:|---|---:|---:|---:|
| 256 | dense | 10.51 | 69.4 | 0.012 |
| 256 | dense_vjp | 2.50 | 45.0 | 0.006 |
| 256 | tiled | 0.51 | 39.3 | 0.007 |
| 512 | dense | 41.02 | 160.5 | 0.040 |
| 512 | dense_vjp | 9.00 | 63.6 | 0.016 |
| 512 | tiled | 1.02 | 40.4 | 0.021 |
| 1024 | dense | 162.03 | 522.5 | 0.160 |
| 1024 | dense_vjp | 34.00 | 137.3 | 0.064 |
| 1024 | tiled | 2.03 | 42.4 | 0.077 |

`dense`：普通 Needle 图；`dense_vjp`：稠密存储概率但手写一阶反向；`tiled`：分块并反向重算。

**结论：本次测试观察到显著内存下降。与较公平的 dense_vjp 对照相比，分块版用更多时间换取更低内存；不声称 GPU 加速。**

RSS 包括解释器、NumPy、临时数组与梯度；图数组统计去除共享 buffer 重复计数，包含输入/输出与算子缓存，不是整个进程峰值。
Windows 用进程 peak working set，并辅以 1 ms RSS 采样。总峰值有框架基线，不能等同于 attention 新增内存。
单线程、冷启动、小规模测试不能外推 GPU 或真实完整模型吞吐。

## Reproduce

运行项目根目录 `run_checks.ps1`；原始数值与环境版本见 `results/*.json`。
