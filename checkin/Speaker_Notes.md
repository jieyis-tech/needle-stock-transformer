# Mentor check-in 讲稿

本文件已整合三页英文讲稿、demo 解说和中文提示，只看这一份即可。
按每分钟 100–115 词讲，加上短演示，整体控制在 3 分钟左右；首次 CUDA 编译可能额外等待。
先按实际语速练习，不必逐字背诵。中文提示和操作说明不用读出来。

## 第 1 页：项目思路（约 40 秒）

Our project is a Transformer for stock prediction, implemented in Needle. We use two types of attention. Temporal attention looks at different dates in the history of one stock. Cross-stock attention allows the model to use information from other stocks on the same date. The goal is to implement these components in Needle and study a GPU attention operation that uses less memory.

**中文理解：** 我们在 Needle 中实现一个股票预测 Transformer。时间注意力看同一只股票不同日期；股票间注意力看同一天不同股票。工程目标是完成这些组件，并研究更省内存的 GPU attention。

**讲的时候：** 把两种 attention 的区别说清即可。老师无需先了解金融，也不需要先了解 MASTER。这里是结构概览，不展示模型中的所有辅助层。

## 第 2 页：实现和短测试（约 65–80 秒，含解说）

### 先介绍实现

We now have a working prototype with temporal and cross-stock attention. For the GPU attention operation, we process the computation in blocks and recompute intermediate values during backward. This reduces the attention values that must be saved.

**中文理解：** 当前是可以运行的原型。GPU attention 分块计算，反向时重算部分中间量，从而减少保存的数据。

### 切换到演示

> I will run a quick check of our implementation.

在文件夹里双击 `Run_Demo.cmd`。首次编译可能稍慢；等结果出来后再按下面的正文解释即可，不用追着窗口滚动逐行念。
如果展示会前运行保留的窗口，要说明这是本机之前的测试输出。

### 看到四条 PASS 后，直接读这一段（固定版，约 40–50 秒）

This demo runs on our NVIDIA GPU using Needle. The first two checks compare the attention outputs and gradients with a PyTorch reference. Both maximum errors are below one millionth. The model then takes twelve days of historical data for thirty-two stocks and returns thirty-two scores. Finally, forty updates on synthetic data reduce the loss from about one point one two to zero point zero zero eight. All four checks pass.

**中文对照：**

- 这是本机 NVIDIA GPU 上运行的 Needle 测试。
- 前两条检查 attention 输出和 Q/K/V 梯度，最大绝对误差分别约为 `2.38e-07` 和 `4.17e-07`，都小于百万分之一。
- 第三条将 32 只股票各 12 天的历史数据输入模型，输出 32 个分数。
- 第四条在合成数据上更新 40 次参数，loss 从 `1.1209` 降到 `0.0077`。英文为了好读，四舍五入说约 `1.12 → 0.008`。
- 四项检查全部通过。

读完后切回第三页。这份固定稿对应已验证的 CUDA 后端和历史面板输入。
测试固定了输入、随机种子和参数更新次数，每次仍会重新进行计算；同一环境下通常会重复得到相同的显示值。
合成训练只检查小样例可以学习，不证明真实股票预测表现。

## 第 3 页：初步结果与剩余工作（约 45 秒）

Our initial tests show a memory and time trade-off. In one full-model test with 1,024 stocks and twelve historical steps, measured peak GPU array allocations fell from about 397 to 159 MiB. The training step increased from about 16 to 22 milliseconds. Our remaining work is to review the numerical differences seen in longer FP32 training, summarize the results clearly, and prepare the final notebook and video. We are keeping the current project scope.

**中文理解：** 初步实验表明这是内存与时间的权衡。给定这一个测试设置，GPU 数组的峰值存活分配从约 397 MiB 降到 159 MiB，但一步训练从约 16 ms 增至 22 ms。剩余工作是复核较长 FP32 训练的数值差异、整理评估和完成报告视频，保持现有项目范围。

**讲的时候：** 不必主动讲复杂指标。397/159 是完整模型测试中的 CuPy 内存池峰值，不是整卡显存；对照也有手写反向。这是 2026-10-26 保存的系统实验，和现场短演示分开。不要声称速度提升，也不要声称长期训练结果完全等价。

## 导师可能追问时的简短回答

### What did you add to Needle?

We added the model components and a custom attention TensorOp with CUDA forward and backward kernels. It uses blocks and recomputation to reduce saved attention values.

中文：模型组件，以及带 CUDA 前后向的自定义 attention 算子。

### Are you using PyTorch to train the model?

No. The model and gradient propagation run in Needle. We use PyTorch only as an independent reference for correctness checks. CuPy provides basic GPU arrays and matrix operations.

中文：训练和求导接入 Needle，PyTorch 只是测试对照。基础 GPU 数组与矩阵运算由 CuPy/cuBLAS 提供。

### Does the lower loss prove the model predicts stocks well?

No. This training example uses synthetic data. It only checks that the model can learn on a small example.

中文：合成数据 loss 下降是可训练性检查，不是预测效果或投资收益证明。

### Is the architecture new?

The architecture is inspired by existing work, especially MASTER. Our focus is the Needle implementation and its memory and time trade-off.

中文：借鉴已有架构，正式报告引用 MASTER 与 FlashAttention；不声称发明这些思想。

### What is the main remaining risk?

Longer FP32 training can follow different trajectories between implementations. We have checked small cases and high-precision references, but we do not claim identical long-run training results.

中文：较长 FP32 训练会分歧，已有小测试与高精度对照，尚不声称长期等价。

### What is your fallback plan?

We will keep the verified input sizes and the current model, and present the measured memory and time trade-off. We do not require a speedup for the project to be useful.

中文：保持已经验证的输入规模与模型，把已测量的权衡讲清楚。

### How is the work divided?

I will focus on implementing the model in Needle and running the stock data experiments. Leif will focus on the GPU attention implementation and performance evaluation. We will review the tests together and jointly prepare the final report and video.

中文：我负责 Needle 模型实现与股票数据实验，Leif 负责 GPU attention 实现和性能评估。我们共同检查测试，并合作完成最终报告和视频。

## 附录：只有等待或运行状态改变时才用

正常展示时不用读这一节。

**等待 CUDA 编译：**

> The first GPU run may take a little longer because the CUDA kernels need to be compiled.

**窗口显示改用 CPU：**

> This run is using the CPU version. The checks are the same, but this is not a GPU performance test.

**输入文件缺失、窗口显示 synthetic inputs：**

> We run the model on a small example and get one score for each stock.

**当前检查失败：**

> This check did not pass in the current run. We will inspect the saved log.

固定稿应与实际窗口状态一致；这些备用句子不属于正常 2–3 分钟主讲内容。
