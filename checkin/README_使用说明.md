# Check-in 材料使用说明

## 文件

- `slides/Needle_Transformer_Checkin.pptx`：恰好 3 页的可编辑英文 PowerPoint，包含演讲者备注。
- `Speaker_Notes.md`：简单英文讲稿、逐页中文理解、演示操作和常见追问。
- `Demo_Narration.md`：短测试配套英文解说、逐条中文解释和等待时可用的话。
- `Run_Demo.cmd`：在 Windows 文件夹中双击运行；最后按任意键关闭窗口。
- `demo_checkin.py`：短测试的代码。
- `demo_runs/`：每次运行单独保存的日志和 JSON，不覆盖旧实验结果。

## 会前

1. 先双击一次 `Run_Demo.cmd`，确认结尾为 `ALL CHECKS PASSED`。
2. 打开 PowerPoint 的演讲者视图，按讲稿练习一遍，确认 2–3 分钟能讲完。
3. 会议中第二页可切到演示窗口。首次 CUDA 编译可能稍慢，先在会前预热。
4. 团队姓名、分工按真实情况口头补充；材料没有虚构成员或角色。

## 测试范围

脚本先尝试 GPU；CUDA 环境不可用时，会清楚标注后改用 CPU。
如果是输出/梯度等正确性失败，不会用 CPU 重试来隐藏失败。

脚本运行一个很小的 attention 输出和 Q/K/V 梯度对照，再做一次小模型前向；
本地历史面板存在时，前向输入来自 2014 年的 32 只股票。
最后进行 40 次小型合成数据参数更新，检查 loss 可以下降。
这不是完整训练、实时市场预测、回测或股票表现证明，也不替代完整测试套件。

依赖已有的 Python 环境：
`C:\Users\15272\.conda\envs\AI_stock_environment\python.exe`。
无需安装新库、下载行情或提供账号。移动到另一台电脑后，应调整 launcher 的解释器路径并准备依赖。

## 命令行

```powershell
# 自动尝试 GPU，不可用时明确降级 CPU
& 'C:\Users\15272\.conda\envs\AI_stock_environment\python.exe' -X utf8 checkin/demo_checkin.py
# 单独验证 CPU 路径
& 'C:\Users\15272\.conda\envs\AI_stock_environment\python.exe' -X utf8 checkin/demo_checkin.py --backend cpu
```

完整项目代码、数据与 CUDA 依赖仍位于父目录 `needle_master`；不要只复制本文件夹到其他电脑后期待它独立运行。
