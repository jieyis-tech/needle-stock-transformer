"""Render checked-in JSON evidence as a short human-readable Markdown report."""
from pathlib import Path
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
results = ROOT/'results'
c = json.loads((results/'correctness.json').read_text())
s = json.loads((results/'smoke_train.json').read_text())
b = json.loads((results/'benchmark.json').read_text())
lines = ['# CPU prototype verification — 2026-10-21', '',
         '这是本机实际运行结果；当前后端为 HW2 Needle + NumPy CPU，非 CUDA。', '',
         '## Correctness', '',
         f"- {c['tests']} 个测试全部通过；包含部分尾块、不同 batch/序列长度、float32/float64、有限差分、共享输入梯度累加。",
         '- 完整小型 MASTER 与独立 PyTorch 实现比较输出、输入/市场梯度及所有参数梯度（float64、dropout=0）。',
         '- 普通与分块模型通过 5 次 Adam 更新的参数轨迹一致性检查；通过股票排列等变性检查。',
         f"- 原样快照的 {c['vendor_files_verified']} 个文件通过 SHA256 检查。", '',
         '| 检查 | 最大绝对误差 |','|---|---:|']
for k,v in c['metrics'].items():
    lines.append(f'| {k} | {v:.3e} |')
lines += ['', '不同检查的 dtype/尺度不同，不应直接用上述误差大小互相比较。', '',
          '## Synthetic training', '',
          f"24 个合成股票、6 个时间步、9 个输入特征，dim=16。40 次 Adam 更新，loss {s['initial_loss']:.6f} → {s['final_loss']:.6f}，耗时 {s['seconds']:.3f} 秒。",
          '同一批样本的拟合检查，不能说明真实数据泛化、RankIC 或收益。', '',
          '## Attention CPU benchmark', '',
          '每组在新子进程中运行 3 次，BLAS 单线程，无 warmup；下面是中位数。T=4、heads=2、head_dim=16、FP32、block=64。', '',
          '| 股票数 | 实现 | 前向图数组 MiB | 进程峰值 RSS MiB | 前向＋反向秒 |',
          '|---:|---|---:|---:|---:|']
for n in (256,512,1024):
    for mode in ('dense','dense_vjp','tiled'):
        rows = [x for x in b['rows'] if x['stocks']==n and x['mode']==mode]
        med = lambda key: statistics.median(x[key] for x in rows)
        sec = statistics.median(x['forward_seconds']+x['backward_seconds'] for x in rows)
        lines.append(f"| {n} | {mode} | {med('forward_graph_array_mib'):.2f} | {med('peak_rss_total_mib'):.1f} | {sec:.3f} |")
lines += ['',
          '`dense`：普通 Needle 图；`dense_vjp`：稠密存储概率但手写一阶反向；`tiled`：分块并反向重算。', '',
          '**结论：本次测试观察到显著内存下降。与较公平的 dense_vjp 对照相比，分块版用更多时间换取更低内存；不声称 GPU 加速。**', '',
          'RSS 包括解释器、NumPy、临时数组与梯度；图数组统计去除共享 buffer 重复计数，包含输入/输出与算子缓存，不是整个进程峰值。',
          'Windows 用进程 peak working set，并辅以 1 ms RSS 采样。总峰值有框架基线，不能等同于 attention 新增内存。',
          '单线程、冷启动、小规模测试不能外推 GPU 或真实完整模型吞吐。', '',
          '## Reproduce', '',
          '运行项目根目录 `run_checks.ps1`；原始数值与环境版本见 `results/*.json`。']
(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
