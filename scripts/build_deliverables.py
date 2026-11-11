"""Create reviewable report, original figures, notebook draft and private bundle."""
from pathlib import Path
import hashlib
import json
import re
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def load(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))


def main():
    bench=load('results/gpu_benchmark.json');check=load('results/gpu_correctness.json')
    alignment=load('results/real_alignment.json');numerics=load('results/training_numerics.json')
    training=load('results/real_training_v1/summary.json');panel=load('data/real_panel_v1/manifest.json')
    rows=bench['rows']
    mainrows=[r for r in rows if r['task']=='model' and r['time_steps']==12 and r['block']==64]
    dense=next(r for r in mainrows if r['stocks']==1024 and r['mode']=='dense_vjp')
    tile=next(r for r in mainrows if r['stocks']==1024 and r['mode']=='tiled')
    saving=100*(1-tile['pool_peak_live_mib']/dense['pool_peak_live_mib'])
    slowdown=100*(tile['wall_median_ms']/dense['wall_median_ms']-1)
    lines=['# GPU 工程结果与复现记录','', '日期：2026-10-26。当前项目位置：`'+str(ROOT)+'`。', '',
           '**已完成 CUDA 后端、自写 attention 内核、真实小面板训练及完整模型实验。以下是工程证据，不代表最终课程提交或投资结果。**','',
           '## GPU 完整模型基准','',
           'RTX 5080 Laptop GPU，FP32，width=32，T=12；每配置独立子进程，2 次 warmup、5 次测量，中位数。训练步含前向、loss、反向与 Adam。', '',
           '| 股票数 | attention 实现 | 内存池存活峰值 MiB | 训练步 wall ms |','|---:|---|---:|---:|']
    for r in mainrows:lines.append(f"| {r['stocks']} | {r['mode']} | {r['pool_peak_live_mib']:.1f} | {r['wall_median_ms']:.2f} |")
    lines += ['',f'1024 只股票时，相对稠密手写 VJP 对照，分块版降低约 **{saving:.1f}%** 的内存池峰值，训练步耗时增加约 **{slowdown:.1f}%**。',
              '内存口径是 CuPy allocator 的存活分配峰值，排除 CUDA context 和 allocator 之外的库工作区；不是整卡显存、缓存保留量或 CPU RSS。',
              '所有 GPU 实现使用同一一阶驱动，避免把不必要的反向图保留算作分块收益。修复前、迁移前基准均已另存，不混入当前结论。','',
              '## 股票数、时间长度、块长消融','',
              '| 类型 | 实现 | N | T | 块长 | 峰值 MiB | wall ms |','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r not in mainrows:lines.append(f"| {r['task']} | {r['mode']} | {r['stocks']} | {r['time_steps']} | {r['block']} | {r['pool_peak_live_mib']:.1f} | {r['wall_median_ms']:.2f} |")
    lines += ['', '## 正确性和内存生命周期','',
              f"- CPU 9 项原有检查通过；GPU 9 个 FP32 attention 形状/块长组合通过，最大输出误差 {max(x['forward_max_abs'] for x in check['attention']):.3e}，最大梯度误差 {max(x['gradient_max_abs'] for x in check['attention']):.3e}。",
              f"- 完整模型 FP64 三种实现与独立 PyTorch CPU 对照；真实日期 {alignment['real_sample_date']} 的全参数梯度最大误差 {alignment['master_fp64_max_parameter_gradient_abs']:.3e}。",
              f"- FP32 全模型单步逐参数 relative-L2 最大误差 {check['fp32_model_max_relative_l2_gradient_error']:.3e}；近零元素的相对误差不能当作严格逐位一致。",
              '- 修复 HW2 驱动对常量保存梯度的问题。一阶 VJP 分离、跳过常量；第 5/20 步常量梯度与内存不增长回归通过。', '',
              '## 真实小面板：数据与 PIT 边界','',
              '固定历史 cohort 为 128 只，选择依据为 2014-01-02 已知上市/退市状态和代码哈希，不使用后来的收益或是否存活。',
              '12 个日级时间步、10 个输入特征；不是原生产月度 PCA 模型。源库只读，没有下载或改表。','',
              '| 段 | 信号日期 | 标签最晚实现日 | 信号日数 | 打分股票日 | 缺标签但仍打分 |','|---|---|---|---:|---:|---:|']
    for split in ('train','validation','test'):
        lo,hi,n=panel['signal_ranges'][split];w=panel['waterfall'][split]
        lines.append(f"| {split} | {lo}–{hi} | {panel['label_exit_ranges'][split][1]} | {n} | {w['score_stock_days']} | {w['unlabelled_scored_stock_days']} |")
    lines += ['', '各段末尾剔除 5 个标签跨界信号日。训练标签最晚 2014-12-31，验证最早信号 2015-01-05；验证标签最晚 2015-06-30，测试最早信号 2016-01-04。',
              '特征与市场输入归一化只拟合 2014 年。2015-03-02 市场特征通过历史前缀重算；数组指纹与打分/标签统计审计通过。',
              '标签为 t 到 t+5 的复权收盘对数收益；未来标签缺失只屏蔽 loss/指标。历史源库可能包含供应商修订，不能声称拥有当年的原始数据 vintage。','',
              '## 固定 3 epoch 训练结果','',
              'seed=42、width=32、Adam lr=0.001、clip=1、dropout=0；没有按验证/测试选择 epoch 或调参。','',
              '| 实现 | train loss 1→3 | 验证 mean daily RankIC | 测试 mean daily RankIC |','|---|---|---:|---:|']
    for r in training['results']:
        lines.append(f"| {r['mode']} | {r['train_loss_curve'][0]:.4f}→{r['train_loss_curve'][-1]:.4f} | {r['metrics']['validation']['mean_daily_rankic']:.4f} | {r['metrics']['test']['mean_daily_rankic']:.4f} |")
    n32=numerics['results']['float32'];n64=numerics['results']['float64']
    lines += ['', '**三种 FP32 长训练轨迹没有保持一致，不能据此选出预测更好的 attention 实现。**',
              f"仅用训练日的 16 步诊断：FP32 参数最大差 {n32['after_16_steps_max_parameter_abs']:.3e}，FP64 参数最大差 {n64['after_16_steps_max_parameter_abs']:.3e}；同一 CUDA 实现重放参数差为 0。",
              '结果支持本测试中舍入误差与自适应更新的敏感性解释；不等于已证明任意长期训练的预测等价。测试集结果未用于改选配置。',
              '权重、分数和日期/股票映射保存在 `results/real_training_v1/`。原始完整训练在迁移前完成；数据和源码随项目保留，迁移后关键检查与 GPU 基准已重跑。','',
              '## 当前边界','',
              '- 新增的是 Needle CUDA/CuPy 数组后端，未直接使用 HW3 Linux NDArray 扩展；CuPy/cuBLAS 提供基础运算，attention 前向/反向是本项目 CUDA 代码。',
              '- 不支持 mixed-device、二阶梯度、attention mask 或训练 dropout。固定长度、dropout=0 的当前模型适用。',
              '- GPU FP32 内核支持的块长/维度有限；FP64 走另一条批量分块参考实现。',
              '- 小型历史股票 cohort、单 seed；没有交易成本、ST/涨跌停规则或可执行回测，不作收益判断。',
              '- notebook 为本地草稿，尚未实跑 Google Colab，组员信息、私有代码链接和 90 秒视频仍需最终交付时补齐。','',
              '## 复现','', '`run_gpu_checks.ps1`、`scripts/benchmark_gpu.py`、`scripts/train_real_gpu.py`。',
              '课程要求：[Final Project](https://dlsyscourse.org/project/)。算法背景：[MASTER](https://doi.org/10.1609/aaai.v38i1.27767)、[FlashAttention](https://arxiv.org/abs/2205.14135)。']
    (ROOT/'GPU_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    figs=ROOT/'figures';figs.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    names={'dense':'Dense Needle graph','dense_vjp':'Dense handwritten VJP','tiled':'Tiled CUDA recomputation'}
    for mode in names:
        rr=sorted([r for r in mainrows if r['mode']==mode],key=lambda r:r['stocks'])
        axes[0].plot([r['stocks'] for r in rr],[r['pool_peak_live_mib'] for r in rr],marker='o',label=names[mode])
        axes[1].plot([r['stocks'] for r in rr],[r['wall_median_ms'] for r in rr],marker='o',label=names[mode])
    axes[0].set(xlabel='Stocks',ylabel='Peak live CuPy pool (MiB)',title='Full MASTER training memory')
    axes[1].set(xlabel='Stocks',ylabel='Median wall step (ms)',title='Full MASTER training latency')
    axes[0].legend(fontsize=8);fig.savefig(figs/'gpu_tradeoff.png',dpi=180);plt.close(fig)
    proposal=('Memory-Efficient MASTER in Needle: Exact Cross-Stock Attention with Recomputation\n\n'
        'We will extend Needle with a CUDA array backend and exact cross-stock attention that uses online softmax and recomputes probabilities during backward to avoid quadratic saved attention matrices. We will integrate the operator into a compact MASTER model and compare dense graph attention, a dense handwritten VJP, and tiled CUDA kernels. Evaluation will cover output/gradient correctness, full-model peak allocation and training latency across stock counts, history lengths and tile sizes, plus a fixed historical stock-panel demonstration. Resources: our completed HW2 Needle library; Li et al., MASTER (AAAI 2024); Dao et al., FlashAttention (2022); CuPy/cuBLAS for basic GPU arrays. The project contribution is implementation and systems evaluation, not a new forecasting algorithm.')
    assert len(proposal)<=1000,len(proposal)
    (ROOT/'docs/PROPOSAL.txt').write_text(proposal+'\n',encoding='utf-8')
    deliver=ROOT/'deliverables';deliver.mkdir(exist_ok=True)
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(keepends=True)))
    def code(s):cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(keepends=True)))
    md('# Needle MASTER report and reproducibility notebook\n\nLocal draft; cloud execution not verified. Fill team details and a private downloadable code link before submission. Never publish Needle homework solutions publicly.')
    report=(ROOT/'docs/REPORT_DRAFT.md').read_text(encoding='utf-8')
    for section in re.split(r'(?m)(?=^## )',report):md(section)
    md('## Measured results\n\n'+(ROOT/'GPU_RESULTS.md').read_text(encoding='utf-8'))
    md('## Reproduce with the private project bundle\n\nChoose a GPU runtime for GPU checks. Upload `needle_master_project.zip` privately. The archive includes the small derived panel, scripts and local evidence, but no CUDA binaries or source warehouse. The following cells have not been executed in Colab.')
    code("from google.colab import files\nfrom pathlib import Path\nimport zipfile, os\nuploaded=files.upload()\narchive=Path('needle_master_project.zip')\nassert archive.exists()\nroot=Path('/content/needle_master').resolve()\nroot.mkdir(exist_ok=True)\nwith zipfile.ZipFile(archive) as z:\n    for item in z.infolist():\n        target=(root/item.filename).resolve()\n        assert target.is_relative_to(root)\n    z.extractall(root)\nos.chdir(root)\nprint(root)\n")
    code("import subprocess,sys,importlib.util\nsubprocess.check_call([sys.executable,'-m','pip','install','-q','-r','requirements.txt'])\nif importlib.util.find_spec('cupy') is None:\n    subprocess.check_call([sys.executable,'-m','pip','install','-q','cupy-cuda12x[ctk]==14.2.0'])\nsubprocess.check_call([sys.executable,'scripts/verify.py'])\n")
    code("import subprocess,sys\nfor script in ['scripts/verify_gpu.py','scripts/verify_real_alignment.py','scripts/check_training_numerics.py']:\n    subprocess.check_call([sys.executable,script])\n")
    code("# This reruns measurements on the current GPU; results may differ from the laptop.\nsubprocess.check_call([sys.executable,'scripts/benchmark_gpu.py'])\n# Optional bounded real training; creates a new versioned run and keeps old evidence.\n# subprocess.check_call([sys.executable,'scripts/train_real_gpu.py'])\n")
    code("from IPython.display import display,Image\ndisplay(Image(filename='figures/gpu_tradeoff.png'))\n")
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python'),accelerator='GPU'),nbformat=4,nbformat_minor=5)
    for i,cell in enumerate(cells):cell['id']=f'cell-{i:03d}'
    npth=deliver/'Needle_MASTER_Report.ipynb';npth.write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        import nbformat
        nbformat.validate(nbformat.read(npth,as_version=4))
    except ImportError:pass
    # Explicit allowlist excludes .git, CUDA binaries, secrets, and the warehouse.
    prefixes=['master_needle','vendor','scripts','tests','docs','results','figures','data/real_panel_v1']
    selected=[]
    for prefix in prefixes:
        selected += [p for p in (ROOT/prefix).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    selected += [ROOT/name for name in ['README.md','GPU_RESULTS.md','RESULTS.md','requirements.txt','run_checks.ps1','run_gpu_checks.ps1']]
    selected.append(npth)
    bundle=deliver/'needle_master_project.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(selected)):z.write(p,p.relative_to(ROOT).as_posix())
    summary=dict(proposal_characters=len(proposal),report_draft_words=len(re.findall(r"\b[\w'-]+\b",report)),
                 bundle_bytes=bundle.stat().st_size,bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
                 bundle_files=len(set(selected)),notebook_cells=len(cells),cloud_run_verified=False)
    (deliver/'MANIFEST.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
