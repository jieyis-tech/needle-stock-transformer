# A Transformer for Stock Prediction in Needle

Private Deep Learning Systems course project: temporal attention and cross-stock attention in Needle, with a custom CUDA attention implementation that trades additional computation for lower saved memory.

This repository contains course-derived Needle code. Keep it private and share it only with authorized project collaborators. It is not a public release of homework solutions.

## Start here

- [Project overview](checkin/leif_handoff_notes/01_Project_Overview.md)
- [Mentor check-in questions and answers](checkin/leif_handoff_notes/02_Leif_Checkin_QA.md)
- [Leif's speaking card](checkin/leif_handoff_notes/03_Leif_Speaking_Card.md)
- [Three check-in slides](checkin/slides/Needle_Transformer_Checkin.pptx)
- [Current speaker notes](checkin/Speaker_Notes.md) (English with Chinese prompts)
- [Implementation notes](docs/IMPLEMENTATION.md)
- [Measured GPU results and limitations](GPU_RESULTS.md)
- [Final-report notebook draft](deliverables/Needle_MASTER_Report.ipynb)

## Model and implementation

Temporal attention connects dates within a stock's history. Cross-stock attention connects stocks observed on the same date; it is self-attention along the stock axis. The model produces one score per stock.

The model is inspired by MASTER. Needle owns the modules, TensorOp integration and gradient propagation. An optional CUDA array adapter uses CuPy/cuBLAS for basic GPU operations. Project-owned CUDA kernels implement attention forward, dQ, and dK/dV, with stable online normalization and backward recomputation. PyTorch is used only as an independent correctness reference.

| Path | Purpose |
|---|---|
| `master_needle/model.py` | Model blocks and prediction head |
| `master_needle/attention.py` | Dense, handwritten-VJP and tiled TensorOps |
| `master_needle/kernels/attention.cu` | Custom FP32 CUDA forward/backward kernels |
| `master_needle/cuda_attention.py` | NVRTC compilation and kernel launches |
| `master_needle/backend.py` | CUDA array adapter and first-order backward driver |
| `master_needle/array_attention.py` | Batched CuPy tiled reference |
| `vendor/needle/` | Hashed completed-HW2 dependency; not new project work |
| `tests/`, `scripts/verify*.py` | Independent correctness checks |
| `scripts/benchmark_gpu.py` | Complete-model and attention-only benchmarks |
| `checkin/` | Slides, script, demo and teammate preparation material |

The backend is process-wide, not a mixed-device implementation or the original HW3 binary backend. Custom kernels support specified FP32 head sizes/blocks; FP64 and other shapes use a separate array reference. Current training is first-order with dropout disabled and no attention mask.

## Run the small demo on CPU

Use an existing Python 3.10+ environment, preferably an isolated environment:

```bash
python -m pip install -r requirements-demo.txt
python checkin/demo_checkin.py --backend cpu
```

The demo checks one small output/gradient case, a model forward pass, and a 40-update synthetic fit. The real stock panel is intentionally omitted from Git, so a clean clone explicitly uses synthetic inputs for the model-forward step. This is not a stock-performance test. The original presenter's demo can use its separately retained historical panel.

On Windows, `checkin/Run_Demo.cmd` uses `NM_PYTHON` if supplied, the presenter's existing conda environment if available, then `py -3` or `python`. It keeps the result window open. On other systems use the Python command above.

## Correctness checks

```bash
python -m pip install -r requirements-dev.txt
python scripts/verify.py
```

CPU tests include output/gradient comparisons, finite differences, partial tiles, shared-input accumulation, permutation equivariance and a short update-trajectory comparison. No source database is required for these checks. Local Windows convenience scripts are provided, but direct Python commands are portable.

## Optional CUDA setup

Follow the [CuPy installation documentation](https://docs.cupy.dev/en/stable/install.html) for your platform and driver. The local validated configuration was CuPy 14.2.0 with CUDA 12.9 runtime components and an RTX 5080 Laptop GPU. No CUDA libraries or virtual environment are stored in Git.

```bash
python scripts/verify_gpu.py
python scripts/benchmark_gpu.py --smoke
```

GPU scripts select the CUDA backend in a fresh process. Direct model use can select it by setting `MASTER_NEEDLE_BACKEND=cuda` before importing `master_needle`. The adapter has a Windows ASCII-junction workaround for non-ASCII CUDA header paths. Full dataset-dependent scripts require the separately retained panel; they are not expected to run from this source-only clone.

## Saved evidence

In one saved full-model test with 1,024 stocks and 12 historical steps, tiled recomputation used 158.7 MiB versus 397.0 MiB for the dense handwritten-VJP control, while the median wall step increased from 15.66 to 21.81 ms. These are peak live CuPy-pool allocations, excluding the CUDA context and external workspaces. They are not whole-device memory figures and do not demonstrate a speedup over PyTorch.

The small historical experiment uses a fixed 128-stock cohort with 2014 training, 2015H1 validation and 2016H1 test. Its raw data, arrays and weights stay outside Git. FP32 long-training trajectories diverge; high-precision and replay diagnostics are reported without claiming identical predictive behavior.

## What is intentionally excluded

- Historical price panels, source databases and data caches.
- CUDA/runtime binaries, virtual environments, compiled caches and build receipts.
- Model weights, score arrays and archives that may contain datasets.
- Credentials, local account configuration and machine-specific migration records.

Scripts, tests, source, small result summaries, figures, check-in slides and the notebook draft are included. The notebook is a local draft referencing a separate private project bundle; it has not been executed in Colab. The private GitHub source repository does not replace the course's final Colab/private-download submission requirements.

## References

- Li et al., *MASTER: Market-Guided Stock Transformer for Stock Price Forecasting*, AAAI 2024: https://doi.org/10.1609/aaai.v38i1.27767
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, 2022: https://arxiv.org/abs/2205.14135
- Course project: https://dlsyscourse.org/project/
