# Likely mentor questions and short answers

## 1. What is the project?

We implement a Transformer for stock prediction in Needle. It learns patterns over time and relationships between stocks. Our GPU attention reduces saved memory by recomputing intermediate values during backward.

## 2. What is your role?

My focus is the GPU attention implementation and performance evaluation. My teammate focuses on the Needle model and stock data experiments. We review the tests and prepare the report and video together.

## 3. What is temporal attention versus cross-stock attention?

Temporal attention connects dates within one stock's history. Cross-stock attention connects stocks on the same date. The latter is self-attention over the stock dimension; cross-stock describes the axis.

## 4. What did you add to Needle?

Model modules, a custom attention TensorOp, CUDA forward/backward kernels, and a CUDA-array adapter. We also use a first-order backward driver that avoids keeping gradients on constant buffers. The code map names the files.

## 5. Why does it save memory?

It does not save the full N-by-N probability matrix. Forward retains Q/K/V, the output and one log-normalizer per query row. Backward reconstructs probability blocks when needed. Other model activations and library workspaces still consume memory.

## 6. How can softmax work across blocks?

For one query, we visit key blocks while maintaining a running maximum, exponential sum and weighted-value sum. When a new maximum appears, previous sums are rescaled. This gives a stable online softmax without the full probability matrix.

## 7. How does backward work?

We reconstruct probability tiles from Q/K and saved log-normalizers. One kernel computes dQ; another computes dK and dV. Each CUDA block owns an output row, avoiding atomic gradient accumulation. The walkthrough contains the formulas.

## 8. Are you using PyTorch to implement or train the project?

No. Needle runs the model and gradient propagation. PyTorch is only an independent reference in tests. CuPy provides GPU arrays and cuBLAS. The attention kernels themselves are project CUDA code. We did not independently implement every basic array primitive.

## 9. Is this a full FlashAttention implementation?

No. It uses stable online softmax, tiling and backward recomputation. The kernels use simple scalar dot products and shared reductions rather than a complete tensor-core/IO-optimized FlashAttention implementation.

## 10. What does the short demo show?

One small output/gradient comparison, a model forward pass, and 40 synthetic-data parameter updates. The presenter's model-forward case has 32 stocks and 12 historical days. The synthetic loss decrease is a learning check, not evidence of stock prediction quality.

## 11. Why does the demo repeat the same numbers?

Seeds, inputs and update count are fixed. The computation is run again each time; it is not a hard-coded success message. The same environment typically reproduces the rounded values. CPU/other environments can differ slightly.

## 12. What is the main measured result?

In one full-model test with 1,024 stocks and 12 time steps, tiled attention used about 159 MiB versus 397 MiB for dense handwritten-VJP attention. A training step took about 22 ms versus 16 ms. It is a memory/time trade-off, not a speedup. These are saved presenter measurements from October 26.

## 13. What does memory mean here?

Peak live allocations through the CuPy pool. This excludes the CUDA context and workspaces outside that allocator. It is not entire GPU usage or reserved pool capacity. All GPU modes use the same first-order driver; the handwritten dense control helps isolate tiling.

## 14. Does 1,024 mean you trained on 1,024 real stocks?

No. The systems benchmark uses synthetic inputs with controlled dimensions. Separate real training uses a fixed historical cohort of 128 stocks: 2014 train, first-half 2015 validation, first-half 2016 test. Do not mix the benchmark size with real training coverage.

## 15. How did you check correctness?

Output/gradient reference comparisons, CPU finite differences, high-precision full-model comparisons and a real-input gradient check. We also check repeated execution and constant-buffer memory retention. Small tests do not prove correctness for every shape or identical long-run training.

## 16. What is the FP32 numerical issue?

Longer FP32 training can follow different trajectories between implementations. A training-only 16-step diagnostic found much smaller FP64 differences and exact same-kernel replay. This supports arithmetic/update sensitivity in that tested case, but the long-run issue is not fully resolved.

## 17. Is the architecture new? What references should we name?

The architecture is inspired by MASTER (Li et al., AAAI 2024), and the memory-saving ideas build on FlashAttention (Dao et al., 2022). We focus on implementation and evaluation in Needle, and cite the background in the report.

## 18. Is there lookahead in the small panel?

Inputs are historical windows, normalization is fit on training observations, and label-boundary dates are purged. All known T-day scoring stocks enter the model; future missing labels mask only loss/metrics. Provider corrections in the warehouse remain a data-vintage limitation.

## 19. Why is there no causal mask?

This is not autoregressive language generation. At the prediction date, every date in the input window has already occurred. The final score can use the whole historical window; no future dates relative to that prediction are inputs.

## 20. What remains, and what is the fallback?

Review validation/numerical behavior, summarize memory and time results, and finish the notebook/video. The fallback keeps verified input sizes and the current model with the custom kernels. We do not require a speedup or an expanded architecture.

## 21. What exactly do you predict, and which loss do you use?

The real experiment predicts a score for the adjusted close-to-close log return five trading days ahead. Real training minimizes one minus the Pearson correlation within each labelled stock cross-section. RankIC evaluation uses rank correlation. The separate tiny synthetic demo uses mean squared error; it is a functional test, not the same financial forecasting task. The benchmark's scalar loss is for measuring a training step, not reporting forecast quality.

## Precise wording

- Memory-saving implementation, not a new attention algorithm.
- Lower measured live-array allocation, not 60% less entire GPU memory.
- Slower training in this test, not a demonstrated speedup.
- Small-case checks passed, not all long-run FP32 results are equivalent.
- Responsibility split is not proof of personal historical authorship.
