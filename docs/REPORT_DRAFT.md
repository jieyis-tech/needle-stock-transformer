# Memory-Efficient MASTER in Needle: Exact Cross-Stock Attention with Recomputation

Local report draft, 26 October 2026. All reported experiments ran on the local
Windows workstation. The accompanying notebook has not yet been executed in
Google Colab. Team information and the final private code-download link remain
to be supplied before course submission.

## Motivation and scope

Cross-stock attention offers a useful systems workload because its sequence axis
is the number of stocks observed together. A forecasting model may use only a
short temporal history yet attend over hundreds or thousands of securities at
each historical step. Materializing every pairwise probability can therefore
consume more memory than the projected features themselves. Our project asks
whether Needle can execute this workload while retaining substantially less
attention state, and what additional computation that saving requires.

We implement a small MASTER-style architecture and a new CUDA execution path for
Needle. The architecture provides a concrete application rather than a claim of
new forecasting methodology. The systems contribution combines device storage,
custom attention forward and backward kernels, an explicit first-order gradient
contract, numerical checks, and complete-model measurements. We intentionally
separate implementation correctness, resource efficiency, and forecasting
behavior. A model that fits a small dataset does not by itself establish either
kernel correctness or an economically useful strategy.

## Background and attribution

Li and colleagues introduced MASTER, a market-guided stock transformer accepted
at AAAI 2024. Its market gate and alternating intra-stock/inter-stock aggregation
motivate the application architecture. We follow the main structure in an existing
local research implementation, while using a compact daily-input variant instead
of claiming reproduction of the paper's complete data pipeline or published
scores. We cite the architecture and distinguish earlier work from this project.

Dao and colleagues' FlashAttention work motivates stable online normalization,
tiling and probability recomputation. Those algorithmic ideas are established
background, not our invention. Our kernels do not implement the complete
FlashAttention scheduling or tensor-core design. The course contribution is a
Needle implementation and its empirical study. The completed local HW2 Needle
library is retained as an unchanged, hashed dependency; its source files are not
counted as new project work. NumPy, CuPy and cuBLAS are external numerical
dependencies, and PyTorch appears only in independent reference checks.

## Framework integration

The CPU feasibility version already used Needle TensorOps and modules, but its
operators ran on NumPy arrays. The GPU version adds a process-wide CUDA array
backend. Tensor construction uploads arrays, Tensor storage remains on the device,
and the basic mathematical and logarithmic operators dispatch to CuPy. The graph
and model interfaces remain Needle's. Basic matrix multiplication uses cuBLAS;
the attention-specific CUDA code is compiled through NVRTC and launched on the
current stream. Training never calls PyTorch autograd or attention routines.

This adapter is a supported path for this model, not a complete replacement for
every Needle backend. It does not load the Linux HW3 extension into Windows or
claim that all HW3 low-level operators have been independently reimplemented.
CPU and CUDA tests run in separate processes, and mixed-device graphs are not
supported. CUDA dependencies are isolated from the existing quantitative research
environment. On Windows, compiler-facing dependency paths use an ASCII junction
when the actual project path contains Chinese characters; the underlying files
remain in the requested project directory.

The model includes a market-status gate, a feature projection, sinusoidal
positions, temporal attention, cross-stock attention, temporal pooling and a
scalar prediction head. Both attention blocks include normalization and a small
feed-forward residual component. Temporal attention is unscaled; stock attention
uses inverse square root of head dimension, matching the local design reference.
All temporal inputs are historical at the signal date. Dropout is fixed to zero,
and this experiment does not require causal or padding masks.

## Tiled forward and backward

Dense attention computes a scaled QK transpose, normalizes each row, then
multiplies by V. Saving the probability matrix costs quadratic storage in the
number of stocks. Our forward kernel assigns one block to a query row and leading
batch. Threads compute scores for a tile of keys, reduce a maximum, exponentiate
relative scores, and accumulate weighted values. When a later tile has a larger
maximum, the earlier numerator and denominator are rescaled before accumulation.
This online softmax produces the same mathematical operation without a global
quadratic probability tensor.

The saved state contains Q, K, V, output O, and row log-normalizers. During
backward, probabilities are reconstructed for the required tile. The softmax
adjoint uses the row correction given by the inner product of dO and O. One
kernel owns each query row and accumulates dQ; another owns each key row and
accumulates dK and dV. This ownership avoids floating-point atomics, but repeats
some probability calculations across the two backward passes.

For fixed head dimension and batch size, saved attention state grows linearly
with stock count. Arithmetic remains quadratic. The implementation uses scalar
dot products and shared reductions rather than sophisticated matrix tiles.
Consequently, memory savings can come with worse latency. Supported FP32 block
sizes are 32, 64, 128 and 256, with feature dimensions no larger than the block.
Other GPU shapes and FP64 use a separate batched CuPy tiled reference.

## Gradient lifetime and correctness

Complete-model testing exposed a framework issue that attention-only testing did
not reveal. The HW2 teaching driver assigned gradients to constant buffers and
constructed a differentiable backward graph. Persistent position tensors could
retain earlier graphs. The CUDA backend therefore uses an explicit first-order
driver: shared-input contributions accumulate, each VJP is detached, and constants
are skipped. All GPU comparison arms use this same driver. Higher-order gradients
are outside the interface. A regression runs twenty updates and verifies that
constant buffers have no gradients and that live allocations remain unchanged
between steps five and twenty. Pre-fix measurements are archived and excluded.

CPU verification covers finite differences, partial tiles, shared inputs,
FP32/FP64 results, stock permutation equivariance and short optimizer trajectories.
GPU kernel checks compare outputs and Q/K/V gradients with an independent
PyTorch CPU oracle across several shapes and block lengths. Full-model checks
use FP64 to reduce cancellation effects. A separate real-date test computes the
correlation loss and checks every parameter gradient against the reference.
The accompanying result files contain the actual maximum errors, not simply a
statement that tests passed.

Exact attention means mathematical equivalence, not identical floating-point
rounding. FP32 checks initially found relative discrepancies on gradient entries
near zero. We retained that observation, added normwise error measurements and
used an FP64 architectural oracle. We did not describe a relaxed FP32 comparison
as bitwise agreement. Training trajectories require a separate investigation
because repeated adaptive updates can amplify small arithmetic differences.

## Real-data protocol and timing safety

The integration panel uses 128 securities chosen by a deterministic code hash
from stocks already listed by 2013 and not delisted as of the initial 2014 signal
date. Later survival and returns do not enter selection. The source warehouse is
opened read-only. A manifest records SQL, source views, dates, features, units,
normalization parameters, exclusions and an array fingerprint. No network
download or production-table mutation occurs.

Inputs contain twelve daily observations of nine price/activity/size features
and a traded indicator. Missing historical values become zero after training-set
standardization; the indicator preserves the missing-trading distinction. Market
inputs use rolling statistics available through each signal date. Scaling is fit
only on 2014 observations. A prefix recomputation checks that a validation-date
market vector is unchanged when subsequent source observations are unavailable.
Historical provider corrections remain a data-vintage limitation.

Training uses 2014 signal dates, validation uses the first half of 2015, and test
uses the first half of 2016. Each segment loses its final five candidate dates so
its forward five-trading-day label is realized within the segment. The latest
training label precedes the earliest validation signal, and the same condition
holds between validation and test. A stock with a known tradable T-day observation
enters scoring even when its future label is missing. Only the loss and metrics
mask that missing target. The report includes counts of those scored-but-unlabeled
rows rather than silently removing them.

The fixed experiment uses width 32, seed 42, three epochs, Adam with learning
rate 0.001, and gradient norm clipping at one. There is no validation-driven
epoch selection and no use of post-2017 observations. This bounded daily variant
is a systems demonstration, not the earlier production monthly PCA model.

## Measurements and findings

GPU benchmarks use the local RTX 5080 Laptop GPU. Each configuration runs in a
fresh process, performs two warmups, and records five timed iterations. A model
step includes forward, scalar loss, backward and Adam. We record synchronized
wall time and CUDA-event time separately. An allocator wrapper captures peak
live CuPy-pool allocations after each allocation. This excludes the CUDA context
and external library workspaces; it is neither whole-device usage nor CPU RSS.

At 1,024 stocks and twelve time steps, the measured full-model live-pool peak
was about 1,461 MiB for ordinary dense graph attention, 397 MiB for dense attention
with a handwritten VJP, and 159 MiB for tiled recomputation. The handwritten dense
control is important: it prevents attributing every difference to tiling when
framework graph construction can also contribute. Relative to that control, the
tiled model used approximately sixty percent less measured live memory. Its
median wall step increased from about 15.5 to 21.6 milliseconds. These local
measurements support a storage/recomputation tradeoff, not a universal speedup.

Predefined ablations vary stock count, temporal length and block length. The
attention-only measurement is reported separately from the complete model.
The model's other activations limit the end-to-end saving, and host launch costs
can dominate small configurations. Five iterations on one Windows laptop are
insufficient for broad performance claims across architectures or power states.

All three implementations trained on the fixed real panel, with complete saved
weights, scores and date/code mappings. Their FP32 trajectories diverged over
hundreds of updates. A train-only sixteen-step diagnostic found exact same-kernel
replay, much smaller differences in FP64, and larger parameter differences in
FP32. This supports numerical sensitivity as an explanation in the tested setting,
but does not establish predictive equivalence after long training. We report the
single-seed RankIC observations without selecting the apparent winner or claiming
an attention method improves forecasting.

## Limitations and deliverables

The project supports first-order, fixed-length, dropout-free execution. Masks,
mixed-device graphs, higher-order derivatives and complete FlashAttention-style
optimization are not implemented. The real cohort is small and historically fixed.
Labels are adjusted close-to-close forecasts, with no execution model, costs,
price limits or ST constraints. Thus no backtest or investment conclusion follows.
The notebook includes reproducible commands and original figures, but cloud
execution remains to be verified. The final course submission also needs team
details, a private code link and the required presentation video.

The current evidence establishes a functioning Needle GPU extension, independently
checked derivatives, real-panel integration, and measured full-model resource
tradeoffs. Future work should target kernel scheduling and memory coalescing,
stronger FP32 stability, additional devices and repeated independent performance
runs. Those extensions should be evaluated on their own evidence rather than
assumed from the memory reductions already observed.

## References

- Tong Li et al., MASTER: Market-Guided Stock Transformer for Stock Price Forecasting,
  AAAI 2024. https://doi.org/10.1609/aaai.v38i1.27767
- Tri Dao et al., FlashAttention: Fast and Memory-Efficient Exact Attention with
  IO-Awareness, 2022. https://arxiv.org/abs/2205.14135
- Deep Learning Systems final project requirements. https://dlsyscourse.org/project/
- CuPy installation and dependency documentation. https://docs.cupy.dev/en/stable/install.html
