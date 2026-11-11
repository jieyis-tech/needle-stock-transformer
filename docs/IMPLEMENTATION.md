# Implementation notes

## Architecture and ownership

The project extends a local completed HW2 Needle snapshot. Vendor files are
unchanged and individually hashed. The execution graph, reverse traversal,
TensorOp interface, modules and optimizer remain Needle concepts. The optional
CUDA adapter rebinds basic array APIs to CuPy and uses CUDA arrays as Tensor storage.
CuPy/cuBLAS supply established allocation/basic math primitives. The attention
forward and two backward kernels in `attention.cu` are project implementations.
No PyTorch training, autograd, SDPA or FlashAttention library call is used.

The adapter is process-wide. Constructing tensors before activation raises an
error; CPU and CUDA checks run in separate processes. CUDA Tensor constructors
route inputs to GPU irrespective of a legacy CPU default argument. Mixed-device
execution is outside the supported interface. Model initialization may generate
NumPy random values before one initial upload. Forward/backward operations stay
on CUDA arrays. Tensor.numpy instrumentation checks explicit host reads; it is
not a complete PCIe hardware profiler.

## Exact attention

Attention is `O = softmax(scale * Q K^T) V`. Temporal MASTER attention uses scale 1;
cross-stock attention uses inverse square root of head dimension, matching the
local MASTER design reference. All temporal tokens are already historical; all
stocks in a cross-section are observed on the same signal date.

FP32 forward assigns one CUDA block to each query row and leading batch.
Threads compute scores for a key tile, reduce a maximum and exponential sum,
then update the row normalizer and weighted output. An online maximum rescales
the previous accumulator. Persistent state is Q/K/V, O, and row log-normalizers;
score/probability tiles use shared memory, never an N by N global-memory allocation.

Backward reconstructs P from Q/K and saved log-normalizers. It uses
`dS = P * (dO V^T - rowsum(dO * O))`, then forms dQ, dK and dV.
One kernel owns query rows for dQ. Another owns key rows for dK/dV. Independent
ownership removes floating-point atomics at the cost of recomputing probabilities
in both passes. FMA/reduction orders differ from cuBLAS and are not bitwise exact.

For fixed head sizes, persistent attention state grows linearly with stock count;
arithmetic remains quadratic. The kernels use scalar dot products and simple
shared reductions, not tensor-core matrix tiles or complete FlashAttention.
More memory efficiency does not imply higher throughput. FP32 supports blocks
32/64/128/256 and dimensions no larger than the block. Tails are guarded.
FP64/unusual shapes use a separate batched CuPy tiled reference.

## First-order driver and constant retention

The HW2 teaching driver builds differentiable gradient expressions and writes
gradients on constants as well as parameters. Persistent position tensors can
therefore keep an older graph reachable. The CUDA adapter installs a first-order
driver: accumulate shared-input VJPs, detach each result, skip requires_grad=False
inputs. This applies equally to all GPU baselines. Double backward is unsupported.
A 20-step regression checks constant buffers have no gradients and live allocation
does not grow between steps 5 and 20. Pre-fix benchmarks are archived and excluded.

## Numerical verification

CPU checks include finite differences, shared Q/K/V accumulation, partial tiles,
FP32/FP64 outputs, stock permutation equivariance and short optimizer trajectories.
GPU FP32 kernels are compared with an independent PyTorch CPU oracle over several
shapes and blocks. Full-model FP64 comparisons isolate architecture/derivatives
from FP32 LayerNorm/reduction cancellation. A real-date correlation-loss example
checks every parameter gradient in FP64.

FP32 whole-model errors are reported as max absolute and relative L2 error per
parameter. Initial elementwise relative checks failed on near-zero gradients;
that motivated a higher-precision oracle and normwise diagnostic, not a claim of
bitwise agreement. The FP32 bound is explicitly weaker than the FP64 check.

After three real-data epochs, trajectories differ. A train-only 16-step precision
experiment and same-kernel replay distinguish arithmetic sensitivity from random
seed/order variation. This limits learning-curve comparisons and does not establish
one attention implementation predicts better. No test score selects a precision,
epoch or hyperparameter.

## Benchmark contract

Each configuration runs in a new child process with two warmups and five timed
iterations. Full-model steps include forward, loss, backward and Adam updates.
CUDA events and synchronized wall time are recorded. An allocator wrapper samples
pool used-byte count after each allocation, measuring live allocations rather than
cached capacity. Context/external library workspaces are excluded. Comparisons
are within this backend, not against PyTorch's optimized training stack.

Predefined subsets vary N=128/512/1024, T=4/12/24 and block=32/64/128, width 32.
Model and attention-only benchmarks are separate. Windows WDDM scheduling, power
states and host launch overhead remain measurement limitations.
