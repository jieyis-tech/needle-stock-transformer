# GPU code walkthrough

## Layout and attention

Model inputs are (N stocks, T dates, F features). Temporal attention treats each stock as a sequence; stock attention treats the stock axis as the sequence. Leading time/head combinations become batches.

For one head: S = scale * Q K^T; P = softmax(S); O = P V.
The stock block uses scale=1/sqrt(head_dim). The local MASTER-inspired temporal block uses scale=1, so do not say all blocks use the same scaling.

## Source reading order

1. `code/master_needle/model.py`: AttentionBlock.forward reshapes/transposes to choose the axis. MASTER.forward combines the blocks and prediction head.
2. `code/master_needle/attention.py`: Needle TensorOp compute/VJP and dispatch between dense and tiled paths.
3. `code/master_needle/cuda_attention.py`: NVRTC compilation, contiguous device inputs, launch arguments.
4. `code/master_needle/kernels/attention.cu`: the three kernels below.
5. `code/master_needle/backend.py`: array binding and first-order backward driver.
6. `code/scripts/benchmark_gpu.py`: warmups, timed steps and live-allocation tracking.

## Forward: attn_forward

One CUDA block owns one query row. Threads compute a key tile's scores, use shared reductions for max/sum, then update the running denominator and weighted-value sum. A new maximum rescales previous accumulators. Output O and row log-sum-exp (LSE) are saved. No full global N-by-N probability matrix is stored in this path.

## Backward: attn_dq and attn_dkv

Let G=dO. Reconstruct probability tiles using P=exp(scale*QK^T-LSE).

    dP = G V^T
    delta = row_sum(G * O)
    dS = P * (dP - delta)
    dQ = scale * dS K
    dK = scale * dS^T Q
    dV = P^T G

The first kernel owns query rows for dQ; the second owns key rows for dK/dV. Ownership avoids atomic gradient writes. The two passes repeat probability work, contributing to the time cost.

## Limits

- Custom kernels: FP32, blocks 32/64/128/256, feature dimensions at most block size.
- FP64 and other shapes use a separate CuPy tiled reference, not double-precision custom kernels.
- First-order only; no attention masks, training dropout or mixed-device graphs.
- Simple scalar dot products/shared reductions, not full tensor-core FlashAttention.
- Saved attention state is linear in stock count with fixed other dimensions; arithmetic is quadratic. Whole-model memory has other terms.

## Backward-driver fix

The HW2 teaching driver could retain gradient graphs on persistent constant position buffers. The CUDA adapter skips constants and detaches VJPs. All GPU comparisons use this same driver. A regression checks stable live allocations after updates 5 and 20. Pre-fix measurements are excluded.

## Three comparison modes

`dense`: ordinary Needle graph attention. `dense_vjp`: dense probabilities with handwritten first-order backward. `tiled`: block recomputation. The slide compares dense_vjp to tiled, so it does not attribute all graph-management overhead to tiling.
