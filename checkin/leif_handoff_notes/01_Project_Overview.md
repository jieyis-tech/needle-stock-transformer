# Project overview for Leif

## One sentence

We implement a Transformer for stock prediction in Needle, with temporal attention and attention across stocks, and study a CUDA implementation that saves attention memory through backward recomputation.

## The idea, without finance background

For each stock, we have a short history of daily price and trading features. The model returns one score for each stock. Temporal attention connects dates within one stock's history. Cross-stock attention connects stocks observed on the same date.

In the real-data experiment, each input contains 12 days and 10 features describing prices, trading activity, size and whether the stock traded. The target is the adjusted close-to-close log return five trading days ahead. The score is a forecasting output, not a trading order. Real training uses a correlation loss; the separate synthetic demo uses mean squared error.

Here, cross-stock attention is self-attention along the stock axis: Q, K and V come from the same set of stock representations. It is not a new encoder-decoder cross-attention mechanism.

Inputs have shape (stocks, time, features). For temporal attention, each stock is a separate sequence. For stock attention, stocks become the sequence axis. The model also has feature projection, positions, normalization, temporal pooling and a prediction head. A market-feature gate exists in the code but is not the main pitch or a separate market-attention invention.

## Why the GPU part matters

Dense stock attention builds weights for all pairs of N stocks. The probability matrix has N squared entries for each head/time batch. We process keys in blocks, keep a stable running softmax normalizer and output, and recompute probabilities during backward. This reduces saved memory but adds computation. Arithmetic remains quadratic; this is mathematically dense attention with different floating-point rounding.

## What runs where

- Needle: model, TensorOp integration and gradient propagation.
- CuPy/cuBLAS: CUDA arrays, basic array operations and matrix multiplication.
- Project CUDA source: specialized attention forward and backward kernels.
- PyTorch: independent correctness reference only, not the training backend.

The CUDA adapter is process-wide and first-order only. It is not the original HW3 binary backend, nor an implementation of every low-level GPU primitive.

## Current scope

There is a working prototype, initial correctness validation, a small real-stock training experiment and GPU memory/time measurements. Remaining work is to review numerical behavior, summarize the evaluation, and finish the notebook and video. We are not adding a new architecture or a large training campaign for check-in.

The architecture is inspired by MASTER (Li et al., AAAI 2024). Tiling, online softmax and recomputation build on established ideas such as FlashAttention (Dao et al., 2022). Our contribution is the Needle implementation and systems evaluation, not invention of these ideas.

## Agreed responsibility split

- Leif: GPU attention implementation and performance evaluation.
- Your teammate: Needle model implementation and stock-data experiments.
- Together: test review, final report and video.

This is the responsibility split, not a claim about which person historically wrote each existing line. Review the relevant code before explaining personal contributions.

## Reading order

Read the speaking card and Q&A first, then view the three slides. For technical details, follow the GPU walkthrough and the actual source in `code/`. Saved evidence is in `evidence/`. Reading this pack requires no installation.
