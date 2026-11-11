# Leif's speaking card

## Your role (about 15 seconds)

My focus is the GPU attention implementation and performance evaluation. My teammate focuses on the Needle model and stock data experiments. We review the tests together and jointly prepare the final report and video.

## How the GPU part works (about 25 seconds)

Dense attention saves weights for every pair of stocks. We process the keys in blocks and keep a running softmax normalization. During backward, we recompute the probability blocks to get the gradients. This reduces saved memory, but it requires additional computation.

## What the result means (about 20 seconds)

In one full-model test, peak live GPU array allocations fell from about 397 to 159 MiB. The training step increased from about 16 to 22 milliseconds. So the current result is a memory and time trade-off, not a speed improvement.

## Main limitation (about 15 seconds)

Longer FP32 training can follow different trajectories between implementations. We have small-case and high-precision checks, but we do not claim identical long-run training results.

## If you need to check a detail

I would like to check the code or the saved result before giving an exact answer.

## Remember the distinctions

- 32 stocks / 12 days: the presenter's small forward demo.
- 40 updates: a separate synthetic learning check.
- 128 stocks: the historical real-data experiment.
- 1,024 stocks: the controlled GPU benchmark with synthetic inputs.
- 397 to 159 MiB: live CuPy-array-pool peak, not total GPU memory.
- 16 to 22 ms: slower, not faster.

Use your own words and review the source. Do not claim personal work you have not reviewed.
