# Demo guide

## During the meeting

The presenter shares slides and runs the original GPU/real-input demo on their own machine. You do not need a CUDA installation to understand the demo. The full pitch including demo is 2-3 minutes; the meeting can last up to 10 minutes.

The four PASS lines mean: output agrees with a reference for one small case; Q/K/V gradients agree; a small model returns finite scores of the expected shape; and a short synthetic same-batch fit reduces loss.

The presenter's saved CUDA run shows output error about 2.38e-7, gradient error about 4.17e-7, and synthetic loss about 1.1209 to 0.0077. These are functional checks, not evidence of profitable predictions or identical long-run training.

## Optional local CPU demo

This pack includes the actual core code and Needle dependency, but no stock panel or CUDA binaries. With Python, NumPy, PyTorch and threadpoolctl already installed, run from `code/`:

    python checkin/demo_checkin.py --backend cpu

Windows users can double-click `code/Run_CPU_Demo.cmd`. It finds `py -3` or `python`, installs nothing, and keeps the output window open. The minimal dependencies are listed in `code/requirements-demo.txt`.

Without the presenter's panel, the model-forward step explicitly uses synthetic inputs. CPU training may end near 0.0075 rather than 0.0077. Do not describe this local CPU/synthetic run as the presenter's GPU/real-input run. No database access is needed.

Other CUDA-check and benchmark scripts are included for reading. Running them needs appropriate CUDA/CuPy; data-dependent diagnostics also need the omitted panel. This preparation pack is not the complete experiment environment.

## If something fails

Report the failure and inspect the saved log. The demo does not silently switch to CPU to disguise a correctness failure; backend-unavailable fallback is explicitly labeled. Avoid spending the meeting debugging the entire environment.
