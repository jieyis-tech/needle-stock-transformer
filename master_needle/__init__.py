"""Needle MASTER extensions; NumPy CPU or optional CUDA array backend."""
from pathlib import Path
import sys
import os

_vendor = str(Path(__file__).resolve().parents[1] / "vendor")
sys.path.insert(0, _vendor)

if os.environ.get('MASTER_NEEDLE_BACKEND','cpu') == 'cuda':
    from .backend import enable_cuda
    enable_cuda()

from .attention import attention, dense_attention
from .model import MASTER

