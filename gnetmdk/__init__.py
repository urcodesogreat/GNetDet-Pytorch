import os
import torch

_REC_VERSION = (1, 9)
_CURR_VERSION = tuple(map(int, torch.__version__.split('.')[:2]))
if _CURR_VERSION < _REC_VERSION:
    print(f"[INFO] Recommended Pytorch Version >= {_REC_VERSION[0]}.{_REC_VERSION[1]}")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
__version__ = "1.1.2"
