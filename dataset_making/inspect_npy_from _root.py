#!/usr/bin/env python3
import sys, io, pickle
from pathlib import Path
import numpy as np
import numpy.lib.format as nf

# 1) Define an Unpickler that redirects numpy._core to numpy.core
class NumpyCoreRedirectUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)

# 2) Low‑level loader for .npy object arrays
def load_npy_object_array(path):
    with open(path, 'rb') as f:
        version = nf.read_magic(f)  # e.g. (1,0) or (2,0)
        shape, fortran_order, dtype = nf._read_array_header(f, version)
        # now the file pointer is at the start of the pickle blob:
        return NumpyCoreRedirectUnpickler(f).load()

# 3) Inspect routine
def inspect_episode(path):
    try:
        ep = load_npy_object_array(path)
    except Exception as e:
        print(f"Failed to load {path.name}: {e}")
        return
    print(f"{path.name}: {type(ep)} with {len(ep) if hasattr(ep,'__len__') else '??'} steps")
    if isinstance(ep, (list, np.ndarray)):
        step0 = ep[0]
        if isinstance(step0, dict):
            for k, v in step0.items():
                if hasattr(v, 'shape'):
                    print(f"  {k!r}: shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"  {k!r}: type={type(v)}")
                    print(v)
        else:
            print("  ! first element is not a dict:", type(step0))
    else:
        print("  ! top‐level object is not a list/ndarray:", type(ep))
    print()

# 4) Run on first few files
if __name__ == "__main__":
    data_dir = Path("data/hanoi_dataset/data")
    for i, path in enumerate(sorted(data_dir.glob("episode_*.npy"))):
        if i >= 5: break       # just inspect first five
        inspect_episode(path)
