## 2024-05-24 - Numpy Memmap Overhead Optimization
**Learning:** In `gguf-py`, `numpy.memmap` introduces significant overhead during repeated array slicing due to `__array_finalize__` and `__getitem__`.
**Action:** Convert the loaded `memmap` directly to an `ndarray` via `.view(type=np.ndarray)` upon initialization to bypass this class overhead while preserving zero-copy behavior.
