## 2024-05-23 - memmap array slicing overhead
**Learning:** In `gguf-py`, `numpy.memmap` introduces significant overhead during repeated array slicing due to `__array_finalize__` and `__getitem__` overhead.
**Action:** Create an `ndarray` view of the loaded `memmap` via `.view(type=np.ndarray)` upon initialization (e.g. `self._fast_data`) to bypass this class overhead while preserving zero-copy behavior.
