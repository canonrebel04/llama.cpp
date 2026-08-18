## 2024-05-24 - Numpy Memmap vs Ndarray Overhead
**Learning:** `numpy.memmap` introduces significant overhead during repeated array slicing due to its `__array_finalize__` and `__getitem__` methods recreating `memmap` instances. This can severely bottleneck GGUF metadata reading (e.g., in `GGUFReader`).
**Action:** To optimize metadata reading while preserving zero-copy behavior, create an `ndarray` view of the loaded `memmap` via `.view(type=np.ndarray)` upon initialization (e.g., store as `self._fast_data`) and use this base array view for internal frequent slice operations instead.
