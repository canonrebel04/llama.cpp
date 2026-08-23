## 2024-03-22 - GGUF Metadata Reading Overhead
**Learning:** `numpy.memmap` introduces significant overhead during repeated array slicing due to `__array_finalize__` and `__getitem__`.
**Action:** In `gguf-py`, create an `ndarray` view of the loaded `memmap` via `.view(type=np.ndarray)` upon initialization (e.g., storing it as `self._fast_data` for internal slicing) to bypass this class overhead while preserving zero-copy behavior.
