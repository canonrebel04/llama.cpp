## 2025-02-20 - Numpy Memmap Overhead in GGUFReader
**Learning:** `numpy.memmap` introduces significant overhead during repeated array slicing due to `__array_finalize__` and `__getitem__`.
**Action:** To optimize metadata reading (e.g., in `GGUFReader`), convert the loaded `memmap` directly to an `ndarray` via `.view(type=np.ndarray)` upon initialization to bypass this class overhead while preserving zero-copy behavior.
