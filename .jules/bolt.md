## 2024-05-24 - Zero-copy NumPy memmap optimization
**Learning:** In `gguf-py`, `numpy.memmap` introduces significant overhead during repeated array slicing due to `__array_finalize__` and `__getitem__` class overheads.
**Action:** To optimize metadata reading (e.g., in `GGUFReader`), create an `ndarray` view of the loaded `memmap` via `.view(type=np.ndarray)` upon initialization and use this view for repeated internal slicing. This bypasses the class overhead while preserving zero-copy behavior, which reduced parsing time by ~70% in tests (0.486s to 0.147s for 10,000 KVs).
