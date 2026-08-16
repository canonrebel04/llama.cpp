## 2024-08-16 - memmap overhead in metadata reading
**Learning:** `numpy.memmap` introduces significant slicing overhead (`__array_finalize__` and `__getitem__`) in Python, which becomes a major bottleneck when doing many small reads/slices as in parsing `gguf` metadata.
**Action:** Always convert loaded `numpy.memmap` objects directly to `np.ndarray` via `.view(type=np.ndarray)` when doing repeated array slicing. It preserves the underlying zero-copy mmap characteristics but entirely bypasses the `memmap` Python class overhead.
