
## 2024-05-18 - Avoid memmap slice overhead
**Learning:** `numpy.memmap` introduces significant overhead during repeated array slicing due to `__array_finalize__` and `__getitem__`.
**Action:** When performing many small reads/slices (e.g., metadata reading in `GGUFReader`), create an `ndarray` view of the `memmap` object (`data.view(type=np.ndarray)`) and slice from that view to bypass overhead while maintaining zero-copy mapping.
