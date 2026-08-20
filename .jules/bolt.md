## 2024-05-24 - Optimizing np.memmap slicing overhead
**Learning:** `numpy.memmap` has significant overhead during repeated array slicing due to its class methods `__array_finalize__` and `__getitem__`.
**Action:** For repeated slicing operations on a memory-mapped file, create an `ndarray` view of the `memmap` object (`data.view(type=np.ndarray)`) and slice that view instead, which preserves zero-copy behavior while bypassing the class overhead.
