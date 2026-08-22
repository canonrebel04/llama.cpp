
## 2024-05-15 - Fast np.memmap slicing
**Learning:** Slicing `numpy.memmap` in a loop introduces significant overhead due to subclass instantiations invoking `__array_finalize__` and `__getitem__`.
**Action:** Create a standard ndarray view of the memmap using `.view(type=np.ndarray)` after initialization, and use this view for internal slicing and fast data access without losing the zero-copy benefits.
