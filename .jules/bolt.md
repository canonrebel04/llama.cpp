## 2026-08-11 - Fast reads from Numpy memmap
**Learning:** In NumPy, calling `.view()` on a memory-mapped array slice creates a new memmap object behind the scenes, triggering expensive `__array_finalize__` and `__getitem__` overheads.
**Action:** For performance-critical code reading small chunks from a memmap, construct an `np.ndarray` directly using `buffer=self.data` and `offset=...` rather than slicing and viewing the memmap.
