## 2024-05-24 - GGUF Metadata Reading Overhead
**Learning:** `numpy.memmap` introduces significant overhead during repeated array slicing due to `__array_finalize__` and `__getitem__`.
**Action:** When performing heavy slicing on memory-mapped files, create an `ndarray` view of the loaded `memmap` upon initialization (e.g., `self.data.view(type=np.ndarray)`) to bypass this class overhead while preserving zero-copy behavior.
