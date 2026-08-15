## 2026-08-11 - Fast reads from Numpy memmap
**Learning:** In NumPy, slicing and viewing a memory-mapped array is a slow operation due to object recreation. However, using `np.ndarray(buffer=...)` directly on a memmap fails with a `ValueError` for unaligned offsets, which happen frequently when reading packed data like GGUF KV pairs.
**Action:** Use `np.dtype(dtype).itemsize` for efficient item size calculation, but stick to `.view(dtype=...)` for slicing to safely handle unaligned offsets.
