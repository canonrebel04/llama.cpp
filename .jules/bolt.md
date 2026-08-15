## 2024-05-18 - Avoid numpy empty for dtype size checking
**Learning:** Using `np.empty([], dtype=dtype).itemsize` to check item sizes in Python creates excessive overhead. However, attempting to read directly from a memory-mapped buffer using `np.ndarray(buffer=self.data, offset=offset)` fails with a `ValueError` on unaligned offsets (which occur in formats like GGUF that pack without alignment).
**Action:** Use the safe `np.dtype(dtype).itemsize` idiom combined with slicing and `.view()` to minimize overhead while safely handling unaligned memory offsets.
