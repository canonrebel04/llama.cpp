## 2024-05-18 - Avoid numpy empty for dtype size checking
**Learning:** Using `np.empty([], dtype=dtype).itemsize` and taking array slices to read from memmap arrays in python creates excessive overhead, especially when parsing files with many small structures (like `gguf_reader.py` parsing GGUF).
**Action:** Use `np.dtype(dtype)` to check itemsize, and ideally read from memmap buffer directly using `np.ndarray(count, dtype=..., buffer=self.data, offset=offset)`. This avoids allocating empty arrays and intermediate view slicing.
