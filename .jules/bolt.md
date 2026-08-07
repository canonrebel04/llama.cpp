## 2024-05-14 - Optimize itemsize extraction in gguf_reader
**Learning:** `np.empty([], dtype=dtype).itemsize` is significantly slower (~30-50%) than `np.dtype(dtype).itemsize` due to array instantiation.
**Action:** Use `np.dtype(dtype).itemsize` when getting item sizes for `npt.DTypeLike` in hot paths like `GGUFReader._get`.
