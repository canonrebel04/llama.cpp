## 2024-08-15 - Optimizing np.memmap reads
**Learning:** In `gguf-py`, when reading binary data from `np.memmap`, using `np.empty([], dtype=dtype).itemsize` and array slicing (`data[start:end].view()`) has significant overhead. Constructing arrays directly with `np.ndarray(..., buffer=data, offset=...)` avoids empty allocations and view overhead.
**Action:** When extracting sub-arrays from `memmap` objects, always use `np.ndarray(..., buffer=memmap_data, offset=...)` and cache `np.dtype(dtype).itemsize` rather than slicing and viewing.
