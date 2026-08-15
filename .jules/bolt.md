## 2025-03-05 - Avoid allocation overhead for array itemsize
**Learning:** `gguf-py` relies heavily on fast metadata reading and is sensitive to loop overhead. Allocating arrays simply to compute the `itemsize` (e.g., via `int(np.empty([], dtype=dtype).itemsize)`) causes a non-negligible performance overhead in tight loops where data offsets and layouts are being calculated.
**Action:** When computing item sizes or offsets dynamically, use `int(np.dtype(dtype).itemsize)` instead to avoid creating intermediary objects entirely.
