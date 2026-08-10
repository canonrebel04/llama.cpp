## 2026-08-10 - GGUF numpy view creations in hot loop
**Learning:** `np.empty([], dtype=dtype).itemsize` creates significant overhead in hot loops compared to `np.dtype(dtype).itemsize`. Additionally, repeated redundant `view()` calls (e.g. slicing with exact size or applying the same byteorder) on ndarrays add measurable overhead in critical parsing loops like `GGUFReader._get`.
**Action:** When working with numpy in tight loops, avoid creating intermediate array objects just to read metadata. Inspect array `.view()` chains for redundancy and short-circuit them when possible.
