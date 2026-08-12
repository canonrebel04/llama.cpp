## 2024-06-25 - Caching np.dtype and np.empty.itemsize in GGUFReader
**Learning:** In `GGUFReader._get()`, the codebase creates a new empty NumPy array on every call just to get its `itemsize`, e.g., `itemsize = int(np.empty([], dtype=dtype).itemsize)`. Since `_get` is called per metadata field (and twice for arrays), this overhead adds up significantly for models with many metadata fields.
**Action:** Cache the `itemsize` and `np.dtype` by `dtype` and `override_order` to prevent repeated object allocation and evaluation overhead during reads.
