## 2024-05-24 - Avoid np.empty for getting dtype itemsize
**Learning:** `np.empty([], dtype=dtype).itemsize` creates a 0-d array just to get the `itemsize`. This incurs the overhead of NumPy array allocation which is significant when called in a tight loop.
**Action:** Use `np.dtype(dtype).itemsize` directly, and reuse the `np.dtype(dtype)` object when performing views using `.view(dtype=dt)`.
