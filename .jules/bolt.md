## 2025-02-27 - Fast rounding for quantization
**Learning:** `np_roundf` custom implementation using `abs`, `floor`, and `sign` takes significantly longer than necessary.
**Action:** Replace it with `np.trunc(n + np.copysign(0.5, n))` to get an exact match for "round half away from zero" (which is C's `roundf()` behavior) but roughly 4x faster since it leverages NumPy's vectorized operations more directly.
