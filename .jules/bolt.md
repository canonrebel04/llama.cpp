## 2024-11-20 - Fast Array Products
**Learning:** Using `np.prod` on small dimensions (e.g. tuples/arrays of length 1-5 like tensor shapes) introduces significant overhead compared to Python's built-in `math.prod`. Testing showed `math.prod` is around 2x faster for small inputs because it avoids numpy's array initialization and internal conversion overhead.
**Action:** When calculating the total elements from small dimension tuples/arrays, use `math.prod(dims.tolist())` instead of `int(np.prod(dims))`.
