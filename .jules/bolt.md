## 2024-05-15 - Numpy Array Creation Overhead
**Learning:** Using list comprehensions inside `np.array([...])` during frequent array operations (like tensor quantization unpacking) creates a measurable performance bottleneck due to python list allocation and subsequent numpy conversion.
**Action:** Replace `np.array([i for i in range(N)])` with `np.arange(N)`. For step functions like `[i for i in range(0, N, S)]` use `np.arange(0, N, S)`. For scaled sequences like `[2 * i for i in range(N)]` use `np.arange(0, 2*N, 2)`.
