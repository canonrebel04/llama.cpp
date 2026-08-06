## 2024-05-17 - O(N^2) Bottleneck in GGUFWriter Shard Sizing
**Learning:** Checking shard size via `sum(ti.nbytes for ti in self.tensors[-1].values())` per added tensor creates an $O(N^2)$ bottleneck when appending many tensors to a single shard.
**Action:** When calculating cumulative properties of iteratively appended collections, always maintain a running total (e.g. `self.tensors_nbytes`) to reduce complexity from $O(N^2)$ to $O(N)$.
