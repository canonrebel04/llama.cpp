## 2024-08-11 - Fast Array Packing with struct
**Learning:** Writing elements of a primitive array one-by-one via individual `struct.pack` calls and concatenating to a `bytearray` is incredibly slow in Python. Passing the unpacked list (`*val`) to a single `struct.pack` format specifier (e.g. `<1000i`) reduces serialization time by roughly ~10x to ~30x for large arrays.
**Action:** Whenever serializing large lists of numeric primitives to bytes using `struct`, construct a length-prefixed format string and bulk-pack them instead of iterating in a loop.
