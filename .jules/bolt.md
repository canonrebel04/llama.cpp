## 2024-05-24 - GGUF Reader Array Bottleneck
**Learning:** Parsing GGUF array metadata fields in pure python with element-wise parsing loops is a major bottleneck for large arrays (e.g. 1M elements take >27s). Since numpy memmaps the entire file, we can drastically optimize reading flat scalar arrays by directly viewing them as numpy arrays.
**Action:** When implementing binary format readers in Python, always try to read arrays of primitive types as single numpy arrays rather than looping in pure python.
