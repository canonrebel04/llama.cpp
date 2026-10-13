#pragma once
#include "common.cuh"
#include "convert.cuh"

// Row-aware dequantizers for IQ1_KT..IQ4_KT (contiguous rows, ne00 % QK_K == 0).
template<typename dst_t>
to_t_nc_cuda_t<dst_t> ggml_cuda_get_dequantize_kt_nc(ggml_type type);
