// CUDA dequantization for the ik_llama.cpp trellis types IQ1_KT..IQ4_KT.
// Ported from ikawrakow/ik_llama.cpp ggml/src/ggml-cuda/convert.cu (Iwan Kawrakow, MIT).
// Rows carry a float scale in front of the blocks (row meta), so the kernels are
// row-aware; rows must be contiguous and a multiple of QK_K (no tail support).
#include "dequant-kt.cuh"

static __device__ __forceinline__ int trellis_next_int(uint32_t & val) {
    constexpr uint32_t ka = 0xCBAC1FED;
    val = ka*val;
    return ggml_cuda_dp4a(val & 0x3f3f3f3f, 0x01010101, -126);
}

template<typename dst_t>
static __global__ void k_dequantize_iq1_kt(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t n_per_row, int64_t row_size) {
    const int64_t ii  = blockIdx.x;
    const int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    const float scale = *(const float *)cx;
    const block_iq1_kt * x = (const block_iq1_kt *)(cx + sizeof(float));
    const int64_t i = ii - (row*n_per_row)/QK_K;
    const int ib = threadIdx.x; // 0...31
    dst_t * y = yy + ii*QK_K + 8*ib;
    uint32_t idx = (x[i].ql[ib] | ((x[i].qh[ib%16] << (8 - 4*(ib/16))) & 0xf00) | ((x[i].sh[ib/4] << (8 - (ib%4))) & 0x1000)) + 4096;
    const float dl = scale * iq4k_values[x[i].sh[ib/4] & 0xf];
    for (int j = 0; j < 8; ++j) {
        y[j] = ggml_cuda_cast<dst_t>(dl * trellis_next_int(idx));
    }
}

template<typename dst_t>
static __global__ void k_dequantize_iq2_kt(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t n_per_row, int64_t row_size) {
    const int64_t ii  = blockIdx.x;
    const int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    const float scale = *(const float *)cx;
    const block_iq2_kt * x = (const block_iq2_kt *)(cx + sizeof(float));
    const int64_t i = ii - (row*n_per_row)/QK_K;
    const int ib = threadIdx.x;
    dst_t * y = yy + ii*QK_K + 8*ib;
    const uint16_t * ql = (const uint16_t *)x[i].ql;
    uint32_t idx = ql[ib] + 4096;
    const float dl = scale * iq4k_values[((x[i].scales[(ib/4)%4] >> 4*(ib/16)) & 0xf)] * 1.05f;
    for (int j = 0; j < 8; ++j) {
        y[j] = ggml_cuda_cast<dst_t>(dl * trellis_next_int(idx));
    }
}

template<typename dst_t>
static __global__ void k_dequantize_iq3_kt(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t n_per_row, int64_t row_size) {
    const int64_t ii  = blockIdx.x;
    const int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    const float scale = *(const float *)cx;
    const block_iq3_kt * x = (const block_iq3_kt *)(cx + sizeof(float));
    const int64_t i = ii - (row*n_per_row)/QK_K;
    const int ib = threadIdx.x;
    dst_t * y = yy + ii*QK_K + 8*ib;
    const uint16_t * ql = (const uint16_t *)x[i].ql;
    uint32_t idx = ql[ib] + 4096;
    const float dl = scale * ((x[i].scales[(ib/4)%4] >> 4*(ib/16)) & 0xf) * 1.01f;
    const uint8_t mask = 1 << (ib/4);
    for (int j = 0; j < 8; ++j) {
        y[j] = ggml_cuda_cast<dst_t>(dl * abs(trellis_next_int(idx)) * (x[i].qh[(8*ib+j)%32] & mask ? -1.f : 1.f));
    }
}

template<typename dst_t>
static __global__ void k_dequantize_iq4_kt(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t n_per_row, int64_t row_size) {
    constexpr int kNumGroups = 64;
    const int64_t ii  = blockIdx.x;
    const int64_t row = (QK_K * ii) / n_per_row;
    const float * dptr = (const float *)((const char *)vx + row * row_size);
    const float scale = dptr[0];
    const block_iq4_kt * x = (const block_iq4_kt *)(dptr + 1);
    const int64_t i = ii - (row*n_per_row)/QK_K;
    const int ib = threadIdx.x;
    dst_t * y = yy + ii*QK_K + 8*ib;
    const int ib32 = ib/4;
    const int ig = ib%4;
    const int jj = ib32*8 + 2*ig;
    const uint32_t * shb = x[i].qs;
    const uint8_t * ql = (const uint8_t *)(shb + 8);
    const uint8_t * qh = ql + kNumGroups;
    const uint32_t qh1 = (qh[(jj+0)%(kNumGroups/2)] << (8 - 4*((jj+0)/(kNumGroups/2)))) & 0xf00;
    const uint32_t qh2 = (qh[(jj+1)%(kNumGroups/2)] << (8 - 4*((jj+1)/(kNumGroups/2)))) & 0xf00;
    const uint32_t sh = shb[ib32];
    const uint32_t offset = sh & 1 ? 4096 + 32768 : 4096;
    uint32_t idx1 = ql[jj+0] + qh1 + (((sh >> (8 + 6*ig+0)) & 7) << 12) + offset;
    uint32_t idx2 = ql[jj+1] + qh2 + (((sh >> (8 + 6*ig+3)) & 7) << 12) + offset;
    const int ls = ((sh & 0xff) >> 1) - 64;
    const float dl = scale * ls;
    for (int j = 0; j < 4; ++j) {
        y[j+0] = ggml_cuda_cast<dst_t>(dl * trellis_next_int(idx1));
        y[j+4] = ggml_cuda_cast<dst_t>(dl * trellis_next_int(idx2));
    }
}

template<ggml_type type, typename dst_t>
static void dequantize_kt_nc_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    GGML_UNUSED(s01); GGML_UNUSED(s02); GGML_UNUSED(s03); // rows must be contiguous (row meta)
    GGML_ASSERT(ne00 % QK_K == 0);
    const int64_t nrows = ne01*ne02*ne03;
    const int64_t nb = nrows*(ne00/QK_K);
    const int64_t row_size = ggml_row_size(type, ne00);
    GGML_ASSERT(nb < (int64_t(1) << 31));
    if constexpr (type == GGML_TYPE_IQ1_KT) {
        k_dequantize_iq1_kt<<<nb, 32, 0, stream>>>(vx, y, ne00, row_size);
    } else if constexpr (type == GGML_TYPE_IQ2_KT) {
        k_dequantize_iq2_kt<<<nb, 32, 0, stream>>>(vx, y, ne00, row_size);
    } else if constexpr (type == GGML_TYPE_IQ3_KT) {
        k_dequantize_iq3_kt<<<nb, 32, 0, stream>>>(vx, y, ne00, row_size);
    } else {
        k_dequantize_iq4_kt<<<nb, 32, 0, stream>>>(vx, y, ne00, row_size);
    }
}

template<typename dst_t>
to_t_nc_cuda_t<dst_t> ggml_cuda_get_dequantize_kt_nc(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_KT: return dequantize_kt_nc_cuda<GGML_TYPE_IQ1_KT, dst_t>;
        case GGML_TYPE_IQ2_KT: return dequantize_kt_nc_cuda<GGML_TYPE_IQ2_KT, dst_t>;
        case GGML_TYPE_IQ3_KT: return dequantize_kt_nc_cuda<GGML_TYPE_IQ3_KT, dst_t>;
        case GGML_TYPE_IQ4_KT: return dequantize_kt_nc_cuda<GGML_TYPE_IQ4_KT, dst_t>;
        default:               return nullptr;
    }
}

template to_t_nc_cuda_t<float>       ggml_cuda_get_dequantize_kt_nc<float>(ggml_type);
template to_t_nc_cuda_t<half>        ggml_cuda_get_dequantize_kt_nc<half>(ggml_type);
template to_t_nc_cuda_t<nv_bfloat16> ggml_cuda_get_dequantize_kt_nc<nv_bfloat16>(ggml_type);
