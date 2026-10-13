#pragma once

// mmvq dot products for the ik_llama.cpp IQ_K types (IQ2_K .. IQ6_K).
// Ported/adapted from ik_llama.cpp (Iwan Kawrakow, MIT). Each call handles one 32-value
// chunk (iqs = 0..7) of a QK_K super-block against one q8_1 block, decoding values with the
// element-wise decoders shared with the dequantize path (dequantize.cuh).

#include "dequantize.cuh"

#define VDR_IQK_Q8_1_MMVQ 1

template <typename block_t, float (*elem)(const block_t *, int)>
static __device__ __forceinline__ float vec_dot_iqk_chunk_q8_1(
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_t * x = (const block_t *) vbq + kbx;
    const block_q8_1 * y = bq8_1 + iqs;
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        sum += elem(x, 32*iqs + j) * (float) y->qs[j];
    }
    return sum * __low2float(y->ds);
}

static __device__ __forceinline__ float vec_dot_iq2_k_q8_1(
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    return vec_dot_iqk_chunk_q8_1<block_iq2_k, iqk_iq2_k_elem>(vbq, bq8_1, kbx, iqs);
}

static __device__ __forceinline__ float vec_dot_iq3_k_q8_1(
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    return vec_dot_iqk_chunk_q8_1<block_iq3_k, iqk_iq3_k_elem>(vbq, bq8_1, kbx, iqs);
}

static __device__ __forceinline__ float vec_dot_iq4_k_q8_1(
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    return vec_dot_iqk_chunk_q8_1<block_iq4_k, iqk_iq4_k_elem>(vbq, bq8_1, kbx, iqs);
}

static __device__ __forceinline__ float vec_dot_iq5_k_q8_1(
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    return vec_dot_iqk_chunk_q8_1<block_iq5_k, iqk_iq5_k_elem>(vbq, bq8_1, kbx, iqs);
}

static __device__ __forceinline__ float vec_dot_iq6_k_q8_1(
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    return vec_dot_iqk_chunk_q8_1<block_iq6_k, iqk_iq6_k_elem>(vbq, bq8_1, kbx, iqs);
}
