#pragma once

// mmvq dot product for TQ3_4S (turbo-tan, charpdev; adapted). Weights stay in the WHT-rotated
// domain: the q8_1 activations were rotated by quantize_q8_1<true>, so the dot is a plain
// sum of centroid*scale*q8 over the 32 values of one block. Each block is decoded once per
// call (scales + 3-bit indices), mirroring the decode-once structure of turbo-tan 920e5b3be.

#include "common.cuh"

#define VDR_TQ3_4S_Q8_1_MMVQ 1

static __device__ __forceinline__ float tq3_4s_scale_dev(const uint8_t b) {
    return b == 0 ? 0.0f : ldexpf(1.0f + (float)(b & 31) / 32.0f, (b >> 5) - 9);
}

static __device__ __forceinline__ float vec_dot_tq3_4s_q8_1(
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    GGML_UNUSED(iqs);
    // same values as TQ3_0_CENTROIDS / TQ3_CENTROIDS_WEIGHT
    constexpr float c[8] = {
        -1.996684f, -1.291398f, -0.740341f, -0.247508f,
         0.230106f,  0.725222f,  1.277503f,  1.988943f };
    const block_tq3_4s * x = (const block_tq3_4s *) vbq + kbx;
    float sum = 0.0f;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        const uint8_t * qp = x->qs + 3*g;
        const uint32_t bits = qp[0] | (qp[1] << 8) | (qp[2] << 16);
        const int8_t * q8 = bq8_1->qs + 8*g;
        float gs = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            gs += c[(bits >> (3*j)) & 7] * (float) q8[j];
        }
        sum += gs * tq3_4s_scale_dev(x->d[g]);
    }
    return sum * __low2float(bq8_1->ds);
}

// Decode-once form for multi-column mmvq (adapted from turbo-tan 920e5b3be, charpdev): the
// weight unpack must not be repeated per activation column.
struct tq3_4s_decoded {
    float w[32];
};

static __device__ __forceinline__ tq3_4s_decoded tq3_4s_decode(const void * __restrict__ vbq, const int & kbx) {
    constexpr float c[8] = {
        -1.996684f, -1.291398f, -0.740341f, -0.247508f,
         0.230106f,  0.725222f,  1.277503f,  1.988943f };
    const block_tq3_4s * x = (const block_tq3_4s *) vbq + kbx;
    tq3_4s_decoded r;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        const uint8_t * qp = x->qs + 3*g;
        const uint32_t bits = qp[0] | (qp[1] << 8) | (qp[2] << 16);
        const float d = tq3_4s_scale_dev(x->d[g]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            r.w[8*g + j] = c[(bits >> (3*j)) & 7] * d;
        }
    }
    return r;
}

static __device__ __forceinline__ float tq3_4s_dot_decoded_q8_1(const tq3_4s_decoded & w, const block_q8_1 * __restrict__ bq8_1) {
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        sum += w.w[j] * (float) bq8_1->qs[j];
    }
    return sum * __low2float(bq8_1->ds);
}
