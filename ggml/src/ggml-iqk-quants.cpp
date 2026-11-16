// IQ2_K .. IQ6_K quantization, dequantization and scalar dot products.
//
// Ported from ik_llama.cpp (ggml/src/iqk/iqk_quantize.cpp)
// Copyright (C) 2024 Iwan Kawrakow, MIT license, SPDX-License-Identifier: MIT
// https://github.com/ikawrakow/ik_llama.cpp
//
// Only the scalar reference paths are carried here (the iqk_mul_mat SIMD GEMM
// and the _r4 interleaved variants are not ported); see docs/fork-types.md.
// The vec_dot functions take Q8_K activations like ik_llama.cpp's fallback.

#include "ggml-quants.h"
#include "ggml-impl.h"
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml-iqk-quants.h"

#include <vector>
#include <utility>
#include <cstdint>
#include <cmath>
#include <cassert>
#include <array>
#include <algorithm>
#include <cstring>

// ik_llama.cpp threads an opaque per-call user pointer through its quantizers;
// it is unused by the IQ_K row quantizers, so an incomplete type is enough here.
struct quantize_user_data;

// The AVX2 search helpers of ik_llama.cpp are not carried; force the scalar
// paths so this file builds identically with any -march.
#ifdef __AVX2__
#undef __AVX2__
#endif

namespace {
// ik_llama.cpp per-type quantization fudge factors (defaults from ik's ggml.c)
inline float ggml_get_quantize_fudge_factor(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_K: return 1.03f;
        case GGML_TYPE_IQ3_K: return 1.01f;
        default:              return 1.0f;
    }
}
const int8_t iq3nl_index[111] = {
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  8,  8,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  9,
  9,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 10, 10,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 11, 11,  4,  4,  4,  4,
  4,  4,  4,  4,  4,  4, 12,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5, 13, 13,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
  6,  6,  6,  6, 14, 14,  7,  7,  7,  7,  7,  7,  7,  7, 7
};
inline int best_index_iq3nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 111) return ix < 0 ? 0 : 7;
    ix = iq3nl_index[ix];
    return ix < 8 ? ix : x - values[ix-8] < values[ix-7] - x ? ix-8 : ix-7;
}
} // namespace


inline int nearest_int(float fval) {
    assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

typedef void (*quantize_func_t)(const float * src, void * qdata, int n_per_row, const float * imatrix, const quantize_user_data * user_data);

struct QHelper {
    QHelper(const float * imatrix, const quantize_user_data * user_data, int n_per_row, int block_size) :
        m_imatrix(imatrix), m_user_data(user_data), m_n_per_row(n_per_row), m_block_size(block_size) {
        if (m_imatrix) {
            m_weight.resize(m_n_per_row);
        }
    }
    const float * row_weights(const float * x) {
        constexpr float kEps  = 1e-9f;
        constexpr float kEps2 = kEps*kEps;
        if (!m_imatrix) return m_imatrix;
        int nblock = m_n_per_row / m_block_size;
        for (int ib = 0; ib < nblock; ++ib) {
            auto wb_in = m_imatrix + ib*m_block_size;
            auto xb = x + ib*m_block_size;
            auto wb = m_weight.data() + ib*m_block_size;
            float sumw2 = 0, sumx2 = 0, sumwx = 0;
            for (int j = 0; j < m_block_size; ++j) {
                wb[j] = wb_in[j];
                sumw2 += wb[j]*wb[j];
                sumx2 += xb[j]*xb[j];
                sumwx += wb[j]*std::abs(xb[j]);
            }
            if (sumw2 > m_block_size*kEps2 && sumx2 > m_block_size*kEps2 && sumwx > m_block_size*kEps2) continue;
            for (int j = 0; j < m_block_size; ++j) {
                wb[j] = kEps;
            }
        }
        return m_weight.data();
    }
    template <typename Func>
    void quantize(int nrows, const float * src, void * dst, int row_size, const Func& qfunc) {
        auto cdst = (char *)dst;
        for (int row = 0; row < nrows; ++row) {
            auto weights = row_weights(src);
            qfunc(src, cdst, m_n_per_row, weights, m_user_data);
            src  += m_n_per_row;
            cdst += row_size;
        }
    }
private:
    const float * m_imatrix;
    const quantize_user_data * m_user_data;
    const int     m_n_per_row;
    const int     m_block_size;
    std::vector<float> m_weight;
};
static float make_qx_quants(int n, int nmax, const float * x, int8_t * L, const float * qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (!amax) { // all zero
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.f;
    }
    float iscale = -nmax / max;
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = std::max(-nmax, std::min(nmax-1, l));
        L[i] = l + nmax;
        sumlx += qw[i]*x[i]*l;
        suml2 += qw[i]*l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.0f;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) continue;
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = std::max(-nmax, std::min(nmax-1, l));
            sumlx += qw[i]*x[i]*l;
            suml2 += qw[i]*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + std::max(-nmax, std::min(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
} // namespace
// ============================================== iq2_K
//

namespace {

inline int best_index_iq2nl(const int8_t * values, float x) {
    int idx = x < values[1] ? 0 : x > values[2] ? 2 : 1;
    return x - values[idx] < values[idx+1] - x ? idx : idx + 1;
}

void quantize_row_iq2_k_impl(const float * x, void * vy, int n_per_row, const float * quant_weights,
        [[maybe_unused]] const quantize_user_data * user_data) {

    constexpr int kBlockSize = 16;

    block_iq2_k * y = (block_iq2_k *)vy;

    float scales[QK_K/kBlockSize];
    float weight[kBlockSize];
    float sumx[kBlockSize+1], sumw[kBlockSize+1];
    float sw[QK_K/kBlockSize];
    int8_t Ls[QK_K/kBlockSize];

    std::array<std::pair<float,int>, kBlockSize> pairs;

    const int8_t * shifted_values = iq2nl_values + 4;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ2_K);

    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq2_k));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;

        uint16_t extra = 0;

        float max_abs_scale = 0;

        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            const float * xb = xbl + kBlockSize*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            sw[ib] = 0;
            float amax = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                sw[ib] += weight[j];
                pairs[j] = {xb[j], j};
                float ax = std::abs(xb[j]);
                amax = std::max(amax, ax);
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            std::sort(pairs.begin(), pairs.end());
            sumx[0] = sumw[0] = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                int jj = pairs[j].second;
                sumw[j+1] = sumw[j] + weight[jj];
                sumx[j+1] = sumx[j] + weight[jj]*xb[jj];
            }
            float best = 0, d = 0;
            bool is_shifted = false;
            float sumqx, sumq2;
            for (int i1 = 0; i1 < kBlockSize; ++i1) {
                for (int i2 = i1; i2 < kBlockSize; ++i2) {
                    for (int i3 = i2; i3 < kBlockSize; ++i3) {
                        sumqx = (sumx[i1] - sumx[ 0])*iq2nl_values[0] + (sumx[i2] - sumx[i1])*iq2nl_values[1]
                              + (sumx[i3] - sumx[i2])*iq2nl_values[2] + (sumx[kBlockSize] - sumx[i3])*iq2nl_values[3];
                        sumq2 = (sumw[i1] - sumw[ 0])*iq2nl_values[0]*iq2nl_values[0] + (sumw[i2] - sumw[i1])*iq2nl_values[1]*iq2nl_values[1]
                              + (sumw[i3] - sumw[i2])*iq2nl_values[2]*iq2nl_values[2] + (sumw[kBlockSize] - sumw[i3])*iq2nl_values[3]*iq2nl_values[3];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = false;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*shifted_values[0] + (sumx[i2] - sumx[i1])*shifted_values[1]
                              + (sumx[i3] - sumx[i2])*shifted_values[2] + (sumx[kBlockSize] - sumx[i3])*shifted_values[3];
                        sumq2 = (sumw[i1] - sumw[ 0])*shifted_values[0]*shifted_values[0] + (sumw[i2] - sumw[i1])*shifted_values[1]*shifted_values[1]
                              + (sumw[i3] - sumw[i2])*shifted_values[2]*shifted_values[2] + (sumw[kBlockSize] - sumw[i3])*shifted_values[3]*shifted_values[3];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = true;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*iq2nl_values[3] + (sumx[i2] - sumx[i1])*iq2nl_values[2]
                              + (sumx[i3] - sumx[i2])*iq2nl_values[1] + (sumx[kBlockSize] - sumx[i3])*iq2nl_values[0];
                        sumq2 = (sumw[i1] - sumw[ 0])*iq2nl_values[3]*iq2nl_values[3] + (sumw[i2] - sumw[i1])*iq2nl_values[2]*iq2nl_values[2]
                              + (sumw[i3] - sumw[i2])*iq2nl_values[1]*iq2nl_values[1] + (sumw[kBlockSize] - sumw[i3])*iq2nl_values[0]*iq2nl_values[0];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = false;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*shifted_values[3] + (sumx[i2] - sumx[i1])*shifted_values[2]
                              + (sumx[i3] - sumx[i2])*shifted_values[1] + (sumx[kBlockSize] - sumx[i3])*shifted_values[0];
                        sumq2 = (sumw[i1] - sumw[ 0])*shifted_values[3]*shifted_values[3] + (sumw[i2] - sumw[i1])*shifted_values[2]*shifted_values[2]
                              + (sumw[i3] - sumw[i2])*shifted_values[1]*shifted_values[1] + (sumw[kBlockSize] - sumw[i3])*shifted_values[0]*shifted_values[0];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = true;
                        }
                    }
                }
            }
            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);

            float abs_scale = fabsf(scales[ib]);
            max_abs_scale = std::max(max_abs_scale, abs_scale);
        }

        if (!max_abs_scale) continue;
        float d = make_qx_quants(QK_K/kBlockSize, 8, scales, Ls, sw);
        if (!d) continue;

        //float d = -max_scale/8;
        y[ibl].extra = extra;
        float id = 1/d;

        float sumqx = 0, sumq2 = 0;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            int ls = nearest_int(id*scales[ib]);
            ls = std::max(-8, std::min(7, ls));
            y[ibl].scales[ib/2] |= ((ls + 8) << 4*(ib%2));
            float dl = d * ls;
            if (dl) {
                const int8_t * block_values = y[ibl].extra & (1 << ib) ? shifted_values : iq2nl_values;
                const float * xb = xbl + kBlockSize*ib;
                if (quant_weights) {
                    const float * qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
                } else {
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
                }
                float idl = 1/dl;
                int ib32 = ib/2;
                int offset = 16*(ib%2);
                uint8_t * qs = y[ibl].qs + 32*(ib32/4) + offset;
                for (int j = 0; j < 16; ++j) {
                    const float al = idl*xb[j];
                    int ibest = best_index_iq2nl(block_values, al);
                    qs[j] |= (ibest << 2*(ib32%4));
                    float w = weight[j];
                    float q = block_values[ibest]*ls;
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
            }
        }
        y[ibl].d = GGML_FP32_TO_FP16(fudge*(sumq2 > 0 ? sumqx/sumq2 : d));

    }
}
}

void quantize_row_iq2_k_ref(const float * x, void * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq2_k(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq2_k(const float * x, void * vy, int64_t k) {
    assert(k % QK_K == 0);
    quantize_row_iq2_k_ref(x, vy, k);
}

size_t quantize_iq2_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    QHelper helper(imatrix, nullptr, n_per_row, 16);
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_K, n_per_row);
    helper.quantize(nrows, src, dst, row_size, quantize_row_iq2_k_impl);
    return nrows * row_size;
}

void dequantize_row_iq2_k(const void * vx_, float * y, int64_t k) {
    const block_iq2_k * x = (const block_iq2_k *)vx_;
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;

        uint16_t extra = x[i].extra;

        int shift = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            float dl1 = d * ((x[i].scales[ib32] & 0xf) - 8);
            float dl2 = d * ((x[i].scales[ib32] >>  4) - 8);
            const int8_t * values1 = extra & 1 ? iq2nl_values + 4 : iq2nl_values;
            const int8_t * values2 = extra & 2 ? iq2nl_values + 4 : iq2nl_values;
            extra >>= 2;
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl1 * values1[(qs[j+ 0] >> shift) & 3];
                y[j+16] = dl2 * values2[(qs[j+16] >> shift) & 3];
            }
            y += 32;
            shift += 2;
            if (shift == 8) { qs += 32; shift = 0; }
        }

    }

}


namespace {
// Scalar fallback used where ik_llama.cpp only has an iqk_mul_mat path:
// dequantize one super-block at a time and dot it with the Q8_K activations.
template <typename Block>
float iqk_vec_dot_dequant_q8_K(int n, const void * vx, const void * vy,
        void (*dequant)(const void *, float *, int64_t)) {
    const int nb = n / QK_K;
    const Block      * x = (const Block *)vx;
    const block_q8_K * y = (const block_q8_K *)vy;
    float tmp[QK_K];
    double sum = 0;
    for (int i = 0; i < nb; ++i) {
        dequant(x + i, tmp, QK_K);
        float sumb = 0;
        for (int j = 0; j < QK_K; ++j) sumb += tmp[j] * y[i].qs[j];
        sum += (double)sumb * y[i].d;
    }
    return (float)sum;
}
} // namespace

void vec_dot_iq2_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    *s = iqk_vec_dot_dequant_q8_K<block_iq2_k>(n, vx, vy, dequantize_row_iq2_k);

}
// ============================================== iq3_k
//
namespace {

static void quantize_row_iq3_k_impl(const float * x, void * vy, int n_per_row, const float * quant_weights,
        [[maybe_unused]] const quantize_user_data * user_data) {

    constexpr int ntry = 3;

#ifdef __AVX2__
    float f_values[16];
    for (int j = 0; j < 16; ++j) f_values[j] = iq3nl_values[j];
#endif

    block_iq3_k * y = (block_iq3_k *)vy;

    float scales[QK_K/16];
    float weight[16];

    const int8_t * shifted_values = iq3nl_values + 8;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ3_K);

    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq3_k));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;

        uint16_t extra = 0;

        float max_abs_scale = 0;

#ifdef __AVX2__
        for (int ib = 0; ib < QK_K/16; ++ib) {
            auto xb = xbl + ib*16;
            auto [d, _, is_shifted] = find_best_scale_avx2<16, 127>(8, iq3nl_values, f_values, k_simple_index_iq3nl, xb,
                    quant_weights ? quant_weights + QK_K*ibl + 16*ib : nullptr, sigma2, ntry, true, 2.0f);

            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);
            float abs_scale = fabsf(scales[ib]);
            max_abs_scale = std::max(max_abs_scale, abs_scale);
        }
#else
        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float * xb = xbl + 16*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*16;
                for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < 16; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/iq3nl_values[0] : max/iq3nl_values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            float best = 0;
            for (int j = 0; j < 16; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq3nl(iq3nl_values, al);
                float q = iq3nl_values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq3nl(iq3nl_values, -al);
                q = iq3nl_values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            if (sumq2_p > 0) {
                d = sumqx_p/sumq2_p;
                best = d*sumqx_p;
            }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            bool is_shifted = false;
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (2*itry + iq3nl_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq3nl(iq3nl_values, al);
                    float q = iq3nl_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq3nl(iq3nl_values, -al);
                    q = iq3nl_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
                }
                id = (2*itry + shifted_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq3nl(shifted_values, al);
                    float q = shifted_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq3nl(shifted_values, -al);
                    q = shifted_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = true;
                }
            }
            if (!d) {
                scales[ib] = 0; continue;
            }

            scales[ib] = d;

            if (is_shifted) extra |= (1 << ib);

            float abs_scale = fabsf(scales[ib]);
            max_abs_scale = MAX(max_abs_scale, abs_scale);
        }
#endif

        if (!max_abs_scale) continue;

        float d = max_abs_scale/31;
        y[ibl].extra = extra;
        float id = 1/d;

        float sumqx = 0, sumq2 = 0;
        for (int ib = 0; ib < QK_K/16; ++ib) {
            int ls = nearest_int(0.5f*(id*fabsf(scales[ib])-1));
            ls = MAX(0, MIN(15, ls));
            y[ibl].scales_l[ib/2] |= (ls << 4*(ib%2));
            if (scales[ib] < 0) y[ibl].scales_h |= (1 << ib);
            ls = (2*ls + 1) * (scales[ib] < 0 ? -1 : 1);
            float dl = d * ls;
            if (dl) {
                const int8_t * block_values = y[ibl].extra & (1 << ib) ? shifted_values : iq3nl_values;
                const float * xb = xbl + 16*ib;
                if (quant_weights) {
                    const float * qw = quant_weights + ibl*QK_K + ib*16;
                    for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
                } else {
                    for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
                }
                float idl = 1/dl;
                int ib32 = ib/2;
                int offset = 16*(ib%2);
                uint8_t * qs = y[ibl].qs + 32*(ib32/4) + offset;
                uint8_t * qh = y[ibl].qh + 32*(ib32/8) + offset;
                for (int j = 0; j < 16; ++j) {
                    const float al = idl*xb[j];
                    int ibest = best_index_iq3nl(block_values, al);
                    qs[j] |= ((ibest &  3) << 2*(ib32%4));
                    qh[j] |= ((ibest >> 2) << (ib32%8));
                    float w = weight[j];
                    float q = block_values[ibest]*ls;
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
            }
        }
        y[ibl].d = GGML_FP32_TO_FP16(fudge*(sumq2 > 0 ? sumqx/sumq2 : d));

    }
}

}

void quantize_row_iq3_k_ref(const float * x, void * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq3_k(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq3_k(const float * x, void * vy, int64_t k) {
    assert(k % QK_K == 0);
    quantize_row_iq3_k_ref(x, vy, k);
}

size_t quantize_iq3_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    QHelper helper(imatrix, nullptr, n_per_row, 16);
    auto row_size = ggml_row_size(GGML_TYPE_IQ3_K, n_per_row);
    helper.quantize(nrows, src, dst, row_size, quantize_row_iq3_k_impl);
    return nrows * row_size;
}

void dequantize_row_iq3_k(const void * vx_, float * y, int64_t k) {
    const block_iq3_k * x = (const block_iq3_k *)vx_;
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;

        uint16_t sh = x[i].scales_h;
        uint16_t extra = x[i].extra;

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            float dl1 = d * ((2*(x[i].scales_l[ib32] & 0xf) + 1) * ((sh & 1) ? -1 : 1));
            float dl2 = d * ((2*(x[i].scales_l[ib32] >>  4) + 1) * ((sh & 2) ? -1 : 1));
            sh >>= 2;
            const int8_t * values1 = extra & 1 ? iq3nl_values + 8 : iq3nl_values;
            const int8_t * values2 = extra & 2 ? iq3nl_values + 8 : iq3nl_values;
            extra >>= 2;
            int shift_l = 2*(ib32%4);
            int shift_h = ib32%8;
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl1 * values1[((qs[j+ 0] >> shift_l) & 3) | (((qh[j+ 0] >> shift_h) & 1) << 2)];
                y[j+16] = dl2 * values2[((qs[j+16] >> shift_l) & 3) | (((qh[j+16] >> shift_h) & 1) << 2)];
            }
            y += 32;
            if (shift_l == 6) qs += 32;
        }

    }
}

void vec_dot_iq3_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    *s = iqk_vec_dot_dequant_q8_K<block_iq3_k>(n, vx, vy, dequantize_row_iq3_k);
}
// ============================================== iq4_K
//
void dequantize_row_iq4_k(const void * vx_, float * y, int64_t k) {
    const block_iq4_k * x = (const block_iq4_k *)vx_;
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const uint8_t * qs = x[i].qs;

        const float d = GGML_FP16_TO_FP32(x[i].d);

        uint16_t extra = x[i].extra;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const uint8_t sh = x[i].scales_h[ib/2] >> 4*(ib%2);
            const float dl1 = d * (((x[i].scales_l[ib] & 0xf) | ((sh << 4) & 0x30)) - 32);
            const float dl2 = d * (((x[i].scales_l[ib] >>  4) | ((sh << 2) & 0x30)) - 32);
            const int8_t * values1 = extra & 1 ? iq4k_values + 16 : iq4k_values;
            const int8_t * values2 = extra & 2 ? iq4k_values + 16 : iq4k_values;
            extra >>= 2;
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl1 * values1[qs[j] & 0xf];
                y[j+16] = dl2 * values2[qs[j] >>  4];
            }
            y  += 32;
            qs += 16;
        }
    }
}

void vec_dot_iq4_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    const int nb = n / QK_K;

    const block_iq4_k * x = (const block_iq4_k *)vx;
    const block_q8_K  * y = (const block_q8_K *)vy;

    float sumf = 0;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const float d4d8 = GGML_FP16_TO_FP32(x[ibl].d) * y[ibl].d;
        uint16_t extra = x[ibl].extra;
        uint32_t h = *((const uint32_t *)x[ibl].scales_h);
        const uint8_t * qs = x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        int32_t sum = 0;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls1 = ((x[ibl].scales_l[ib] & 0xf) | ((h << 4) & 0x30)) - 32;
            const int ls2 = ((x[ibl].scales_l[ib] >>  4) | ((h << 2) & 0x30)) - 32;
            h >>= 4;
            const int8_t * values1 = iq4k_values + 16*(extra & 1);
            const int8_t * values2 = iq4k_values +  8*(extra & 2);
            extra >>= 2;
            int sumi1 = 0, sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * values1[qs[j] & 0xf];
                sumi2 += q8[j+16] * values2[qs[j] >>  4];
            }
            sum += ls1*sumi1 + ls2*sumi2;
            qs += 16;
            q8 += 32;
        }
        sumf += d4d8 * sum;
    }
    *s = sumf;

}


namespace {
const int8_t iq4nl_index[241] = {
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 16, 16,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1, 17, 17,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 18,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
     3,  3,  3,  3,  3,  3, 19,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4, 20,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
     5,  5, 21, 21,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6, 22,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 23, 23,  8,  8,  8,  8,
     8,  8,  8,  8,  8,  8, 24,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 25, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 26, 26,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 27, 27, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 28, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 29, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 30, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15
};
inline int best_index_iq4nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 241) return ix < 0 ? 0 : 15;
    ix = iq4nl_index[ix];
    return ix < 16 ? ix : x - values[ix-16] < values[ix-15] - x ? ix-16 : ix-15;
}
#ifdef __AVX2__
template <int block_size>
inline void best_index_iq4k(float id, const int8_t * values, const float * f_values,
        const __m256 * vx, /*__m256i * vidx,*/ __m256 * vq) {
    static_assert(block_size == 32 || block_size == 16);
    static const int8_t k_simple_index[256] = {
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5,
         5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,10,10,10,10,10,10,10,10,
        10,10,10,10,10,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,12,12,12,12,12,12,12,12,12,12,12,12,
        12,12,12,12,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,14,14,14,14,14,14,14,14,
        14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,
    };

    auto vid = _mm256_set1_ps(id);
    auto flip = _mm256_set1_ps(-0.0f);
    auto val0 = _mm256_set1_epi32(values[0]);
    for (int i = 0; i < block_size/8; ++i) {
        auto vxi_p = _mm256_mul_ps(vid, vx[i]);
        auto vxi_m = _mm256_xor_ps(vxi_p, flip);
        auto vidx_p = _mm256_sub_epi32(_mm256_cvttps_epi32(vxi_p), val0);
        auto vidx_m = _mm256_sub_epi32(_mm256_cvttps_epi32(vxi_m), val0);
        vidx_p = _mm256_min_epi32(_mm256_max_epi32(vidx_p, _mm256_setzero_si256()), _mm256_set1_epi32(240));
        vidx_m = _mm256_min_epi32(_mm256_max_epi32(vidx_m, _mm256_setzero_si256()), _mm256_set1_epi32(240));
        auto tab_p0 = _mm256_i32gather_epi32((const int32_t *)k_simple_index, vidx_p, 1);
        auto tab_m0 = _mm256_i32gather_epi32((const int32_t *)k_simple_index, vidx_m, 1);
        tab_p0 = _mm256_and_si256(tab_p0, _mm256_set1_epi32(0xff));
        tab_m0 = _mm256_and_si256(tab_m0, _mm256_set1_epi32(0xff));
        auto tab_p1 = _mm256_add_epi32(tab_p0, _mm256_set1_epi32(1));
        auto tab_m1 = _mm256_add_epi32(tab_m0, _mm256_set1_epi32(1));
        auto cand0_p = _mm256_castsi256_ps(_mm256_i32gather_epi32((const int32_t *)f_values, tab_p0,  4));
        auto cand1_p = _mm256_castsi256_ps(_mm256_i32gather_epi32((const int32_t *)f_values, tab_p1,  4));
        auto cand0_m = _mm256_castsi256_ps(_mm256_i32gather_epi32((const int32_t *)f_values, tab_m0,  4));
        auto cand1_m = _mm256_castsi256_ps(_mm256_i32gather_epi32((const int32_t *)f_values, tab_m1,  4));
        auto mask_p  = _mm256_castps_si256(_mm256_cmp_ps(_mm256_sub_ps(vxi_p, cand0_p), _mm256_sub_ps(cand1_p, vxi_p), _CMP_LT_OQ));
        auto mask_m  = _mm256_castps_si256(_mm256_cmp_ps(_mm256_sub_ps(vxi_m, cand0_m), _mm256_sub_ps(cand1_m, vxi_m), _CMP_LT_OQ));
        //vidx[2*i+0]  = _mm256_blendv_epi8(tab_p0, tab_p1, mask_p);
        //vidx[2*i+1]  = _mm256_blendv_epi8(tab_m0, tab_m1, mask_m);
        vq[2*i+0]    = _mm256_castsi256_ps(_mm256_blendv_epi8(_mm256_castps_si256(cand1_p), _mm256_castps_si256(cand0_p), mask_p));
        vq[2*i+1]    = _mm256_castsi256_ps(_mm256_blendv_epi8(_mm256_castps_si256(cand1_m), _mm256_castps_si256(cand0_m), mask_m));
    }
}
template <int block_size>
inline bool evaluate_block_iq4k(float id, const int8_t * values, const float * f_values,
        const __m256 * vx, const __m256 * vw, float & d, float & best) {
    __m256  vq[block_size/4];
    //__m256i vidx[block_size/4];
    //best_index_iq4k<block_size>(id, values, f_values, vx, vidx, vq);
    best_index_iq4k<block_size>(id, values, f_values, vx, vq);
    __m256 sumqx_p = _mm256_setzero_ps(), sumq2_p = _mm256_setzero_ps();
    __m256 sumqx_m = _mm256_setzero_ps(), sumq2_m = _mm256_setzero_ps();
    for (int i = 0; i < block_size/8; ++i) {
        auto wp = _mm256_mul_ps(vw[i], vq[2*i+0]);
        auto wm = _mm256_mul_ps(vw[i], vq[2*i+1]);
        sumqx_p = _mm256_fmadd_ps(wp, vx[i],     sumqx_p);
        sumq2_p = _mm256_fmadd_ps(wp, vq[2*i+0], sumq2_p);
        sumqx_m = _mm256_fmadd_ps(wm, vx[i],     sumqx_m);
        sumq2_m = _mm256_fmadd_ps(wm, vq[2*i+1], sumq2_m);
    }
    auto sqx_p = hsum_float_8(sumqx_p);
    auto sq2_p = hsum_float_8(sumq2_p);
    auto sqx_m = hsum_float_8(sumqx_m);
    auto sq2_m = hsum_float_8(sumq2_m);
    bool updated = false;
    if (sq2_p > 0 && sqx_p*sqx_p > sq2_p*best) {
        d = sqx_p/sq2_p; best = d * sqx_p; updated = true;
    }
    if (sq2_m > 0 && sqx_m*sqx_m > sq2_m*best) {
        d = sqx_m/sq2_m; best = d * sqx_m; updated = true;
    }
    return updated;
}
#endif

static void quantize_row_iq4_k_impl_bs16(const float * x,
        block_iq4_k * y,
        float * scales, float * weight, uint8_t * L,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    constexpr int super_block_size = 256;
    constexpr int block_size = 16;

    float sigma2 = 0;
    for (int j = 0; j < super_block_size; ++j) sigma2 += x[j]*x[j];
    sigma2 *= 2.f/super_block_size;

    memset(y, 0, sizeof(block_iq4_k));
    y->d = GGML_FP32_TO_FP16(0.f);

    uint16_t * scales_h = (uint16_t *)y->scales_h;

    const int8_t * shifted_values = values + 16;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_K);

#ifdef __AVX2__
    float f_values[32];
    for (int j = 0; j < 32; ++j) f_values[j] = values[j];
#endif

    float max_scale = 0, amax_scale = 0;
    uint16_t extra = 0;
    for (int ib = 0; ib < super_block_size/block_size; ++ib) {
        const float * xb = x + ib*block_size;
        if (quant_weights) {
            const float * qw = quant_weights + ib*block_size;
            for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        } else {
            for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
        }
        float amax = 0, max = 0;
        for (int j = 0; j < block_size; ++j) {
            float ax = fabsf(xb[j]);
            if (ax > amax) {
                amax = ax; max = xb[j];
            }
        }
        if (amax < 1e-16f) {
            scales[ib] = 0;
            continue;
        }
        float d = ntry > 0 ? -max/values[0] : max/values[0];
        float id = 1/d;
        bool is_shifted = false;
#ifdef __AVX2__
        float best = 0;
        __m256 vx[block_size/8], vw[block_size/8];
        for (int j = 0; j < block_size/8; ++j) vx[j] = _mm256_loadu_ps(xb + 8*j);
        for (int j = 0; j < block_size/8; ++j) vw[j] = _mm256_loadu_ps(weight + 8*j);
        if (!evaluate_block_iq4k<block_size>(id, values, f_values, vx, vw, d, best)) {
            GGML_ABORT("Fatal error");
        }
#else
        float sumqx_p = 0, sumq2_p = 0;
        float sumqx_m = 0, sumq2_m = 0;
        for (int j = 0; j < block_size; ++j) {
            float w = weight[j];
            float al = id*xb[j];
            int l = best_index_iq4nl(values, al);
            float q = values[l];
            sumqx_p += w*q*xb[j];
            sumq2_p += w*q*q;
            l = best_index_iq4nl(values, -al);
            q = values[l];
            sumqx_m += w*q*xb[j];
            sumq2_m += w*q*q;
        }
        d = sumqx_p/sumq2_p;
        float best = d*sumqx_p;
        if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
            d = sumqx_m/sumq2_m; best = d*sumqx_m;
        }
#endif
        for (int itry = -ntry; itry <= ntry; ++itry) {
            id = (itry + values[0])/max;
#ifdef __AVX2__
            if (evaluate_block_iq4k<block_size>(id, values, f_values, vx, vw, d, best)) {
                is_shifted = false;
            }
#else
            sumqx_p = sumq2_p = 0;
            sumqx_m = sumq2_m = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq4nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq4nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
            }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
            }
#endif
            id = (itry + shifted_values[0])/max;
#ifdef __AVX2__
            if (evaluate_block_iq4k<block_size>(id, values+16, f_values+16, vx, vw, d, best)) {
                is_shifted = true;
            }
#else
            sumqx_p = sumq2_p = 0;
            sumqx_m = sumq2_m = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq4nl(shifted_values, al);
                float q = shifted_values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq4nl(shifted_values, -al);
                q = shifted_values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = true;
            }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = true;
            }
#endif
        }
        if (is_shifted) extra |= (1 << ib);
        scales[ib] = d;
        float abs_d = fabsf(d);
        if (abs_d > amax_scale) {
            amax_scale = abs_d; max_scale = d;
        }
    }
    float d = -max_scale/32;
    y->d = GGML_FP32_TO_FP16(fudge*d);
    y->extra = extra;
    float id = d ? 1/d : 0.f;
    float sumqx = 0, sumq2 = 0;
    for (int ib = 0; ib < super_block_size/block_size; ++ib) {
        const int8_t * block_values = extra & (1 << ib) ? shifted_values : values;
        int l = nearest_int(id*scales[ib]);
        l = MAX(-32, MIN(31, l));
        float dl = d * l;
        float idl = dl ? 1/dl : 0.f;
        uint8_t * Lb = L + ib*block_size;
        const float * xb = x + ib*block_size;
        if (quant_weights) {
            const float * qw = quant_weights + ib*block_size;
            for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        } else {
            for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
        }
        for (int j = 0; j < block_size; ++j) {
            Lb[j] = best_index_iq4nl(block_values, idl*xb[j]);
            float w = weight[j];
            float q = block_values[Lb[j]]*l;
            sumqx += w*q*xb[j];
            sumq2 += w*q*q;
        }
        l += 32;
        uint8_t l_l = l & 0xf;
        uint8_t l_h = l >>  4;
        if (ib%2 == 0) y->scales_l[ib/2] = l_l;
        else y->scales_l[ib/2] |= (l_l << 4);
        scales_h[ib/8] |= (l_h << 2*(ib%8));
    }
    if (sumq2 > 0) y->d = GGML_FP32_TO_FP16(fudge*sumqx/sumq2);

    for (int i = 0; i < super_block_size/32; ++i) {
        for (int j = 0; j < 16; ++j) {
            y->qs[16*i + j] = L[32*i + j] | (L[32*i + 16 + j] << 4);
        }
    }
}

}

void quantize_row_iq4_k_ref(const float * x, void * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq4_k(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq4_k(const float * x, void * vy, int64_t k) {
    assert(k % QK_K == 0);
    quantize_row_iq4_k_ref(x, vy, k);
}

size_t quantize_iq4_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    uint8_t L[QK_K];
    float weight[16];
    float scales[QK_K/16];
    auto q_func = [&L, &weight, &scales] (const float * x, void * vy, int n_per_row, const float * imatrix,
            [[maybe_unused]] const quantize_user_data * user_data) {
        block_iq4_k * iq4 = (block_iq4_k *)vy;
        int nblock = n_per_row/QK_K;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * qw = imatrix ? imatrix + QK_K*ibl : nullptr;
            quantize_row_iq4_k_impl_bs16(x + QK_K*ibl, iq4 + ibl,
                    scales, weight, L, iq4k_values, qw, 7);
        }
    };
    auto row_size = ggml_row_size(GGML_TYPE_IQ4_K, n_per_row);
    QHelper helper(imatrix, nullptr, n_per_row, 16);
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}

//
// ============================================== iq5_K
//
void dequantize_row_iq5_k(const void * vx_, float * y, int64_t k) {
    const block_iq5_k * x = (const block_iq5_k *)vx_;
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * sl = x[i].scales_l;
        const uint8_t * sh = x[i].scales_h;

        uint16_t extra = x[i].extra;

        int shift = 0;
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {

            float dl1 = d * (((sl[2*ib64+0] & 0xf) | ((sh[ib64] << 4) & 0x30)) - 32);
            float dl2 = d * (((sl[2*ib64+0] >>  4) | ((sh[ib64] << 2) & 0x30)) - 32);
            float dl3 = d * (((sl[2*ib64+1] & 0xf) | ((sh[ib64] >> 0) & 0x30)) - 32);
            float dl4 = d * (((sl[2*ib64+1] >>  4) | ((sh[ib64] >> 2) & 0x30)) - 32);
            const int8_t * values1 = iq5nl_values + ((extra & 1) << 5);
            const int8_t * values2 = iq5nl_values + ((extra & 2) << 4);
            const int8_t * values3 = iq5nl_values + ((extra & 4) << 3);
            const int8_t * values4 = iq5nl_values + ((extra & 8) << 2);
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl1 * values1[(qs[j+ 0] & 0xf) | (((qh[j+ 0] >> shift) & 1) << 4)];
                y[j+16] = dl2 * values2[(qs[j+16] & 0xf) | (((qh[j+16] >> shift) & 1) << 4)];
                y[j+32] = dl3 * values3[(qs[j+ 0] >>  4) | (((qh[j+ 0] >> shift) & 2) << 3)];
                y[j+48] = dl4 * values4[(qs[j+16] >>  4) | (((qh[j+16] >> shift) & 2) << 3)];
            }
            y  += 64;
            qs += 32;
            extra >>= 4;
            shift += 2;
            if (shift == 8) { qh += 32; shift = 0; }
        }

    }
}

void vec_dot_iq5_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    const int nb = n / QK_K;

    const block_iq5_k * x = (const block_iq5_k *)vx;
    const block_q8_K  * y = (const block_q8_K  *)vy;

    float sumf = 0;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * sl = x[i].scales_l;
        const uint8_t * sh = x[i].scales_h;
        const int8_t  * q8 = y[i].qs;

        uint16_t extra = x[i].extra;

        int shift = 0;
        int sumb  = 0;
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {

            int dl1 = (((sl[2*ib64+0] & 0xf) | ((sh[ib64] << 4) & 0x30)) - 32);
            int dl2 = (((sl[2*ib64+0] >>  4) | ((sh[ib64] << 2) & 0x30)) - 32);
            int dl3 = (((sl[2*ib64+1] & 0xf) | ((sh[ib64] >> 0) & 0x30)) - 32);
            int dl4 = (((sl[2*ib64+1] >>  4) | ((sh[ib64] >> 2) & 0x30)) - 32);
            const int8_t * values1 = iq5nl_values + ((extra & 1) << 5);
            const int8_t * values2 = iq5nl_values + ((extra & 2) << 4);
            const int8_t * values3 = iq5nl_values + ((extra & 4) << 3);
            const int8_t * values4 = iq5nl_values + ((extra & 8) << 2);
            int sumi1 = 0, sumi2 = 0, sumi3 = 0, sumi4 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * values1[(qs[j+ 0] & 0xf) | (((qh[j+ 0] >> shift) & 1) << 4)];
                sumi2 += q8[j+16] * values2[(qs[j+16] & 0xf) | (((qh[j+16] >> shift) & 1) << 4)];
                sumi3 += q8[j+32] * values3[(qs[j+ 0] >>  4) | (((qh[j+ 0] >> shift) & 2) << 3)];
                sumi4 += q8[j+48] * values4[(qs[j+16] >>  4) | (((qh[j+16] >> shift) & 2) << 3)];
            }
            sumb += dl1 * sumi1 + dl2 * sumi2 + dl3 * sumi3 + dl4 * sumi4;
            q8 += 64;
            qs += 32;
            extra >>= 4;
            shift += 2;
        }
        sumf += d * sumb;

    }

    *s = sumf;

}

namespace {
const int8_t iq5nl_index[248] = {
     0,  0,  0,  0,  0,  0, 32,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 33, 33,  2,  2,  2,  2,  2,  2,  2,  2,  2, 34, 34,  3,  3,
     3,  3,  3,  3,  3,  3, 35, 35,  4,  4,  4,  4,  4,  4,  4, 36, 36,  5,  5,  5,  5,  5,  5,  5, 37, 37,  6,  6,  6,  6,  6,  6,
     6, 38,  7,  7,  7,  7,  7,  7, 39, 39,  8,  8,  8,  8,  8, 40, 40,  9,  9,  9,  9,  9, 41, 41, 10, 10, 10, 10, 10, 42, 11, 11,
    11, 11, 11, 43, 12, 12, 12, 12, 12, 44, 13, 13, 13, 13, 13, 45, 14, 14, 14, 14, 14, 46, 15, 15, 15, 15, 47, 47, 16, 16, 16, 16,
    48, 17, 17, 17, 17, 17, 49, 18, 18, 18, 18, 18, 50, 19, 19, 19, 19, 19, 51, 20, 20, 20, 20, 20, 52, 21, 21, 21, 21, 21, 53, 53,
    22, 22, 22, 22, 22, 54, 54, 23, 23, 23, 23, 23, 23, 55, 24, 24, 24, 24, 24, 24, 24, 56, 25, 25, 25, 25, 25, 25, 25, 57, 57, 26,
    26, 26, 26, 26, 26, 26, 58, 58, 27, 27, 27, 27, 27, 27, 27, 27, 59, 28, 28, 28, 28, 28, 28, 28, 28, 28, 60, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 61, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 62, 31, 31, 31, 31, 31, 31
};
inline int best_index_iq5nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 247) return ix < 0 ? 0 : 31;
    ix = iq5nl_index[ix];
    return ix < 32 ? ix : x - values[ix-32] < values[ix-31] - x ? ix-32 : ix-31;
}
#ifdef __AVX2__
static const int8_t k_simple_index_iq5nl[256] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6,
    6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9,10,10,10,10,10,10,
    11,11,11,11,11,11,12,12,12,12,12,12,13,13,13,13,13,13,14,14,14,14,14,14,15,15,15,15,15,16,16,16,
    16,16,16,17,17,17,17,17,17,18,18,18,18,18,18,19,19,19,19,19,19,20,20,20,20,20,20,21,21,21,21,21,
    21,21,22,22,22,22,22,22,22,23,23,23,23,23,23,23,23,24,24,24,24,24,24,24,24,25,25,25,25,25,25,25,
    25,25,26,26,26,26,26,26,26,26,26,27,27,27,27,27,27,27,27,27,27,28,28,28,28,28,28,28,28,28,28,29,
    29,29,29,29,29,29,29,29,29,29,29,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,
};
#endif

void quantize_row_iq5_k_impl(const float * x, void * vy, int n_per_row, const float * quant_weights,
        [[maybe_unused]] const quantize_user_data * user_data) {
    const int ntry = 5;

    block_iq5_k * y = (block_iq5_k *)vy;

    float scales[QK_K/16];
    float weight[16];

#ifdef __AVX2__
    float f_values[64];
    for (int j = 0; j < 64; ++j) f_values[j] = iq5nl_values[j];
#else
    const float step = 1.f;
#endif

    const int8_t * shifted_values = iq5nl_values + 32;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ5_K);

    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq5_k));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 2*sumx2/QK_K;

        float max_scale = 0, max_abs_scale = 0;
        uint16_t extra = 0;

#ifdef __AVX2__
        for (int ib = 0; ib < QK_K/16; ++ib) {
            auto [d, _, is_shifted] = find_best_scale_avx2<16, 240>(32, iq5nl_values, f_values, k_simple_index_iq5nl, xbl + ib*16,
                    quant_weights ? quant_weights + QK_K*ibl + 16*ib : nullptr, sigma2, ntry); //, true);
            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);
            float abs_scale = fabsf(scales[ib]);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale; max_scale = scales[ib];
            }
        }
#else
        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float * xb = xbl + 16*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*16;
                for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < 16; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/iq5nl_values[0] : max/iq5nl_values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            for (int j = 0; j < 16; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq5nl(iq5nl_values, al);
                float q = iq5nl_values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq5nl(iq5nl_values, -al);
                q = iq5nl_values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            d = sumqx_p/sumq2_p;
            float best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            bool is_shifted = false;
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry*step + iq5nl_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq5nl(iq5nl_values, al);
                    float q = iq5nl_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq5nl(iq5nl_values, -al);
                    q = iq5nl_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
                }
                id = (itry*step + shifted_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq5nl(shifted_values, al);
                    float q = shifted_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq5nl(shifted_values, -al);
                    q = shifted_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = true;
                }
            }
            if (d) {
                const int8_t * block_values = is_shifted ? shifted_values : iq5nl_values;
                float sumqx = 0, sumq2 = 0;
                id = 1/d;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq5nl(block_values, al);
                    float q = block_values[l];
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) d = sumqx/sumq2;
            }
            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);

            float abs_scale = fabsf(scales[ib]);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale; max_scale = scales[ib];
            }

        }
#endif

        if (!max_abs_scale) continue;
        float d = -max_scale/32;
        y[ibl].d = GGML_FP32_TO_FP16(fudge*d);
        y[ibl].extra = extra;

        float id = 1/d;

        float sumqx = 0, sumq2 = 0;
        for (int ib = 0; ib < QK_K/16; ++ib) {
            int ls = nearest_int(id*scales[ib]);
            ls = MAX(-32, MIN(31, ls));
            int uls = ls + 32;
            y[ibl].scales_l[ib/2] |= ((uls & 0xf) << 4*(ib%2));
            y[ibl].scales_h[ib/4] |= ((uls >>  4) << 2*(ib%4));
            float dl = d * ls;
            if (dl) {
                const int8_t * block_values = y[ibl].extra & (1 << ib) ? shifted_values : iq5nl_values;
                const float * xb = xbl + 16*ib;
                if (quant_weights) {
                    const float * qw = quant_weights + ibl*QK_K + ib*16;
                    for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
                } else {
                    for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
                }
                float idl = 1/dl;
                int ib32 = ib/2;
                int offset = 16*(ib%2);
                uint8_t * qs = y[ibl].qs + 32*(ib32/2) + offset;
                uint8_t * qh = y[ibl].qh + 32*(ib32/8) + offset;
                for (int j = 0; j < 16; ++j) {
                    const float al = idl*xb[j];
                    int ibest = best_index_iq5nl(block_values, al);
                    qs[j] |= ((ibest & 0xf) << 4*(ib32%2));
                    qh[j] |= ((ibest >>  4) << (ib32%8));
                    float w = weight[j];
                    float q = block_values[ibest]*ls;
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
            }
        }
        if (sumq2 > 0) y[ibl].d = GGML_FP32_TO_FP16(fudge*sumqx/sumq2);

    }

}

}

void quantize_row_iq5_k_ref(const float * x, void * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq5_k(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq5_k(const float * x, void * vy, int64_t k) {
    assert(k % QK_K == 0);
    quantize_row_iq5_k_ref(x, vy, k);
}

size_t quantize_iq5_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    QHelper helper(imatrix, nullptr, n_per_row, 16);
    auto row_size = ggml_row_size(GGML_TYPE_IQ5_K, n_per_row);
    helper.quantize(nrows, src, dst, row_size, quantize_row_iq5_k_impl);
    return nrows * row_size;
}

//
// ============================================== iq6_K
//
#define A_IQ6K -127.f
#define B_IQ6K 6.2568f
#define C_IQ6K 0.11218f
#define D_IQ6K 0.0011972f
#define S_IQ6K 1.f

void dequantize_row_iq6_k(const void * vx_, float * y, int64_t k) {
    const block_iq6_k * x = (const block_iq6_k *)vx_;
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const int8_t  * sl = x[i].scales;

        uint16_t extra = x[i].extra;

        int shift = 0;
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {

            float dl1 = d * sl[4*ib64 + 0];
            float dl2 = d * sl[4*ib64 + 1];
            float dl3 = d * sl[4*ib64 + 2];
            float dl4 = d * sl[4*ib64 + 3];
            float m1 = extra & 1 ? S_IQ6K : 0;
            float m2 = extra & 2 ? S_IQ6K : 0;
            float m3 = extra & 4 ? S_IQ6K : 0;
            float m4 = extra & 8 ? S_IQ6K : 0;
            for (int j = 0; j < 16; ++j) {
                float q1 = ((qs[j+ 0] & 0xf) | (((qh[j+ 0] >> shift) & 0x03) << 4));
                float q2 = ((qs[j+16] & 0xf) | (((qh[j+16] >> shift) & 0x03) << 4));
                float q3 = ((qs[j+ 0] >>  4) | (((qh[j+ 0] >> shift) & 0x0c) << 2));
                float q4 = ((qs[j+16] >>  4) | (((qh[j+16] >> shift) & 0x0c) << 2));
                y[j+ 0] = dl1 * (A_IQ6K + q1*(B_IQ6K + q1*(-C_IQ6K + q1*D_IQ6K)) + m1);
                y[j+16] = dl2 * (A_IQ6K + q2*(B_IQ6K + q2*(-C_IQ6K + q2*D_IQ6K)) + m2);
                y[j+32] = dl3 * (A_IQ6K + q3*(B_IQ6K + q3*(-C_IQ6K + q3*D_IQ6K)) + m3);
                y[j+48] = dl4 * (A_IQ6K + q4*(B_IQ6K + q4*(-C_IQ6K + q4*D_IQ6K)) + m4);
            }
            y  += 64;
            qs += 32;
            extra >>= 4;
            shift += 4;
            if (shift == 8) { qh += 32; shift = 0; }
        }

    }
}

void vec_dot_iq6_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    *s = iqk_vec_dot_dequant_q8_K<block_iq6_k>(n, vx, vy, dequantize_row_iq6_k);

    // TODO
    //const int nb = n / QK_K;

    //const block_iq5_k * x = (const block_iq5_k *)vx;
    //const block_q8_K  * y = (const block_q8_K  *)vy;

    //float sumf = 0;

    //for (int i = 0; i < nb; i++) {

    //    const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
    //    const uint8_t * qs = x[i].qs;
    //    const uint8_t * qh = x[i].qh;
    //    const uint8_t * sl = x[i].scales_l;
    //    const uint8_t * sh = x[i].scales_h;
    //    const int8_t  * q8 = y[i].qs;

    //    uint16_t extra = x[i].extra;

    //    int shift = 0;
    //    int sumb  = 0;
    //    for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {

    //        int dl1 = (((sl[2*ib64+0] & 0xf) | ((sh[ib64] << 4) & 0x30)) - 32);
    //        int dl2 = (((sl[2*ib64+0] >>  4) | ((sh[ib64] << 2) & 0x30)) - 32);
    //        int dl3 = (((sl[2*ib64+1] & 0xf) | ((sh[ib64] >> 0) & 0x30)) - 32);
    //        int dl4 = (((sl[2*ib64+1] >>  4) | ((sh[ib64] >> 2) & 0x30)) - 32);
    //        const int8_t * values1 = iq5nl_values + ((extra & 1) << 5);
    //        const int8_t * values2 = iq5nl_values + ((extra & 2) << 4);
    //        const int8_t * values3 = iq5nl_values + ((extra & 4) << 3);
    //        const int8_t * values4 = iq5nl_values + ((extra & 8) << 2);
    //        int sumi1 = 0, sumi2 = 0, sumi3 = 0, sumi4 = 0;
    //        for (int j = 0; j < 16; ++j) {
    //            sumi1 += q8[j+ 0] * values1[(qs[j+ 0] & 0xf) | (((qh[j+ 0] >> shift) & 1) << 4)];
    //            sumi2 += q8[j+16] * values2[(qs[j+16] & 0xf) | (((qh[j+16] >> shift) & 1) << 4)];
    //            sumi3 += q8[j+32] * values3[(qs[j+ 0] >>  4) | (((qh[j+ 0] >> shift) & 2) << 3)];
    //            sumi4 += q8[j+48] * values4[(qs[j+16] >>  4) | (((qh[j+16] >> shift) & 2) << 3)];
    //        }
    //        sumb += dl1 * sumi1 + dl2 * sumi2 + dl3 * sumi3 + dl4 * sumi4;
    //        q8 += 64;
    //        qs += 32;
    //        extra >>= 4;
    //        shift += 2;
    //    }
    //    sumf += d * sumb;

    //}

    //*s = sumf;

}

namespace {

inline int best_index(int n, const float * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}
static const uint8_t iq6nl_index[249] = {
   0,   0,   0,  64,   1,   1,   1,   1,   1,  65,   2,   2,   2,   2,   2,  66,   3,   3,   3,   3,  67,  67,   4,   4,   4,   4,  68,   5,   5,   5,   5,  69,
  69,   6,   6,   6,  70,  70,   7,   7,   7,  71,   8,   8,   8,  72,  72,   9,   9,   9,  73,  73,  10,  10,  10,  74,  11,  11,  11,  75,  12,  12,  12,  76,
  13,  13,  13,  77,  14,  14,  14,  78,  15,  15,  79,  79,  16,  16,  80,  17,  17,  81,  81,  18,  18,  82,  19,  19,  83,  83,  20,  84,  84,  21,  85,  85,
  22,  86,  86,  23,  87,  87,  24,  88,  88,  25,  89,  89,  26,  90,  90,  27,  91,  91,  28,  92,  29,  93,  93,  30,  94,  94,  31,  95,  95,  32,  96,  33,
  97,  97,  34,  98,  98,  35,  99,  99,  36, 100, 100,  37, 101,  38, 102, 102,  39, 103, 103,  40, 104, 104,  41,  41, 105,  42,  42, 106, 106,  43, 107, 107,
  44, 108, 108,  45,  45, 109,  46,  46,  46, 110,  47,  47, 111, 111,  48,  48, 112,  49,  49,  49, 113,  50,  50,  50, 114,  51,  51,  51, 115,  52,  52,  52,
 116, 116,  53,  53,  53, 117,  54,  54,  54, 118, 118,  55,  55,  55, 119, 119,  56,  56,  56, 120, 120,  57,  57,  57, 121, 121,  58,  58,  58,  58, 122,  59,
  59,  59,  59, 123, 123,  60,  60,  60,  60, 124,  61,  61,  61,  61,  61, 125,  62,  62,  62,  62,  62, 126,  63,  63, 63,
};
inline int best_index_iq6nl(const float * values, float x) {
    int ix = (int)(x - values[0]);
    if (ix < 0 || ix >= 249) return ix < 0 ? 0 : 63;
    ix = iq6nl_index[ix];
    return ix < 64 ? ix : x - values[ix-64] < values[ix-63] - x ? ix-64 : ix-63;
}
#ifdef __AVX2__
static int8_t k_simple_index_iq6nl[256] = {
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5,
    5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9,10,10,10,10,11,11,11,11,12,12,12,
    12,13,13,13,13,14,14,14,14,15,15,15,16,16,16,16,17,17,17,18,18,18,18,19,19,19,20,20,20,21,21,21,
    22,22,22,23,23,23,24,24,24,25,25,25,26,26,26,27,27,27,28,28,29,29,29,30,30,30,31,31,31,32,32,33,
    33,33,34,34,34,35,35,35,36,36,36,37,37,38,38,38,39,39,39,40,40,40,41,41,41,41,42,42,42,43,43,43,
    44,44,44,45,45,45,45,46,46,46,46,47,47,47,48,48,48,48,49,49,49,49,50,50,50,50,51,51,51,51,52,52,
    52,52,52,53,53,53,53,54,54,54,54,54,55,55,55,55,55,56,56,56,56,56,57,57,57,57,57,58,58,58,58,58,
    58,59,59,59,59,59,60,60,60,60,60,60,61,61,61,61,61,61,62,62,62,62,62,62,63,63,63,63,63,63,63,63,
};
#endif


void quantize_row_iq6_k_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, const float * values, const float * shifted_values) {
    const int ntry = 5;

    block_iq6_k * y = (block_iq6_k *)vy;

    float scales[QK_K/16];
    float weight[16];

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ6_K);

#ifdef __AVX2__
    float f_values[128];
    for (int j = 0; j < 128; ++j) f_values[j] = iq6nl_values[j];
#else
    const float step = 1.f;
#endif

    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq6_k));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 2*sumx2/QK_K;

        float max_scale = 0, max_abs_scale = 0;
        uint16_t extra = 0;

#ifdef __AVX2__
        for (int ib = 0; ib < QK_K/16; ++ib) {
            auto [d, _, is_shifted] = find_best_scale_avx2<16, 248>(64, iq6nl_values, f_values, k_simple_index_iq6nl, xbl + ib*16,
                    quant_weights ? quant_weights + QK_K*ibl + 16*ib : nullptr, sigma2, ntry); // , true, 1.0f, true) -> corresponds to not AVX2, but not really better.
            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);
            float abs_scale = fabsf(scales[ib]);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale; max_scale = scales[ib];
            }
        }
#else
        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float * xb = xbl + 16*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*16;
                for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < 16; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/values[0] : max/values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            for (int j = 0; j < 16; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq6nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq6nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            d = sumqx_p/sumq2_p;
            float best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            bool is_shifted = false;
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry*step + values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq6nl(values, al);
                    float q = values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq6nl(values, -al);
                    q = values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
                }
                id = (itry*step + shifted_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq6nl(shifted_values, al);
                    float q = shifted_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq6nl(shifted_values, -al);
                    q = shifted_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = true;
                }
            }
            if (d) {
                const float * block_values = is_shifted ? shifted_values : values;
                float sumqx = 0, sumq2 = 0;
                id = 1/d;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq6nl(block_values, al);
                    float q = block_values[l];
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) d = sumqx/sumq2;
            }
            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);

            float abs_scale = fabsf(scales[ib]);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale; max_scale = scales[ib];
            }

        }
#endif

        if (!max_abs_scale) continue;
        float d = -max_scale/127;
        y[ibl].d = GGML_FP32_TO_FP16(fudge*d);
        y[ibl].extra = extra;

        float id = 1/d;

        float sumqx = 0, sumq2 = 0;
        for (int ib = 0; ib < QK_K/16; ++ib) {
            int ls = nearest_int(id*scales[ib]);
            ls = MAX(-127, MIN(127, ls));
            y[ibl].scales[ib] |= ls;
            float dl = d * ls;
            if (dl) {
                const float * block_values = y[ibl].extra & (1 << ib) ? shifted_values : values;
                const float * xb = xbl + 16*ib;
                if (quant_weights) {
                    const float * qw = quant_weights + ibl*QK_K + ib*16;
                    for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
                } else {
                    for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
                }
                float idl = 1/dl;
                int ib32 = ib/2;
                int offset = 16*(ib%2);
                uint8_t * qs = y[ibl].qs + 32*(ib32/2) + offset;
                uint8_t * qh = y[ibl].qh + 32*(ib32/4) + offset;
                for (int j = 0; j < 16; ++j) {
                    const float al = idl*xb[j];
                    //int ibest = best_index(64, block_values, al);
                    int ibest = best_index_iq6nl(block_values, al);
                    qs[j] |= ((ibest & 0xf) << 4*(ib32%2));
                    qh[j] |= ((ibest >>  4) << 2*(ib32%4));
                    float w = weight[j];
                    float q = block_values[ibest]*ls;
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
            }
        }
        if (sumq2 > 0) y[ibl].d = GGML_FP32_TO_FP16(fudge*sumqx/sumq2);

    }
}

}

void quantize_row_iq6_k_ref(const float * x, void * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq6_k(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq6_k(const float * x, void * vy, int64_t k) {
    assert(k % QK_K == 0);
    quantize_row_iq6_k_ref(x, vy, k);
}

size_t quantize_iq6_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    float values[128];
    for (int i = 0; i < 64; ++i) {
        values[i] = iq6nl_values[i];
        values[i+64] = values[i] + S_IQ6K;
    }
    auto q_func = [values] (const float * x, void * vy, int n_per_row, const float * imatrix,
             [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_row_iq6_k_impl(x, vy, n_per_row, imatrix, values, values + 64);
    };
    auto row_size = ggml_row_size(GGML_TYPE_IQ6_K, n_per_row);
    QHelper helper(imatrix, nullptr, n_per_row, 16);
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}
