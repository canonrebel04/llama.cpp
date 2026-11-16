// ik_llama.cpp row-scaled IQ_KS / IQ4_KSS / IQ2_KL quantization types.
// Ported from ikawrakow/ik_llama.cpp ggml/src/iqk/iqk_quantize.cpp (Iwan Kawrakow, MIT).
// Scalar paths only (the AVX2 search helpers are compiled out). These types store a
// per-row float/half scale ("row meta") in front of the blocks: see ggml_row_meta_size().

#include "ggml-quants.h"
#include "ggml-impl.h"
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml-iqk-quants.h"

#include <vector>
#include <mutex>
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
        case GGML_TYPE_IQ2_KL: return 1.025f;
        case GGML_TYPE_IQ4_KSS: return 1.01f;
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
inline int best_index_iq4nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 241) return ix < 0 ? 0 : 15;
    ix = iq4nl_index[ix];
    return ix < 16 ? ix : x - values[ix-16] < values[ix-15] - x ? ix-16 : ix-15;
}
inline int best_index_iq5nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 247) return ix < 0 ? 0 : 31;
    ix = iq5nl_index[ix];
    return ix < 32 ? ix : x - values[ix-32] < values[ix-31] - x ? ix-32 : ix-31;
}
} // namespace

namespace {
// Row-level scalar fallback for row-scaled types (the per-row scale precedes the blocks).
float iqk_row_vec_dot_dequant_q8_K(int n, const void * vx, const void * vy, void (*dequant)(const void *, float *, int64_t)) {
    thread_local std::vector<float> tmp;
    tmp.resize(n);
    dequant(vx, tmp.data(), n);
    const block_q8_K * y = (const block_q8_K *)vy;
    double sum = 0;
    for (int i = 0; i < n/QK_K; ++i) {
        float sumb = 0;
        for (int j = 0; j < QK_K; ++j) sumb += tmp[i*QK_K + j] * y[i].qs[j];
        sum += (double)sumb * y[i].d;
    }
    return (float)sum;
}
} // namespace

namespace {
#if defined(__AVX2__)
inline void to_values_i32(__m256i idx, __m256i ivalues, __m256i * iv) {
    auto ival = _mm256_shuffle_epi8(ivalues, idx);
    auto ival_1 = _mm256_srli_si256(ival, 8);
    iv[0] = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(ival));
    iv[1] = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(ival_1));
    iv[2] = _mm256_cvtepi8_epi32(_mm256_extracti128_si256(ival, 1));
    iv[3] = _mm256_cvtepi8_epi32(_mm256_extracti128_si256(ival_1, 1));
}
inline __m256i to_int8(const __m256i * ibest) {
    auto i0 = _mm256_packs_epi32(ibest[0], ibest[1]); // 0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15
    auto i1 = _mm256_packs_epi32(ibest[2], ibest[3]); // 16, 17, 18, 19, 24, 25, 26, 27, 20, 21, 22, 23, 28, 29, 30, 31
    auto idx = _mm256_packs_epi16(i0, i1); // 0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31
    auto idx_l = _mm256_castsi256_si128(idx);
    auto idx_h = _mm256_extracti128_si256(idx, 1);
    auto idx1  = _mm_unpacklo_epi32(idx_l, idx_h); // 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    auto idx2  = _mm_unpackhi_epi32(idx_l, idx_h);
    return MM256_SET_M128I(idx2, idx1);
}
bool compute_1block_iq2ks(float d, const __m256 * vx, const __m256 * vw, const int8_t * values, __m256i & this_idx, float & best_d, float & score) {
    constexpr int kBlockSize = 32;
    uint32_t aux32;
    std::memcpy(&aux32, values, sizeof(aux32));
    auto ivalues = _mm256_set1_epi32(aux32);
    __m256 vbest[8];
    __m256i ibest[8];
    auto val = _mm256_set1_ps(d*values[0]);
    auto ival = _mm256_set1_epi32(0);
    for (int k = 0; k < kBlockSize/8; ++k) {
        auto diff = _mm256_sub_ps(vx[k], val);
        vbest[k] = _mm256_mul_ps(diff, diff);
        ibest[k] = ival;
        diff = _mm256_add_ps(vx[k], val);
        vbest[k+4] = _mm256_mul_ps(diff, diff);
        ibest[k+4] = ival;
    }
    for (int j = 1; j < 4; ++j) {
        val = _mm256_set1_ps(d*values[j]);
        ival = _mm256_set1_epi32(j);
        for (int k = 0; k < kBlockSize/8; ++k) {
            auto diff = _mm256_sub_ps(vx[k], val);
            diff = _mm256_mul_ps(diff, diff);
            auto mask = _mm256_cmp_ps(diff, vbest[k], _CMP_LT_OQ);
            vbest[k] = _mm256_or_ps(_mm256_and_ps(mask, diff), _mm256_andnot_ps(mask, vbest[k]));
            auto imask = _mm256_castps_si256(mask);
            ibest[k] = _mm256_or_si256(_mm256_and_si256(imask, ival), _mm256_andnot_si256(imask, ibest[k]));
            diff = _mm256_add_ps(vx[k], val);
            diff = _mm256_mul_ps(diff, diff);
            mask = _mm256_cmp_ps(diff, vbest[k+4], _CMP_LT_OQ);
            vbest[k+4] = _mm256_or_ps(_mm256_and_ps(mask, diff), _mm256_andnot_ps(mask, vbest[k+4]));
            imask = _mm256_castps_si256(mask);
            ibest[k+4] = _mm256_or_si256(_mm256_and_si256(imask, ival), _mm256_andnot_si256(imask, ibest[k+4]));
        }
    }
    bool result = false;
    auto idx1 = to_int8(ibest+0);
    auto idx2 = to_int8(ibest+4);
    to_values_i32(idx1, ivalues, ibest+0);
    to_values_i32(idx2, ivalues, ibest+4);
    auto vsqx_1 = _mm256_setzero_ps();
    auto vsq2_1 = _mm256_setzero_ps();
    auto vsqx_2 = _mm256_setzero_ps();
    auto vsq2_2 = _mm256_setzero_ps();
    for (int k = 0; k < 4; ++k) {
        auto vq1  = _mm256_cvtepi32_ps(ibest[k+0]);
        auto vwq1 = _mm256_mul_ps(vw[k], vq1);
        auto vq2  = _mm256_cvtepi32_ps(ibest[k+4]);
        auto vwq2 = _mm256_mul_ps(vw[k], vq2);
        vsqx_1 = _mm256_fmadd_ps(vwq1, vx[k], vsqx_1);
        vsq2_1 = _mm256_fmadd_ps(vwq1, vq1,   vsq2_1);
        vsqx_2 = _mm256_fmadd_ps(vwq2, vx[k], vsqx_2);
        vsq2_2 = _mm256_fmadd_ps(vwq2, vq2,   vsq2_2);
    }
    auto sumqx_1 = hsum_float_8(vsqx_1);
    auto sumq2_1 = hsum_float_8(vsq2_1);
    auto sumqx_2 = hsum_float_8(vsqx_2);
    auto sumq2_2 = hsum_float_8(vsq2_2);
    if (sumq2_1 > 0) {
        best_d = sumqx_1/sumq2_1;
        score = sumqx_1 * best_d;
        this_idx = idx1;
        result = true;
    }
    if (sumq2_2 > 0 && (!result || sumqx_2*sumqx_2 > score*sumq2_2)) {
        best_d = sumqx_2/sumq2_2;
        score = sumqx_2 * best_d;
        this_idx = idx2;
        result = true;
    }
    return result;
}
float compute_1block_iq2ks_rmse(float d, const __m256 * vx, const __m256 * vw, const int8_t * values, __m256i & this_idx) {
    constexpr int kBlockSize = 32;
    uint32_t aux32;
    std::memcpy(&aux32, values, sizeof(aux32));
    auto ivalues = _mm256_set1_epi32(aux32);
    __m256 vbest[4];
    __m256i ibest[4];
    auto val = _mm256_set1_ps(d*values[0]);
    auto ival = _mm256_set1_epi32(0);
    for (int k = 0; k < kBlockSize/8; ++k) {
        auto diff = _mm256_sub_ps(vx[k], val);
        vbest[k] = _mm256_mul_ps(diff, diff);
        ibest[k] = ival;
    }
    for (int j = 1; j < 4; ++j) {
        val = _mm256_set1_ps(d*values[j]);
        ival = _mm256_set1_epi32(j);
        for (int k = 0; k < kBlockSize/8; ++k) {
            auto diff = _mm256_sub_ps(vx[k], val);
            diff = _mm256_mul_ps(diff, diff);
            auto mask = _mm256_cmp_ps(diff, vbest[k], _CMP_LT_OQ);
            vbest[k] = _mm256_or_ps(_mm256_and_ps(mask, diff), _mm256_andnot_ps(mask, vbest[k]));
            auto imask = _mm256_castps_si256(mask);
            ibest[k] = _mm256_or_si256(_mm256_and_si256(imask, ival), _mm256_andnot_si256(imask, ibest[k]));
        }
    }
    auto idx = to_int8(ibest);
    to_values_i32(idx, ivalues, ibest);
    auto vd = _mm256_set1_ps(-d);
    auto vrmse = _mm256_setzero_ps();
    for (int k = 0; k < 4; ++k) {
        auto vq   = _mm256_cvtepi32_ps(ibest[k]);
        auto diff = _mm256_fmadd_ps(vd, vq, vx[k]);
        auto wdiff = _mm256_mul_ps(vw[k], diff);
        vrmse = _mm256_fmadd_ps(wdiff, diff, vrmse);
    }
    this_idx = idx;
    return hsum_float_8(vrmse);
}
void quantize_row_iq2_ks_fast_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales, float * all_sw, int8_t * all_Ls) {

    constexpr int kBlockSize = 32;

    ggml_half * dptr = (ggml_half *)vy;
    *dptr = GGML_FP32_TO_FP16(0.f);

    block_iq2_ks * y = (block_iq2_ks *)(dptr + 1);

    float weight[kBlockSize];

    const int8_t * shifted_values = iq2nl_values + 4;

    const int nblock = n_per_row/QK_K;

    __m256 vx[4], vw[4];

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq2_ks));

        auto scales = all_scales + ibl*(QK_K/kBlockSize);
        auto sw = all_sw + ibl*(QK_K/kBlockSize);

        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;

        uint16_t extra = 0;

        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            const float * xb = xbl + kBlockSize*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            float amax = 0, max = 0, sumw = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
                sumw += weight[j];
            }
            sw[ib] = sumw;
            if (amax < 1e-14f) {
                scales[ib] = 0;
                continue;
            }
            for (int k = 0; k < 4; ++k) {
                vx[k] = _mm256_loadu_ps(xb + 8*k);
                vw[k] = _mm256_loadu_ps(weight + 8*k);
            }
            float d = max/iq2nl_values[7];
            float best = 0;
            __m256i this_idx;
            float this_d, this_score;
            if (compute_1block_iq2ks(d, vx, vw, iq2nl_values, this_idx, this_d, this_score)) {
                best = this_score; d = this_d;
            }
            for (int itry = -13; itry <= 13; ++itry) {
                if (compute_1block_iq2ks(max/(iq2nl_values[0] + 0.5f*itry), vx, vw, iq2nl_values, this_idx, this_d, this_score)) {
                    if (this_score > best) {
                        best = this_score; d = this_d;
                    }
                }
            }
            bool is_shifted = false;
            for (int itry = -13; itry <= 13; ++itry) {
                if (compute_1block_iq2ks(max/(iq2nl_values[4] + 0.5f*itry), vx, vw, iq2nl_values + 4, this_idx, this_d, this_score)) {
                    if (this_score > best) {
                        best = this_score; d = this_d; is_shifted = true;
                    }
                }
            }
            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);
        }
        y[ibl].extra = extra;
    }

    float d = make_qx_quants(nblock*(QK_K/kBlockSize), 16, all_scales, all_Ls, all_sw);

    if (!d) return;

    auto vsumqx = _mm256_setzero_ps();
    auto vsumq2 = _mm256_setzero_ps();
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto scales = all_scales + ibl*(QK_K/kBlockSize);
        auto xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;
        auto Ls = all_Ls + ibl*(QK_K/kBlockSize);
        __m256i idx[4];
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            const int8_t * block_values = y[ibl].extra & (1 << ib) ? shifted_values : iq2nl_values;
            uint32_t aux32;
            std::memcpy(&aux32, block_values, sizeof(aux32));
            auto ivalues = _mm256_set1_epi32(aux32);
            const float * xb = xbl + kBlockSize*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            for (int k = 0; k < 4; ++k) {
                vx[k] = _mm256_loadu_ps(xb + 8*k);
                vw[k] = _mm256_loadu_ps(weight + 8*k);
            }
            int ls = Ls[ib] - 16;
            float dl = d*ls;
            __m256i idx1, idx2;
            auto rmse1 = compute_1block_iq2ks_rmse(dl, vx, vw, block_values, idx1);
            if (Ls[ib] > 0 && dl > scales[ib]) {
                auto rmse2 = compute_1block_iq2ks_rmse(d*(Ls[ib] - 17), vx, vw, block_values, idx2);
                if (rmse2 < rmse1) {
                    --Ls[ib]; idx1 = idx2;
                }
            }
            else if (Ls[ib] < 15 && dl < scales[ib]) {
                auto rmse2 = compute_1block_iq2ks_rmse(d*(Ls[ib] - 15), vx, vw, block_values, idx2);
                if (rmse2 < rmse1) {
                    ++Ls[ib]; idx1 = idx2;
                }
            }
            __m256i iv[4];
            to_values_i32(idx1, ivalues, iv);
            auto vd = _mm256_set1_ps(Ls[ib] - 16);
            for (int k = 0; k < 4; ++k) {
                auto vq = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(iv[k]));
                auto wvq = _mm256_mul_ps(vw[k], vq);
                vsumqx = _mm256_fmadd_ps(wvq, vx[k], vsumqx);
                vsumq2 = _mm256_fmadd_ps(wvq, vq,    vsumq2);
            }
            ls = Ls[ib];
            y[ibl].scales[ib/2] |= ((ls & 0xf) << 4*(ib%2));
            y[ibl].extra |= ((ls >> 4) << (8 + ib));
            idx[ib % 4] = idx1;
            if ((ib % 4) == 3) {
                auto vqs1 = _mm256_or_si256(idx[0], _mm256_slli_epi16(idx[1], 2));
                auto vqs2 = _mm256_or_si256(_mm256_slli_epi16(idx[2], 4), _mm256_slli_epi16(idx[3], 6));
                auto vqs = _mm256_or_si256(vqs1, vqs2);
                _mm256_storeu_si256((__m256i *)y[ibl].qs + ib/4, vqs);
            }
        }
    }
    float sumqx = hsum_float_8(vsumqx);
    float sumq2 = hsum_float_8(vsumq2);
    *dptr = GGML_FP32_TO_FP16(1.000f*(sumq2 > 0 ? sumqx/sumq2 : d));
}
#endif
void quantize_row_iq2_ks_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales, float * all_sw, int8_t * all_Ls) {

    constexpr int kBlockSize = 32;
    constexpr int kMax_i1 = 3*kBlockSize/4;
    constexpr int kMin_i3 = kBlockSize/4;

    ggml_half * dptr = (ggml_half *)vy;
    *dptr = GGML_FP32_TO_FP16(0.f);

    block_iq2_ks * y = (block_iq2_ks *)(dptr + 1);

    float weight[kBlockSize];
    float sumx[kBlockSize+1], sumw[kBlockSize+1];

    std::array<std::pair<float,int>, kBlockSize> pairs;

    float val [4] = {float(iq2nl_values[0]), float(iq2nl_values[1]), float(iq2nl_values[2]), float(iq2nl_values[3])};
    float sval[4] = {float(iq2nl_values[4]), float(iq2nl_values[5]), float(iq2nl_values[6]), float(iq2nl_values[7])};

    const int8_t * shifted_values = iq2nl_values + 4;

    const int nblock = n_per_row/QK_K;

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq2_ks));

        auto scales = all_scales + ibl*(QK_K/kBlockSize);
        auto sw = all_sw + ibl*(QK_K/kBlockSize);

        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;

        uint16_t extra = 0;

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
            for (int i1 = 0; i1 < kMax_i1; ++i1) {
                for (int i2 = i1; i2 < kBlockSize; ++i2) {
                    for (int i3 = std::max(i2, kMin_i3); i3 < kBlockSize; ++i3) {
                        sumqx = (sumx[i1] - sumx[ 0])*val[0] + (sumx[i2] - sumx[i1])*val[1]
                              + (sumx[i3] - sumx[i2])*val[2] + (sumx[kBlockSize] - sumx[i3])*val[3];
                        sumq2 = (sumw[i1] - sumw[ 0])*val[0]*val[0] + (sumw[i2] - sumw[i1])*val[1]*val[1]
                              + (sumw[i3] - sumw[i2])*val[2]*val[2] + (sumw[kBlockSize] - sumw[i3])*val[3]*val[3];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = false;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*sval[0] + (sumx[i2] - sumx[i1])*sval[1]
                              + (sumx[i3] - sumx[i2])*sval[2] + (sumx[kBlockSize] - sumx[i3])*sval[3];
                        sumq2 = (sumw[i1] - sumw[ 0])*sval[0]*sval[0] + (sumw[i2] - sumw[i1])*sval[1]*sval[1]
                              + (sumw[i3] - sumw[i2])*sval[2]*sval[2] + (sumw[kBlockSize] - sumw[i3])*sval[3]*sval[3];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = true;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*val[3] + (sumx[i2        ] - sumx[i1])*val[2]
                              + (sumx[i3] - sumx[i2])*val[1] + (sumx[kBlockSize] - sumx[i3])*val[0];
                        sumq2 = (sumw[i1] - sumw[ 0])*val[3]*val[3] + (sumw[i2        ] - sumw[i1])*val[2]*val[2]
                              + (sumw[i3] - sumw[i2])*val[1]*val[1] + (sumw[kBlockSize] - sumw[i3])*val[0]*val[0];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = false;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*sval[3] + (sumx[i2        ] - sumx[i1])*sval[2]
                              + (sumx[i3] - sumx[i2])*sval[1] + (sumx[kBlockSize] - sumx[i3])*sval[0];
                        sumq2 = (sumw[i1] - sumw[ 0])*sval[3]*sval[3] + (sumw[i2        ] - sumw[i1])*sval[2]*sval[2]
                              + (sumw[i3] - sumw[i2])*sval[1]*sval[1] + (sumw[kBlockSize] - sumw[i3])*sval[0]*sval[0];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = true;
                        }
                    }
                }
            }
            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);
        }
        y[ibl].extra = extra;
    }

    float d = make_qx_quants(nblock*(QK_K/kBlockSize), 16, all_scales, all_Ls, all_sw);

    if (!d) return;

    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;
        auto Ls = all_Ls + ibl*(QK_K/kBlockSize);
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            int ls = Ls[ib];
            y[ibl].scales[ib/2] |= ((ls & 0xf) << 4*(ib%2));
            y[ibl].extra |= ((ls >> 4) << (8 + ib));
            ls -= 16;
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
                uint8_t * qs = y[ibl].qs + 32*(ib/4);
                for (int j = 0; j < 32; ++j) {
                    const float al = idl*xb[j];
                    int ibest = best_index_iq2nl(block_values, al);
                    qs[j] |= (ibest << 2*(ib%4));
                    float w = weight[j];
                    float q = block_values[ibest]*ls;
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
            }
        }
    }
    *dptr = GGML_FP32_TO_FP16(1.030f*(sumq2 > 0 ? sumqx/sumq2 : d));
}
}

void quantize_row_iq2_ks_ref(const float * x, void * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq2_ks(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq2_ks(const float * x, void * vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq2_ks * y = (block_iq2_ks *)vy;
    quantize_row_iq2_ks_ref(x, y, k);
}

size_t quantize_iq2_ks(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_KS, n_per_row);
    int nblock = n_per_row/QK_K;
    std::vector<float> all_scales(nblock*(QK_K/kBlockSize)), all_sw(nblock*(QK_K/kBlockSize));
    std::vector<int8_t> all_Ls(nblock*(QK_K/kBlockSize));
    auto q_func = [&all_scales, &all_sw, &all_Ls] (const float * x, void * vy, int n_per_row, const float * imatrix,
            [[maybe_unused]] const quantize_user_data * user_data) {
#ifdef __AVX2__
        if (user_data && !user_data->slow_iq2_ks) {
            quantize_row_iq2_ks_fast_impl(x, vy, n_per_row, imatrix, all_scales.data(), all_sw.data(), all_Ls.data());
            return;
        }
#endif
        quantize_row_iq2_ks_impl(x, vy, n_per_row, imatrix, all_scales.data(), all_sw.data(), all_Ls.data());
    };
    QHelper helper(imatrix, nullptr, n_per_row, kBlockSize);
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}

void dequantize_row_iq2_ks(const void * vx_, float * y, int64_t k) {
    const block_iq2_ks * x = (const block_iq2_ks *) vx_;
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    const ggml_half * dptr = (const ggml_half *)x;
    const float d = GGML_FP16_TO_FP32(*dptr);
    x = (const block_iq2_ks *)(dptr + 1);

    for (int i = 0; i < nb; i++) {

        const uint8_t * qs = x[i].qs;

        uint16_t extra = x[i].extra;

        int shift = 0;
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
            float dl1 = d * (((x[i].scales[ib64] & 0xf) | ((extra >> 4) & 0x10)) - 16);
            float dl2 = d * (((x[i].scales[ib64] >>  4) | ((extra >> 5) & 0x10)) - 16);
            const int8_t * values1 = extra & 1 ? iq2nl_values + 4 : iq2nl_values;
            const int8_t * values2 = extra & 2 ? iq2nl_values + 4 : iq2nl_values;
            extra >>= 2;
            for (int j = 0; j < 32; ++j) {
                y[j+ 0] = dl1 * values1[(qs[j] >> (shift+0)) & 3];
                y[j+32] = dl2 * values2[(qs[j] >> (shift+2)) & 3];
            }
            y += 64;
            shift += 4;
            if (shift == 8) { qs += 32; shift = 0; }
        }

    }

}

void vec_dot_iq2_ks_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ2_KS, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif

    const ggml_half * dptr = (const ggml_half *)vx;
    const float d = GGML_FP16_TO_FP32(*dptr);
    const block_iq2_ks * x = (const block_iq2_ks *)(dptr + 1);
    const block_q8_K   * y = (const block_q8_K *)vy;

    const int nb = n / QK_K;
    float sumf = 0;
    for (int i = 0; i < nb; i++) {
        const uint8_t * qs = x[i].qs;
        const  int8_t * q8 = y[i].qs;
        uint16_t extra = x[i].extra;
        int sumi = 0;
        for (int ib128 = 0; ib128 < QK_K/128; ++ib128) {
            int d1 = (((x[i].scales[2*ib128+0] & 0xf) | ((extra >> 4) & 0x10)) - 16);
            int d2 = (((x[i].scales[2*ib128+0] >>  4) | ((extra >> 5) & 0x10)) - 16);
            int d3 = (((x[i].scales[2*ib128+1] & 0xf) | ((extra >> 6) & 0x10)) - 16);
            int d4 = (((x[i].scales[2*ib128+1] >>  4) | ((extra >> 7) & 0x10)) - 16);
            const int8_t * values1 = extra & 1 ? iq2nl_values + 4 : iq2nl_values;
            const int8_t * values2 = extra & 2 ? iq2nl_values + 4 : iq2nl_values;
            const int8_t * values3 = extra & 4 ? iq2nl_values + 4 : iq2nl_values;
            const int8_t * values4 = extra & 8 ? iq2nl_values + 4 : iq2nl_values;
            extra >>= 4;
            int sumi1 = 0, sumi2 = 0, sumi3 = 0, sumi4 = 0;
            for (int j = 0; j < 32; ++j) {
                sumi1 += q8[j+ 0] * values1[(qs[j] >> 0) & 3];
                sumi2 += q8[j+32] * values2[(qs[j] >> 2) & 3];
                sumi3 += q8[j+64] * values3[(qs[j] >> 4) & 3];
                sumi4 += q8[j+96] * values4[(qs[j] >> 6) & 3];
            }
            sumi += d1*sumi1 + d2*sumi2 + d3*sumi3 + d4*sumi4;
            q8 += 128;
            qs +=  32;
        }
        sumf += y[i].d * sumi;
    }

    *s = d * sumf;

}

//
// ======================================== iq2_kl
//
namespace {


void quantize_row_iq2_kl_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales) {
    constexpr int kBlockSize = 32;
    constexpr float kSigmaFactor = 2.25f;
    constexpr int ntry = 5;
    static const int k_index[64] = {-1, -2, 0, -3, -4, 1, -5, -6, 2, -7, -8, 3, -9, 4, -10, 5, -11, 6, 7, -12, 8, 9, 10, -13, 11, -14, -15, -16, 12, 13, -17,
        14, -18, -19, 15, 16, 17, 18, 19, -20, -21, 20, 21, 22, 23, 24, -22, -23, 25, -24, 26, -25, 27, -26, 28, 29, -27, -28, 30, -29, -30, 31, -31, -32};
    static const std::vector<std::vector<int>> k_neighbours = {
        { 2, 0, 6, 11, 7, 3, 8, 15,  },
        { 0, 2, 3, 6, 7, 1, 8, 4,  },
        { 0, 1, 3, 4, 8, 7, 9, 6,  },
        { 1, 0, 3, 4, 8, 9, 7, 10,  },
        { 1, 4, 5, 10, 9, 3, 8, 0,  },
        { 5, 1, 4, 10, 9, 14, 8, 3,  },
        { 6, 2, 7, 0, 3, 11, 8, 15,  },
        { 3, 7, 0, 6, 8, 4, 12, 9,  },
        { 3, 4, 8, 9, 1, 7, 12, 10,  },
        { 4, 10, 5, 9, 1, 8, 13, 14,  },
        { 11, 2, 6, 7, 20, 15, 25, 21,  },
        { 8, 7, 3, 12, 9, 16, 17, 13,  },
        { 14, 5, 10, 19, 9, 13, 4, 18,  },
        { 6, 15, 7, 11, 20, 21, 16, 2,  },
        { 15, 7, 16, 6, 21, 12, 17, 22,  },
        { 12, 16, 17, 8, 15, 7, 13, 22,  },
        { 19, 10, 13, 18, 14, 9, 12, 24,  },
        { 11, 20, 25, 6, 15, 2, 21, 7,  },
        { 20, 15, 21, 6, 11, 7, 16, 26,  },
        { 14, 19, 29, 10, 28, 18, 13, 24,  },
        { 25, 11, 20, 21, 15, 6, 26, 30,  },
        { 19, 24, 28, 18, 29, 23, 13, 17,  },
        { 29, 19, 14, 28, 24, 18, 10, 13,  },
        { 20, 26, 21, 25, 30, 15, 22, 16,  },
        { 27, 26, 22, 23, 21, 30, 16, 24,  },
        { 27, 24, 28, 31, 23, 18, 22, 17,  },
        { 25, 30, 20, 26, 21, 11, 15, 22,  },
        { 30, 26, 25, 20, 21, 27, 22, 15,  },
        { 30, 27, 31, 26, 22, 23, 21, 24,  },
        { 31, 27, 30, 26, 28, 23, 22, 24,  },
        { 31, 28, 29, 27, 24, 23, 19, 18,  },
        { 29, 28, 31, 24, 19, 27, 14, 18,  },
    };
    auto values = iq3nl_values;
    std::pair<int8_t, int8_t> grid[32];
    for (int j = 0; j < 64; ++j) {
        if (int i = k_index[j]; i >= 0) {
            int i1 = j/8, i2 = j%8;
            grid[i] = {values[i1], values[i2]};
        }
    }

    ggml_half * dptr = (ggml_half *)vy;
    auto y = (block_iq2_kl *)(dptr + 1);

    float weight[kBlockSize];

    auto index = [&grid, values] (float id, float x1, float x2, float w1, float w2) {
        float sx1 = id*x1;
        float sx2 = id*x2;
        int l1 = best_index_iq3nl(values, sx1);
        int l2 = best_index_iq3nl(values, sx2);
        int i = k_index[8*l1 + l2];
        if (i >= 0) return i;
        auto& neigh = k_neighbours[-i-1];
        float best = std::numeric_limits<float>::max();
        int ibest = -1;
        for (auto& n : neigh) {
            float diff1 = grid[n].first  - sx1;
            float diff2 = grid[n].second - sx2;
            float score = w1*diff1*diff1 + w2*diff2*diff2;
            if (score < best) {
                best = score; ibest = n;
            }
        }
        GGML_ASSERT(ibest >= 0);
        return ibest;
    };

    float max_scale = 0, max_abs_scale = 0;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ2_KL);

    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {
        std::memset(&y[ibl], 0, sizeof(block_iq2_kl));
        auto scales = all_scales + ibl*(QK_K/kBlockSize);
        auto xbl = x + ibl*QK_K;
        float sigma2 = 0;
        for (int j = 0; j < QK_K; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= kSigmaFactor/QK_K;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            auto xb = xbl + ib*kBlockSize;
            if (quant_weights) {
                auto qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j]*sqrt(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = std::abs(xb[j]); //xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                float ax = std::abs(xb[j]);
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
            for (int j = 0; j < kBlockSize; j += 2) {
                float w1 = weight[j+0];
                float w2 = weight[j+1];
                int idx = index(id, xb[j+0], xb[j+1], w1, w2);
                float q1 = grid[idx].first ;
                float q2 = grid[idx].second;
                sumqx_p += w1*q1*xb[j] + w2*q2*xb[j+1];
                sumq2_p += w1*q1*q1 + w2*q2*q2;
                idx = index(-id, xb[j+0], xb[j+1], w1, w2);
                q1 = grid[idx].first ;
                q2 = grid[idx].second;
                sumqx_m += w1*q1*xb[j] + w2*q2*xb[j+1];
                sumq2_m += w1*q1*q1 + w2*q2*q2;
            }
            d = sumqx_p/sumq2_p;
            float best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < kBlockSize; j += 2) {
                    float w1 = weight[j+0];
                    float w2 = weight[j+1];
                    int idx = index(id, xb[j+0], xb[j+1], w1, w2);
                    float q1 = grid[idx].first ;
                    float q2 = grid[idx].second;
                    sumqx_p += w1*q1*xb[j] + w2*q2*xb[j+1];
                    sumq2_p += w1*q1*q1 + w2*q2*q2;
                    idx = index(-id, xb[j+0], xb[j+1], w1, w2);
                    q1 = grid[idx].first ;
                    q2 = grid[idx].second;
                    sumqx_m += w1*q1*xb[j] + w2*q2*xb[j+1];
                    sumq2_m += w1*q1*q1 + w2*q2*q2;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m;
                }
            }
            scales[ib] = d;
            float ad = std::abs(d);
            if (ad > max_abs_scale) {
                max_abs_scale = ad; max_scale = d;
            }
        }
    }

    if (!max_abs_scale) {
        dptr[0] = GGML_FP32_TO_FP16(0.f);
        return;
    }

    float d = -max_scale/32;
    float id = 1/d;

    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {
        auto scales = all_scales + ibl*(QK_K/kBlockSize);
        auto xbl = x + ibl*QK_K;
        float sigma2 = 0;
        for (int j = 0; j < QK_K; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= kSigmaFactor/QK_K;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            auto xb = xbl + ib*kBlockSize;
            if (quant_weights) {
                auto qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j]*sqrt(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = std::abs(xb[j]); //xb[j]*xb[j];
            }
            int ls = nearest_int(id*scales[ib]);
            ls = std::max(-32, std::min(31, ls));
            int lsmin = std::max(-32, ls-1);
            int lsmax = std::min( 31, ls+1);
            float best_score = std::numeric_limits<float>::max();
            int best_ls = ls;
            for (int ils = lsmin; ils <= lsmax; ++ils) {
                float dl = d*ils;
                float idl = dl ? 1/dl : 0.f;
                float score = 0;
                for (int j = 0; j < kBlockSize/2; ++j) {
                    float w1 = weight[2*j+0];
                    float w2 = weight[2*j+1];
                    int idx = index(idl, xb[2*j+0], xb[2*j+1], w1, w2);
                    float diff1 = dl*grid[idx].first  - xb[2*j+0];
                    float diff2 = dl*grid[idx].second - xb[2*j+1];
                    score += w1*diff1*diff1 + w2*diff2*diff2;
                }
                if (score < best_score) {
                    best_score = score;
                    best_ls = ils;
                }
            }
            ls = best_ls;
            int uls = ls + 32;
            y[ibl].scales_l[ib%4] |= ((uls & 0xf) << 4*(ib/4));
            y[ibl].scales_h |= ((uls >> 4) << 2*ib);
            if (ls == 0) continue;
            float dl = d*ls;
            float idl = 1/dl;
            for (int j = 0; j < kBlockSize/2; ++j) {
                float w1 = weight[2*j+0];
                float w2 = weight[2*j+1];
                int idx = index(idl, xb[2*j+0], xb[2*j+1], w1, w2);
                y[ibl].qs[16*(ib/2) + j] |= ((idx & 0xf) << 4*(ib%2));
                y[ibl].qh[j] |= ((idx >> 4) << ib);
                float q1 = ls*grid[idx].first ;
                float q2 = ls*grid[idx].second;
                sumqx += w1*q1*xb[2*j] + w2*q2*xb[2*j+1];
                sumq2 += w1*q1*q1 + w2*q2*q2;
            }
        }
    }
    if (sumq2 > 0) d = sumqx/sumq2;

    dptr[0] = GGML_FP32_TO_FP16(fudge * d);

}
}

void quantize_row_iq2_kl_ref(const float * x, void * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq2_kl(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq2_kl(const float * x, void * vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq2_kl * y = (block_iq2_kl *)vy;
    quantize_row_iq2_kl_ref(x, y, k);
}

size_t quantize_iq2_kl(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_KL, n_per_row);
    int nblock = n_per_row/QK_K;
    std::vector<float> all_scales(nblock*(QK_K/kBlockSize));
    auto q_func = [&all_scales] (const float * x, void * vy, int n_per_row, const float * imatrix,
            [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_row_iq2_kl_impl(x, vy, n_per_row, imatrix, all_scales.data());
    };
    QHelper helper(imatrix, nullptr, n_per_row, kBlockSize);
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}

void dequantize_row_iq2_kl(const void * vx_, float * y, int64_t k) {
    const block_iq2_kl * x = (const block_iq2_kl *) vx_;
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    const ggml_half * dptr = (const ggml_half *)x;
    const float d = GGML_FP16_TO_FP32(*dptr);
    x = (const block_iq2_kl *)(dptr + 1);

    for (int i = 0; i < nb; i++) {

        auto qs = x[i].qs;
        auto qh = x[i].qh;
        auto scales_h = x[i].scales_h;

        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
            float dl1 = d * (int(((x[i].scales_l[(2*ib64+0)%4] >> 4*(ib64/2)) & 0xf) | (((scales_h >> (4*ib64+0)) & 3) << 4)) - 32);
            float dl2 = d * (int(((x[i].scales_l[(2*ib64+1)%4] >> 4*(ib64/2)) & 0xf) | (((scales_h >> (4*ib64+2)) & 3) << 4)) - 32);
            for (int j = 0; j < 16; ++j) {
                const int8_t * val1 = (const int8_t *)(iq2kl_values + ((qs[j] & 0xf) | (((qh[j] >> (2*ib64+0)) & 1) << 4)));
                const int8_t * val2 = (const int8_t *)(iq2kl_values + ((qs[j] >>  4) | (((qh[j] >> (2*ib64+1)) & 1) << 4)));
                y[2*j+ 0] = dl1 * val1[0];
                y[2*j+ 1] = dl1 * val1[1];
                y[2*j+32] = dl2 * val2[0];
                y[2*j+33] = dl2 * val2[1];
            }
            y  += 64;
            qs += 16;
        }

    }
}

void vec_dot_iq2_kl_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    // ik_llama.cpp only has an iqk_mul_mat path for this type: dequantize the row and dot with Q8_K
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs); GGML_UNUSED(bx); GGML_UNUSED(by); GGML_UNUSED(nrc);
    *s = iqk_row_vec_dot_dequant_q8_K(n, vx, vy, dequantize_row_iq2_kl);
}

//
// ============================================== iq3_ks
//
namespace {
static void quantize_row_iq3_ks_impl(int n_per_row, const float * x, char * cy,
        float * all_scales, float * weight,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    constexpr int super_block_size = 256;
    constexpr int block_size = 32;

#ifdef __AVX2__
    float f_values[16];
    for (int j = 0; j < 16; ++j) f_values[j] = values[j];
#endif

    ggml_half * dptr = (ggml_half *)cy;
    block_iq3_ks * y = (block_iq3_ks *)(dptr + 1);

    const int8_t * shifted_values = values + 8;

    float amax_scale = 0;
    float max_scale = 0;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ3_KS);

    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        memset(&y[ibl], 0, sizeof(block_iq3_ks));
        const float * xbl = x + ibl*super_block_size;
        auto scales = all_scales + ibl*(super_block_size/block_size);
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
#ifdef __AVX2__
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            auto [d, _, is_shifted] = find_best_scale_avx2<block_size, 127>(8, values, f_values, k_simple_index_iq3nl, xbl + ib*block_size,
                    quant_weights ? quant_weights + super_block_size*ibl + block_size*ib : nullptr, sigma2, ntry);
            if (is_shifted) y[ibl].extra |= (1 << (8 + ib));
            scales[ib] = d;
            float ascale = std::abs(d);
            if (ascale > amax_scale) {
                amax_scale = ascale; max_scale = d;
            }
        }
#else
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
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
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            float best = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq3nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq3nl(values, -al);
                q = values[l];
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
                id = (itry + values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq3nl(values, al);
                    float q = values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq3nl(values, -al);
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
                id = (itry + shifted_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
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
            if (is_shifted) y[ibl].extra |= (1 << (8 + ib));
            scales[ib] = d;
            float ascale = std::abs(d);
            if (ascale > amax_scale) {
                amax_scale = ascale; max_scale = d;
            }
        }
#endif
    }
    float d = -max_scale/16;
    *dptr = GGML_FP32_TO_FP16(d);
    if (!d) return;
    float id = d ? 1/d : 0.f;
    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        const float * xbl = x + ibl*super_block_size;
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        auto scales = all_scales + (super_block_size/block_size)*ibl;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const int8_t * block_values = (y[ibl].extra >> (8 + ib)) & 0x01 ? shifted_values : values;
            int l = nearest_int(id*scales[ib]);
            l = std::max(-16, std::min(15, l));
            uint8_t ul = l + 16;
            y[ibl].scales[ib%4] |= (ul & 0xf) << 4*(ib/4);
            y[ibl].extra |= (ul >> 4) << ib;
            float dl = d * l;
            float idl = dl ? 1/dl : 0.f;
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            auto qs = y[ibl].qs + (ib/4)*block_size;
            auto qh = y[ibl].qh + (ib/8)*block_size;
            for (int j = 0; j < block_size; ++j) {
                uint8_t i = best_index_iq3nl(block_values, idl*xb[j]);
                qs[j] |= ((i &  3) << 2*(ib%4));
                qh[j] |= ((i >> 2) << (ib%8));
                float w = weight[j];
                float q = block_values[i]*l;
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
        }
    }
    if (sumq2 > 0) *dptr = GGML_FP32_TO_FP16(fudge*sumqx/sumq2);
}
}

void quantize_row_iq3_ks_ref(const float * x, void * y, int64_t k) {
    quantize_iq3_ks(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq3_ks(const float * x, void * y, int64_t k) {
    quantize_iq3_ks(x, (void *)y, 1, k, nullptr);
}

size_t quantize_iq3_ks(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row%QK_K == 0);
    float weight[kBlockSize];
    std::vector<float> all_scales(n_per_row/kBlockSize);
    auto row_size = ggml_row_size(GGML_TYPE_IQ3_KS, n_per_row);
    QHelper helper(imatrix, nullptr, n_per_row, kBlockSize);
    auto q_func = [&all_scales, &weight] (const float * x, void * vy, int n_per_row, const float * imatrix,
            [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_row_iq3_ks_impl(n_per_row, x, (char *)vy, all_scales.data(), weight, iq3nl_values, imatrix, 5);
    };
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}

void dequantize_row_iq3_ks(const void * vx_, float * y, int64_t k) {
    const block_iq3_ks * x = (const block_iq3_ks *) vx_;
    constexpr int kBlockSize = 32;
    static_assert(QK_K/kBlockSize == 8);
    GGML_ASSERT(k%QK_K == 0);
    const ggml_half * dptr = (const ggml_half *)x;
    float d = GGML_FP16_TO_FP32(*dptr);
    x = (const block_iq3_ks *)(dptr + 1);
    float dl[8];
    int nblock = k/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int j = 0; j < 4; ++j) {
            int ls1 = (x[ibl].scales[j] & 0xf) | (((x[ibl].extra >> (j+0)) & 1) << 4);
            int ls2 = (x[ibl].scales[j] >>  4) | (((x[ibl].extra >> (j+4)) & 1) << 4);
            dl[j+0] = d*(ls1 - 16);
            dl[j+4] = d*(ls2 - 16);
        }
        auto qs = x[ibl].qs;
        auto qh = x[ibl].qh;
        for (int i128 = 0; i128 < QK_K/128; ++i128) {
            for (int ib = 0; ib < 4; ++ib) {
                const int8_t * values = iq3nl_values + ((x[ibl].extra >> (8 + (4*i128+ib)) & 1) << 3);
                for (int j = 0; j < kBlockSize; ++j) {
                    y[j] = dl[4*i128 + ib] * values[((qs[j] >> 2*ib) & 3) | (((qh[j] >> (4*i128+ib)) & 1) << 2)];
                }
                y += kBlockSize;
            }
            qs += kBlockSize;
        }
    }
}

void vec_dot_iq3_ks_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    // ik_llama.cpp only has an iqk_mul_mat path for this type: dequantize the row and dot with Q8_K
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs); GGML_UNUSED(bx); GGML_UNUSED(by); GGML_UNUSED(nrc);
    *s = iqk_row_vec_dot_dequant_q8_K(n, vx, vy, dequantize_row_iq3_ks);
}
namespace {
static void quantize_row_iq4_k_impl_bs32(int n_per_row, const float * x, char * cy,
        float * all_scales, float * weight,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    constexpr int super_block_size = 256;
    constexpr int block_size = 32;

    float * dptr = (float *)cy;
    block_iq4_ks * y = (block_iq4_ks *)(dptr + 1);

    const int8_t * shifted_values = values + 16;

    float amax_scale = 0;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_KS);

#ifdef __AVX2__
    float f_values[32];
    for (int j = 0; j < 32; ++j) f_values[j] = values[j];
#endif

    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        memset(&y[ibl], 0, sizeof(block_iq4_ks));
        const float * xbl = x + ibl*super_block_size;
        auto scales = all_scales + ibl*(super_block_size/block_size);
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
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
            float best = 0;
#ifdef __AVX2__
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
            best = d*sumqx_p;
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
                id = (itry + shifted_values[0])/max;
                if (evaluate_block_iq4k<block_size>(id, values+16, f_values+16, vx, vw, d, best)) {
                    is_shifted = true;
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
                id = (itry + shifted_values[0])/max;
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
            if (is_shifted) y[ibl].scales[ib] = 0x01;
            scales[ib] = d;
            amax_scale = std::max(amax_scale, std::abs(d));
        }
    }
    float d = amax_scale/127;
    *dptr = fudge*d;
    if (!d) return;
    float id = d ? 1/d : 0.f;
    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        const float * xbl = x + ibl*super_block_size;
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        auto scales = all_scales + (super_block_size/block_size)*ibl;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const int8_t * block_values = y[ibl].scales[ib] & 0x01 ? shifted_values : values;
            int l = nearest_int(0.5f*(id*scales[ib]+127.f));
            l = std::max(0, std::min(127, l)) << 1;
            y[ibl].scales[ib] |= l;
            l -= 127;
            float dl = d * l;
            float idl = dl ? 1/dl : 0.f;
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            auto qs = y[ibl].qs + ib*(block_size/2);
            for (int j = 0; j < block_size/2; ++j) {
                uint8_t i1 = best_index_iq4nl(block_values, idl*xb[j]);
                uint8_t i2 = best_index_iq4nl(block_values, idl*xb[j+block_size/2]);
                qs[j] = i1 | (i2 << 4);
                float w1 = weight[j];
                float w2 = weight[j+block_size/2];
                float q1 = block_values[i1]*l;
                float q2 = block_values[i2]*l;
                sumqx += w1*q1*xb[j] + w2*q2*xb[j+block_size/2];
                sumq2 += w1*q1*q1 + w2*q2*q2;
            }
        }
    }
    if (sumq2 > 0) *dptr = fudge*sumqx/sumq2;
}
}

void quantize_row_iq4_ks_ref(const float * x, void * y, int64_t k) {
    quantize_iq4_ks(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq4_ks(const float * x, void * y, int64_t k) {
    quantize_iq4_ks(x, (void *)y, 1, k, nullptr);
}

size_t quantize_iq4_ks(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ4_KS, n_per_row);
    float weight[kBlockSize];
    std::vector<float> all_scales(n_per_row/kBlockSize);
    QHelper helper(imatrix, nullptr, n_per_row, kBlockSize);
    auto q_func = [&all_scales, &weight] (const float * x, void * vy, int n_per_row, const float * imatrix,
             [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_row_iq4_k_impl_bs32(n_per_row, x, (char *)vy, all_scales.data(), weight, iq4k_values, imatrix, 7);
    };
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}

void dequantize_row_iq4_ks(const void * vx_, float * y, int64_t k) {
    const block_iq4_ks * x = (const block_iq4_ks *) vx_;
    constexpr int kBlockSize = 32; //128;
    GGML_ASSERT(k%QK_K == 0);
    const float * dptr = (const float *)x;
    float d = *dptr;
    x = (const block_iq4_ks *)(dptr + 1);
    int nblock = k/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto qs = x[ibl].qs;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            float dl = d * ((int)(x[ibl].scales[ib] & 254) - 127);
            const int8_t * values = iq4k_values + ((x[ibl].scales[ib] & 1) << 4);
            for (int j = 0; j < kBlockSize/2; ++j) {
                y[j             ] = dl * values[qs[j] & 0xf];
                y[j+kBlockSize/2] = dl * values[qs[j] >>  4];
            }
            y  += kBlockSize;
            qs += kBlockSize/2;
        }
    }
}

void  vec_dot_iq4_ks_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    constexpr int kBlockSize = 32;
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ4_KS, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    const float * dptr = (const float *)vx;
    const float d = *dptr;
    //printf("%s: n = %d, d = %g\n", __func__, n, d);
    const block_iq4_ks * x = (const block_iq4_ks *)(dptr + 1);
    const block_q8_K    * y = (const block_q8_K    *)vy;
    int nblock = n/QK_K;
    float sumf = 0;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        //int sumi = 0;
        auto qy = y[ibl].qs;
        auto qx = x[ibl].qs;
        float db = d * y[ibl].d;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            float dl = db * ((x[ibl].scales[ib] & 254) - 127);
            //int ls = (x[ibl].scales[ib] & 254) - 127;
            const int8_t * values = iq4k_values + ((x[ibl].scales[ib] & 1) << 4);
            int suml = 0;
            for (int j = 0; j < kBlockSize/2; ++j) {
                suml += qy[j               ] * values[qx[j] & 0xf]
                      + qy[j + kBlockSize/2] * values[qx[j] >>  4];
            }
            sumf += dl * suml;
            //sumi += ls * suml;
            qy += kBlockSize;
            qx += kBlockSize/2;
        }
        //sumf += d * y[ibl].d * sumi;
    }
    *s = sumf;
}

namespace {
static void quantize_row_iq5_ks_impl(int n_per_row, const float * x, char * cy,
        float * all_scales, float * weight,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    constexpr int super_block_size = 256;
    constexpr int block_size = 32;

#ifdef __AVX2__
    float f_values[64];
    for (int j = 0; j < 64; ++j) f_values[j] = values[j];
#endif
    float * dptr = (float *)cy;
    dptr[0] = 0;
    block_iq5_ks * y = (block_iq5_ks *)(dptr + 1);

    const int8_t * shifted_values = values + 32;

    float amax_scale = 0;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ5_KS);

    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        memset(&y[ibl], 0, sizeof(block_iq5_ks));
        const float * xbl = x + ibl*super_block_size;
        auto scales = all_scales + ibl*(super_block_size/block_size);
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
#ifdef __AVX2__
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            auto [d, _, is_shifted] = find_best_scale_avx2<block_size, 240>(32, values, f_values, k_simple_index_iq5nl, xbl + ib*block_size,
                    quant_weights ? quant_weights + super_block_size*ibl + block_size*ib : nullptr, sigma2, ntry);
            if (is_shifted) y[ibl].scales[ib] = 0x01;
            scales[ib] = d;
            amax_scale = std::max(amax_scale, std::abs(d));
        }
#else
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
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
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq5nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq5nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            d = sumqx_p/sumq2_p;
            bool is_shifted = false;
            float best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq5nl(values, al);
                    float q = values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq5nl(values, -al);
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
                id = (itry + shifted_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
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
            if (is_shifted) y[ibl].scales[ib] = 0x01;
            scales[ib] = d;
            amax_scale = std::max(amax_scale, std::abs(d));
        }
#endif
    }
    float d = amax_scale/127;
    *dptr = fudge*d;
    if (!d) return;
    float id = d ? 1/d : 0.f;
    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        const float * xbl = x + ibl*super_block_size;
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        auto scales = all_scales + (super_block_size/block_size)*ibl;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const int8_t * block_values = y[ibl].scales[ib] & 0x01 ? shifted_values : values;
            int l = nearest_int(0.5f*(id*scales[ib]+127.f));
            l = std::max(0, std::min(127, l)) << 1;
            y[ibl].scales[ib] |= l;
            l -= 127;
            float dl = d * l;
            float idl = dl ? 1/dl : 0.f;
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            for (int j = 0; j < block_size; ++j) {
                uint8_t idx = best_index_iq5nl(block_values, idl*xb[j]);
                y[ibl].qs[block_size*(ib/2) + j] |= ((idx & 0xf) << 4*(ib%2));
                y[ibl].qh[j] |= ((idx >> 4) << ib);
                float w = weight[j];
                float q = block_values[idx]*l;
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
        }
    }
    if (sumq2 > 0) *dptr = fudge*sumqx/sumq2;
}
}

void quantize_row_iq5_ks_ref(const float * x, void * y, int64_t k) {
    quantize_iq5_ks(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq5_ks(const float * x, void * y, int64_t k) {
    quantize_iq5_ks(x, (void *)y, 1, k, nullptr);
}

size_t quantize_iq5_ks(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ5_KS, n_per_row);
    float weight[kBlockSize];
    std::vector<float> all_scales(n_per_row/kBlockSize);
    QHelper helper(imatrix, nullptr, n_per_row, kBlockSize);
    auto q_func = [&all_scales, &weight] (const float * x, void * vy, int n_per_row, const float * imatrix,
             [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_row_iq5_ks_impl(n_per_row, x, (char *)vy, all_scales.data(), weight, iq5nl_values, imatrix, 5);
    };
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}

void dequantize_row_iq5_ks(const void * vx_, float * y, int64_t k) {
    const block_iq5_ks * x = (const block_iq5_ks *) vx_;
    constexpr int kBlockSize = 32;
    GGML_ASSERT(k%QK_K == 0);
    const float * dptr = (const float *)x;
    float d = *dptr;
    x = (const block_iq5_ks *)(dptr + 1);
    int nblock = k/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto qs = x[ibl].qs;
        auto qh = x[ibl].qh;
        for (int ib64 = 0; ib64 < QK_K/(2*kBlockSize); ++ib64) {
            float dl1 = d * ((int)(x[ibl].scales[2*ib64+0] & 254) - 127);
            float dl2 = d * ((int)(x[ibl].scales[2*ib64+1] & 254) - 127);
            const int8_t * values1 = iq5nl_values + ((x[ibl].scales[2*ib64+0] & 1) << 5);
            const int8_t * values2 = iq5nl_values + ((x[ibl].scales[2*ib64+1] & 1) << 5);
            for (int j = 0; j < kBlockSize; ++j) {
                y[j           ] = dl1 * values1[(qs[j] & 0xf) | (((qh[j] >> (2*ib64+0)) & 1) << 4)];
                y[j+kBlockSize] = dl2 * values2[(qs[j] >>  4) | (((qh[j] >> (2*ib64+1)) & 1) << 4)];
            }
            y  += 2*kBlockSize;
            qs += kBlockSize;
        }
    }
}

void  vec_dot_iq5_ks_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    constexpr int kBlockSize = 32;
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ5_KS, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    const float * dptr = (const float *)vx;
    const float d = *dptr;
    const block_iq5_ks * x = (const block_iq5_ks *)(dptr + 1);
    const block_q8_K   * y = (const block_q8_K    *)vy;
    int nblock = n/QK_K;
    float sumf = 0;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto qy = y[ibl].qs;
        auto qs = x[ibl].qs;
        auto qh = x[ibl].qh;
        float db = d * y[ibl].d;
        for (int ib64 = 0; ib64 < QK_K/(2*kBlockSize); ++ib64) {
            float dl1 = db * ((int)(x[ibl].scales[2*ib64+0] & 254) - 127);
            float dl2 = db * ((int)(x[ibl].scales[2*ib64+1] & 254) - 127);
            const int8_t * values1 = iq5nl_values + ((x[ibl].scales[2*ib64+0] & 1) << 5);
            const int8_t * values2 = iq5nl_values + ((x[ibl].scales[2*ib64+1] & 1) << 5);
            int suml1 = 0;
            int suml2 = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                suml1 += qy[j           ] * values1[(qs[j] & 0xf) | (((qh[j] >> (2*ib64+0)) & 1) << 4)];
                suml2 += qy[j+kBlockSize] * values2[(qs[j] >>  4) | (((qh[j] >> (2*ib64+1)) & 1) << 4)];
            }
            sumf += dl1*suml1 + dl2*suml2;
            qy += 2*kBlockSize;
            qs += kBlockSize;
        }
    }
    *s = sumf;
}

namespace {
inline int popcount(uint8_t x)  { return __builtin_popcount(x); }
inline int popcount(uint16_t x) { return __builtin_popcount(x); }
inline int popcount(uint32_t x) { return __builtin_popcount(x); }
const uint16_t * scramble_table() {
    static std::mutex mutex;
    static std::vector<uint16_t> table;
    std::lock_guard<std::mutex> lock(mutex);
    if (table.empty()) {
        table.resize(1 << 15);
        for (int i = 0; i < int(table.size()); ++i) {
            uint16_t val = i;
            int non = popcount(val);
            if (non%2) val |= (1 << 15);
            bool found = false;
            for (int j = 0; j < int(table.size()); ++j) {
                if ((j ^ (j << 1)) == val) {
                    table[i] = j; found = true; break;
                }
            }
            if (!found) {
                printf("Oops: did not find for %d %u\n", i, val);
                exit(1);
            }
        }
    }
    return table.data();
}
uint16_t prune_iq4ks(uint16_t v, const int8_t * values, const float * x, const float * w, float dl) {
    if (popcount(v)%2 == 0) return v;
    float best_score = std::numeric_limits<float>::max();
    uint8_t q4[4];
    int jbest = -1;
    uint8_t bestq = 0;
    for (int j = 0; j < 4; ++j) {
        uint8_t q = (v >> 4*j) & 0xf;
        q4[j] = q;
        auto pc = popcount(q);
        float diff0 = dl*iq4k_values[q] - x[j];
        int qmin = std::max(int(q)-2,  0);
        int qmax = std::min(int(q)+2, 15);
        for (int iq = qmin; iq <= qmax; ++iq) {
            uint8_t qq = iq;
            if (qq == q) continue;
            int pci = popcount(qq);
            if (std::abs(pci - pc)%2) {
                float diff1 = dl*values[qq] - x[j];
                float score = w[j]*(diff1*diff1 - diff0*diff0);
                if (score < best_score) {
                    best_score = score; jbest = j; bestq = qq;
                }
            }
        }
    }
    GGML_ASSERT(jbest >= 0);
    q4[jbest] = bestq;
    return (q4[0] | (q4[1] << 4) | (q4[2] << 8) | (q4[3] << 12));
}
static void quantize_row_iq4_kss_impl(int n_per_row, const float * x, char * cy,
        float * all_scales, float * weight,
        const int8_t * values,
        const float * quant_weights,
        const uint16_t * table,
        const int ntry) {

    constexpr int super_block_size = 256;
    constexpr int block_size = 32;

    float * dptr = (float *)cy;
    *dptr = 0;
    block_iq4_kss * y = (block_iq4_kss *)(dptr + 1);

    const int8_t * shifted_values = values + 16;

    uint16_t vps[block_size/2], vms[block_size/2], vs[block_size/2];
    float xv[4], wv[4];

    float amax_scale = 0;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_KSS);

    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        memset(&y[ibl], 0, sizeof(block_iq4_kss));
        const float * xbl = x + ibl*super_block_size;
        auto scales = all_scales + ibl*(super_block_size/block_size);
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
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
            float best = 0;
            float d = -max/iq4k_values[0];
            std::memset(vs, 0, block_size);
            for (int itry = -ntry; itry <= ntry; ++itry) {
                float id = (itry + values[0])/max;
                float sumqx_p = 0, sumq2_p = 0;
                float sumqx_m = 0, sumq2_m = 0;
                float this_d = 1/id;
                for (int k = 0; k < block_size/4; ++k) {
                    xv[0] =     xb[2*k+0]; xv[1] =     xb[2*k+0+block_size/2]; xv[2] =     xb[2*k+1]; xv[3] =     xb[2*k+1+block_size/2];
                    wv[0] = weight[2*k+0]; wv[1] = weight[2*k+0+block_size/2]; wv[2] = weight[2*k+1]; wv[3] = weight[2*k+1+block_size/2];
                    uint16_t vp = 0, vm = 0;
                    for (int j = 0; j < 4; ++j) {
                        float al = id*xv[j];
                        vp |= (best_index_iq4nl(values,  al) << 4*j);
                        vm |= (best_index_iq4nl(values, -al) << 4*j);
                    }
                    vp = prune_iq4ks(vp, values, xv, wv,  this_d);
                    vm = prune_iq4ks(vm, values, xv, wv,  this_d);
                    for (int j = 0; j < 4; ++j) {
                        float w = wv[j];
                        float q = values[(vp >> 4*j) & 0xf];
                        sumqx_p += w*q*xv[j];
                        sumq2_p += w*q*q;
                        q = values[(vm >> 4*j) & 0xf];
                        sumqx_m += w*q*xv[j];
                        sumq2_m += w*q*q;
                    }
                    vps[k] = vp;
                    vms[k] = vm;
                }
                bool copy_p = false, copy_m = false;
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; copy_p = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; copy_m = true;
                }
                if (copy_m) {
                    std::memcpy(vs, vms, block_size);
                } else if (copy_p) {
                    std::memcpy(vs, vps, block_size);
                }

                id = (itry + shifted_values[0])/max;
                this_d = 1/id;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int k = 0; k < block_size/4; ++k) {
                    xv[0] =     xb[2*k+0]; xv[1] =     xb[2*k+0+block_size/2]; xv[2] =     xb[2*k+1]; xv[3] =     xb[2*k+1+block_size/2];
                    wv[0] = weight[2*k+0]; wv[1] = weight[2*k+0+block_size/2]; wv[2] = weight[2*k+1]; wv[3] = weight[2*k+1+block_size/2];
                    uint16_t vp = 0, vm = 0;
                    for (int j = 0; j < 4; ++j) {
                        float al = id*xv[j];
                        vp |= (best_index_iq4nl(shifted_values,  al) << 4*j);
                        vm |= (best_index_iq4nl(shifted_values, -al) << 4*j);
                    }
                    vp = prune_iq4ks(vp, shifted_values, xv, wv,  this_d);
                    vm = prune_iq4ks(vm, shifted_values, xv, wv,  this_d);
                    for (int j = 0; j < 4; ++j) {
                        float w = wv[j];
                        float q = shifted_values[(vp >> 4*j) & 0xf];
                        sumqx_p += w*q*xv[j];
                        sumq2_p += w*q*q;
                        q = shifted_values[(vm >> 4*j) & 0xf];
                        sumqx_m += w*q*xv[j];
                        sumq2_m += w*q*q;
                    }
                    vps[k] = vp;
                    vms[k] = vm;
                }
                copy_p = copy_m = false;
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; copy_p = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; copy_m = true;
                }
                if (copy_m) {
                    std::memcpy(vs, vms, block_size);
                } else if (copy_p) {
                    std::memcpy(vs, vps, block_size);
                }
            }
            scales[ib] = d;
            amax_scale = std::max(amax_scale, std::abs(d));
        }
    }
    float d = amax_scale/127;
    *dptr = fudge*d;
    if (!d) return;
    float id = 1/d;
    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        auto scales = all_scales + (super_block_size/block_size)*ibl;
        const float * xbl = x + ibl*super_block_size;
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            int l = nearest_int(0.5f*(id*scales[ib]+127.f));
            l = (std::max(0, std::min(127, l)) << 1) - 127;
            if (l) {
                float dl = d*l;
                float idl = 1/dl;
                float mse_p = 0, mse_m = 0;
                for (int k = 0; k < block_size/4; ++k) {
                    xv[0] =     xb[2*k+0]; xv[1] =     xb[2*k+0+block_size/2]; xv[2] =     xb[2*k+1]; xv[3] =     xb[2*k+1+block_size/2];
                    wv[0] = weight[2*k+0]; wv[1] = weight[2*k+0+block_size/2]; wv[2] = weight[2*k+1]; wv[3] = weight[2*k+1+block_size/2];
                    uint16_t vp = 0, vm = 0;
                    for (int j = 0; j < 4; ++j) {
                        float al = idl*xv[j];
                        vp |= (best_index_iq4nl(        values, al) << 4*j);
                        vm |= (best_index_iq4nl(shifted_values, al) << 4*j);
                    }
                    vp = prune_iq4ks(vp,         values, xv, wv,  dl);
                    vm = prune_iq4ks(vm, shifted_values, xv, wv,  dl);
                    for (int j = 0; j < 4; ++j) {
                        float w = wv[j];
                        float q = values[(vp >> 4*j) & 0xf];
                        mse_p += w*(xv[j] - dl*q)*(xv[j] - dl*q);
                        q = shifted_values[(vm >> 4*j) & 0xf];
                        mse_m += w*(xv[j] - dl*q)*(xv[j] - dl*q);
                    }
                    vps[k] = vp;
                    vms[k] = vm;
                }
                const uint16_t * v = vps;
                const int8_t * block_values = values;
                if (mse_m < mse_p) {
                    v = vms;
                    block_values = values + 16;
                }
                for (int k = 0; k < block_size/4; ++k) {
                    xv[0] =     xb[2*k+0]; xv[1] =     xb[2*k+0+block_size/2]; xv[2] =     xb[2*k+1]; xv[3] =     xb[2*k+1+block_size/2];
                    wv[0] = weight[2*k+0]; wv[1] = weight[2*k+0+block_size/2]; wv[2] = weight[2*k+1]; wv[3] = weight[2*k+1+block_size/2];
                    for (int j = 0; j < 4; ++j) {
                        float q = block_values[(v[k] >> 4*j) & 0xf] * l;
                        sumqx += wv[j]*q*xv[j];
                        sumq2 += wv[j]*q*q;
                    }
                }
                l += 127;
                if (mse_m < mse_p) l |= 1;
                uint16_t * q16 = (uint16_t *)y[ibl].qs + (block_size/4)*ib;
                for (int k = 0; k < block_size/4; ++k) {
                    auto val = table[v[k] & 0x7fff];
                    q16[k] = (val << 1) | ((l >> k) & 1);
                }
            } else {
                l += 127;
                uint16_t * q16 = (uint16_t *)y[ibl].qs + (block_size/4)*ib;
                for (int k = 0; k < block_size/4; ++k) {
                    q16[k] = ((l >> k) & 1);
                }
            }
        }
    }
    if (sumq2 > 0) *dptr = fudge*sumqx/sumq2;
}
}

size_t quantize_iq4_kss(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size    = ggml_row_size(GGML_TYPE_IQ4_KSS, n_per_row);
    std::vector<float> all_scales(n_per_row/kBlockSize);
    float weight[kBlockSize];
    auto table = scramble_table();
    QHelper helper(imatrix, nullptr, n_per_row, kBlockSize);
    auto q_func = [&all_scales, &weight, table] (const float * x, void * vy, int n_per_row, const float * imatrix,
             [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_row_iq4_kss_impl(n_per_row, x, (char *)vy, all_scales.data(), weight, iq4k_values, imatrix, table, 7);
    };
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}

void quantize_row_iq4_kss_ref(const float * x, void * y, int64_t k) {
    quantize_iq4_kss(x, y, 1, k, nullptr);
}

void quantize_row_iq4_kss(const float * x, void * y, int64_t k) {
    quantize_iq4_kss(x, y, 1, k, nullptr);
}

void dequantize_row_iq4_kss(const void * vx_, float * y, int64_t k) {
    const block_iq4_kss * x = (const block_iq4_kss *) vx_;
    const float * dptr = (const float *)x;
    const float d = *dptr;
    x = (const block_iq4_kss *)(dptr + 1);
    uint16_t aux16[8];
    const uint8_t * aux8 = (const uint8_t *)aux16;
    for (int ibl = 0; ibl < k/QK_K; ++ibl) {
        auto qs = (const uint16_t *)x[ibl].qs;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            int16_t ls = 0;
            for (int k = 0; k < 8; ++k) {
                aux16[k] = qs[k] & 0xfffe;
                aux16[k] ^= (aux16[k] >> 1);
                ls |= (qs[k] & 1) << k;
            }
            const int8_t * values = iq4k_values + ((ls & 1) << 4);
            float dl = d * ((ls & 254) - 127);
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl * values[aux8[j] & 0xf];
                y[j+16] = dl * values[aux8[j] >>  4];
            }
            y  += 32;
            qs += 8;
        }
    }
}

void vec_dot_iq4_kss_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    // ik_llama.cpp only has an iqk_mul_mat path for this type: dequantize the row and dot with Q8_K
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs); GGML_UNUSED(bx); GGML_UNUSED(by); GGML_UNUSED(nrc);
    *s = iqk_row_vec_dot_dequant_q8_K(n, vx, vy, dequantize_row_iq4_kss);
}
