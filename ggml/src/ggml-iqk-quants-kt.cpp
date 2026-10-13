// ik_llama.cpp trellis quantization types IQ1_KT, IQ2_KT, IQ3_KT, IQ4_KT.
// Ported from ikawrakow/ik_llama.cpp ggml/src/iqk/iqk_quantize.cpp (Iwan Kawrakow, MIT).
// Scalar paths only (AVX2 helpers compiled out); rows must be a multiple of QK_K.
// per-row float scale (row meta) in front of the blocks.

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
#include <thread>
#include <random>
namespace {
// ik_llama.cpp default fudge factors for the trellis types (ik ggml.c)
inline float ggml_get_quantize_fudge_factor(enum ggml_type type) {
    return type == GGML_TYPE_IQ1_KT ? 1.07f : 1.0f;
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
inline int best_index_iq4nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 241) return ix < 0 ? 0 : 15;
    ix = iq4nl_index[ix];
    return ix < 16 ? ix : x - values[ix-16] < values[ix-15] - x ? ix-16 : ix-15;
}
void iqk_kt_finish_row(int type, const void * ypad, void * qrow, int64_t n_per_row);
}
#include <memory>
#include <cstdio>

namespace {
// Row-level dot through dequantization (ik_llama.cpp computes these types only in
// iqk_mul_mat). The per-row scale sits in front of the blocks, so dequantize whole rows.
float iqk_kt_row_vec_dot_dequant_q8_K(int n, const void * vx, const void * vy, void (*dequant)(const void *, float *, int64_t)) {
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
}
namespace {
inline int nearest_int(float fval) {
    assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}
}
namespace {
template <int block_size, int group_size, int num_bits, bool is_abs = false, bool is_int = false>
class QuantizerIQKT {
    static_assert(group_size == 8 || group_size == 4);
    static_assert(block_size >= 8 && block_size%8 == 0);
public:
    constexpr static int kSuperBlockSize = QK_K;
    constexpr static int kBlockSize = block_size;
    constexpr static int kGroupSize = group_size;
    constexpr static int kNg = kBlockSize/kGroupSize;
    constexpr static int kNblock = kSuperBlockSize/kBlockSize;
    constexpr static int kNumVal = 1 << num_bits; // i.e, 16 bits per group of 8
    constexpr static float kScale = is_int ? 1.f : 31.75f;
    constexpr static bool kVerbose = false;

    QuantizerIQKT(int num_clusters, int num_neighbours, int offset = 4096);
    const float * values() const { return m_values.data(); }
    const std::vector<std::vector<int>>& in_cluster() const { return m_in_cluster; }
    const float * mid() const { return m_mid; }

    inline void find_best_match(float d, const float * xb, const float * weight, int * best_idx) const;
    inline std::pair<float, float> find_best_scale(const float * xb, const float * weight, const int * best_idx) const;
    inline float find_best_inverse_scale(const float * xb, const float * weight, const int * best_idx) const;

    static inline void set_values(uint32_t i, float * result, float scale, int offset = 4096) {
        uint32_t x = i + offset;
        if constexpr (is_int) {
            constexpr uint32_t ka = 0xCBAC1FED;
            uint32_t s;
            auto i8 = (const int8_t *)&s;
            for (int k = 0; k < kGroupSize; ++k) {
                x = ka*x;
                s = x & 0x3f3f3f3f;
                if constexpr (is_abs) {
                    result[k] = scale*std::abs(i8[0] + i8[1] + i8[2] + i8[3] - 126.f);
                } else {
                    result[k] = scale*(i8[0] + i8[1] + i8[2] + i8[3] - 126.f);
                }
            }
        } else {
            constexpr uint32_t ka = 89226354;
            constexpr uint32_t kb = 64248484;
            constexpr uint32_t kmask = 0x8fff8fff;
            constexpr uint32_t km32 = 0x3b603b60;
            for (int k = 0; k < kGroupSize; ++k) {
                x = ka*x + kb;
                uint32_t s = (x & kmask) ^ km32;
                float val = GGML_FP16_TO_FP32(s & 65535) + GGML_FP16_TO_FP32(s >> 16);
                if constexpr (is_abs) result[k] = scale*std::abs(val);
                else result[k] = scale*val;
            }
        }
    }

    static inline int bin4(float x) {
        if constexpr (is_abs) {
            return x < 16.f ? 0 : x < 32.f ? 1 : x < 64.f ? 2 : 3;
        } else {
            return x < -24.f ? 0 : x < 0.0f ? 1 : x < 24.f ? 2 : 3;
        }
    }
    static inline int bin5(float x) {
        if constexpr (is_abs) {
            return x < 11.2f ? 0 : x < 24.f ? 1 : x < 39.f ? 2 : x < 58.f ? 3 : 4;
        } else {
            return x < -48.f ? 0 : x < -16.f ? 1 : x < 16.f ? 2 : x < 48.f ? 3 : 4;
        }
    }
    inline int bin3(int idim, float x) const { return x < m_mid[2*idim+0] ? 0 : x < m_mid[2*idim+1] ? 1 : 2; }

    static inline void set_weights(float sigma2_scale, int nblock, const float * x, const float * imatrix, float * row_weights, int n_per_row) {
        constexpr float kEps2   = 1e-14f;
        constexpr float kWeight = 1e-4f;
        for (int ibl = 0; ibl < nblock; ++ibl) {

            const float * xbl = x + ibl*kSuperBlockSize;
            float * wbl = row_weights + ibl*kSuperBlockSize;
            const int n = std::min(kSuperBlockSize, n_per_row - ibl*kSuperBlockSize);
            for (int j = n; j < kSuperBlockSize; ++j) wbl[j] = 0;

            float sumx2 = 0;
            for (int j = 0; j < n; ++j) sumx2 += xbl[j]*xbl[j];
            if (sumx2 < kEps2*n) {
                // all x in th super block are (almost) zero
                for (int j = 0; j < n; ++j) wbl[j] = kWeight;
                continue;
            }
            const float sigma2 = sigma2_scale*sumx2/n;

            if (imatrix) {
                for (int ib = 0; ib < n/kBlockSize; ++ib) {
                    const float * qw = imatrix + ibl*kSuperBlockSize + ib*kBlockSize;
                    const float * xb = xbl + ib*kBlockSize;
                    float * wb = wbl + ib*kBlockSize;
                    float sumwx = 0, sumw2 = 0, sumx2 = 0;
                    for (int j = 0; j < kBlockSize; ++j) {
                        wb[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
                        sumwx += wb[j]*std::abs(xb[j]);
                        sumw2 += wb[j]*wb[j];
                        sumx2 += xb[j]*xb[j];
                    }
                    if (sumx2 < kEps2 || sumw2 < kEps2 || sumwx < kEps2) {
                        for (int j = 0; j < kBlockSize; ++j) wb[j] = kWeight;
                    }
                }
            } else {
                for (int j = 0; j < n; ++j) wbl[j] = 0.25f*sigma2 + xbl[j]*xbl[j];
            }
        }
    }
private:
    static std::vector<float> cluster_points(const std::vector<float>& points, int ncluster, int niter, float * mid);
    static std::vector<std::vector<int>> finalize_clusters(int num_neighbours, const std::vector<float>& points, const std::vector<float>& clusters,
            std::vector<std::vector<float>>& c_values);
    std::vector<float> m_values;
    std::vector<float> m_clusters;
    std::vector<std::vector<int>> m_in_cluster;
    std::vector<std::vector<float>> m_c_values;
    float m_mid[4*kGroupSize];
};

template <int block_size, int group_size, int num_bits, bool is_abs, bool is_int>
QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::QuantizerIQKT(int num_clusters, int num_neighbours, int offset) {
    m_values.resize(kNumVal*kGroupSize);
    float * data = m_values.data();
    for (int i = 0; i < kNumVal; ++i) {
        set_values(i, data, kScale, offset);
        data += kGroupSize;
    }
    if (num_clusters == 0) return;
    // Make 128 clusters.
    // Note: we get a slightly better result by using 64 clusters
    //       at the expense of almost doubling the quantization time.
    m_clusters = cluster_points(m_values, num_clusters, 200, m_mid);
    GGML_ASSERT(!m_clusters.empty());
    m_in_cluster = finalize_clusters(num_neighbours, m_values, m_clusters, m_c_values);
}

template <int block_size, int group_size, int num_bits, bool is_abs, bool is_int>
std::pair<float, float> QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::find_best_scale(
        const float * xb, const float * weight, const int * best_idx) const {
    float sumqx = 0, sumq2 = 0;
#ifdef __AVX2__
    auto vqx = _mm256_setzero_ps();
    auto vq2 = _mm256_setzero_ps();
    for (int l = 0; l < kBlockSize; l += 8) {
        auto vx = _mm256_loadu_ps(xb+l);
        auto vw = _mm256_loadu_ps(weight+l);
        auto vq = kGroupSize == 8 ? _mm256_loadu_ps(m_values.data() + kGroupSize*best_idx[l/kGroupSize]) :
            _mm256_set_m128(_mm_loadu_ps(m_values.data() + kGroupSize*best_idx[l/kGroupSize+1]),
                            _mm_loadu_ps(m_values.data() + kGroupSize*best_idx[l/kGroupSize+0]));
        auto vqw = _mm256_mul_ps(vq, vw);
        vqx = _mm256_fmadd_ps(vqw, vx, vqx);
        vq2 = _mm256_fmadd_ps(vqw, vq, vq2);
    }
    sumqx = hsum_float_8(vqx);
    sumq2 = hsum_float_8(vq2);
#else
    for (int l = 0; l < kNg; ++l) {
        auto xl = xb + kGroupSize*l;
        auto wl = weight + kGroupSize*l;
        auto ql = m_values.data() + kGroupSize*best_idx[l];
        for (int k = 0; k < kGroupSize; ++k) {
            sumqx += wl[k]*ql[k]*xl[k];
            sumq2 += wl[k]*ql[k]*ql[k];
        }
    }
#endif
    return sumq2 > 0 ? std::make_pair(sumqx/sumq2, sumqx*sumqx/sumq2) : std::make_pair(0.f, 0.f);
}

template <int block_size, int group_size, int num_bits, bool is_abs, bool is_int>
float QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::find_best_inverse_scale(
        const float * xb, const float * weight, const int * best_idx) const {
    float sumqx = 0, sumx2 = 0;
#ifdef __AVX2__
    auto vqx = _mm256_setzero_ps();
    auto vx2 = _mm256_setzero_ps();
    for (int l = 0; l < kBlockSize; l += 8) {
        auto vx = _mm256_loadu_ps(xb+l);
        auto vw = _mm256_loadu_ps(weight+l);
        auto vq = kGroupSize == 8 ? _mm256_loadu_ps(m_values.data() + kGroupSize*best_idx[l/kGroupSize]) :
            _mm256_set_m128(_mm_loadu_ps(m_values.data() + kGroupSize*best_idx[l/kGroupSize+1]),
                            _mm_loadu_ps(m_values.data() + kGroupSize*best_idx[l/kGroupSize+0]));
        auto vxw = _mm256_mul_ps(vx, vw);
        vx2 = _mm256_fmadd_ps(vxw, vx, vx2);
        vqx = _mm256_fmadd_ps(vxw, vq, vqx);
    }
    sumqx = hsum_float_8(vqx);
    sumx2 = hsum_float_8(vx2);
#else
    for (int l = 0; l < kNg; ++l) {
        auto xl = xb + kGroupSize*l;
        auto wl = weight + kGroupSize*l;
        auto ql = m_values.data() + kGroupSize*best_idx[l];
        for (int k = 0; k < kGroupSize; ++k) {
            sumqx += wl[k]*ql[k]*xl[k];
            sumx2 += wl[k]*xl[k]*xl[k];
        }
    }
#endif
    return sumx2 > 0 ? sumqx/sumx2 : 0.f;
}

template <int block_size, int group_size, int num_bits, bool is_abs, bool is_int>
void QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::find_best_match(float d,
        [[maybe_unused]] const float * xb, [[maybe_unused]] const float * weight, int * best_idx) const {
    if (!d) {
        std::memset(best_idx, 0, kNg*sizeof(int));
        return;
    }
    [[maybe_unused]] int ncluster = m_clusters.size()/kGroupSize;
    [[maybe_unused]] float id = 1/d;
#ifdef __AVX2__
    if constexpr (kGroupSize == 8) {
        __m256 sqx[8];
        const __m256i add_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        float sx[8];
        int   index[8];
        auto vid = _mm256_set1_ps(id);
        auto add8 = _mm256_set1_epi32(8);
        for (int l = 0; l < kNg; ++l) {
            auto xl = xb + 8*l;
            auto wl = weight + 8*l;
            auto vx = _mm256_mul_ps(vid, _mm256_loadu_ps(xl));
            auto vw = _mm256_loadu_ps(wl);
            int jbest = -1;
            if (kGroupSize == 8 && (ncluster == 256 || ncluster == 6561)) {
                _mm256_store_ps(sx, vx);
                uint16_t u = 0;
                if (ncluster == 256) {
                    for (int j = 0; j < 8; ++j) if (sx[j] > m_mid[j]) u |= (1 << j);
                } else {
                    int s = 1;
                    for (int j = 0; j < 8; ++j) { u += s*bin3(j, sx[j]); s *= 3; }
                }
                jbest = u;
            } else {
                auto vbest = _mm256_set1_ps(INFINITY);
                auto best_index = _mm256_set1_epi32(-1);
                float best = INFINITY;
                auto idx = add_idx;
                for (int j = 0; j < ncluster; j += 8) {
                    for (int i = 0; i < 8; ++i) {
                        auto vq = _mm256_loadu_ps(m_clusters.data() + kGroupSize*(j+i));
                        auto vdiff = _mm256_sub_ps(vq, vx);
                        sqx[i] = _mm256_mul_ps(vw, _mm256_mul_ps(vdiff, vdiff));
                    }
                    auto score = hsum_float_8x8(sqx);
                    auto mask  = _mm256_cmp_ps(score, vbest, _CMP_LT_OQ);
                    best_index = _mm256_or_si256(_mm256_and_si256(_mm256_castps_si256(mask), idx),
                            _mm256_andnot_si256(_mm256_castps_si256(mask), best_index));
                    vbest = _mm256_min_ps(vbest, score);
                    idx = _mm256_add_epi32(idx, add8);
                }
                _mm256_store_ps(sx, vbest);
                _mm256_store_si256((__m256i *)index, best_index);
                for (int i = 0; i < 8; ++i) {
                    if (sx[i] < best) { best = sx[i]; jbest = index[i]; }
                }
            }
            auto& points = m_in_cluster[jbest];
            auto& values = points.empty() ? m_values : m_c_values[jbest];
            int npoint = values.size()/kGroupSize;
            GGML_ASSERT(npoint > 0 && npoint%8 == 0);
            int jbest_cluster = jbest;
            auto vbest = _mm256_set1_ps(INFINITY);
            auto best_index = _mm256_set1_epi32(-1);
            auto best = INFINITY; jbest = -1;
            auto idx = add_idx;
            for (int j = 0; j < npoint; j += 8) {
                for (int i = 0; i < 8; ++i) {
                    auto vq = _mm256_loadu_ps(values.data() + kGroupSize*(j+i));
                    auto vdiff = _mm256_sub_ps(vq, vx);
                    sqx[i] = _mm256_mul_ps(vw, _mm256_mul_ps(vdiff, vdiff));
                }
                auto score = hsum_float_8x8(sqx);
                auto mask  = _mm256_cmp_ps(score, vbest, _CMP_LT_OQ);
                best_index = _mm256_or_si256(_mm256_and_si256(_mm256_castps_si256(mask), idx),
                        _mm256_andnot_si256(_mm256_castps_si256(mask), best_index));
                vbest = _mm256_min_ps(vbest, score);
                idx = _mm256_add_epi32(idx, add8);
            }
            _mm256_store_ps(sx, vbest);
            _mm256_store_si256((__m256i *)index, best_index);
            for (int i = 0; i < 8; ++i) {
                if (sx[i] < best) { best = sx[i]; jbest = index[i]; }
            }
            if (jbest < 0) {
                fprintf(stderr, "Oops: jbest = %d for cluster %d with %d points\n", jbest, jbest_cluster, int(points.size()));
                GGML_ASSERT(false);
            }
            best_idx[l] = points.empty() ? jbest : points[jbest];
        }
    } else {
        __m256 sqx[4];
        const __m256i add_idx = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
        const __m256 sign_bit = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        float sx[8];
        int   index[8];
        auto vid_p = _mm256_set1_ps(id);
        auto add8 = _mm256_set1_epi32(8);
        for (int l = 0; l < kNg; ++l) {
            auto xl = xb + 4*l;
            auto wl = weight + 4*l;
            auto vx4 = _mm_loadu_ps(xl);
            auto vx = _mm256_mul_ps(vid_p, _mm256_set_m128(vx4, vx4));
            auto vw4 = _mm_loadu_ps(wl);
            auto vw = _mm256_set_m128(vw4, vw4);
            int jbest = -1;
            if (ncluster == 256 || ncluster == 625) {
                _mm256_storeu_ps(sx, vx);
                uint16_t u = 0;
                if (ncluster == 256) {
                    for (int k = 0; k < 4; ++k) u |= (bin4(sx[k]) << 2*k);
                } else {
                    int l = 1;
                    for (int k = 0; k < 4; ++k) { u += bin5(sx[k])*l; l *= 5; }
                }
                jbest = u;
            } else {
                auto vbest = _mm256_set1_ps(INFINITY);
                auto best_index = _mm256_set1_epi32(-1);
                float best = INFINITY;
                auto idx = add_idx;
                for (int j = 0; j < ncluster; j += 8) {
                    for (int i = 0; i < 4; ++i) {
                        auto vq = _mm256_loadu_ps(m_clusters.data() + kGroupSize*(j+2*i));
                        auto vdiff = _mm256_sub_ps(vq, vx);
                        vdiff = _mm256_and_ps(sign_bit, vdiff);
                        sqx[i] = _mm256_mul_ps(vw, _mm256_mul_ps(vdiff, _mm256_mul_ps(vdiff, vdiff)));
                    }
                    auto score = hsum_float_4x8(sqx);
                    auto mask  = _mm256_cmp_ps(score, vbest, _CMP_LT_OQ);
                    best_index = _mm256_or_si256(_mm256_and_si256(_mm256_castps_si256(mask), idx),
                            _mm256_andnot_si256(_mm256_castps_si256(mask), best_index));
                    vbest = _mm256_min_ps(vbest, score);
                    idx = _mm256_add_epi32(idx, add8);
                }
                _mm256_store_ps(sx, vbest);
                _mm256_store_si256((__m256i *)index, best_index);
                for (int i = 0; i < 8; ++i) {
                    if (sx[i] < best) { best = sx[i]; jbest = index[i]; }
                }
            }
            auto& points = m_in_cluster[jbest];
            auto& values = m_c_values[jbest];
            GGML_ASSERT(!points.empty() && points.size()%8 == 0);
            int jbest_cluster = jbest;
            auto vbest = _mm256_set1_ps(INFINITY);
            auto best_index = _mm256_set1_epi32(-1);
            float best = INFINITY; jbest = -1;
            auto idx = add_idx;
            for (int j = 0; j < int(points.size()); j += 8) {
                for (int i = 0; i < 4; ++i) {
                    auto vq = _mm256_loadu_ps(values.data() + kGroupSize*(j+2*i));
                    auto vdiff = _mm256_sub_ps(vq, vx);
                    sqx[i] = _mm256_mul_ps(vw, _mm256_mul_ps(vdiff, vdiff));
                }
                auto score = hsum_float_4x8(sqx);
                auto mask  = _mm256_cmp_ps(score, vbest, _CMP_LT_OQ);
                best_index = _mm256_or_si256(_mm256_and_si256(_mm256_castps_si256(mask), idx),
                                       _mm256_andnot_si256(_mm256_castps_si256(mask), best_index));
                vbest = _mm256_min_ps(vbest, score);
                idx = _mm256_add_epi32(idx, add8);
            }
            _mm256_store_ps(sx, vbest);
            _mm256_store_si256((__m256i *)index, best_index);
            for (int i = 0; i < 8; ++i) {
                if (sx[i] < best) { best = sx[i]; jbest = index[i]; }
            }
            if (jbest < 0) {
                fprintf(stderr, "Oops: jbest = %d for cluster %d with %d points\n", jbest, jbest_cluster, int(points.size()));
                GGML_ASSERT(false);
            }
            best_idx[l] = points[jbest];
        }
    }
#else
    // Scalar port of the AVX2 search above (fork addition): same cluster pre-selection
    // (direct binning for the structured cluster counts, else nearest cluster), then the
    // weighted nearest trellis point inside the chosen cluster.
    auto wdist2 = [&](const float * q, const float * x, const float * w) {
        float s = 0;
        for (int k = 0; k < kGroupSize; ++k) { const float t = q[k] - x[k]; s += w[k]*t*t; }
        return s;
    };
    float sx[8];
    for (int l = 0; l < kNg; ++l) {
        const float * xl = xb + kGroupSize*l;
        const float * wl = weight + kGroupSize*l;
        for (int k = 0; k < kGroupSize; ++k) sx[k] = id*xl[k];
        int jbest = -1;
        if constexpr (kGroupSize == 8) {
            if (ncluster == 256 || ncluster == 6561) {
                uint16_t u = 0;
                if (ncluster == 256) {
                    for (int j = 0; j < 8; ++j) if (sx[j] > m_mid[j]) u |= (1 << j);
                } else {
                    int sc = 1;
                    for (int j = 0; j < 8; ++j) { u += sc*bin3(j, sx[j]); sc *= 3; }
                }
                jbest = u;
            } else {
                float best = INFINITY;
                for (int j = 0; j < ncluster; ++j) {
                    const float d2 = wdist2(m_clusters.data() + kGroupSize*j, sx, wl);
                    if (d2 < best) { best = d2; jbest = j; }
                }
            }
        } else {
            if (ncluster == 256 || ncluster == 625) {
                uint16_t u = 0;
                if (ncluster == 256) {
                    for (int k = 0; k < 4; ++k) u |= (bin4(sx[k]) << 2*k);
                } else {
                    int sc = 1;
                    for (int k = 0; k < 4; ++k) { u += bin5(sx[k])*sc; sc *= 5; }
                }
                jbest = u;
            } else {
                float best = INFINITY;
                for (int j = 0; j < ncluster; ++j) {
                    const float * q = m_clusters.data() + kGroupSize*j;
                    float d3 = 0;
                    for (int k = 0; k < 4; ++k) { const float t = std::abs(q[k] - sx[k]); d3 += wl[k]*t*t*t; }
                    if (d3 < best) { best = d3; jbest = j; }
                }
            }
        }
        const auto & points = m_in_cluster[jbest];
        const auto & values = (kGroupSize == 8 && points.empty()) ? m_values : m_c_values[jbest];
        const int npoint = values.size()/kGroupSize;
        GGML_ASSERT(npoint > 0);
        float best = INFINITY; int ibest = -1;
        for (int j = 0; j < npoint; ++j) {
            const float d2 = wdist2(values.data() + kGroupSize*j, sx, wl);
            if (d2 < best) { best = d2; ibest = j; }
        }
        GGML_ASSERT(ibest >= 0);
        best_idx[l] = points.empty() ? ibest : points[ibest];
    }
#endif
}

template <int block_size, int group_size, int num_bits, bool is_abs, bool is_int>
std::vector<std::vector<int>> QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::finalize_clusters(int num_neighbours,
        const std::vector<float>& values, const std::vector<float>& clusters, std::vector<std::vector<float>>& c_values) {
    int ncluster = clusters.size()/kGroupSize;
    std::vector<std::vector<int>> p_in_cluster(ncluster);
    std::vector<int> which_cluster(num_neighbours*kNumVal);
    std::vector<int> ibest(num_neighbours);
    std::vector<float> best(num_neighbours);
    for (int ip = 0; ip < kNumVal; ++ip) {
        auto vp = values.data() + ip*kGroupSize;
        for (int j = 0; j < num_neighbours; ++j) {
            best[j] = INFINITY; ibest[j] = -1;
        }
        for (int ic = 0; ic < ncluster; ++ic) {
            auto vc = clusters.data() + ic*kGroupSize;
            float dist2 = 0;
            for (int k = 0; k < kGroupSize; ++k) {
                float d = vp[k] - vc[k]; dist2 += d*d;
            }
            for (int j = 0; j < num_neighbours; ++j) {
                if (dist2 < best[j]) {
                    for (int k = num_neighbours-1; k > j; --k) {
                        best[k] = best[k-1]; ibest[k] = ibest[k-1];
                    }
                    best[j] = dist2; ibest[j] = ic;
                    break;
                }
            }
        }
        for (int j = 0; j < num_neighbours; ++j) {
            if (ibest[j] < 0) {
                printf("Oops: ibest[%d] = %d\n", j, ibest[j]);
            }
            GGML_ASSERT(ibest[j] >= 0);
            p_in_cluster[ibest[j]].push_back(ip);
        }
        std::memcpy(which_cluster.data() + num_neighbours*ip, ibest.data(), num_neighbours*sizeof(int));
    }
    std::vector<std::pair<float, int>> extra;
    extra.reserve(kNumVal);
    for (int ic = 0; ic < ncluster; ++ic) {
        auto& points = p_in_cluster[ic];
        if (!points.empty() && points.size()%8 == 0) continue;
        extra.clear();
        auto vc = clusters.data() + ic*kGroupSize;
        for (int ip = 0; ip < kNumVal; ++ip) {
            bool can_add = true;
            for (int j = 0; j < num_neighbours; ++j) {
                if (which_cluster[num_neighbours*ip+j] == ic) { can_add = false; break; }
            }
            if (!can_add) continue;
            auto vp = values.data() + ip*kGroupSize;
            float dist2 = 0;
            for (int k = 0; k < kGroupSize; ++k) {
                float d = vp[k] - vc[k]; dist2 += d*d;
            }
            extra.push_back(std::make_pair(dist2, ip));
        }
        std::sort(extra.begin(), extra.end());
        int nadd = 8*((points.size()+7)/8) - points.size();
        for (int i = 0; i < nadd; ++i) points.push_back(extra[i].second);
        GGML_ASSERT(points.size()%8 == 0);
    }
    auto min = p_in_cluster.front().size(), max = p_in_cluster.front().size();
    for (auto& points : p_in_cluster) {
        min = std::min(min, points.size());
        max = std::max(max, points.size());
    }
    c_values.resize(p_in_cluster.size());
    for (int i = 0; i < int(p_in_cluster.size()); ++i) {
        auto& points = p_in_cluster[i];
        c_values[i].resize(points.size()*kGroupSize);
        auto ptr = c_values[i].data();
        for (auto j : points) {
            std::memcpy(ptr, values.data() + j*kGroupSize, kGroupSize*sizeof(float));
            ptr += kGroupSize;
        }
    }

    if (kVerbose) {
        printf("%s: prepared %d clusters\n", __func__, ncluster);
        printf("    min number of points in a cluster: %d\n", int(min));
        printf("    max number of points in a cluster: %d\n", int(max));
    }
    return p_in_cluster;
}

template <int block_size, int group_size, int num_bits, bool is_abs, bool is_int>
std::vector<float> QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::cluster_points(const std::vector<float>& points, int ncluster, int niter, float * mid) {
    constexpr int ndim = kGroupSize;
    GGML_ASSERT(points.size() % ndim == 0);
    int npoint = points.size() / ndim;
    GGML_ASSERT(npoint >= 2*ncluster);
    std::vector<std::pair<float, float>> range(ndim, std::make_pair(INFINITY, -INFINITY));
    double Fo = 0;
    for (int i = 0; i < npoint; ++i) {
        auto v = points.data() + i*ndim;
        for (int k = 0; k < ndim; ++k) {
            Fo += v[k]*v[k];
            range[k].first  = std::min(range[k].first, v[k]);
            range[k].second = std::max(range[k].second, v[k]);
        }
    }
    if (kVerbose) printf("%s (ndim = %d, npoint = %d): Fo = %g\n", __func__, ndim, npoint, Fo/points.size());
    if constexpr (is_abs) {
        std::vector<int> P(npoint);
        for (int idim = 0; idim < ndim; ++idim) {
            for (int ip = 0; ip < npoint; ++ip) P[ip] = points[ip*ndim+idim];
            std::sort(P.begin(), P.end());
            if (ndim == 8 && ncluster == 6561) {
                mid[2*idim + 0] = P[npoint/3];
                mid[2*idim + 1] = P[2*npoint/3];
            } else {
                mid[idim] = npoint%2 == 0 ? 0.5f*(P[npoint/2] + P[npoint/2-1]) : P[npoint/2];
                if (kVerbose) printf("%s: mid[%d] = %g\n", __func__, idim, mid[idim]);
            }
        }
    } else {
        for (int k = 0; k < ndim; ++k) mid[k] = 0.5f*(range[k].first + range[k].second);
    }
    std::vector<float> sump(ncluster*ndim);
    std::vector<int> counts(ncluster);
    std::vector<float> result(ncluster*ndim);
    if (ndim == 8 && (ncluster == 256 || ncluster == 6561)) {
        std::memset(sump.data(), 0, sump.size()*sizeof(float));
        std::memset(counts.data(), 0, counts.size()*sizeof(int));
        for (int ip = 0; ip < npoint; ++ip) {
            auto vp = points.data() + ndim*ip;
            uint16_t u = 0;
            if (ncluster == 256) {
                for (int k = 0; k < ndim; ++k) if (vp[k] > mid[k]) u |= (1 << k);
            } else {
                int s = 1;
                for (int k = 0; k < ndim; ++k) {
                    int bin = vp[k] < mid[2*k+0] ? 0 : vp[k] < mid[2*k+1] ? 1 : 2;
                    u += s*bin; s *= 3;
                }
            }
            ++counts[u];
            for (int k = 0; k < ndim; ++k) sump[ndim*u + k] += vp[k];
        }
        for (int ic = 0; ic < ncluster; ++ic) {
            if (!counts[ic]) {
                printf("%s: Oops. Cluster %d has no points\n", __func__, ic);
                GGML_ABORT("fatal error");
            }
            for (int k = 0; k < ndim; ++k) result[ic*ndim + k] = sump[ic*ndim + k]/counts[ic];
        }
        return result;
    }
    else if (ndim == 4 && (ncluster == 256 || ncluster == 625)) {
        std::memset(sump.data(), 0, sump.size()*sizeof(float));
        std::memset(counts.data(), 0, counts.size()*sizeof(int));
        for (int ip = 0; ip < npoint; ++ip) {
            auto vp = points.data() + ndim*ip;
            uint16_t u = 0;
            if (ncluster == 256) {
                for (int k = 0; k < ndim; ++k) u |= (bin4(vp[k]) << 2*k);
            } else {
                int s = 1;
                for (int k = 0; k < ndim; ++k) { u += s*bin5(vp[k]); s *= 5; }
            }
            if (u >= int(counts.size())) {
                printf("Oops: u = %u, vp = %g, %g, %g, %g\n", u, vp[0], vp[1], vp[2], vp[3]);
                u = 0;
                if (ncluster == 256) {
                    for (int k = 0; k < ndim; ++k) {
                        auto bin = bin4(vp[k]); u |= (bin << 2*k);
                        printf(" bin[%d] = %d, u = %u", k, bin, u);
                    }
                } else {
                    for (int k = 0; k < ndim; ++k) printf(" bin[%d] = %d", k, bin5(vp[k]));
                }
                printf("\n");
                GGML_ABORT("fatal error");
            }
            ++counts[u];
            for (int k = 0; k < ndim; ++k) sump[ndim*u + k] += vp[k];
        }
        int nzero = 0;
        for (int ic = 0; ic < ncluster; ++ic) {
            if (!counts[ic]) {
                ++nzero;
                printf("%s: Oops. Cluster %d has no points: ", __func__, ic);
                for (int k = 0; k < ndim; ++k) {
                    int l = (ic >> 2*k) & 3;
                    printf(" %d", l);
                }
                printf("\n");
            } else {
                for (int k = 0; k < ndim; ++k) result[ic*ndim + k] = sump[ic*ndim + k]/counts[ic];
            }
        }
        if (nzero > 0) printf("%s: %d out of %d clusters dir not have any points\n", __func__, nzero, ncluster);
        return result;
    }
    std::mt19937 rndm(1234);
    float scale = 1.f/4294967296.f;
    for (int i = 0; i < ncluster; ++i) {
        auto v = result.data() + i*ndim;
        for (int k = 0; k < ndim; ++k) v[k] = range[k].first + (range[k].second - range[k].first)*scale*rndm();
    }
    std::vector<int> which_cluster(npoint, -1);
    double Flast = Fo;
    for (int iter = 0; iter < niter; ++iter) {
        std::memset(sump.data(), 0, sump.size()*sizeof(float));
        std::memset(counts.data(), 0, counts.size()*sizeof(int));
        int nchanged = 0;
        double F = 0;
        for (int ip = 0; ip < npoint; ++ip) {
            auto vp = points.data() + ndim*ip;
            float best = INFINITY; int ibest = -1;
            for (int ic = 0; ic < ncluster; ++ic) {
                auto vc = result.data() + ndim*ic;
                float dist2 = 0;
                for (int k = 0; k < ndim; ++k) {
                    float d = vp[k] - vc[k]; dist2 += d*d;
                }
                if (dist2 < best) {
                    best = dist2; ibest = ic;
                }
            }
            if (ibest < 0) {
                printf("Oops(iteration %d) - failed to find cluster for point", iter);
                for (int k = 0; k < ndim; ++k) printf(" %g", vp[k]);
                printf("\nHave %d clusters\n", ncluster);
            }
            GGML_ASSERT(ibest >= 0);
            F += best;
            if (which_cluster[ip] != ibest) ++nchanged;
            which_cluster[ip] = ibest;
            ++counts[ibest];
            auto vc = sump.data() + ndim*ibest;
            for (int k = 0; k < ndim; ++k) vc[k] += vp[k];
        }
        if (nchanged == 0) break;
        for (int ic = 0; ic < ncluster; ++ic) {
            float norm = counts[ic] > 0 ? 1.f/counts[ic] : 0.f;
            auto vc = sump.data() + ndim*ic;
            auto r  = result.data() + ndim*ic;
            for (int k = 0; k < ndim; ++k) r[k] = vc[k]*norm;
        }
        if (kVerbose) printf("%s(iteration %d): F = %g, nchanged = %d\n", __func__, iter+1, F/points.size(), nchanged);
        if (iter > 1 && Flast/F - 1 < 1e-6) break;
        Flast = F;
    }
    int nzero = 0;
    for (int ic = 0; ic < ncluster; ++ic) {
        if (!counts[ic]) ++nzero;
    }
    if (nzero > 0) printf("%s: there are %d empty clusters\n", __func__, nzero);
    return result;
}

// ========================================== iq1_kt ====================================================

namespace {
void iqk_kt_can_quantize([[maybe_unused]] ggml_type type) {
    // fork: a scalar find_best_match is provided, so no AVX2 requirement
}
}

using QuantizerIQ1KT = QuantizerIQKT<32, 8, 13, false, true>;

const QuantizerIQ1KT& iq1kt_quantizer() {
    static std::mutex mutex;
    static std::unique_ptr<QuantizerIQ1KT> quantizer;
    std::lock_guard<std::mutex> lock(mutex);
    if (!quantizer) quantizer = std::make_unique<QuantizerIQ1KT>(256, 32);
    return *quantizer;
}

void quantize_row_iq1_kt_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales, float * all_weights,
        int * all_idx) {

    constexpr float kSigmaScale = 2.0f;
    using Q = QuantizerIQ1KT;

    static_assert(Q::kNumVal%8 == 0);

    float * dptr = (float *)vy;

    block_iq1_kt * y = (block_iq1_kt *)(dptr + 1);

    int   best_idx[2*Q::kNg];

    auto& quantizer = iq1kt_quantizer();

    int nblock = n_per_row / Q::kSuperBlockSize;

    Q::set_weights(kSigmaScale, nblock, x, quant_weights, all_weights, n_per_row);

    float amax_row = 0;
    for (int j = 0; j < n_per_row; ++j) {
        amax_row = std::max(amax_row, std::abs(x[j]));
    }

    float amax_scale = 0, max_scale = 0;

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq1_kt));

        const float * xbl = x + ibl*Q::kSuperBlockSize;
        auto scales = all_scales + ibl*Q::kNblock;

        for (int ib = 0; ib < Q::kNblock; ++ib) {
            const float * xb = xbl + Q::kBlockSize*ib;
            const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
            float amax = 0;
            for (int j = 0; j < Q::kBlockSize; ++j) {
                float ax = std::abs(xb[j]);
                amax = std::max(amax, ax);
            }
            if (amax < 1e-16f) {
                scales[ib] = 0.0f;
                for (int ig = 0; ig < Q::kNg; ++ig) all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig] = 0;
                continue;
            }
            float scale_0 = std::max(90.f, 124.f*amax/amax_row);
            quantizer.find_best_match( amax/scale_0, xb, weight, best_idx);
            auto [dp, score_p] = quantizer.find_best_scale(xb, weight, best_idx);
            quantizer.find_best_match(-amax/scale_0, xb, weight, best_idx + Q::kNg);
            auto [dm, score_m] = quantizer.find_best_scale(xb, weight, best_idx + Q::kNg);

            auto idx = best_idx;
            if (score_p > score_m) scales[ib] = dp;
            else {
                scales[ib] = dm; idx += Q::kNg; score_p = score_m;
            }
            for (int ig = 0; ig < Q::kNg; ++ig) all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig] = idx[ig];

            scale_0 -= 8;
            quantizer.find_best_match( amax/scale_0, xb, weight, best_idx);
            auto [dp1, score_p1] = quantizer.find_best_scale(xb, weight, best_idx);
            quantizer.find_best_match(-amax/scale_0, xb, weight, best_idx + Q::kNg);
            auto [dm1, score_m1] = quantizer.find_best_scale(xb, weight, best_idx + Q::kNg);

            if (score_p1 > score_p || score_m1 > score_p) {
                idx = best_idx;
                if (score_p1 > score_m1) scales[ib] = dp1;
                else {
                    scales[ib] = dm1; idx += Q::kNg;
                }
                for (int ig = 0; ig < Q::kNg; ++ig) all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig] = idx[ig];
            }

            float abs_scale = std::abs(scales[ib]);
            if (abs_scale > amax_scale) {
                amax_scale = abs_scale;
                max_scale = scales[ib];
            }
        }

    }

    if (!max_scale) {
        *dptr = 0;
        return;
    }

    float d = max_scale/iq4k_values[0];
    float best = 0;
    for (int itry = -9; itry <= 9; ++itry) {
        float id = (itry + iq4k_values[0])/max_scale;
        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * xb = x + ibl*Q::kSuperBlockSize;
            const float * wb = all_weights + ibl*Q::kSuperBlockSize;
            auto scales = all_scales + ibl*Q::kNblock;
            for (int ib = 0; ib < Q::kNblock; ++ib) {
                int ls = best_index_iq4nl(iq4k_values, id*scales[ib]);
                float dl = iq4k_values[ls];
                for (int ig = 0; ig < Q::kNg; ++ig) {
                    auto qb = quantizer.values() + Q::kGroupSize*all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig];
                    for (int j = 0; j < Q::kGroupSize; ++j) {
                        int jj = ig*Q::kGroupSize + j;
                        float q = dl*qb[j];
                        sumqx += wb[jj]*xb[jj]*q;
                        sumq2 += wb[jj]*q*q;
                    }
                }
                xb += Q::kBlockSize;
                wb += Q::kBlockSize;
            }
        }
        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
            d = sumqx/sumq2; best = d*sumqx;
        }
    }

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ1_KT);

    float id = d ? 1/d : 0.f;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto scales = all_scales + ibl*Q::kNblock;
        for (int ib = 0; ib < Q::kNblock; ++ib) {
            int ls = best_index_iq4nl(iq4k_values, id*scales[ib]);
            y[ibl].sh[ib] = ls;
        }
    }

    *dptr = fudge*d;
    if (!d) return;

    for (int iloop = 0; iloop < 1; ++iloop) {

        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {

            const float * xbl = x + ibl*Q::kSuperBlockSize;

            for (int ib = 0; ib < Q::kNblock; ++ib) {
                const float * xb = xbl + Q::kBlockSize*ib;
                const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
                int ls = iq4k_values[y[ibl].sh[ib] & 0xf];
                float dl = d*ls;
                quantizer.find_best_match(dl, xb, weight, best_idx);

                auto prev_idx = all_idx + (ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize;

                float mse1 = 0, mse2 = 0;
                for (int ig = 0; ig < Q::kNg; ++ig) {
                    auto q1 = quantizer.values() + Q::kGroupSize*prev_idx[ig];
                    auto q2 = quantizer.values() + Q::kGroupSize*best_idx[ig];
                    for (int j = 0; j < Q::kGroupSize; ++j) {
                        int jj = ig*Q::kGroupSize + j;
                        float diff1 = xb[jj] - dl*q1[j];
                        float diff2 = xb[jj] - dl*q2[j];
                        mse1 += weight[jj]*diff1*diff1;
                        mse2 += weight[jj]*diff2*diff2;
                    }
                }
                if (mse1 < mse2) {
                    for (int ig = 0; ig < Q::kNg; ++ig) best_idx[ig] = prev_idx[ig];
                } else {
                    for (int ig = 0; ig < Q::kNg; ++ig) prev_idx[ig] = best_idx[ig];
                }

                for (int j = 0; j < Q::kNg; ++j) {
                    y[ibl].ql[ib*Q::kNg+j] = best_idx[j] & 0xff;
                    y[ibl].qh[(ib%(Q::kNblock/2))*Q::kNg+j] |= (((best_idx[j] >> 8) & 0xf) << 4*(ib/(Q::kNblock/2)));
                    y[ibl].sh[ib] |= ((best_idx[j] >> 12) << (4+j));
                    auto xl = xb + Q::kGroupSize*j;
                    auto wl = weight + Q::kGroupSize*j;
                    auto ql = quantizer.values() + best_idx[j]*Q::kGroupSize;
                    for (int k = 0; k < Q::kGroupSize; ++k) {
                        float q = ql[k]*ls;
                        sumqx += wl[k]*xl[k]*q;
                        sumq2 += wl[k]*q*q;
                    }
                }
            }
        }
        if (sumq2 > 0) {
            d = sumqx/sumq2;
            *dptr = d * fudge;
            if (!d) return;
        } else {
            break;
        }

    }

}
}

void quantize_row_iq1_kt_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq1_kt(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq1_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq1_kt * y = (block_iq1_kt *)vy;
    quantize_row_iq1_kt_ref(x, y, k);
}

size_t quantize_iq1_kt(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    iqk_kt_can_quantize(GGML_TYPE_IQ1_KT);
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ1_KT, n_per_row);
    std::vector<float> scales(n_per_row/QuantizerIQ1KT::kBlockSize);
    std::vector<float> weights(n_per_row);
    std::vector<int> idx(n_per_row/QuantizerIQ1KT::kGroupSize);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_iq1_kt_impl(src, (void *)qrow, n_per_row, imatrix, scales.data(), weights.data(), idx.data());
        src += n_per_row;
        qrow += row_size;
    }
    return nrows * row_size;
}

void dequantize_row_iq1_kt(const void * vx_, float * y, int64_t k) {
    const block_iq1_kt * x = (const block_iq1_kt *)vx_;
    assert(k % QuantizerIQ1KT::kSuperBlockSize == 0);
    using Q = QuantizerIQ1KT;
    const int nb = k / Q::kSuperBlockSize;
    const float * dptr = (const float *)x;
    const float d = *dptr * Q::kScale;
    x = (const block_iq1_kt *)(dptr + 1);
    auto& deq = iq1kt_quantizer();
    for (int ibl = 0; ibl < nb; ++ibl) {
        for (int ib = 0; ib < Q::kNblock; ++ib) {
            float sl = d * iq4k_values[x[ibl].sh[ib] & 0xf];
            for (int ig = 0; ig < Q::kNg; ++ig) {
                uint16_t idx = x[ibl].ql[ib*Q::kNg + ig] | ((x[ibl].qh[(ib%(Q::kNblock/2))*Q::kNg + ig] << (8 - 4*(ib/(Q::kNblock/2)))) & 0xf00);
                idx |= (x[ibl].sh[ib] << (8 - ig) & 0x1000);
                deq.set_values(idx, y, sl);
                y += Q::kGroupSize;
            }
        }
    }
}

void vec_dot_iq1_kt_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    *s = iqk_kt_row_vec_dot_dequant_q8_K(n, vx, vy, dequantize_row_iq1_kt);

}

// ========================================== iq2_kt ====================================================

namespace {

using QuantizerIQ2KT = QuantizerIQKT<32, 8, 16, false, true>;

const QuantizerIQ2KT& iq2kt_quantizer() {
    static std::mutex mutex;
    static std::unique_ptr<QuantizerIQ2KT> quantizer;
    std::lock_guard<std::mutex> lock(mutex);
    if (!quantizer) quantizer = std::make_unique<QuantizerIQ2KT>(256, 8);
    return *quantizer;
}

void quantize_row_iq2_kt_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales, float * all_weights,
        int * all_idx) {

    constexpr float kSigmaScale = 2.0f;
    using Q = QuantizerIQ2KT;

    static_assert(Q::kNumVal%8 == 0);

    float * dptr = (float *)vy;

    block_iq2_kt * y = (block_iq2_kt *)(dptr + 1);

    int   best_idx[2*Q::kNg];

    auto& quantizer = iq2kt_quantizer();

    int nblock = n_per_row / Q::kSuperBlockSize;

    Q::set_weights(kSigmaScale, nblock, x, quant_weights, all_weights, n_per_row);

    float amax_row = 0;
    for (int j = 0; j < n_per_row; ++j) {
        amax_row = std::max(amax_row, std::abs(x[j]));
    }

    float amax_scale = 0, max_scale = 0;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ2_KT);

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq2_kt));

        const float * xbl = x + ibl*Q::kSuperBlockSize;
        auto scales = all_scales + ibl*Q::kNblock;

        for (int ib = 0; ib < Q::kNblock; ++ib) {
            const float * xb = xbl + Q::kBlockSize*ib;
            const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
            float amax = 0;
            for (int j = 0; j < Q::kBlockSize; ++j) {
                float ax = std::abs(xb[j]);
                amax = std::max(amax, ax);
            }
            if (amax < 1e-16f) {
                scales[ib] = 0.0f;
                for (int ig = 0; ig < Q::kNg; ++ig) all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig] = 0;
                continue;
            }
            float scale_0 = std::max(90.f, 124.f*amax/amax_row);
            quantizer.find_best_match( amax/scale_0, xb, weight, best_idx);
            auto [dp, score_p] = quantizer.find_best_scale(xb, weight, best_idx);
            quantizer.find_best_match(-amax/scale_0, xb, weight, best_idx + Q::kNg);
            auto [dm, score_m] = quantizer.find_best_scale(xb, weight, best_idx + Q::kNg);

            auto idx = best_idx;
            if (score_p > score_m) scales[ib] = dp;
            else {
                scales[ib] = dm; idx += Q::kNg;
            }
            for (int ig = 0; ig < Q::kNg; ++ig) all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig] = idx[ig];

            float abs_scale = std::abs(scales[ib]);
            if (abs_scale > amax_scale) {
                amax_scale = abs_scale;
                max_scale = scales[ib];
            }
        }

    }

    if (!max_scale) {
        *dptr = 0;
        return;
    }

    float d = max_scale/iq4k_values[0];
    float best = 0;
    for (int itry = -9; itry <= 9; ++itry) {
        float id = (itry + iq4k_values[0])/max_scale;
        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * xb = x + ibl*Q::kSuperBlockSize;
            const float * wb = all_weights + ibl*Q::kSuperBlockSize;
            auto scales = all_scales + ibl*Q::kNblock;
            for (int ib = 0; ib < Q::kNblock; ++ib) {
                int ls = best_index_iq4nl(iq4k_values, id*scales[ib]);
                float dl = iq4k_values[ls];
                for (int ig = 0; ig < Q::kNg; ++ig) {
                    auto qb = quantizer.values() + Q::kGroupSize*all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig];
                    for (int j = 0; j < Q::kGroupSize; ++j) {
                        int jj = ig*Q::kGroupSize + j;
                        float q = dl*qb[j];
                        sumqx += wb[jj]*xb[jj]*q;
                        sumq2 += wb[jj]*q*q;
                    }
                }
                xb += Q::kBlockSize;
                wb += Q::kBlockSize;
            }
        }
        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
            d = sumqx/sumq2; best = d*sumqx;
        }
    }

    float id = d ? 1/d : 0.f;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto scales = all_scales + ibl*Q::kNblock;
        for (int ib = 0; ib < Q::kNblock/2; ++ib) {
            int ls1 = best_index_iq4nl(iq4k_values, id*scales[ib]);
            int ls2 = best_index_iq4nl(iq4k_values, id*scales[ib + Q::kNblock/2]);
            y[ibl].scales[ib] = ls1 | (ls2 << 4);
        }
    }

    *dptr = d * fudge;
    if (!d) return;

    for (int iloop = 0; iloop < 1; ++iloop) {

        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {

            auto qs = (uint16_t *)y[ibl].ql;
            const float * xbl = x + ibl*Q::kSuperBlockSize;

            for (int ib = 0; ib < Q::kNblock; ++ib) {
                const float * xb = xbl + Q::kBlockSize*ib;
                const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
                int ls = iq4k_values[(y[ibl].scales[ib%(Q::kNblock/2)] >> 4*(ib/(Q::kNblock/2))) & 0xf];
                float dl = d*ls;
                quantizer.find_best_match(dl, xb, weight, best_idx);

                auto prev_idx = all_idx + (ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize;

                float mse1 = 0, mse2 = 0;
                for (int ig = 0; ig < Q::kNg; ++ig) {
                    auto q1 = quantizer.values() + Q::kGroupSize*prev_idx[ig];
                    auto q2 = quantizer.values() + Q::kGroupSize*best_idx[ig];
                    for (int j = 0; j < Q::kGroupSize; ++j) {
                        int jj = ig*Q::kGroupSize + j;
                        float diff1 = xb[jj] - dl*q1[j];
                        float diff2 = xb[jj] - dl*q2[j];
                        mse1 += weight[jj]*diff1*diff1;
                        mse2 += weight[jj]*diff2*diff2;
                    }
                }
                if (mse1 < mse2) {
                    for (int ig = 0; ig < Q::kNg; ++ig) best_idx[ig] = prev_idx[ig];
                } else {
                    for (int ig = 0; ig < Q::kNg; ++ig) prev_idx[ig] = best_idx[ig];
                }

                for (int j = 0; j < Q::kNg; ++j) {
                    qs[j] = best_idx[j];
                    auto xl = xb + Q::kGroupSize*j;
                    auto wl = weight + Q::kGroupSize*j;
                    auto ql = quantizer.values() + best_idx[j]*Q::kGroupSize;
                    for (int k = 0; k < Q::kGroupSize; ++k) {
                        float q = ql[k]*ls;
                        sumqx += wl[k]*xl[k]*q;
                        sumq2 += wl[k]*q*q;
                    }
                }
                qs += Q::kNg;
            }
        }
        if (sumq2 > 0) {
            d = sumqx/sumq2;
            *dptr = d * fudge;
            if (!d) return;
        } else {
            break;
        }

        if (false) {
            for (int ibl = 0; ibl < nblock; ++ibl) {
                const float * xbl = x + ibl*Q::kSuperBlockSize;
                auto scales = all_scales + ibl*Q::kNblock;
                auto qs = (uint16_t *)y[ibl].ql;
                for (int ib = 0; ib < Q::kNblock; ++ib) {
                    const float * xb = xbl + Q::kBlockSize*ib;
                    const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
                    for (int j = 0; j < Q::kNg; ++j) best_idx[j] = qs[ib*Q::kNg+j];
                    auto pair = quantizer.find_best_scale(xb, weight, best_idx);
                    scales[ib] = pair.first;
                }
            }
            float id = d ? 1/d : 0.f;
            for (int ibl = 0; ibl < nblock; ++ibl) {
                auto scales = all_scales + ibl*Q::kNblock;
                for (int ib = 0; ib < Q::kNblock/2; ++ib) {
                    int ls1 = best_index_iq4nl(iq4k_values, id*scales[ib]);
                    int ls2 = best_index_iq4nl(iq4k_values, id*scales[ib + Q::kNblock/2]);
                    y[ibl].scales[ib] = ls1 | (ls2 << 4);
                }
            }
        }

    }

}
}

namespace {

template <typename Impl>
size_t quantize_kt_rows(ggml_type type, const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix, Impl impl) {
    GGML_ASSERT(n_per_row%32 == 0);
    const size_t row_size = ggml_row_size(type, n_per_row);
    const int nt = (n_per_row & (QK_K - 1))/32;
    char * qrow = (char *)dst;
    if (nt == 0) {
        for (int64_t row = 0; row < nrows; ++row) {
            impl(src, qrow, imatrix);
            src += n_per_row;
            qrow += row_size;
        }
        return nrows*row_size;
    }
    std::vector<float> xpad(GGML_PAD(n_per_row, QK_K), 0.f), wpad;
    if (imatrix) {
        wpad.assign(imatrix, imatrix + n_per_row);
        wpad.resize(xpad.size(), 0.f);
    }
    std::vector<char> ypad(ggml_row_size(type, GGML_PAD(n_per_row, QK_K)));
    for (int64_t row = 0; row < nrows; ++row) {
        std::memcpy(xpad.data(), src, n_per_row*sizeof(float));
        impl(xpad.data(), ypad.data(), imatrix ? wpad.data() : nullptr);
        iqk_kt_finish_row(type, ypad.data(), qrow, n_per_row);
        src += n_per_row;
        qrow += row_size;
    }
    return nrows*row_size;
}

void repack_tail_iq2_kt(const block_iq2_kt * x, int nt, uint8_t * tail) {
    if (nt < 1) return;
    std::memcpy(tail+0, x->scales, 4);
    std::memcpy(tail+4, x->ql, 8*nt);
}

inline void repack_kt_scales_iq3kt(const uint8_t * scales, int nt, uint8_t * tail) {
    std::memset(tail, 0, (nt + 1)/2);
    for (int ib = 0; ib < nt; ++ib) {
        tail[ib/2] |= ((scales[ib%4] >> 4*(ib/4)) & 0xf) << 4*(ib&1);
    }
}

void repack_tail_iq3_kt(const block_iq3_kt * x, int nt, uint8_t * tail) {
    std::memcpy(tail, x->ql, 8*nt);
    uint8_t * qh = tail + 8*nt;
    std::memset(qh, 0, 4*nt);
    for (int ib = 0; ib < nt; ++ib) {
        for (int j = 0; j < 32; ++j) {
            if (x->qh[j] & (1 << ib)) qh[4*ib + j/8] |= 1 << (j&7);
        }
    }
    repack_kt_scales_iq3kt(x->scales, nt, tail + 12*nt);
}

void repack_tail_iq4_kt(const block_iq4_kt * x, int nt, uint8_t * tail) {
    const uint8_t * ql = (const uint8_t *)(x->qs + 8);
    const uint8_t * qh = ql + 64;
    for (int ib = 0; ib < nt; ++ib) {
        uint8_t * record = tail + 16*ib;
        std::memcpy(record, x->qs + ib, 4);
        std::memcpy(record + 4, ql + 8*ib, 8);
        uint8_t * qht = record + 12;
        std::memset(qht, 0, 4);
        for (int jj = 0; jj < 8; ++jj) {
            const int j = 8*ib + jj;
            qht[jj/2] |= ((qh[j%32] >> 4*(j/32)) & 0xf) << 4*(jj&1);
        }
    }
}

}

namespace {
void iqk_kt_finish_row(int type, const void * ypad, void * qrow, int64_t n_per_row) {
    const int nb = n_per_row/QK_K;
    const int nt = (n_per_row & (QK_K - 1))/32;
    const size_t head = ggml_row_size(ggml_type(type), nb*QK_K);
    std::memset(qrow, 0, ggml_row_size(ggml_type(type), n_per_row));
    std::memcpy(qrow, ypad, head);
    if (nt == 0) return;
    const char * blocks = (const char *)ypad + sizeof(float);
    if (type == GGML_TYPE_IQ2_KT) {
        repack_tail_iq2_kt((const block_iq2_kt *)blocks + nb, nt, (uint8_t *)qrow + head);
    } else if (type == GGML_TYPE_IQ3_KT) {
        repack_tail_iq3_kt((const block_iq3_kt *)blocks + nb, nt, (uint8_t *)qrow + head);
    } else {
        repack_tail_iq4_kt((const block_iq4_kt *)blocks + nb, nt, (uint8_t *)qrow + head);
    }
}
}

void quantize_row_iq2_kt_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq2_kt(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq2_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq2_kt * y = (block_iq2_kt *)vy;
    quantize_row_iq2_kt_ref(x, y, k);
}

size_t quantize_iq2_kt(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    iqk_kt_can_quantize(GGML_TYPE_IQ2_KT);
    GGML_ASSERT(n_per_row%32== 0);
    int n_per_row_padded = GGML_PAD(n_per_row, QK_K);
    std::vector<float> scales(n_per_row_padded/QuantizerIQ2KT::kBlockSize);
    std::vector<float> weights(n_per_row_padded);
    std::vector<int>   idx(n_per_row_padded/QuantizerIQ2KT::kGroupSize);
    if (n_per_row % QK_K == 0) {
    return quantize_kt_rows(GGML_TYPE_IQ2_KT, src, dst, nrows, n_per_row, imatrix,
            [&] (const float * x, void * y, const float * w) { quantize_row_iq2_kt_impl(x, y, n_per_row, w, scales.data(), weights.data(), idx.data()); });
    }
    size_t row_size = ggml_row_size(GGML_TYPE_IQ2_KT, n_per_row);
    size_t row_size_padded = ggml_row_size(GGML_TYPE_IQ2_KT, n_per_row_padded);
    size_t row_size_nb = ggml_row_size(GGML_TYPE_IQ2_KT, QK_K*(n_per_row/QK_K));
    std::vector<float> xpad(n_per_row_padded);
    std::vector<char>  ypad(row_size_padded);
    int n_extra = n_per_row_padded - n_per_row;
    int n_padding = n_extra/32;
    std::vector<float> padded_imatrix;
    if (imatrix) {
        padded_imatrix.resize(n_per_row_padded);
        std::memcpy(padded_imatrix.data(), imatrix, n_per_row*sizeof(float));
        if (n_per_row >= n_extra) {
            std::memcpy(padded_imatrix.data() + n_per_row, imatrix + n_per_row - n_extra, n_extra*sizeof(float));
        } else {
            for (int ib = 0; ib < n_padding; ++ib) std::memcpy(padded_imatrix.data() + n_per_row + 32*ib, imatrix + n_per_row - 32, 32*sizeof(float));
        }
    }
    int nblock256 = n_per_row/QK_K;
    int nt = (n_per_row - nblock256*QK_K)/32;
    size_t size = 0;
    auto cy = (char *)dst;
    for (int row = 0; row < nrows; ++row) {
        std::memcpy(xpad.data(), src, n_per_row*sizeof(float));
        if (n_per_row >= n_extra) {
            std::memcpy(xpad.data() + n_per_row, src + n_per_row - n_extra, n_extra*sizeof(float));
        } else {
            for (int ib = 0; ib < n_padding; ++ib) std::memcpy(xpad.data() + n_per_row + 32*ib, src + n_per_row - 32, 32*sizeof(float));
        }
        quantize_row_iq2_kt_impl(xpad.data(), ypad.data(), n_per_row_padded, imatrix ? padded_imatrix.data() : nullptr, scales.data(), weights.data(), idx.data());
        block_iq2_kt *qy, *qy_padded;
        if (nblock256 >= 1) {
            std::memcpy(cy, ypad.data(), row_size_nb);
            qy = (block_iq2_kt *)(cy + row_size_nb);
            qy_padded = (block_iq2_kt *)(ypad.data() + row_size_nb);
        } else {
            std::memcpy(cy, ypad.data(), 4);
            qy = (block_iq2_kt *)(cy + sizeof(float));
            qy_padded = (block_iq2_kt *)(ypad.data() + sizeof(float));
        }
        std::memcpy(qy->scales, qy_padded->scales, 4);
        std::memcpy(qy->ql, qy_padded->ql, 8*nt);
        src += n_per_row;
        cy  += row_size;
        size += row_size;
    }
    return size;
}

void dequantize_row_iq2_kt(const void * vx_, float * y, int64_t k) {
    const block_iq2_kt * x = (const block_iq2_kt *)vx_;
    using Q = QuantizerIQ2KT;
    assert(k % Q::kBlockSize == 0);
    const int nb = k / Q::kSuperBlockSize;
    const float * dptr = (const float *)x;
    // ik_llama.cpp inference kernels (iqk_gemm_ktquants.cpp, CUDA) apply 1.05f at run time
    const float d = *dptr * Q::kScale * 1.05f;
    x = (const block_iq2_kt *)(dptr + 1);
    auto& deq = iq2kt_quantizer();
    for (int ibl = 0; ibl < nb; ++ibl) {
        auto yl = y + ibl*Q::kSuperBlockSize;
        auto yh = yl + Q::kSuperBlockSize/2;
        const uint16_t * ql = (const uint16_t *)x[ibl].ql;
        const uint16_t * qh = ql + Q::kNg*QuantizerIQ2KT::kNblock/2;
        for (int ib = 0; ib < Q::kNblock/2; ++ib) {
            float sl = d * iq4k_values[x[ibl].scales[ib] & 0xf];
            float sh = d * iq4k_values[x[ibl].scales[ib] >>  4];
            for (int ig = 0; ig < Q::kNg; ++ig) {
                deq.set_values(ql[ig], yl, sl);
                deq.set_values(qh[ig], yh, sh);
                yl += QuantizerIQ2KT::kGroupSize;
                yh += QuantizerIQ2KT::kGroupSize;
            }
            ql += QuantizerIQ2KT::kNg;
            qh += QuantizerIQ2KT::kNg;
        }
    }
    const int nt = (k % QK_K)/32;
    if (nt == 0) return;
    const uint8_t * scales = (const uint8_t *)(x + nb);
    const uint16_t * ql = (const uint16_t *)(scales + 4);
    y += nb*QK_K;
    for (int ib = 0; ib < nt; ++ib) {
        float s = d * iq4k_values[(scales[ib%4] >> 4*(ib/4)) & 0xf];
        for (int ig = 0; ig < Q::kNg; ++ig) {
            deq.set_values(ql[Q::kNg*ib + ig], y, s);
            y += Q::kGroupSize;
        }
    }
}

void vec_dot_iq2_kt_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    *s = iqk_kt_row_vec_dot_dequant_q8_K(n, vx, vy, dequantize_row_iq2_kt);

}

namespace {

using QuantizerIQ3KT = QuantizerIQKT<32, 8, 16, true, true>;
const QuantizerIQ3KT& iq3kt_quantizer() {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unique_ptr<QuantizerIQ3KT> quantizer;
    if (!quantizer) quantizer = std::make_unique<QuantizerIQ3KT>(256, 8);
    return *quantizer;
}

void quantize_row_iq3_kt_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales,
        float * all_weights, float * qtmp) {

    constexpr float kSigmaScale = 2.0f;
    constexpr float kStep = 8.0f;

    using Q = QuantizerIQ3KT;

    static_assert(Q::kNumVal%8 == 0);

    constexpr int kNumGroups = Q::kSuperBlockSize/Q::kGroupSize;

    float * dptr = (float *)vy;

    block_iq3_kt * y = (block_iq3_kt *)(dptr + 1);

    int   best_idx[2*Q::kNg];

    auto& quantizer = iq3kt_quantizer();

    int nblock = GGML_PAD(n_per_row, Q::kSuperBlockSize) / Q::kSuperBlockSize;

    float amax_row = 0;
    for (int j = 0; j < n_per_row; ++j) amax_row = std::max(amax_row, std::abs(x[j]));
    if (!amax_row) {
        *dptr = 0.f;
        std::memset(y, 0, nblock*sizeof(block_iq3_kt));
        return;
    }

    Q::set_weights(kSigmaScale, nblock, x, quant_weights, all_weights, n_per_row);

    float amax_scale = 0, max_scale = 0;

    float xaux[Q::kBlockSize];

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ3_KT);

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq3_kt));

        auto scales = all_scales + ibl*Q::kNblock;
        auto xbl = x + ibl*Q::kSuperBlockSize;

        for (int ib = 0; ib < Q::kNblock; ++ib) {
            const float * xb = xbl + Q::kBlockSize*ib;
            const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
            float amax = 0;
            for (int j = 0; j < Q::kBlockSize; ++j) {
                float ax = std::abs(xb[j]);
                xaux[j] = ax;
                amax = std::max(amax, ax);
            }
            if (amax < 1e-16f) {
                scales[ib] = 0.0f;
                continue;
            }

            //quantizer.find_best_match(amax/96.f, xaux, weight, best_idx+Q::kNg);
            //scales[ib] = quantizer.find_best_scale(xaux, weight, best_idx+Q::kNg).first;

            float scale_0 = std::max(84.f, 123.f*amax/amax_row);
            //float scale_0 = std::max(64.f, 123.f*amax/amax_row);
            float best = 0;
            bool found_solution = false;
            for (int itry = -3; itry <= 3; ++itry) {
                quantizer.find_best_match(amax/(scale_0 + kStep*itry), xaux, weight, best_idx);
                auto [d, score] = quantizer.find_best_scale(xaux, weight, best_idx);
                if (score > best) {
                    best = score;
                    found_solution = true;
                    scales[ib] = d;
                    std::memcpy(best_idx+Q::kNg, best_idx, Q::kNg*sizeof(int));
                }
            }
            if (!found_solution) {
                fprintf(stderr, "======================= %s: failed to find solution for a block\n", __func__);
                fprintf(stderr, "Model weights and importances:\n");
                for (int j = 0; j < Q::kBlockSize; ++j) {
                    fprintf(stderr, "%2d  %g  %g\n", j, xaux[j], weight[j]);
                }
                GGML_ASSERT(false);
            }

            auto xt = qtmp + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
            for (int ig = 0; ig < Q::kNg; ++ig) {
                auto q = quantizer.values() + Q::kGroupSize*best_idx[Q::kNg+ig];
                for (int j = 0; j < Q::kGroupSize; ++j) *xt++ = q[j];
            }

            float abs_scale = std::abs(scales[ib]);
            if (abs_scale > amax_scale) {
                amax_scale = abs_scale;
                max_scale = scales[ib];
            }
        }

    }

    GGML_ASSERT(max_scale >= 0);
    float d = max_scale/15;
    float best = 0;
    for (int itry = -9; itry <= 9; ++itry) {
        float id = (itry*0.2f + 15)/max_scale;
        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * xb = x + ibl*Q::kSuperBlockSize;
            const float * qb = qtmp + ibl*Q::kSuperBlockSize;
            const float * wb = all_weights + ibl*Q::kSuperBlockSize;
            auto scales = all_scales + ibl*Q::kNblock;
            for (int ib = 0; ib < Q::kNblock; ++ib) {
                int ls = nearest_int(id*scales[ib]);
                ls = std::max(0, std::min(15, ls));
                float dl = ls;
                for (int j = 0; j < Q::kBlockSize; ++j) {
                    float q = dl*qb[j];
                    sumqx += wb[j]*std::abs(xb[j])*q;
                    sumq2 += wb[j]*q*q;
                }
                xb += Q::kBlockSize;
                wb += Q::kBlockSize;
                qb += Q::kBlockSize;
            }
        }
        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
            d = sumqx/sumq2; best = d*sumqx;
        }
    }

    float id = d ? 1/d : 0.f;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto scales = all_scales + ibl*Q::kNblock;
        for (int ib = 0; ib < Q::kNblock/2; ++ib) {
            int ls1 = nearest_int(id*scales[ib]);
            int ls2 = nearest_int(id*scales[ib + Q::kNblock/2]);
            ls1 = std::max(0, std::min(15, ls1));
            ls2 = std::max(0, std::min(15, ls2));
            y[ibl].scales[ib] = ls1 | (ls2 << 4);
        }
    }

    *dptr = d * fudge;

    for (int iloop = 0; iloop < 1; ++iloop) {

        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {

            uint16_t * ql = (uint16_t *)y[ibl].ql;

            std::memset(y[ibl].qh, 0, kNumGroups/2);
            const float * xbl = x + ibl*Q::kSuperBlockSize;

            for (int ib = 0; ib < Q::kNblock; ++ib) {
                const float * xb = xbl + Q::kBlockSize*ib;
                const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
                for (int j = 0; j < Q::kBlockSize; ++j) {
                    xaux[j] = std::abs(xb[j]);
                    if (xb[j] < 0) y[ibl].qh[j] |= (1 << ib);
                }
                int ls = (y[ibl].scales[ib%(Q::kNblock/2)] >> 4*(ib/(Q::kNblock/2))) & 0xf;
                float dl = d*ls;
                quantizer.find_best_match(dl, xaux, weight, best_idx);

                for (int j = 0; j < Q::kNg; ++j) {
                    ql[ib*Q::kNg+j] = best_idx[j];
                    auto xl = xaux + Q::kGroupSize*j;
                    auto wl = weight + Q::kGroupSize*j;
                    auto ql = quantizer.values() + best_idx[j]*Q::kGroupSize;
                    for (int k = 0; k < Q::kGroupSize; ++k) {
                        float q = ql[k]*ls;
                        sumqx += wl[k]*xl[k]*q;
                        sumq2 += wl[k]*q*q;
                    }
                }
            }
        }
        if (sumq2 > 0) {
            d = sumqx/sumq2;
            *dptr = d * fudge;
            if (!d) break;
        } else {
            break;
        }
    }
}
}

void quantize_row_iq3_kt_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    assert(k % 32 == 0);
    quantize_iq3_kt(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq3_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % 32 == 0);
    block_iq3_kt * y = (block_iq3_kt *)vy;
    quantize_row_iq3_kt_ref(x, y, k);
}

size_t quantize_iq3_kt(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    iqk_kt_can_quantize(GGML_TYPE_IQ3_KT);
    std::vector<float> scales(GGML_PAD(n_per_row, QK_K)/QuantizerIQ3KT::kBlockSize);
    std::vector<float> weights(GGML_PAD(n_per_row, QK_K));
    std::vector<float> xtmp(GGML_PAD(n_per_row, QK_K));
    return quantize_kt_rows(GGML_TYPE_IQ3_KT, src, dst, nrows, n_per_row, imatrix,
            [&] (const float * x, void * y, const float * w) { quantize_row_iq3_kt_impl(x, y, n_per_row, w, scales.data(), weights.data(), xtmp.data()); });
}

void dequantize_row_iq3_kt(const void * vx_, float * y, int64_t k) {
    const block_iq3_kt * x = (const block_iq3_kt *)vx_;
#ifdef __AVX2__
    //if (iqk_dequantize_ktquants(GGML_TYPE_IQ3_KT, k, x, 0, y, 0, 1)) return;
#endif
    using Q = QuantizerIQ3KT;
    constexpr int kNumGroups = Q::kSuperBlockSize/Q::kGroupSize;
    assert(k % 32 == 0);
    const int nb = k / Q::kSuperBlockSize;
    const float * dptr = (const float *)x;
    const float d = *dptr * Q::kScale * 1.01f;
    x = (const block_iq3_kt *)(dptr + 1);
    auto& deq = iq3kt_quantizer();
    for (int ibl = 0; ibl < nb; ++ibl) {
        auto yl = y + ibl*Q::kSuperBlockSize;
        auto yh = yl + Q::kSuperBlockSize/2;
        auto qll = (const uint16_t *)x[ibl].ql;
        auto qlh = qll + kNumGroups/2;
        int jj = 0;
        for (int ib = 0; ib < Q::kNblock/2; ++ib) {
            float sl = d * (x[ibl].scales[ib] & 0xf);
            float sh = d * (x[ibl].scales[ib] >>  4);
            uint8_t l_mask = 1 << ib;
            uint8_t h_mask = l_mask << (Q::kNblock/2);
            for (int ig = 0; ig < Q::kNg; ++ig) {
                deq.set_values(qll[jj], yl, sl);
                deq.set_values(qlh[jj], yh, sh);
                for (int j = 0; j < Q::kGroupSize; ++j) {
                    if (x[ibl].qh[ig*Q::kGroupSize+j] & l_mask) yl[j] = -yl[j];
                    if (x[ibl].qh[ig*Q::kGroupSize+j] & h_mask) yh[j] = -yh[j];
                }
                yl += Q::kGroupSize;
                yh += Q::kGroupSize;
                ++jj;
            }
        }
    }
    const int nt = (k % QK_K)/32;
    if (nt > 0) {
        const uint8_t * tail = (const uint8_t *)(x + nb);
        const uint16_t * ql = (const uint16_t *)tail;
        const uint8_t * qh = tail + 8*nt;
        const uint8_t * scales = tail + 12*nt;
        y += nb*QK_K;
        for (int ib = 0; ib < nt; ++ib) {
            float s = d * ((scales[ib/2] >> 4*(ib&1)) & 0xf);
            for (int ig = 0; ig < Q::kNg; ++ig) {
                deq.set_values(ql[Q::kNg*ib + ig], y, s);
                for (int j = 0; j < Q::kGroupSize; ++j) {
                    if (qh[4*ib + ig] & (1 << j)) y[j] = -y[j];
                }
                y += Q::kGroupSize;
            }
        }
    }
}

void vec_dot_iq3_kt_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % 32 == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    *s = iqk_kt_row_vec_dot_dequant_q8_K(n, vx, vy, dequantize_row_iq3_kt);

}

// ======================================== iq4_kt

namespace{

using QuantizerIQ4KT = QuantizerIQKT<32, 4, 15, false, true>;

const QuantizerIQ4KT& iq4kt_quantizer(bool with_offset = false) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unique_ptr<QuantizerIQ4KT> quantizer1;
    static std::unique_ptr<QuantizerIQ4KT> quantizer2;
    if (with_offset) {
        if (!quantizer2) quantizer2 = std::make_unique<QuantizerIQ4KT>(625, 6, 4096+32768);
        return *quantizer2;
    }
    if (!quantizer1) quantizer1 = std::make_unique<QuantizerIQ4KT>(625, 6, 4096);
    return *quantizer1;
}

const QuantizerIQ4KT& iq4kt_dequantizer() {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unique_ptr<QuantizerIQ4KT> dequantizer;
    if (!dequantizer) dequantizer = std::make_unique<QuantizerIQ4KT>(0, 0, 4096);
    return *dequantizer;
}

void quantize_row_iq4_kt_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales, float * all_weights) {

    constexpr float kSigmaScale = 2.0f;
    constexpr int kNtry = 2;
    using Q = QuantizerIQ4KT;

    static_assert(Q::kNumVal%8 == 0);

    float * dptr = (float *)vy;

    block_iq4_kt * y = (block_iq4_kt *)(dptr + 1);

    auto& quantizer1 = iq4kt_quantizer();
    auto& quantizer2 = iq4kt_quantizer(true);

    int nblock = GGML_PAD(n_per_row, Q::kSuperBlockSize) / Q::kSuperBlockSize;

    Q::set_weights(kSigmaScale, nblock, x, quant_weights, all_weights, n_per_row);

    float amax_row = 0;
    for (int j = 0; j < n_per_row; ++j) {
        amax_row = std::max(amax_row, std::abs(x[j]));
    }
    if (!amax_row) {
        dptr[0] = 0.f;
        std::memset(y, 0, nblock*sizeof(block_iq4_kt));
        return;
    }

    int   best_idx[2*Q::kNg];
    float xaux[Q::kBlockSize];

    float amax_scale = 0, max_scale = 0;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_KT);

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq4_kt));

        const float * xbl = x + ibl*Q::kSuperBlockSize;
        auto scales = all_scales + ibl*Q::kNblock;

        for (int ib = 0; ib < Q::kNblock; ++ib) {
            const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
            float amax = 0;
            for (int j = 0; j < Q::kBlockSize; ++j) {
                xaux[j] = xbl[ib*Q::kBlockSize+j];
                float ax = std::abs(xaux[j]);
                amax = std::max(amax, ax);
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            float best = 0;
            float scale_0 = std::max(90.f, 124.f*amax/amax_row);
            for (int itry = -kNtry; itry <= kNtry; ++itry) {
                quantizer1.find_best_match( amax/(8.f*itry + scale_0), xaux, weight, best_idx);
                auto [dp, score_p] = quantizer1.find_best_scale(xaux, weight, best_idx);
                if (score_p > best) {
                    best = score_p; scales[ib] = dp;
                }
                quantizer1.find_best_match(-amax/(8.f*itry + scale_0), xaux, weight, best_idx);
                auto [dm, score_m] = quantizer1.find_best_scale(xaux, weight, best_idx);
                if (score_m > best) {
                    best = score_m; scales[ib] = dm;
                }
            }

            quantizer2.find_best_match(scales[ib], xaux, weight, best_idx);
            auto [d, score] = quantizer2.find_best_scale(xaux, weight, best_idx);
            if (score > best) {
                best = score;
                scales[ib] = d;
                y[ibl].qs[ib] = 1;
            }
            bool with_offset = false;
            for (int itry = -kNtry; itry <= kNtry; ++itry) {
                quantizer2.find_best_match( amax/(8.f*itry + scale_0), xaux, weight, best_idx);
                auto [dp, score_p] = quantizer2.find_best_scale(xaux, weight, best_idx);
                if (score_p > best) {
                    best = score_p; scales[ib] = dp; with_offset = true;
                }
                quantizer2.find_best_match(-amax/(8.f*itry + scale_0), xaux, weight, best_idx);
                auto [dm, score_m] = quantizer2.find_best_scale(xaux, weight, best_idx);
                if (score_m > best) {
                    best = score_m; scales[ib] = dm; with_offset = true;
                }
            }
            if (with_offset) y[ibl].qs[ib] = 1;

            float abs_scale = std::abs(scales[ib]);
            if (abs_scale > amax_scale) {
                amax_scale = abs_scale;
                max_scale = scales[ib];
            }
        }

    }

    float d = -max_scale/64;

    dptr[0] = d * fudge;
    if (!d) return;

    constexpr int kNumGroups = Q::kSuperBlockSize/Q::kGroupSize;

    for (int iloop = 0; iloop < 1; ++iloop) {

        const float id = 1/d;

        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {

            // high 3 bits + scales
            // each block of 32 needs 8 x 3 (high bits) + 1 x 8 (scale) = 32 bits = 1 x uint32_t
            // we have 8 blocks
            auto shb = y[ibl].qs;  // high 3 bits + scales
            auto ql = (uint8_t *)(shb + Q::kNblock);
            auto qh = ql + kNumGroups;
            std::memset(qh, 0, kNumGroups/2);
            const float * xbl = x + ibl*Q::kSuperBlockSize;
            auto scales = all_scales + ibl*Q::kNblock;

            for (int ib = 0; ib < Q::kNblock; ++ib) {
                auto& quantizer = y[ibl].qs[ib] & 1 ? quantizer2 : quantizer1;
                const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
                for (int j = 0; j < Q::kBlockSize; ++j) xaux[j] = xbl[ib*Q::kBlockSize+j];
                int ls = nearest_int(id*scales[ib]);
                ls = std::min(ls, 63);
                *(uint8_t *)(shb + ib) = ((ls + 64) << 1) | (shb[ib] & 1);
                float dl = d*ls;
                quantizer.find_best_match(dl, xaux, weight, best_idx);

                for (int j = 0; j < Q::kNg; ++j) {
                    shb[ib] |= ((best_idx[j] >> 12) << (8 + 3*j));
                    ql[Q::kNg*ib + j] = best_idx[j] & 255;
                    qh[(Q::kNg*ib + j)%(kNumGroups/2)] |= ((best_idx[j] >> 8) & 0xf) << 4*((Q::kNg*ib + j)/(kNumGroups/2));
                    auto xl = xaux + Q::kGroupSize*j;
                    auto wl = weight + Q::kGroupSize*j;
                    auto ql = quantizer.values() + Q::kGroupSize*best_idx[j];
                    for (int k = 0; k < Q::kGroupSize; ++k) {
                        float q = ql[k]*ls;
                        sumqx += wl[k]*xl[k]*q;
                        sumq2 += wl[k]*q*q;
                    }
                }
            }
        }
        if (sumq2 > 0) {
            d = sumqx/sumq2;
            dptr[0] = d * fudge;
            if (!d) break;
        } else {
            break;
        }
    }
}
}

[[maybe_unused]] static int iqk_kt_codebook(int type, int bank, const int ** offsets, const int ** points, const float ** values, const float ** mid) {
    struct Flat { std::vector<int> offsets, points; };
    static std::mutex mutex;
    static Flat flat[3];
    std::lock_guard<std::mutex> lock(mutex);
    auto flatten = [offsets, points, values, mid] (const auto& q, Flat& f) {
        if (f.offsets.empty()) {
            f.offsets.push_back(0);
            for (auto& p : q.in_cluster()) {
                f.points.insert(f.points.end(), p.begin(), p.end());
                f.offsets.push_back(f.points.size());
            }
        }
        *offsets = f.offsets.data();
        *points = f.points.data();
        *values = q.values();
        *mid = q.mid();
        return int(f.offsets.size()) - 1;
    };
    if (type == GGML_TYPE_IQ4_KT) return flatten(iq4kt_quantizer(bank == 1), flat[bank]);
    if (type == GGML_TYPE_IQ3_KT) return flatten(iq3kt_quantizer(), flat[2]);
    return 0;
}

void quantize_row_iq4_kt_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    assert(k % 32 == 0);
    quantize_iq4_kt(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq4_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % 32 == 0);
    block_iq4_kt * y = (block_iq4_kt *)vy;
    quantize_row_iq4_kt_ref(x, y, k);
}

size_t quantize_iq4_kt(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    iqk_kt_can_quantize(GGML_TYPE_IQ4_KT);
    std::vector<float> scales(GGML_PAD(n_per_row, QK_K)/QuantizerIQ4KT::kBlockSize);
    std::vector<float> weights(GGML_PAD(n_per_row, QK_K));
    return quantize_kt_rows(GGML_TYPE_IQ4_KT, src, dst, nrows, n_per_row, imatrix,
            [&] (const float * x, void * y, const float * w) { quantize_row_iq4_kt_impl(x, y, n_per_row, w, scales.data(), weights.data()); });
}

void dequantize_row_iq4_kt(const void * vx_, float * y, int64_t k) {
    const block_iq4_kt * x = (const block_iq4_kt *)vx_;
#ifdef __AVX2__
    //if (iqk_dequantize_ktquants(GGML_TYPE_IQ4_KT, k, x, 0, y, 0, 1)) return;
#endif
    using Q = QuantizerIQ4KT;
    assert(k % 32 == 0);
    constexpr int kNumGroups = Q::kSuperBlockSize/Q::kGroupSize;
    const int nb = k / Q::kSuperBlockSize;
    const float * dptr = (const float *)x;
    const float d = dptr[0] * Q::kScale;
    x = (const block_iq4_kt *)(dptr + 1);
    auto& deq = iq4kt_dequantizer();
    for (int ibl = 0; ibl < nb; ++ibl) {
        auto shb = x[ibl].qs;
        auto ql = (const uint8_t *)(shb + Q::kNblock);
        auto qh = ql + kNumGroups;
        for (int ib = 0; ib < Q::kNblock; ++ib) {
            int offset = shb[ib] & 1 ? 32768 + 4096 : 4096;
            int ls = int((shb[ib] & 0xff) >> 1) - 64;
            float sl = d * ls;
            for (int ig = 0; ig < Q::kNg; ++ig) {
                int jj = ib*Q::kNg+ig;
                uint16_t idx = ql[jj] | ((qh[jj%(kNumGroups/2)] << (8 - 4*(jj/(kNumGroups/2)))) & 0xf00) | (((shb[ib] >> (8 + 3*ig)) & 7) << 12);
                deq.set_values(idx, y, sl, offset);
                y += Q::kGroupSize;
            }
        }
    }
    const int nt = (k % QK_K)/32;
    if (nt > 0) {
        const uint8_t * tail = (const uint8_t *)(x + nb);
        const uint32_t * shb = (const uint32_t *)tail;
        for (int ib = 0; ib < nt; ++ib) {
            const uint32_t sh = shb[4*ib];
            const uint8_t * ql = tail + 16*ib + 4;
            const uint8_t * qh = tail + 16*ib + 12;
            int offset = sh & 1 ? 32768 + 4096 : 4096;
            int ls = int((sh & 0xff) >> 1) - 64;
            float sl = d * ls;
            for (int ig = 0; ig < Q::kNg; ++ig) {
                uint16_t idx = ql[ig] | (((qh[ig/2] >> 4*(ig&1)) & 0xf) << 8) | (((sh >> (8 + 3*ig)) & 7) << 12);
                deq.set_values(idx, y, sl, offset);
                y += Q::kGroupSize;
            }
        }
    }
}

void vec_dot_iq4_kt_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % 32 == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    *s = iqk_kt_row_vec_dot_dequant_q8_K(n, vx, vy, dequantize_row_iq4_kt);

}

