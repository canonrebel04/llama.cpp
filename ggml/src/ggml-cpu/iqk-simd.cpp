// SIMD (AVX2 / NEON) vec_dot kernels for the ik_llama.cpp IQ_K types
// (IQ2_K, IQ3_K, IQ4_K, IQ5_K) against Q8_K activations.
//
// Approach (after ik_llama.cpp iqk_mul_mat.cpp, Iwan Kawrakow, MIT): the
// non-linear codebooks have at most 32 entries, so each 16-byte group of packed
// indices is decoded with byte shuffles (pshufb / vqtbl1q) into int8 values,
// and the int8 x int8 products are accumulated in int32 with the per-16 block
// scale applied in integer arithmetic. The result is exact in int32 and only
// scaled by d * d8 at the end. IQ6_K (polynomial codebook) keeps the scalar
// path. Without AVX2/NEON these forward to the scalar kernels in
// ggml-iqk-quants.cpp.
#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml-iqk-quants.h"
#include "iqk-simd.h"

#include <cstdint>
#include <cstring>
#include <cassert>

#if defined(__AVX2__)
#include <immintrin.h>
#define IQK_SIMD 1
typedef __m128i v16;
static inline v16 v_load(const void * p) { return _mm_loadu_si128((const __m128i *)p); }
static inline v16 v_and(v16 a, uint8_t m) { return _mm_and_si128(a, _mm_set1_epi8((char)m)); }
static inline v16 v_shr(v16 a, int n) { return _mm_and_si128(_mm_srli_epi16(a, n), _mm_set1_epi8((char)(0xFF >> n))); }
static inline v16 v_shl(v16 a, int n) { return _mm_and_si128(_mm_slli_epi16(a, n), _mm_set1_epi8((char)((0xFF << n) & 0xFF))); }
static inline v16 v_or(v16 a, v16 b) { return _mm_or_si128(a, b); }
static inline v16 v_add8(v16 a, uint8_t b) { return _mm_add_epi8(a, _mm_set1_epi8((char)b)); }
// table lookup, idx in [0,16)
static inline v16 v_lut(v16 tbl, v16 idx) { return _mm_shuffle_epi8(tbl, idx); }
// table lookup over 32 entries, idx in [0,32)
static inline v16 v_lut32(v16 lo, v16 hi, v16 idx) {
    const v16 i4 = v_and(idx, 0x0F);
    const v16 m  = _mm_cmpeq_epi8(v_and(idx, 0x10), _mm_set1_epi8(0x10));
    return _mm_blendv_epi8(_mm_shuffle_epi8(lo, i4), _mm_shuffle_epi8(hi, i4), m);
}
#elif defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#define IQK_SIMD 1
typedef uint8x16_t v16;
static inline v16 v_load(const void * p) { return vld1q_u8((const uint8_t *)p); }
static inline v16 v_and(v16 a, uint8_t m) { return vandq_u8(a, vdupq_n_u8(m)); }
static inline v16 v_shr(v16 a, int n) { return vshlq_u8(a, vdupq_n_s8((int8_t)-n)); }
static inline v16 v_shl(v16 a, int n) { return vshlq_u8(a, vdupq_n_s8((int8_t)n)); }
static inline v16 v_or(v16 a, v16 b) { return vorrq_u8(a, b); }
static inline v16 v_add8(v16 a, uint8_t b) { return vaddq_u8(a, vdupq_n_u8(b)); }
static inline v16 v_lut(v16 tbl, v16 idx) { return vqtbl1q_u8(tbl, idx); }
static inline v16 v_lut32(v16 lo, v16 hi, v16 idx) {
    uint8x16x2_t t = { { lo, hi } };
    return vqtbl2q_u8(t, idx);
}
#endif

#ifdef IQK_SIMD

// Accumulates ls0 * dot(g0, q8[0..15]) + ls1 * dot(g1, q8[16..31]) in int32.
// Decoded values go straight from registers into the dot product (no staging
// buffer, which would stall on store-to-load forwarding).
#if defined(__AVX2__)
struct iqk_acc {
    __m256i a = _mm256_setzero_si256();
    inline void put2(v16 g0, v16 g1, const int8_t * q8, int ls0, int ls1) {
        const __m256i x = _mm256_set_m128i(g1, g0);
        const __m256i y = _mm256_loadu_si256((const __m256i *)q8);
        // |y| * sign(x, y): q8 may be -128 (|y| = 128 as u8), codebook values are within [-127, 127]
        const __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(y, y), _mm256_sign_epi8(x, y));
        const __m256i sc  = _mm256_set_m128i(_mm_set1_epi16((short)ls1), _mm_set1_epi16((short)ls0));
        a = _mm256_add_epi32(a, _mm256_madd_epi16(p16, sc));
    }
    inline int32_t sum() const {
        __m128i s = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
        return _mm_cvtsi128_si32(s);
    }
};
#else
struct iqk_acc {
    int32x4_t a = vdupq_n_s32(0);
    static inline int32x4_t dot16(int8x16_t x, int8x16_t y) {
#if defined(__ARM_FEATURE_DOTPROD)
        return vdotq_s32(vdupq_n_s32(0), x, y);
#else
        int32x4_t s32 = vpaddlq_s16(vmull_s8(vget_low_s8(x), vget_low_s8(y)));
        return vpadalq_s16(s32, vmull_s8(vget_high_s8(x), vget_high_s8(y)));
#endif
    }
    inline void put2(v16 g0, v16 g1, const int8_t * q8, int ls0, int ls1) {
        a = vmlaq_n_s32(a, dot16(vreinterpretq_s8_u8(g0), vld1q_s8(q8)),      ls0);
        a = vmlaq_n_s32(a, dot16(vreinterpretq_s8_u8(g1), vld1q_s8(q8 + 16)), ls1);
    }
    inline int32_t sum() const { return vaddvq_s32(a); }
};
#endif

static inline v16 v_tbl16(const int8_t * src, int n) {
    int8_t t[16] = {0};
    memcpy(t, src, n);
    return v_load(t);
}

void ggml_cpu_vec_dot_iq2_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0); GGML_UNUSED(bs); GGML_UNUSED(bx); GGML_UNUSED(by); GGML_UNUSED(nrc);
    const block_iq2_k * x = (const block_iq2_k *)vx;
    const block_q8_K  * y = (const block_q8_K  *)vy;
    const v16 tbl = v_tbl16(iq2nl_values, 8); // idx | (shift << 2)
    float sumf = 0;
    for (int i = 0; i < n / QK_K; ++i) {
        iqk_acc acc;
        const int8_t * q8 = y[i].qs;
        uint16_t extra = x[i].extra;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const int ls0 = (x[i].scales[ib32] & 0xf) - 8;
            const int ls1 = (x[i].scales[ib32] >>  4) - 8;
            const int shift = 2*(ib32 % 4);
            const uint8_t * qs = x[i].qs + 32*(ib32/4);
            const v16 i1 = v_add8(v_and(v_shr(v_load(qs     ), shift), 3), (extra & 1) ? 4 : 0);
            const v16 i2 = v_add8(v_and(v_shr(v_load(qs + 16), shift), 3), (extra & 2) ? 4 : 0);
            extra >>= 2;
            acc.put2(v_lut(tbl, i1), v_lut(tbl, i2), q8 + 32*ib32, ls0, ls1);
        }
        sumf += GGML_FP16_TO_FP32(x[i].d) * y[i].d * acc.sum();
    }
    *s = sumf;
}

void ggml_cpu_vec_dot_iq3_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0); GGML_UNUSED(bs); GGML_UNUSED(bx); GGML_UNUSED(by); GGML_UNUSED(nrc);
    const block_iq3_k * x = (const block_iq3_k *)vx;
    const block_q8_K  * y = (const block_q8_K  *)vy;
    const v16 tbl = v_load(iq3nl_values); // idx | (shift << 3)
    float sumf = 0;
    for (int i = 0; i < n / QK_K; ++i) {
        iqk_acc acc;
        const int8_t * q8 = y[i].qs;
        uint16_t extra = x[i].extra;
        uint16_t sh = x[i].scales_h;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const int ls0 = (2*(x[i].scales_l[ib32] & 0xf) + 1) * ((sh & 1) ? -1 : 1);
            const int ls1 = (2*(x[i].scales_l[ib32] >>  4) + 1) * ((sh & 2) ? -1 : 1);
            sh >>= 2;
            const int shift_l = 2*(ib32 % 4);
            const int shift_h = ib32 % 8;
            const uint8_t * qs = x[i].qs + 32*(ib32/4);
            const uint8_t * qh = x[i].qh;
            v16 i1 = v_or(v_and(v_shr(v_load(qs     ), shift_l), 3), v_shl(v_and(v_shr(v_load(qh     ), shift_h), 1), 2));
            v16 i2 = v_or(v_and(v_shr(v_load(qs + 16), shift_l), 3), v_shl(v_and(v_shr(v_load(qh + 16), shift_h), 1), 2));
            i1 = v_add8(i1, (extra & 1) ? 8 : 0);
            i2 = v_add8(i2, (extra & 2) ? 8 : 0);
            extra >>= 2;
            acc.put2(v_lut(tbl, i1), v_lut(tbl, i2), q8 + 32*ib32, ls0, ls1);
        }
        sumf += GGML_FP16_TO_FP32(x[i].d) * y[i].d * acc.sum();
    }
    *s = sumf;
}

void ggml_cpu_vec_dot_iq4_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0); GGML_UNUSED(bs); GGML_UNUSED(bx); GGML_UNUSED(by); GGML_UNUSED(nrc);
    const block_iq4_k * x = (const block_iq4_k *)vx;
    const block_q8_K  * y = (const block_q8_K  *)vy;
    const v16 t0 = v_load(iq4k_values), t1 = v_load(iq4k_values + 16);
    float sumf = 0;
    for (int i = 0; i < n / QK_K; ++i) {
        iqk_acc acc;
        const int8_t * q8 = y[i].qs;
        uint16_t extra = x[i].extra;
        uint32_t h; memcpy(&h, x[i].scales_h, 4);
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls0 = ((x[i].scales_l[ib] & 0xf) | ((h << 4) & 0x30)) - 32;
            const int ls1 = ((x[i].scales_l[ib] >>  4) | ((h << 2) & 0x30)) - 32;
            h >>= 4;
            const v16 b = v_load(x[i].qs + 16*ib);
            acc.put2(v_lut((extra & 1) ? t1 : t0, v_and(b, 0x0F)), v_lut((extra & 2) ? t1 : t0, v_shr(b, 4)), q8 + 32*ib, ls0, ls1);
            extra >>= 2;
        }
        sumf += GGML_FP16_TO_FP32(x[i].d) * y[i].d * acc.sum();
    }
    *s = sumf;
}

void ggml_cpu_vec_dot_iq5_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0); GGML_UNUSED(bs); GGML_UNUSED(bx); GGML_UNUSED(by); GGML_UNUSED(nrc);
    const block_iq5_k * x = (const block_iq5_k *)vx;
    const block_q8_K  * y = (const block_q8_K  *)vy;
    const v16 t[4] = { v_load(iq5nl_values), v_load(iq5nl_values + 16), v_load(iq5nl_values + 32), v_load(iq5nl_values + 48) };
    float sumf = 0;
    for (int i = 0; i < n / QK_K; ++i) {
        iqk_acc acc;
        const int8_t * q8 = y[i].qs;
        uint16_t extra = x[i].extra;
        const uint8_t * sl = x[i].scales_l;
        const uint8_t * sh = x[i].scales_h;
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
            const int ls0 = ((sl[2*ib64+0] & 0xf) | ((sh[ib64] << 4) & 0x30)) - 32;
            const int ls1 = ((sl[2*ib64+0] >>  4) | ((sh[ib64] << 2) & 0x30)) - 32;
            const int ls2 = ((sl[2*ib64+1] & 0xf) | ((sh[ib64] >> 0) & 0x30)) - 32;
            const int ls3 = ((sl[2*ib64+1] >>  4) | ((sh[ib64] >> 2) & 0x30)) - 32;
            const int shift = 2*(ib64 % 4);
            const uint8_t * qs = x[i].qs + 32*ib64;
            const uint8_t * qh = x[i].qh + 32*(ib64/4);
            const v16 b0 = v_load(qs), b1 = v_load(qs + 16);
            const v16 h0 = v_shr(v_load(qh), shift), h1 = v_shr(v_load(qh + 16), shift);
            const v16 i1 = v_or(v_and(b0, 0x0F), v_shl(v_and(h0, 1), 4));
            const v16 i2 = v_or(v_and(b1, 0x0F), v_shl(v_and(h1, 1), 4));
            const v16 i3 = v_or(v_shr(b0, 4),    v_shl(v_and(h0, 2), 3));
            const v16 i4 = v_or(v_shr(b1, 4),    v_shl(v_and(h1, 2), 3));
            const int s1 = (extra & 1) ? 2 : 0, s2 = (extra & 2) ? 2 : 0, s3 = (extra & 4) ? 2 : 0, s4 = (extra & 8) ? 2 : 0;
            acc.put2(v_lut32(t[s1], t[s1+1], i1), v_lut32(t[s2], t[s2+1], i2), q8 + 64*ib64,      ls0, ls1);
            acc.put2(v_lut32(t[s3], t[s3+1], i3), v_lut32(t[s4], t[s4+1], i4), q8 + 64*ib64 + 32, ls2, ls3);
            extra >>= 4;
        }
        sumf += GGML_FP16_TO_FP32(x[i].d) * y[i].d * acc.sum();
    }
    *s = sumf;
}

#else // !IQK_SIMD

void ggml_cpu_vec_dot_iq2_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) { vec_dot_iq2_k_q8_k(n, s, bs, vx, bx, vy, by, nrc); }
void ggml_cpu_vec_dot_iq3_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) { vec_dot_iq3_k_q8_k(n, s, bs, vx, bx, vy, by, nrc); }
void ggml_cpu_vec_dot_iq4_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) { vec_dot_iq4_k_q8_k(n, s, bs, vx, bx, vy, by, nrc); }
void ggml_cpu_vec_dot_iq5_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) { vec_dot_iq5_k_q8_k(n, s, bs, vx, bx, vy, by, nrc); }

#endif
