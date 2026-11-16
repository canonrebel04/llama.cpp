#pragma once

// ik_llama.cpp IQ_K quants (IQ2_K .. IQ6_K), ported from ik_llama.cpp
// Copyright (C) 2024 Iwan Kawrakow, MIT license. See docs/fork-types.md.

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

GGML_API void   quantize_row_iq2_k_ref(const float * x, void * y, int64_t k);
GGML_API void   quantize_row_iq2_k(const float * x, void * y, int64_t k);
GGML_API size_t quantize_iq2_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void   dequantize_row_iq2_k(const void * x, float * y, int64_t k);
GGML_API void   vec_dot_iq2_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

GGML_API void   quantize_row_iq3_k_ref(const float * x, void * y, int64_t k);
GGML_API void   quantize_row_iq3_k(const float * x, void * y, int64_t k);
GGML_API size_t quantize_iq3_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void   dequantize_row_iq3_k(const void * x, float * y, int64_t k);
GGML_API void   vec_dot_iq3_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

GGML_API void   quantize_row_iq4_k_ref(const float * x, void * y, int64_t k);
GGML_API void   quantize_row_iq4_k(const float * x, void * y, int64_t k);
GGML_API size_t quantize_iq4_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void   dequantize_row_iq4_k(const void * x, float * y, int64_t k);
GGML_API void   vec_dot_iq4_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

GGML_API void   quantize_row_iq5_k_ref(const float * x, void * y, int64_t k);
GGML_API void   quantize_row_iq5_k(const float * x, void * y, int64_t k);
GGML_API size_t quantize_iq5_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void   dequantize_row_iq5_k(const void * x, float * y, int64_t k);
GGML_API void   vec_dot_iq5_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

GGML_API void   quantize_row_iq6_k_ref(const float * x, void * y, int64_t k);
GGML_API void   quantize_row_iq6_k(const float * x, void * y, int64_t k);
GGML_API size_t quantize_iq6_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void   dequantize_row_iq6_k(const void * x, float * y, int64_t k);
GGML_API void   vec_dot_iq6_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

// row-scaled types (ggml-iqk-quants-ks.cpp); x/y hold exactly one row
GGML_API void   quantize_row_iq4_ks_ref(const float * x, void * y, int64_t k);
GGML_API void   quantize_row_iq4_ks(const float * x, void * y, int64_t k);
GGML_API size_t quantize_iq4_ks(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void   dequantize_row_iq4_ks(const void * x, float * y, int64_t k);
GGML_API void   vec_dot_iq4_ks_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
GGML_API void   quantize_row_iq2_ks_ref(const float * x, void * y, int64_t k);
GGML_API void   quantize_row_iq2_ks(const float * x, void * y, int64_t k);
GGML_API size_t quantize_iq2_ks(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void   dequantize_row_iq2_ks(const void * x, float * y, int64_t k);
GGML_API void   vec_dot_iq2_ks_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
GGML_API void   quantize_row_iq4_kss_ref(const float * x, void * y, int64_t k);
GGML_API void   quantize_row_iq4_kss(const float * x, void * y, int64_t k);
GGML_API size_t quantize_iq4_kss(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void   dequantize_row_iq4_kss(const void * x, float * y, int64_t k);
GGML_API void   vec_dot_iq4_kss_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
GGML_API void   quantize_row_iq5_ks_ref(const float * x, void * y, int64_t k);
GGML_API void   quantize_row_iq5_ks(const float * x, void * y, int64_t k);
GGML_API size_t quantize_iq5_ks(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void   dequantize_row_iq5_ks(const void * x, float * y, int64_t k);
GGML_API void   vec_dot_iq5_ks_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
// trellis types (ggml-iqk-quants-kt.cpp)
GGML_API void   quantize_row_iq1_kt_ref(const float * x, void * y, int64_t k);
GGML_API void   quantize_row_iq1_kt(const float * x, void * y, int64_t k);
GGML_API size_t quantize_iq1_kt(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void   dequantize_row_iq1_kt(const void * x, float * y, int64_t k);
GGML_API void   vec_dot_iq1_kt_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
GGML_API void   quantize_row_iq2_kt_ref(const float * x, void * y, int64_t k);
GGML_API void   quantize_row_iq2_kt(const float * x, void * y, int64_t k);
GGML_API size_t quantize_iq2_kt(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void   dequantize_row_iq2_kt(const void * x, float * y, int64_t k);
GGML_API void   vec_dot_iq2_kt_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
GGML_API void   quantize_row_iq3_kt_ref(const float * x, void * y, int64_t k);
GGML_API void   quantize_row_iq3_kt(const float * x, void * y, int64_t k);
GGML_API size_t quantize_iq3_kt(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void   dequantize_row_iq3_kt(const void * x, float * y, int64_t k);
GGML_API void   vec_dot_iq3_kt_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
GGML_API void   quantize_row_iq4_kt_ref(const float * x, void * y, int64_t k);
GGML_API void   quantize_row_iq4_kt(const float * x, void * y, int64_t k);
GGML_API size_t quantize_iq4_kt(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void   dequantize_row_iq4_kt(const void * x, float * y, int64_t k);
GGML_API void   vec_dot_iq4_kt_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
GGML_API void   quantize_row_iq3_ks_ref(const float * x, void * y, int64_t k);
GGML_API void   quantize_row_iq3_ks(const float * x, void * y, int64_t k);
GGML_API size_t quantize_iq3_ks(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void   dequantize_row_iq3_ks(const void * x, float * y, int64_t k);
GGML_API void   vec_dot_iq3_ks_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
GGML_API void   quantize_row_iq2_kl_ref(const float * x, void * y, int64_t k);
GGML_API void   quantize_row_iq2_kl(const float * x, void * y, int64_t k);
GGML_API size_t quantize_iq2_kl(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void   dequantize_row_iq2_kl(const void * x, float * y, int64_t k);
GGML_API void   vec_dot_iq2_kl_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif
