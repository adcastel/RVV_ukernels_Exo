
#pragma once
#ifndef KERNELS_RVV_24X8_FP16_H
#define KERNELS_RVV_24X8_FP16_H

#ifdef __cplusplus
extern "C" {
#endif


#include <stdint.h>
#include <stdbool.h>

// Compiler feature macros adapted from Hedley (public domain)
// https://github.com/nemequ/hedley

#if defined(__has_builtin)
#  define EXO_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
#  define EXO_HAS_BUILTIN(builtin) (0)
#endif

#if EXO_HAS_BUILTIN(__builtin_assume)
#  define EXO_ASSUME(expr) __builtin_assume(expr)
#elif EXO_HAS_BUILTIN(__builtin_unreachable)
#  define EXO_ASSUME(expr) \
      ((void)((expr) ? 1 : (__builtin_unreachable(), 1)))
#else
#  define EXO_ASSUME(expr) ((void)(expr))
#endif


struct exo_win_1f16{
    _Float16 * const data;
    const int_fast32_t strides[1];
};
struct exo_win_1f16c{
    const _Float16 * const data;
    const int_fast32_t strides[1];
};
struct exo_win_2f16{
    _Float16 * const data;
    const int_fast32_t strides[2];
};
struct exo_win_2f16c{
    const _Float16 * const data;
    const int_fast32_t strides[2];
};
// gemm_RISCV_10x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 10] @DRAM
// )
void gemm_RISCV_10x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_10x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 10] @DRAM
// )
void gemm_RISCV_10x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_10x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 10] @DRAM
// )
void gemm_RISCV_10x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_10x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 10] @DRAM
// )
void gemm_RISCV_10x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_10x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 10] @DRAM
// )
void gemm_RISCV_10x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_10x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 10] @DRAM
// )
void gemm_RISCV_10x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_10x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 10] @DRAM
// )
void gemm_RISCV_10x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_10x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 10] @DRAM
// )
void gemm_RISCV_10x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_10x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 10] @DRAM
// )
void gemm_RISCV_10x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_10x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 10] @DRAM
// )
void gemm_RISCV_10x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_10x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 10] @DRAM
// )
void gemm_RISCV_10x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_10x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 10] @DRAM
// )
void gemm_RISCV_10x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_10x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 10] @DRAM
// )
void gemm_RISCV_10x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_10x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 10] @DRAM
// )
void gemm_RISCV_10x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_10x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 10] @DRAM
// )
void gemm_RISCV_10x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_10x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 10] @DRAM
// )
void gemm_RISCV_10x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_11x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 11] @DRAM
// )
void gemm_RISCV_11x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_11x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 11] @DRAM
// )
void gemm_RISCV_11x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_11x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 11] @DRAM
// )
void gemm_RISCV_11x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_11x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 11] @DRAM
// )
void gemm_RISCV_11x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_11x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 11] @DRAM
// )
void gemm_RISCV_11x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_11x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 11] @DRAM
// )
void gemm_RISCV_11x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_11x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 11] @DRAM
// )
void gemm_RISCV_11x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_11x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 11] @DRAM
// )
void gemm_RISCV_11x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_11x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 11] @DRAM
// )
void gemm_RISCV_11x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_11x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 11] @DRAM
// )
void gemm_RISCV_11x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_11x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 11] @DRAM
// )
void gemm_RISCV_11x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_11x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 11] @DRAM
// )
void gemm_RISCV_11x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_11x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 11] @DRAM
// )
void gemm_RISCV_11x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_11x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 11] @DRAM
// )
void gemm_RISCV_11x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_11x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 11] @DRAM
// )
void gemm_RISCV_11x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_11x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 11] @DRAM
// )
void gemm_RISCV_11x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_12x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 12] @DRAM
// )
void gemm_RISCV_12x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_12x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 12] @DRAM
// )
void gemm_RISCV_12x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_12x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 12] @DRAM
// )
void gemm_RISCV_12x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_12x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 12] @DRAM
// )
void gemm_RISCV_12x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_12x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 12] @DRAM
// )
void gemm_RISCV_12x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_12x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 12] @DRAM
// )
void gemm_RISCV_12x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_12x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 12] @DRAM
// )
void gemm_RISCV_12x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_12x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 12] @DRAM
// )
void gemm_RISCV_12x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_12x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 12] @DRAM
// )
void gemm_RISCV_12x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_12x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 12] @DRAM
// )
void gemm_RISCV_12x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_12x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 12] @DRAM
// )
void gemm_RISCV_12x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_12x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 12] @DRAM
// )
void gemm_RISCV_12x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_12x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 12] @DRAM
// )
void gemm_RISCV_12x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_12x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 12] @DRAM
// )
void gemm_RISCV_12x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_12x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 12] @DRAM
// )
void gemm_RISCV_12x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_12x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 12] @DRAM
// )
void gemm_RISCV_12x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_13x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 13] @DRAM
// )
void gemm_RISCV_13x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_13x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 13] @DRAM
// )
void gemm_RISCV_13x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_13x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 13] @DRAM
// )
void gemm_RISCV_13x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_13x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 13] @DRAM
// )
void gemm_RISCV_13x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_13x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 13] @DRAM
// )
void gemm_RISCV_13x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_13x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 13] @DRAM
// )
void gemm_RISCV_13x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_13x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 13] @DRAM
// )
void gemm_RISCV_13x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_13x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 13] @DRAM
// )
void gemm_RISCV_13x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_13x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 13] @DRAM
// )
void gemm_RISCV_13x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_13x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 13] @DRAM
// )
void gemm_RISCV_13x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_13x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 13] @DRAM
// )
void gemm_RISCV_13x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_13x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 13] @DRAM
// )
void gemm_RISCV_13x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_13x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 13] @DRAM
// )
void gemm_RISCV_13x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_13x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 13] @DRAM
// )
void gemm_RISCV_13x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_13x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 13] @DRAM
// )
void gemm_RISCV_13x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_13x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 13] @DRAM
// )
void gemm_RISCV_13x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_14x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 14] @DRAM
// )
void gemm_RISCV_14x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_14x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 14] @DRAM
// )
void gemm_RISCV_14x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_14x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 14] @DRAM
// )
void gemm_RISCV_14x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_14x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 14] @DRAM
// )
void gemm_RISCV_14x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_14x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 14] @DRAM
// )
void gemm_RISCV_14x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_14x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 14] @DRAM
// )
void gemm_RISCV_14x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_14x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 14] @DRAM
// )
void gemm_RISCV_14x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_14x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 14] @DRAM
// )
void gemm_RISCV_14x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_14x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 14] @DRAM
// )
void gemm_RISCV_14x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_14x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 14] @DRAM
// )
void gemm_RISCV_14x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_14x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 14] @DRAM
// )
void gemm_RISCV_14x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_14x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 14] @DRAM
// )
void gemm_RISCV_14x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_14x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 14] @DRAM
// )
void gemm_RISCV_14x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_14x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 14] @DRAM
// )
void gemm_RISCV_14x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_14x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 14] @DRAM
// )
void gemm_RISCV_14x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_14x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 14] @DRAM
// )
void gemm_RISCV_14x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_15x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 15] @DRAM
// )
void gemm_RISCV_15x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_15x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 15] @DRAM
// )
void gemm_RISCV_15x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_15x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 15] @DRAM
// )
void gemm_RISCV_15x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_15x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 15] @DRAM
// )
void gemm_RISCV_15x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_15x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 15] @DRAM
// )
void gemm_RISCV_15x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_15x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 15] @DRAM
// )
void gemm_RISCV_15x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_15x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 15] @DRAM
// )
void gemm_RISCV_15x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_15x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 15] @DRAM
// )
void gemm_RISCV_15x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_15x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 15] @DRAM
// )
void gemm_RISCV_15x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_15x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 15] @DRAM
// )
void gemm_RISCV_15x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_15x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 15] @DRAM
// )
void gemm_RISCV_15x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_15x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 15] @DRAM
// )
void gemm_RISCV_15x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_15x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 15] @DRAM
// )
void gemm_RISCV_15x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_15x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 15] @DRAM
// )
void gemm_RISCV_15x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_15x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 15] @DRAM
// )
void gemm_RISCV_15x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_15x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 15] @DRAM
// )
void gemm_RISCV_15x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_16x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 16] @DRAM
// )
void gemm_RISCV_16x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_16x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 16] @DRAM
// )
void gemm_RISCV_16x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_16x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 16] @DRAM
// )
void gemm_RISCV_16x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_16x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 16] @DRAM
// )
void gemm_RISCV_16x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_16x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 16] @DRAM
// )
void gemm_RISCV_16x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_16x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 16] @DRAM
// )
void gemm_RISCV_16x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_16x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 16] @DRAM
// )
void gemm_RISCV_16x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_16x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 16] @DRAM
// )
void gemm_RISCV_16x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_16x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 16] @DRAM
// )
void gemm_RISCV_16x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_16x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 16] @DRAM
// )
void gemm_RISCV_16x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_16x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 16] @DRAM
// )
void gemm_RISCV_16x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_16x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 16] @DRAM
// )
void gemm_RISCV_16x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_16x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 16] @DRAM
// )
void gemm_RISCV_16x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_16x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 16] @DRAM
// )
void gemm_RISCV_16x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_16x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 16] @DRAM
// )
void gemm_RISCV_16x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_16x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 16] @DRAM
// )
void gemm_RISCV_16x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_17x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 17] @DRAM
// )
void gemm_RISCV_17x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_17x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 17] @DRAM
// )
void gemm_RISCV_17x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_17x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 17] @DRAM
// )
void gemm_RISCV_17x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_17x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 17] @DRAM
// )
void gemm_RISCV_17x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_17x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 17] @DRAM
// )
void gemm_RISCV_17x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_17x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 17] @DRAM
// )
void gemm_RISCV_17x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_17x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 17] @DRAM
// )
void gemm_RISCV_17x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_17x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 17] @DRAM
// )
void gemm_RISCV_17x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_17x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 17] @DRAM
// )
void gemm_RISCV_17x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_17x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 17] @DRAM
// )
void gemm_RISCV_17x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_17x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 17] @DRAM
// )
void gemm_RISCV_17x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_17x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 17] @DRAM
// )
void gemm_RISCV_17x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_17x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 17] @DRAM
// )
void gemm_RISCV_17x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_17x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 17] @DRAM
// )
void gemm_RISCV_17x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_17x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 17] @DRAM
// )
void gemm_RISCV_17x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_17x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 17] @DRAM
// )
void gemm_RISCV_17x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_18x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 18] @DRAM
// )
void gemm_RISCV_18x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_18x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 18] @DRAM
// )
void gemm_RISCV_18x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_18x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 18] @DRAM
// )
void gemm_RISCV_18x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_18x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 18] @DRAM
// )
void gemm_RISCV_18x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_18x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 18] @DRAM
// )
void gemm_RISCV_18x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_18x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 18] @DRAM
// )
void gemm_RISCV_18x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_18x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 18] @DRAM
// )
void gemm_RISCV_18x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_18x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 18] @DRAM
// )
void gemm_RISCV_18x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_18x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 18] @DRAM
// )
void gemm_RISCV_18x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_18x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 18] @DRAM
// )
void gemm_RISCV_18x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_18x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 18] @DRAM
// )
void gemm_RISCV_18x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_18x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 18] @DRAM
// )
void gemm_RISCV_18x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_18x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 18] @DRAM
// )
void gemm_RISCV_18x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_18x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 18] @DRAM
// )
void gemm_RISCV_18x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_18x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 18] @DRAM
// )
void gemm_RISCV_18x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_18x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 18] @DRAM
// )
void gemm_RISCV_18x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_19x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 19] @DRAM
// )
void gemm_RISCV_19x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_19x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 19] @DRAM
// )
void gemm_RISCV_19x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_19x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 19] @DRAM
// )
void gemm_RISCV_19x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_19x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 19] @DRAM
// )
void gemm_RISCV_19x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_19x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 19] @DRAM
// )
void gemm_RISCV_19x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_19x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 19] @DRAM
// )
void gemm_RISCV_19x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_19x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 19] @DRAM
// )
void gemm_RISCV_19x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_19x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 19] @DRAM
// )
void gemm_RISCV_19x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_19x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 19] @DRAM
// )
void gemm_RISCV_19x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_19x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 19] @DRAM
// )
void gemm_RISCV_19x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_19x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 19] @DRAM
// )
void gemm_RISCV_19x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_19x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 19] @DRAM
// )
void gemm_RISCV_19x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_19x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 19] @DRAM
// )
void gemm_RISCV_19x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_19x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 19] @DRAM
// )
void gemm_RISCV_19x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_19x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 19] @DRAM
// )
void gemm_RISCV_19x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_19x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 19] @DRAM
// )
void gemm_RISCV_19x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_1x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 1] @DRAM
// )
void gemm_RISCV_1x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_1x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 1] @DRAM
// )
void gemm_RISCV_1x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_1x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 1] @DRAM
// )
void gemm_RISCV_1x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_1x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 1] @DRAM
// )
void gemm_RISCV_1x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_1x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 1] @DRAM
// )
void gemm_RISCV_1x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_1x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 1] @DRAM
// )
void gemm_RISCV_1x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_1x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 1] @DRAM
// )
void gemm_RISCV_1x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_1x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 1] @DRAM
// )
void gemm_RISCV_1x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_1x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 1] @DRAM
// )
void gemm_RISCV_1x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_1x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 1] @DRAM
// )
void gemm_RISCV_1x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_1x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 1] @DRAM
// )
void gemm_RISCV_1x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_1x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 1] @DRAM
// )
void gemm_RISCV_1x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_1x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 1] @DRAM
// )
void gemm_RISCV_1x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_1x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 1] @DRAM
// )
void gemm_RISCV_1x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_1x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 1] @DRAM
// )
void gemm_RISCV_1x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_1x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 1] @DRAM
// )
void gemm_RISCV_1x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_20x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 20] @DRAM
// )
void gemm_RISCV_20x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_20x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 20] @DRAM
// )
void gemm_RISCV_20x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_20x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 20] @DRAM
// )
void gemm_RISCV_20x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_20x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 20] @DRAM
// )
void gemm_RISCV_20x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_20x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 20] @DRAM
// )
void gemm_RISCV_20x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_20x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 20] @DRAM
// )
void gemm_RISCV_20x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_20x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 20] @DRAM
// )
void gemm_RISCV_20x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_20x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 20] @DRAM
// )
void gemm_RISCV_20x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_20x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 20] @DRAM
// )
void gemm_RISCV_20x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_20x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 20] @DRAM
// )
void gemm_RISCV_20x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_20x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 20] @DRAM
// )
void gemm_RISCV_20x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_20x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 20] @DRAM
// )
void gemm_RISCV_20x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_20x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 20] @DRAM
// )
void gemm_RISCV_20x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_20x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 20] @DRAM
// )
void gemm_RISCV_20x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_20x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 20] @DRAM
// )
void gemm_RISCV_20x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_20x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 20] @DRAM
// )
void gemm_RISCV_20x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_21x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 21] @DRAM
// )
void gemm_RISCV_21x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_21x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 21] @DRAM
// )
void gemm_RISCV_21x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_21x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 21] @DRAM
// )
void gemm_RISCV_21x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_21x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 21] @DRAM
// )
void gemm_RISCV_21x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_21x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 21] @DRAM
// )
void gemm_RISCV_21x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_21x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 21] @DRAM
// )
void gemm_RISCV_21x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_21x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 21] @DRAM
// )
void gemm_RISCV_21x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_21x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 21] @DRAM
// )
void gemm_RISCV_21x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_21x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 21] @DRAM
// )
void gemm_RISCV_21x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_21x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 21] @DRAM
// )
void gemm_RISCV_21x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_21x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 21] @DRAM
// )
void gemm_RISCV_21x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_21x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 21] @DRAM
// )
void gemm_RISCV_21x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_21x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 21] @DRAM
// )
void gemm_RISCV_21x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_21x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 21] @DRAM
// )
void gemm_RISCV_21x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_21x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 21] @DRAM
// )
void gemm_RISCV_21x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_21x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 21] @DRAM
// )
void gemm_RISCV_21x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_22x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 22] @DRAM
// )
void gemm_RISCV_22x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_22x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 22] @DRAM
// )
void gemm_RISCV_22x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_22x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 22] @DRAM
// )
void gemm_RISCV_22x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_22x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 22] @DRAM
// )
void gemm_RISCV_22x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_22x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 22] @DRAM
// )
void gemm_RISCV_22x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_22x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 22] @DRAM
// )
void gemm_RISCV_22x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_22x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 22] @DRAM
// )
void gemm_RISCV_22x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_22x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 22] @DRAM
// )
void gemm_RISCV_22x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_22x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 22] @DRAM
// )
void gemm_RISCV_22x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_22x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 22] @DRAM
// )
void gemm_RISCV_22x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_22x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 22] @DRAM
// )
void gemm_RISCV_22x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_22x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 22] @DRAM
// )
void gemm_RISCV_22x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_22x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 22] @DRAM
// )
void gemm_RISCV_22x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_22x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 22] @DRAM
// )
void gemm_RISCV_22x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_22x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 22] @DRAM
// )
void gemm_RISCV_22x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_22x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 22] @DRAM
// )
void gemm_RISCV_22x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_23x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 23] @DRAM
// )
void gemm_RISCV_23x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_23x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 23] @DRAM
// )
void gemm_RISCV_23x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_23x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 23] @DRAM
// )
void gemm_RISCV_23x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_23x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 23] @DRAM
// )
void gemm_RISCV_23x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_23x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 23] @DRAM
// )
void gemm_RISCV_23x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_23x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 23] @DRAM
// )
void gemm_RISCV_23x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_23x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 23] @DRAM
// )
void gemm_RISCV_23x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_23x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 23] @DRAM
// )
void gemm_RISCV_23x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_23x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 23] @DRAM
// )
void gemm_RISCV_23x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_23x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 23] @DRAM
// )
void gemm_RISCV_23x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_23x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 23] @DRAM
// )
void gemm_RISCV_23x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_23x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 23] @DRAM
// )
void gemm_RISCV_23x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_23x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 23] @DRAM
// )
void gemm_RISCV_23x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_23x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 23] @DRAM
// )
void gemm_RISCV_23x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_23x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 23] @DRAM
// )
void gemm_RISCV_23x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_23x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 23] @DRAM
// )
void gemm_RISCV_23x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_24x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 24] @DRAM
// )
void gemm_RISCV_24x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_24x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 24] @DRAM
// )
void gemm_RISCV_24x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_24x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 24] @DRAM
// )
void gemm_RISCV_24x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_24x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 24] @DRAM
// )
void gemm_RISCV_24x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_24x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 24] @DRAM
// )
void gemm_RISCV_24x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_24x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 24] @DRAM
// )
void gemm_RISCV_24x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_24x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 24] @DRAM
// )
void gemm_RISCV_24x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_24x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 24] @DRAM
// )
void gemm_RISCV_24x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_24x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 24] @DRAM
// )
void gemm_RISCV_24x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_24x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 24] @DRAM
// )
void gemm_RISCV_24x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_24x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 24] @DRAM
// )
void gemm_RISCV_24x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_24x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 24] @DRAM
// )
void gemm_RISCV_24x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_24x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 24] @DRAM
// )
void gemm_RISCV_24x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_24x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 24] @DRAM
// )
void gemm_RISCV_24x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_24x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 24] @DRAM
// )
void gemm_RISCV_24x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_24x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 24] @DRAM
// )
void gemm_RISCV_24x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_2x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 2] @DRAM
// )
void gemm_RISCV_2x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_2x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 2] @DRAM
// )
void gemm_RISCV_2x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_2x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 2] @DRAM
// )
void gemm_RISCV_2x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_2x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 2] @DRAM
// )
void gemm_RISCV_2x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_2x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 2] @DRAM
// )
void gemm_RISCV_2x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_2x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 2] @DRAM
// )
void gemm_RISCV_2x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_2x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 2] @DRAM
// )
void gemm_RISCV_2x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_2x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 2] @DRAM
// )
void gemm_RISCV_2x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_2x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 2] @DRAM
// )
void gemm_RISCV_2x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_2x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 2] @DRAM
// )
void gemm_RISCV_2x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_2x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 2] @DRAM
// )
void gemm_RISCV_2x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_2x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 2] @DRAM
// )
void gemm_RISCV_2x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_2x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 2] @DRAM
// )
void gemm_RISCV_2x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_2x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 2] @DRAM
// )
void gemm_RISCV_2x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_2x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 2] @DRAM
// )
void gemm_RISCV_2x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_2x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 2] @DRAM
// )
void gemm_RISCV_2x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_3x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 3] @DRAM
// )
void gemm_RISCV_3x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_3x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 3] @DRAM
// )
void gemm_RISCV_3x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_3x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 3] @DRAM
// )
void gemm_RISCV_3x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_3x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 3] @DRAM
// )
void gemm_RISCV_3x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_3x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 3] @DRAM
// )
void gemm_RISCV_3x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_3x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 3] @DRAM
// )
void gemm_RISCV_3x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_3x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 3] @DRAM
// )
void gemm_RISCV_3x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_3x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 3] @DRAM
// )
void gemm_RISCV_3x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_3x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 3] @DRAM
// )
void gemm_RISCV_3x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_3x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 3] @DRAM
// )
void gemm_RISCV_3x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_3x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 3] @DRAM
// )
void gemm_RISCV_3x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_3x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 3] @DRAM
// )
void gemm_RISCV_3x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_3x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 3] @DRAM
// )
void gemm_RISCV_3x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_3x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 3] @DRAM
// )
void gemm_RISCV_3x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_3x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 3] @DRAM
// )
void gemm_RISCV_3x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_3x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 3] @DRAM
// )
void gemm_RISCV_3x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_4x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 4] @DRAM
// )
void gemm_RISCV_4x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_4x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 4] @DRAM
// )
void gemm_RISCV_4x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_4x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 4] @DRAM
// )
void gemm_RISCV_4x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_4x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 4] @DRAM
// )
void gemm_RISCV_4x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_4x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 4] @DRAM
// )
void gemm_RISCV_4x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_4x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 4] @DRAM
// )
void gemm_RISCV_4x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_4x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 4] @DRAM
// )
void gemm_RISCV_4x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_4x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 4] @DRAM
// )
void gemm_RISCV_4x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_4x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 4] @DRAM
// )
void gemm_RISCV_4x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_4x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 4] @DRAM
// )
void gemm_RISCV_4x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_4x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 4] @DRAM
// )
void gemm_RISCV_4x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_4x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 4] @DRAM
// )
void gemm_RISCV_4x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_4x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 4] @DRAM
// )
void gemm_RISCV_4x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_4x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 4] @DRAM
// )
void gemm_RISCV_4x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_4x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 4] @DRAM
// )
void gemm_RISCV_4x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_4x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 4] @DRAM
// )
void gemm_RISCV_4x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_5x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 5] @DRAM
// )
void gemm_RISCV_5x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_5x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 5] @DRAM
// )
void gemm_RISCV_5x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_5x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 5] @DRAM
// )
void gemm_RISCV_5x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_5x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 5] @DRAM
// )
void gemm_RISCV_5x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_5x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 5] @DRAM
// )
void gemm_RISCV_5x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_5x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 5] @DRAM
// )
void gemm_RISCV_5x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_5x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 5] @DRAM
// )
void gemm_RISCV_5x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_5x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 5] @DRAM
// )
void gemm_RISCV_5x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_5x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 5] @DRAM
// )
void gemm_RISCV_5x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_5x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 5] @DRAM
// )
void gemm_RISCV_5x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_5x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 5] @DRAM
// )
void gemm_RISCV_5x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_5x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 5] @DRAM
// )
void gemm_RISCV_5x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_5x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 5] @DRAM
// )
void gemm_RISCV_5x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_5x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 5] @DRAM
// )
void gemm_RISCV_5x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_5x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 5] @DRAM
// )
void gemm_RISCV_5x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_5x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 5] @DRAM
// )
void gemm_RISCV_5x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_6x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 6] @DRAM
// )
void gemm_RISCV_6x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_6x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 6] @DRAM
// )
void gemm_RISCV_6x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_6x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 6] @DRAM
// )
void gemm_RISCV_6x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_6x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 6] @DRAM
// )
void gemm_RISCV_6x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_6x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 6] @DRAM
// )
void gemm_RISCV_6x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_6x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 6] @DRAM
// )
void gemm_RISCV_6x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_6x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 6] @DRAM
// )
void gemm_RISCV_6x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_6x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 6] @DRAM
// )
void gemm_RISCV_6x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_6x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 6] @DRAM
// )
void gemm_RISCV_6x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_6x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 6] @DRAM
// )
void gemm_RISCV_6x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_6x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 6] @DRAM
// )
void gemm_RISCV_6x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_6x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 6] @DRAM
// )
void gemm_RISCV_6x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_6x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 6] @DRAM
// )
void gemm_RISCV_6x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_6x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 6] @DRAM
// )
void gemm_RISCV_6x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_6x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 6] @DRAM
// )
void gemm_RISCV_6x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_6x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 6] @DRAM
// )
void gemm_RISCV_6x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_7x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 7] @DRAM
// )
void gemm_RISCV_7x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_7x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 7] @DRAM
// )
void gemm_RISCV_7x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_7x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 7] @DRAM
// )
void gemm_RISCV_7x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_7x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 7] @DRAM
// )
void gemm_RISCV_7x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_7x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 7] @DRAM
// )
void gemm_RISCV_7x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_7x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 7] @DRAM
// )
void gemm_RISCV_7x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_7x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 7] @DRAM
// )
void gemm_RISCV_7x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_7x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 7] @DRAM
// )
void gemm_RISCV_7x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_7x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 7] @DRAM
// )
void gemm_RISCV_7x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_7x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 7] @DRAM
// )
void gemm_RISCV_7x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_7x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 7] @DRAM
// )
void gemm_RISCV_7x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_7x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 7] @DRAM
// )
void gemm_RISCV_7x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_7x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 7] @DRAM
// )
void gemm_RISCV_7x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_7x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 7] @DRAM
// )
void gemm_RISCV_7x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_7x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 7] @DRAM
// )
void gemm_RISCV_7x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_7x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 7] @DRAM
// )
void gemm_RISCV_7x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_8x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 8] @DRAM
// )
void gemm_RISCV_8x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_8x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 8] @DRAM
// )
void gemm_RISCV_8x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_8x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 8] @DRAM
// )
void gemm_RISCV_8x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_8x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 8] @DRAM
// )
void gemm_RISCV_8x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_8x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 8] @DRAM
// )
void gemm_RISCV_8x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_8x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 8] @DRAM
// )
void gemm_RISCV_8x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_8x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 8] @DRAM
// )
void gemm_RISCV_8x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_8x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 8] @DRAM
// )
void gemm_RISCV_8x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_8x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 8] @DRAM
// )
void gemm_RISCV_8x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_8x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 8] @DRAM
// )
void gemm_RISCV_8x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_8x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 8] @DRAM
// )
void gemm_RISCV_8x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_8x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 8] @DRAM
// )
void gemm_RISCV_8x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_8x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 8] @DRAM
// )
void gemm_RISCV_8x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_8x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 8] @DRAM
// )
void gemm_RISCV_8x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_8x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 8] @DRAM
// )
void gemm_RISCV_8x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_8x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 8] @DRAM
// )
void gemm_RISCV_8x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_9x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 9] @DRAM
// )
void gemm_RISCV_9x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_9x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 9] @DRAM
// )
void gemm_RISCV_9x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_9x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 9] @DRAM
// )
void gemm_RISCV_9x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_9x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 9] @DRAM
// )
void gemm_RISCV_9x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_9x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 9] @DRAM
// )
void gemm_RISCV_9x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_9x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 9] @DRAM
// )
void gemm_RISCV_9x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_9x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 9] @DRAM
// )
void gemm_RISCV_9x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_9x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 9] @DRAM
// )
void gemm_RISCV_9x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_9x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 9] @DRAM
// )
void gemm_RISCV_9x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_9x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 9] @DRAM
// )
void gemm_RISCV_9x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_9x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 9] @DRAM
// )
void gemm_RISCV_9x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_9x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 9] @DRAM
// )
void gemm_RISCV_9x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_9x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 9] @DRAM
// )
void gemm_RISCV_9x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_9x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 9] @DRAM
// )
void gemm_RISCV_9x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_9x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 9] @DRAM
// )
void gemm_RISCV_9x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_RISCV_9x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 9] @DRAM
// )
void gemm_RISCV_9x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );



#ifdef __cplusplus
}
#endif
#endif  // KERNELS_RVV_24X8_FP16_H
