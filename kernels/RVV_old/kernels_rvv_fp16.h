
#pragma once
#ifndef AAA_H
#define AAA_H

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
// gemm_RISCV_10x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 10] @DRAM
// )
void gemm_RISCV_10x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 10] @DRAM
// )
void gemm_RISCV_10x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 10] @DRAM
// )
void gemm_RISCV_10x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 10] @DRAM
// )
void gemm_RISCV_10x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 10] @DRAM
// )
void gemm_RISCV_10x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 10] @DRAM
// )
void gemm_RISCV_10x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 10] @DRAM
// )
void gemm_RISCV_10x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 10] @DRAM
// )
void gemm_RISCV_10x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 10] @DRAM
// )
void gemm_RISCV_10x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 10] @DRAM
// )
void gemm_RISCV_10x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 10] @DRAM
// )
void gemm_RISCV_10x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 10] @DRAM
// )
void gemm_RISCV_10x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 10] @DRAM
// )
void gemm_RISCV_10x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 10] @DRAM
// )
void gemm_RISCV_10x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 10] @DRAM
// )
void gemm_RISCV_10x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 10] @DRAM
// )
void gemm_RISCV_10x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 10] @DRAM
// )
void gemm_RISCV_10x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 10] @DRAM
// )
void gemm_RISCV_10x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 10] @DRAM
// )
void gemm_RISCV_10x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 10] @DRAM
// )
void gemm_RISCV_10x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 10] @DRAM
// )
void gemm_RISCV_10x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 10] @DRAM
// )
void gemm_RISCV_10x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 10] @DRAM
// )
void gemm_RISCV_10x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 10] @DRAM
// )
void gemm_RISCV_10x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 10] @DRAM
// )
void gemm_RISCV_10x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 10] @DRAM
// )
void gemm_RISCV_10x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 10] @DRAM
// )
void gemm_RISCV_10x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 10] @DRAM
// )
void gemm_RISCV_10x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 10] @DRAM
// )
void gemm_RISCV_10x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 10] @DRAM
// )
void gemm_RISCV_10x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 10] @DRAM
// )
void gemm_RISCV_10x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_10x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 10] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 10] @DRAM
// )
void gemm_RISCV_10x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 11] @DRAM
// )
void gemm_RISCV_11x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 11] @DRAM
// )
void gemm_RISCV_11x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 11] @DRAM
// )
void gemm_RISCV_11x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 11] @DRAM
// )
void gemm_RISCV_11x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 11] @DRAM
// )
void gemm_RISCV_11x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 11] @DRAM
// )
void gemm_RISCV_11x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 11] @DRAM
// )
void gemm_RISCV_11x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 11] @DRAM
// )
void gemm_RISCV_11x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 11] @DRAM
// )
void gemm_RISCV_11x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 11] @DRAM
// )
void gemm_RISCV_11x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 11] @DRAM
// )
void gemm_RISCV_11x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 11] @DRAM
// )
void gemm_RISCV_11x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 11] @DRAM
// )
void gemm_RISCV_11x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 11] @DRAM
// )
void gemm_RISCV_11x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 11] @DRAM
// )
void gemm_RISCV_11x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 11] @DRAM
// )
void gemm_RISCV_11x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 11] @DRAM
// )
void gemm_RISCV_11x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 11] @DRAM
// )
void gemm_RISCV_11x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 11] @DRAM
// )
void gemm_RISCV_11x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 11] @DRAM
// )
void gemm_RISCV_11x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 11] @DRAM
// )
void gemm_RISCV_11x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 11] @DRAM
// )
void gemm_RISCV_11x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 11] @DRAM
// )
void gemm_RISCV_11x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 11] @DRAM
// )
void gemm_RISCV_11x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 11] @DRAM
// )
void gemm_RISCV_11x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 11] @DRAM
// )
void gemm_RISCV_11x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 11] @DRAM
// )
void gemm_RISCV_11x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 11] @DRAM
// )
void gemm_RISCV_11x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 11] @DRAM
// )
void gemm_RISCV_11x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 11] @DRAM
// )
void gemm_RISCV_11x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 11] @DRAM
// )
void gemm_RISCV_11x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_11x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 11] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 11] @DRAM
// )
void gemm_RISCV_11x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 12] @DRAM
// )
void gemm_RISCV_12x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 12] @DRAM
// )
void gemm_RISCV_12x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 12] @DRAM
// )
void gemm_RISCV_12x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 12] @DRAM
// )
void gemm_RISCV_12x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 12] @DRAM
// )
void gemm_RISCV_12x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 12] @DRAM
// )
void gemm_RISCV_12x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 12] @DRAM
// )
void gemm_RISCV_12x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 12] @DRAM
// )
void gemm_RISCV_12x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 12] @DRAM
// )
void gemm_RISCV_12x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 12] @DRAM
// )
void gemm_RISCV_12x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 12] @DRAM
// )
void gemm_RISCV_12x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 12] @DRAM
// )
void gemm_RISCV_12x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 12] @DRAM
// )
void gemm_RISCV_12x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 12] @DRAM
// )
void gemm_RISCV_12x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 12] @DRAM
// )
void gemm_RISCV_12x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 12] @DRAM
// )
void gemm_RISCV_12x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 12] @DRAM
// )
void gemm_RISCV_12x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 12] @DRAM
// )
void gemm_RISCV_12x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 12] @DRAM
// )
void gemm_RISCV_12x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 12] @DRAM
// )
void gemm_RISCV_12x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 12] @DRAM
// )
void gemm_RISCV_12x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 12] @DRAM
// )
void gemm_RISCV_12x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 12] @DRAM
// )
void gemm_RISCV_12x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 12] @DRAM
// )
void gemm_RISCV_12x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 12] @DRAM
// )
void gemm_RISCV_12x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 12] @DRAM
// )
void gemm_RISCV_12x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 12] @DRAM
// )
void gemm_RISCV_12x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 12] @DRAM
// )
void gemm_RISCV_12x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 12] @DRAM
// )
void gemm_RISCV_12x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 12] @DRAM
// )
void gemm_RISCV_12x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 12] @DRAM
// )
void gemm_RISCV_12x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_12x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 12] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 12] @DRAM
// )
void gemm_RISCV_12x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 13] @DRAM
// )
void gemm_RISCV_13x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 13] @DRAM
// )
void gemm_RISCV_13x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 13] @DRAM
// )
void gemm_RISCV_13x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 13] @DRAM
// )
void gemm_RISCV_13x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 13] @DRAM
// )
void gemm_RISCV_13x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 13] @DRAM
// )
void gemm_RISCV_13x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 13] @DRAM
// )
void gemm_RISCV_13x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 13] @DRAM
// )
void gemm_RISCV_13x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 13] @DRAM
// )
void gemm_RISCV_13x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 13] @DRAM
// )
void gemm_RISCV_13x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 13] @DRAM
// )
void gemm_RISCV_13x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 13] @DRAM
// )
void gemm_RISCV_13x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 13] @DRAM
// )
void gemm_RISCV_13x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 13] @DRAM
// )
void gemm_RISCV_13x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 13] @DRAM
// )
void gemm_RISCV_13x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 13] @DRAM
// )
void gemm_RISCV_13x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 13] @DRAM
// )
void gemm_RISCV_13x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 13] @DRAM
// )
void gemm_RISCV_13x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 13] @DRAM
// )
void gemm_RISCV_13x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 13] @DRAM
// )
void gemm_RISCV_13x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 13] @DRAM
// )
void gemm_RISCV_13x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 13] @DRAM
// )
void gemm_RISCV_13x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 13] @DRAM
// )
void gemm_RISCV_13x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 13] @DRAM
// )
void gemm_RISCV_13x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 13] @DRAM
// )
void gemm_RISCV_13x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 13] @DRAM
// )
void gemm_RISCV_13x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 13] @DRAM
// )
void gemm_RISCV_13x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 13] @DRAM
// )
void gemm_RISCV_13x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 13] @DRAM
// )
void gemm_RISCV_13x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 13] @DRAM
// )
void gemm_RISCV_13x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 13] @DRAM
// )
void gemm_RISCV_13x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_13x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 13] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 13] @DRAM
// )
void gemm_RISCV_13x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 14] @DRAM
// )
void gemm_RISCV_14x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 14] @DRAM
// )
void gemm_RISCV_14x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 14] @DRAM
// )
void gemm_RISCV_14x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 14] @DRAM
// )
void gemm_RISCV_14x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 14] @DRAM
// )
void gemm_RISCV_14x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 14] @DRAM
// )
void gemm_RISCV_14x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 14] @DRAM
// )
void gemm_RISCV_14x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 14] @DRAM
// )
void gemm_RISCV_14x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 14] @DRAM
// )
void gemm_RISCV_14x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 14] @DRAM
// )
void gemm_RISCV_14x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 14] @DRAM
// )
void gemm_RISCV_14x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 14] @DRAM
// )
void gemm_RISCV_14x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 14] @DRAM
// )
void gemm_RISCV_14x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 14] @DRAM
// )
void gemm_RISCV_14x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 14] @DRAM
// )
void gemm_RISCV_14x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 14] @DRAM
// )
void gemm_RISCV_14x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 14] @DRAM
// )
void gemm_RISCV_14x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 14] @DRAM
// )
void gemm_RISCV_14x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 14] @DRAM
// )
void gemm_RISCV_14x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 14] @DRAM
// )
void gemm_RISCV_14x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 14] @DRAM
// )
void gemm_RISCV_14x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 14] @DRAM
// )
void gemm_RISCV_14x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 14] @DRAM
// )
void gemm_RISCV_14x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 14] @DRAM
// )
void gemm_RISCV_14x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 14] @DRAM
// )
void gemm_RISCV_14x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 14] @DRAM
// )
void gemm_RISCV_14x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 14] @DRAM
// )
void gemm_RISCV_14x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 14] @DRAM
// )
void gemm_RISCV_14x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 14] @DRAM
// )
void gemm_RISCV_14x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 14] @DRAM
// )
void gemm_RISCV_14x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 14] @DRAM
// )
void gemm_RISCV_14x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_14x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 14] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 14] @DRAM
// )
void gemm_RISCV_14x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 15] @DRAM
// )
void gemm_RISCV_15x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 15] @DRAM
// )
void gemm_RISCV_15x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 15] @DRAM
// )
void gemm_RISCV_15x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 15] @DRAM
// )
void gemm_RISCV_15x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 15] @DRAM
// )
void gemm_RISCV_15x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 15] @DRAM
// )
void gemm_RISCV_15x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 15] @DRAM
// )
void gemm_RISCV_15x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 15] @DRAM
// )
void gemm_RISCV_15x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 15] @DRAM
// )
void gemm_RISCV_15x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 15] @DRAM
// )
void gemm_RISCV_15x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 15] @DRAM
// )
void gemm_RISCV_15x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 15] @DRAM
// )
void gemm_RISCV_15x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 15] @DRAM
// )
void gemm_RISCV_15x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 15] @DRAM
// )
void gemm_RISCV_15x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 15] @DRAM
// )
void gemm_RISCV_15x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 15] @DRAM
// )
void gemm_RISCV_15x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 15] @DRAM
// )
void gemm_RISCV_15x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 15] @DRAM
// )
void gemm_RISCV_15x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 15] @DRAM
// )
void gemm_RISCV_15x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 15] @DRAM
// )
void gemm_RISCV_15x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 15] @DRAM
// )
void gemm_RISCV_15x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 15] @DRAM
// )
void gemm_RISCV_15x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 15] @DRAM
// )
void gemm_RISCV_15x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 15] @DRAM
// )
void gemm_RISCV_15x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 15] @DRAM
// )
void gemm_RISCV_15x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 15] @DRAM
// )
void gemm_RISCV_15x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 15] @DRAM
// )
void gemm_RISCV_15x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 15] @DRAM
// )
void gemm_RISCV_15x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 15] @DRAM
// )
void gemm_RISCV_15x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 15] @DRAM
// )
void gemm_RISCV_15x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 15] @DRAM
// )
void gemm_RISCV_15x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_15x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 15] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 15] @DRAM
// )
void gemm_RISCV_15x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 16] @DRAM
// )
void gemm_RISCV_16x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 16] @DRAM
// )
void gemm_RISCV_16x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 16] @DRAM
// )
void gemm_RISCV_16x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 16] @DRAM
// )
void gemm_RISCV_16x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 16] @DRAM
// )
void gemm_RISCV_16x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 16] @DRAM
// )
void gemm_RISCV_16x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 16] @DRAM
// )
void gemm_RISCV_16x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 16] @DRAM
// )
void gemm_RISCV_16x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 16] @DRAM
// )
void gemm_RISCV_16x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 16] @DRAM
// )
void gemm_RISCV_16x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 16] @DRAM
// )
void gemm_RISCV_16x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 16] @DRAM
// )
void gemm_RISCV_16x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 16] @DRAM
// )
void gemm_RISCV_16x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 16] @DRAM
// )
void gemm_RISCV_16x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 16] @DRAM
// )
void gemm_RISCV_16x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 16] @DRAM
// )
void gemm_RISCV_16x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 16] @DRAM
// )
void gemm_RISCV_16x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 16] @DRAM
// )
void gemm_RISCV_16x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 16] @DRAM
// )
void gemm_RISCV_16x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 16] @DRAM
// )
void gemm_RISCV_16x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 16] @DRAM
// )
void gemm_RISCV_16x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 16] @DRAM
// )
void gemm_RISCV_16x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 16] @DRAM
// )
void gemm_RISCV_16x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 16] @DRAM
// )
void gemm_RISCV_16x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 16] @DRAM
// )
void gemm_RISCV_16x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 16] @DRAM
// )
void gemm_RISCV_16x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 16] @DRAM
// )
void gemm_RISCV_16x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 16] @DRAM
// )
void gemm_RISCV_16x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 16] @DRAM
// )
void gemm_RISCV_16x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 16] @DRAM
// )
void gemm_RISCV_16x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 16] @DRAM
// )
void gemm_RISCV_16x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_16x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 16] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 16] @DRAM
// )
void gemm_RISCV_16x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_17x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 17] @DRAM
// )
void gemm_RISCV_17x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_17x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 17] @DRAM
// )
void gemm_RISCV_17x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_17x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 17] @DRAM
// )
void gemm_RISCV_17x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_17x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 17] @DRAM
// )
void gemm_RISCV_17x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_17x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 17] @DRAM
// )
void gemm_RISCV_17x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_17x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 17] @DRAM
// )
void gemm_RISCV_17x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_17x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 17] @DRAM
// )
void gemm_RISCV_17x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_17x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 17] @DRAM
// )
void gemm_RISCV_17x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_17x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 17] @DRAM
// )
void gemm_RISCV_17x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_17x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 17] @DRAM
// )
void gemm_RISCV_17x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_17x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 17] @DRAM
// )
void gemm_RISCV_17x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_17x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 17] @DRAM
// )
void gemm_RISCV_17x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_17x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 17] @DRAM
// )
void gemm_RISCV_17x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_17x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 17] @DRAM
// )
void gemm_RISCV_17x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_17x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 17] @DRAM
// )
void gemm_RISCV_17x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_17x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 17] @DRAM
// )
void gemm_RISCV_17x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_17x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 17] @DRAM
// )
void gemm_RISCV_17x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_17x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 17] @DRAM
// )
void gemm_RISCV_17x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_17x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 17] @DRAM
// )
void gemm_RISCV_17x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_17x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 17] @DRAM
// )
void gemm_RISCV_17x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_17x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 17] @DRAM
// )
void gemm_RISCV_17x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_17x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 17] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 17] @DRAM
// )
void gemm_RISCV_17x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_18x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 18] @DRAM
// )
void gemm_RISCV_18x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_18x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 18] @DRAM
// )
void gemm_RISCV_18x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_18x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 18] @DRAM
// )
void gemm_RISCV_18x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_18x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 18] @DRAM
// )
void gemm_RISCV_18x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_18x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 18] @DRAM
// )
void gemm_RISCV_18x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_18x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 18] @DRAM
// )
void gemm_RISCV_18x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_18x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 18] @DRAM
// )
void gemm_RISCV_18x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_18x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 18] @DRAM
// )
void gemm_RISCV_18x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_18x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 18] @DRAM
// )
void gemm_RISCV_18x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_18x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 18] @DRAM
// )
void gemm_RISCV_18x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_18x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 18] @DRAM
// )
void gemm_RISCV_18x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_18x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 18] @DRAM
// )
void gemm_RISCV_18x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_18x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 18] @DRAM
// )
void gemm_RISCV_18x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_18x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 18] @DRAM
// )
void gemm_RISCV_18x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_18x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 18] @DRAM
// )
void gemm_RISCV_18x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_18x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 18] @DRAM
// )
void gemm_RISCV_18x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_18x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 18] @DRAM
// )
void gemm_RISCV_18x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_18x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 18] @DRAM
// )
void gemm_RISCV_18x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_18x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 18] @DRAM
// )
void gemm_RISCV_18x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_18x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 18] @DRAM
// )
void gemm_RISCV_18x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_18x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 18] @DRAM
// )
void gemm_RISCV_18x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_18x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 18] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 18] @DRAM
// )
void gemm_RISCV_18x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_19x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 19] @DRAM
// )
void gemm_RISCV_19x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_19x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 19] @DRAM
// )
void gemm_RISCV_19x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_19x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 19] @DRAM
// )
void gemm_RISCV_19x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_19x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 19] @DRAM
// )
void gemm_RISCV_19x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_19x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 19] @DRAM
// )
void gemm_RISCV_19x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_19x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 19] @DRAM
// )
void gemm_RISCV_19x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_19x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 19] @DRAM
// )
void gemm_RISCV_19x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_19x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 19] @DRAM
// )
void gemm_RISCV_19x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_19x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 19] @DRAM
// )
void gemm_RISCV_19x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_19x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 19] @DRAM
// )
void gemm_RISCV_19x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_19x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 19] @DRAM
// )
void gemm_RISCV_19x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_19x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 19] @DRAM
// )
void gemm_RISCV_19x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_19x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 19] @DRAM
// )
void gemm_RISCV_19x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_19x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 19] @DRAM
// )
void gemm_RISCV_19x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_19x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 19] @DRAM
// )
void gemm_RISCV_19x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_19x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 19] @DRAM
// )
void gemm_RISCV_19x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_19x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 19] @DRAM
// )
void gemm_RISCV_19x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_19x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 19] @DRAM
// )
void gemm_RISCV_19x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_19x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 19] @DRAM
// )
void gemm_RISCV_19x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_19x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 19] @DRAM
// )
void gemm_RISCV_19x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_19x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 19] @DRAM
// )
void gemm_RISCV_19x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_19x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 19] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 19] @DRAM
// )
void gemm_RISCV_19x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 1] @DRAM
// )
void gemm_RISCV_1x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 1] @DRAM
// )
void gemm_RISCV_1x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 1] @DRAM
// )
void gemm_RISCV_1x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 1] @DRAM
// )
void gemm_RISCV_1x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 1] @DRAM
// )
void gemm_RISCV_1x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 1] @DRAM
// )
void gemm_RISCV_1x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 1] @DRAM
// )
void gemm_RISCV_1x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 1] @DRAM
// )
void gemm_RISCV_1x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 1] @DRAM
// )
void gemm_RISCV_1x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 1] @DRAM
// )
void gemm_RISCV_1x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 1] @DRAM
// )
void gemm_RISCV_1x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 1] @DRAM
// )
void gemm_RISCV_1x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 1] @DRAM
// )
void gemm_RISCV_1x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 1] @DRAM
// )
void gemm_RISCV_1x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x17_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 1] @DRAM
// )
void gemm_RISCV_1x17_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x17_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 1] @DRAM
// )
void gemm_RISCV_1x17_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x18_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 1] @DRAM
// )
void gemm_RISCV_1x18_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x18_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 1] @DRAM
// )
void gemm_RISCV_1x18_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x19_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 1] @DRAM
// )
void gemm_RISCV_1x19_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x19_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 1] @DRAM
// )
void gemm_RISCV_1x19_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 1] @DRAM
// )
void gemm_RISCV_1x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 1] @DRAM
// )
void gemm_RISCV_1x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x20_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 1] @DRAM
// )
void gemm_RISCV_1x20_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x20_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 1] @DRAM
// )
void gemm_RISCV_1x20_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x21_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 1] @DRAM
// )
void gemm_RISCV_1x21_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x21_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 1] @DRAM
// )
void gemm_RISCV_1x21_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x22_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 1] @DRAM
// )
void gemm_RISCV_1x22_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x22_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 1] @DRAM
// )
void gemm_RISCV_1x22_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x23_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 1] @DRAM
// )
void gemm_RISCV_1x23_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x23_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 1] @DRAM
// )
void gemm_RISCV_1x23_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x24_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 1] @DRAM
// )
void gemm_RISCV_1x24_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x24_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 1] @DRAM
// )
void gemm_RISCV_1x24_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 1] @DRAM
// )
void gemm_RISCV_1x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 1] @DRAM
// )
void gemm_RISCV_1x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 1] @DRAM
// )
void gemm_RISCV_1x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 1] @DRAM
// )
void gemm_RISCV_1x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 1] @DRAM
// )
void gemm_RISCV_1x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 1] @DRAM
// )
void gemm_RISCV_1x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 1] @DRAM
// )
void gemm_RISCV_1x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 1] @DRAM
// )
void gemm_RISCV_1x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 1] @DRAM
// )
void gemm_RISCV_1x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 1] @DRAM
// )
void gemm_RISCV_1x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 1] @DRAM
// )
void gemm_RISCV_1x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 1] @DRAM
// )
void gemm_RISCV_1x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 1] @DRAM
// )
void gemm_RISCV_1x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 1] @DRAM
// )
void gemm_RISCV_1x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 1] @DRAM
// )
void gemm_RISCV_1x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_1x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 1] @DRAM
// )
void gemm_RISCV_1x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_20x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 20] @DRAM
// )
void gemm_RISCV_20x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_20x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 20] @DRAM
// )
void gemm_RISCV_20x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_20x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 20] @DRAM
// )
void gemm_RISCV_20x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_20x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 20] @DRAM
// )
void gemm_RISCV_20x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_20x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 20] @DRAM
// )
void gemm_RISCV_20x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_20x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 20] @DRAM
// )
void gemm_RISCV_20x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_20x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 20] @DRAM
// )
void gemm_RISCV_20x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_20x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 20] @DRAM
// )
void gemm_RISCV_20x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_20x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 20] @DRAM
// )
void gemm_RISCV_20x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_20x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 20] @DRAM
// )
void gemm_RISCV_20x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_20x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 20] @DRAM
// )
void gemm_RISCV_20x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_20x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 20] @DRAM
// )
void gemm_RISCV_20x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_20x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 20] @DRAM
// )
void gemm_RISCV_20x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_20x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 20] @DRAM
// )
void gemm_RISCV_20x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_20x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 20] @DRAM
// )
void gemm_RISCV_20x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_20x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 20] @DRAM
// )
void gemm_RISCV_20x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_20x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 20] @DRAM
// )
void gemm_RISCV_20x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_20x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 20] @DRAM
// )
void gemm_RISCV_20x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_20x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 20] @DRAM
// )
void gemm_RISCV_20x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_20x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 20] @DRAM
// )
void gemm_RISCV_20x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_20x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 20] @DRAM
// )
void gemm_RISCV_20x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_20x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 20] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 20] @DRAM
// )
void gemm_RISCV_20x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_21x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 21] @DRAM
// )
void gemm_RISCV_21x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_21x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 21] @DRAM
// )
void gemm_RISCV_21x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_21x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 21] @DRAM
// )
void gemm_RISCV_21x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_21x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 21] @DRAM
// )
void gemm_RISCV_21x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_21x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 21] @DRAM
// )
void gemm_RISCV_21x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_21x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 21] @DRAM
// )
void gemm_RISCV_21x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_21x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 21] @DRAM
// )
void gemm_RISCV_21x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_21x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 21] @DRAM
// )
void gemm_RISCV_21x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_21x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 21] @DRAM
// )
void gemm_RISCV_21x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_21x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 21] @DRAM
// )
void gemm_RISCV_21x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_21x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 21] @DRAM
// )
void gemm_RISCV_21x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_21x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 21] @DRAM
// )
void gemm_RISCV_21x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_21x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 21] @DRAM
// )
void gemm_RISCV_21x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_21x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 21] @DRAM
// )
void gemm_RISCV_21x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_21x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 21] @DRAM
// )
void gemm_RISCV_21x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_21x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 21] @DRAM
// )
void gemm_RISCV_21x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_21x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 21] @DRAM
// )
void gemm_RISCV_21x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_21x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 21] @DRAM
// )
void gemm_RISCV_21x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_21x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 21] @DRAM
// )
void gemm_RISCV_21x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_21x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 21] @DRAM
// )
void gemm_RISCV_21x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_21x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 21] @DRAM
// )
void gemm_RISCV_21x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_21x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 21] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 21] @DRAM
// )
void gemm_RISCV_21x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_22x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 22] @DRAM
// )
void gemm_RISCV_22x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_22x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 22] @DRAM
// )
void gemm_RISCV_22x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_22x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 22] @DRAM
// )
void gemm_RISCV_22x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_22x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 22] @DRAM
// )
void gemm_RISCV_22x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_22x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 22] @DRAM
// )
void gemm_RISCV_22x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_22x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 22] @DRAM
// )
void gemm_RISCV_22x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_22x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 22] @DRAM
// )
void gemm_RISCV_22x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_22x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 22] @DRAM
// )
void gemm_RISCV_22x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_22x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 22] @DRAM
// )
void gemm_RISCV_22x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_22x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 22] @DRAM
// )
void gemm_RISCV_22x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_22x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 22] @DRAM
// )
void gemm_RISCV_22x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_22x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 22] @DRAM
// )
void gemm_RISCV_22x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_22x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 22] @DRAM
// )
void gemm_RISCV_22x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_22x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 22] @DRAM
// )
void gemm_RISCV_22x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_22x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 22] @DRAM
// )
void gemm_RISCV_22x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_22x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 22] @DRAM
// )
void gemm_RISCV_22x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_22x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 22] @DRAM
// )
void gemm_RISCV_22x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_22x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 22] @DRAM
// )
void gemm_RISCV_22x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_22x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 22] @DRAM
// )
void gemm_RISCV_22x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_22x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 22] @DRAM
// )
void gemm_RISCV_22x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_22x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 22] @DRAM
// )
void gemm_RISCV_22x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_22x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 22] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 22] @DRAM
// )
void gemm_RISCV_22x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_23x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 23] @DRAM
// )
void gemm_RISCV_23x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_23x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 23] @DRAM
// )
void gemm_RISCV_23x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_23x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 23] @DRAM
// )
void gemm_RISCV_23x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_23x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 23] @DRAM
// )
void gemm_RISCV_23x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_23x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 23] @DRAM
// )
void gemm_RISCV_23x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_23x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 23] @DRAM
// )
void gemm_RISCV_23x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_23x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 23] @DRAM
// )
void gemm_RISCV_23x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_23x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 23] @DRAM
// )
void gemm_RISCV_23x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_23x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 23] @DRAM
// )
void gemm_RISCV_23x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_23x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 23] @DRAM
// )
void gemm_RISCV_23x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_23x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 23] @DRAM
// )
void gemm_RISCV_23x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_23x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 23] @DRAM
// )
void gemm_RISCV_23x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_23x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 23] @DRAM
// )
void gemm_RISCV_23x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_23x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 23] @DRAM
// )
void gemm_RISCV_23x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_23x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 23] @DRAM
// )
void gemm_RISCV_23x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_23x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 23] @DRAM
// )
void gemm_RISCV_23x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_23x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 23] @DRAM
// )
void gemm_RISCV_23x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_23x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 23] @DRAM
// )
void gemm_RISCV_23x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_23x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 23] @DRAM
// )
void gemm_RISCV_23x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_23x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 23] @DRAM
// )
void gemm_RISCV_23x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_23x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 23] @DRAM
// )
void gemm_RISCV_23x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_23x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 23] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 23] @DRAM
// )
void gemm_RISCV_23x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_24x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 24] @DRAM
// )
void gemm_RISCV_24x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_24x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 24] @DRAM
// )
void gemm_RISCV_24x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_24x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 24] @DRAM
// )
void gemm_RISCV_24x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_24x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 24] @DRAM
// )
void gemm_RISCV_24x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_24x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 24] @DRAM
// )
void gemm_RISCV_24x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_24x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 24] @DRAM
// )
void gemm_RISCV_24x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_24x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 24] @DRAM
// )
void gemm_RISCV_24x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_24x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 24] @DRAM
// )
void gemm_RISCV_24x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_24x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 24] @DRAM
// )
void gemm_RISCV_24x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_24x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 24] @DRAM
// )
void gemm_RISCV_24x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_24x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 24] @DRAM
// )
void gemm_RISCV_24x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_24x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 24] @DRAM
// )
void gemm_RISCV_24x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_24x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 24] @DRAM
// )
void gemm_RISCV_24x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_24x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 24] @DRAM
// )
void gemm_RISCV_24x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_24x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 24] @DRAM
// )
void gemm_RISCV_24x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_24x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 24] @DRAM
// )
void gemm_RISCV_24x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_24x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 24] @DRAM
// )
void gemm_RISCV_24x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_24x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 24] @DRAM
// )
void gemm_RISCV_24x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_24x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 24] @DRAM
// )
void gemm_RISCV_24x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_24x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 24] @DRAM
// )
void gemm_RISCV_24x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_24x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 24] @DRAM
// )
void gemm_RISCV_24x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_24x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 24] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 24] @DRAM
// )
void gemm_RISCV_24x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_25x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 25] @DRAM
// )
void gemm_RISCV_25x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_25x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 25] @DRAM
// )
void gemm_RISCV_25x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_25x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 25] @DRAM
// )
void gemm_RISCV_25x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_25x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 25] @DRAM
// )
void gemm_RISCV_25x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_25x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 25] @DRAM
// )
void gemm_RISCV_25x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_25x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 25] @DRAM
// )
void gemm_RISCV_25x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_25x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 25] @DRAM
// )
void gemm_RISCV_25x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_25x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 25] @DRAM
// )
void gemm_RISCV_25x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_25x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 25] @DRAM
// )
void gemm_RISCV_25x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_25x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 25] @DRAM
// )
void gemm_RISCV_25x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_25x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 25] @DRAM
// )
void gemm_RISCV_25x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_25x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 25] @DRAM
// )
void gemm_RISCV_25x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_25x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 25] @DRAM
// )
void gemm_RISCV_25x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_25x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 25] @DRAM
// )
void gemm_RISCV_25x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_25x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 25] @DRAM
// )
void gemm_RISCV_25x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_25x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 25] @DRAM
// )
void gemm_RISCV_25x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_25x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 25] @DRAM
// )
void gemm_RISCV_25x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_25x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 25] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 25] @DRAM
// )
void gemm_RISCV_25x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_26x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 26] @DRAM
// )
void gemm_RISCV_26x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_26x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 26] @DRAM
// )
void gemm_RISCV_26x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_26x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 26] @DRAM
// )
void gemm_RISCV_26x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_26x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 26] @DRAM
// )
void gemm_RISCV_26x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_26x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 26] @DRAM
// )
void gemm_RISCV_26x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_26x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 26] @DRAM
// )
void gemm_RISCV_26x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_26x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 26] @DRAM
// )
void gemm_RISCV_26x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_26x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 26] @DRAM
// )
void gemm_RISCV_26x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_26x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 26] @DRAM
// )
void gemm_RISCV_26x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_26x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 26] @DRAM
// )
void gemm_RISCV_26x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_26x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 26] @DRAM
// )
void gemm_RISCV_26x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_26x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 26] @DRAM
// )
void gemm_RISCV_26x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_26x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 26] @DRAM
// )
void gemm_RISCV_26x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_26x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 26] @DRAM
// )
void gemm_RISCV_26x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_26x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 26] @DRAM
// )
void gemm_RISCV_26x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_26x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 26] @DRAM
// )
void gemm_RISCV_26x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_26x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 26] @DRAM
// )
void gemm_RISCV_26x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_26x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 26] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 26] @DRAM
// )
void gemm_RISCV_26x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_27x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 27] @DRAM
// )
void gemm_RISCV_27x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_27x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 27] @DRAM
// )
void gemm_RISCV_27x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_27x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 27] @DRAM
// )
void gemm_RISCV_27x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_27x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 27] @DRAM
// )
void gemm_RISCV_27x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_27x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 27] @DRAM
// )
void gemm_RISCV_27x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_27x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 27] @DRAM
// )
void gemm_RISCV_27x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_27x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 27] @DRAM
// )
void gemm_RISCV_27x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_27x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 27] @DRAM
// )
void gemm_RISCV_27x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_27x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 27] @DRAM
// )
void gemm_RISCV_27x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_27x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 27] @DRAM
// )
void gemm_RISCV_27x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_27x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 27] @DRAM
// )
void gemm_RISCV_27x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_27x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 27] @DRAM
// )
void gemm_RISCV_27x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_27x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 27] @DRAM
// )
void gemm_RISCV_27x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_27x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 27] @DRAM
// )
void gemm_RISCV_27x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_27x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 27] @DRAM
// )
void gemm_RISCV_27x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_27x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 27] @DRAM
// )
void gemm_RISCV_27x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_27x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 27] @DRAM
// )
void gemm_RISCV_27x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_27x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 27] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 27] @DRAM
// )
void gemm_RISCV_27x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_28x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 28] @DRAM
// )
void gemm_RISCV_28x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_28x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 28] @DRAM
// )
void gemm_RISCV_28x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_28x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 28] @DRAM
// )
void gemm_RISCV_28x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_28x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 28] @DRAM
// )
void gemm_RISCV_28x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_28x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 28] @DRAM
// )
void gemm_RISCV_28x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_28x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 28] @DRAM
// )
void gemm_RISCV_28x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_28x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 28] @DRAM
// )
void gemm_RISCV_28x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_28x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 28] @DRAM
// )
void gemm_RISCV_28x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_28x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 28] @DRAM
// )
void gemm_RISCV_28x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_28x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 28] @DRAM
// )
void gemm_RISCV_28x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_28x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 28] @DRAM
// )
void gemm_RISCV_28x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_28x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 28] @DRAM
// )
void gemm_RISCV_28x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_28x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 28] @DRAM
// )
void gemm_RISCV_28x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_28x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 28] @DRAM
// )
void gemm_RISCV_28x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_28x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 28] @DRAM
// )
void gemm_RISCV_28x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_28x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 28] @DRAM
// )
void gemm_RISCV_28x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_28x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 28] @DRAM
// )
void gemm_RISCV_28x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_28x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 28] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 28] @DRAM
// )
void gemm_RISCV_28x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_29x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 29] @DRAM
// )
void gemm_RISCV_29x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_29x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 29] @DRAM
// )
void gemm_RISCV_29x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_29x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 29] @DRAM
// )
void gemm_RISCV_29x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_29x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 29] @DRAM
// )
void gemm_RISCV_29x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_29x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 29] @DRAM
// )
void gemm_RISCV_29x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_29x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 29] @DRAM
// )
void gemm_RISCV_29x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_29x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 29] @DRAM
// )
void gemm_RISCV_29x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_29x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 29] @DRAM
// )
void gemm_RISCV_29x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_29x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 29] @DRAM
// )
void gemm_RISCV_29x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_29x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 29] @DRAM
// )
void gemm_RISCV_29x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_29x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 29] @DRAM
// )
void gemm_RISCV_29x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_29x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 29] @DRAM
// )
void gemm_RISCV_29x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_29x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 29] @DRAM
// )
void gemm_RISCV_29x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_29x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 29] @DRAM
// )
void gemm_RISCV_29x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_29x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 29] @DRAM
// )
void gemm_RISCV_29x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_29x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 29] @DRAM
// )
void gemm_RISCV_29x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_29x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 29] @DRAM
// )
void gemm_RISCV_29x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_29x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 29] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 29] @DRAM
// )
void gemm_RISCV_29x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 2] @DRAM
// )
void gemm_RISCV_2x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 2] @DRAM
// )
void gemm_RISCV_2x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 2] @DRAM
// )
void gemm_RISCV_2x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 2] @DRAM
// )
void gemm_RISCV_2x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 2] @DRAM
// )
void gemm_RISCV_2x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 2] @DRAM
// )
void gemm_RISCV_2x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 2] @DRAM
// )
void gemm_RISCV_2x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 2] @DRAM
// )
void gemm_RISCV_2x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 2] @DRAM
// )
void gemm_RISCV_2x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 2] @DRAM
// )
void gemm_RISCV_2x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 2] @DRAM
// )
void gemm_RISCV_2x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 2] @DRAM
// )
void gemm_RISCV_2x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 2] @DRAM
// )
void gemm_RISCV_2x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 2] @DRAM
// )
void gemm_RISCV_2x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x17_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 2] @DRAM
// )
void gemm_RISCV_2x17_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x17_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 2] @DRAM
// )
void gemm_RISCV_2x17_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x18_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 2] @DRAM
// )
void gemm_RISCV_2x18_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x18_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 2] @DRAM
// )
void gemm_RISCV_2x18_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x19_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 2] @DRAM
// )
void gemm_RISCV_2x19_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x19_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 2] @DRAM
// )
void gemm_RISCV_2x19_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 2] @DRAM
// )
void gemm_RISCV_2x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 2] @DRAM
// )
void gemm_RISCV_2x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x20_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 2] @DRAM
// )
void gemm_RISCV_2x20_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x20_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 2] @DRAM
// )
void gemm_RISCV_2x20_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x21_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 2] @DRAM
// )
void gemm_RISCV_2x21_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x21_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 2] @DRAM
// )
void gemm_RISCV_2x21_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x22_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 2] @DRAM
// )
void gemm_RISCV_2x22_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x22_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 2] @DRAM
// )
void gemm_RISCV_2x22_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x23_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 2] @DRAM
// )
void gemm_RISCV_2x23_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x23_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 2] @DRAM
// )
void gemm_RISCV_2x23_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x24_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 2] @DRAM
// )
void gemm_RISCV_2x24_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x24_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 2] @DRAM
// )
void gemm_RISCV_2x24_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 2] @DRAM
// )
void gemm_RISCV_2x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 2] @DRAM
// )
void gemm_RISCV_2x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 2] @DRAM
// )
void gemm_RISCV_2x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 2] @DRAM
// )
void gemm_RISCV_2x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 2] @DRAM
// )
void gemm_RISCV_2x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 2] @DRAM
// )
void gemm_RISCV_2x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 2] @DRAM
// )
void gemm_RISCV_2x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 2] @DRAM
// )
void gemm_RISCV_2x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 2] @DRAM
// )
void gemm_RISCV_2x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 2] @DRAM
// )
void gemm_RISCV_2x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 2] @DRAM
// )
void gemm_RISCV_2x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 2] @DRAM
// )
void gemm_RISCV_2x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 2] @DRAM
// )
void gemm_RISCV_2x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 2] @DRAM
// )
void gemm_RISCV_2x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 2] @DRAM
// )
void gemm_RISCV_2x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_2x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 2] @DRAM
// )
void gemm_RISCV_2x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_30x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 30] @DRAM
// )
void gemm_RISCV_30x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_30x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 30] @DRAM
// )
void gemm_RISCV_30x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_30x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 30] @DRAM
// )
void gemm_RISCV_30x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_30x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 30] @DRAM
// )
void gemm_RISCV_30x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_30x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 30] @DRAM
// )
void gemm_RISCV_30x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_30x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 30] @DRAM
// )
void gemm_RISCV_30x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_30x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 30] @DRAM
// )
void gemm_RISCV_30x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_30x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 30] @DRAM
// )
void gemm_RISCV_30x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_30x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 30] @DRAM
// )
void gemm_RISCV_30x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_30x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 30] @DRAM
// )
void gemm_RISCV_30x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_30x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 30] @DRAM
// )
void gemm_RISCV_30x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_30x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 30] @DRAM
// )
void gemm_RISCV_30x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_30x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 30] @DRAM
// )
void gemm_RISCV_30x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_30x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 30] @DRAM
// )
void gemm_RISCV_30x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_30x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 30] @DRAM
// )
void gemm_RISCV_30x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_30x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 30] @DRAM
// )
void gemm_RISCV_30x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_30x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 30] @DRAM
// )
void gemm_RISCV_30x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_30x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 30] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 30] @DRAM
// )
void gemm_RISCV_30x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_31x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 31] @DRAM
// )
void gemm_RISCV_31x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_31x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 31] @DRAM
// )
void gemm_RISCV_31x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_31x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 31] @DRAM
// )
void gemm_RISCV_31x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_31x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 31] @DRAM
// )
void gemm_RISCV_31x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_31x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 31] @DRAM
// )
void gemm_RISCV_31x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_31x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 31] @DRAM
// )
void gemm_RISCV_31x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_31x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 31] @DRAM
// )
void gemm_RISCV_31x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_31x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 31] @DRAM
// )
void gemm_RISCV_31x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_31x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 31] @DRAM
// )
void gemm_RISCV_31x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_31x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 31] @DRAM
// )
void gemm_RISCV_31x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_31x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 31] @DRAM
// )
void gemm_RISCV_31x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_31x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 31] @DRAM
// )
void gemm_RISCV_31x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_31x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 31] @DRAM
// )
void gemm_RISCV_31x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_31x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 31] @DRAM
// )
void gemm_RISCV_31x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_31x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 31] @DRAM
// )
void gemm_RISCV_31x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_31x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 31] @DRAM
// )
void gemm_RISCV_31x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_31x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 31] @DRAM
// )
void gemm_RISCV_31x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_31x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 31] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 31] @DRAM
// )
void gemm_RISCV_31x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_32x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 32] @DRAM
// )
void gemm_RISCV_32x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_32x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 32] @DRAM
// )
void gemm_RISCV_32x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_32x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 32] @DRAM
// )
void gemm_RISCV_32x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_32x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 32] @DRAM
// )
void gemm_RISCV_32x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_32x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 32] @DRAM
// )
void gemm_RISCV_32x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_32x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 32] @DRAM
// )
void gemm_RISCV_32x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_32x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 32] @DRAM
// )
void gemm_RISCV_32x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_32x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 32] @DRAM
// )
void gemm_RISCV_32x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_32x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 32] @DRAM
// )
void gemm_RISCV_32x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_32x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 32] @DRAM
// )
void gemm_RISCV_32x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_32x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 32] @DRAM
// )
void gemm_RISCV_32x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_32x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 32] @DRAM
// )
void gemm_RISCV_32x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_32x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 32] @DRAM
// )
void gemm_RISCV_32x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_32x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 32] @DRAM
// )
void gemm_RISCV_32x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_32x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 32] @DRAM
// )
void gemm_RISCV_32x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_32x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 32] @DRAM
// )
void gemm_RISCV_32x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_32x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 32] @DRAM
// )
void gemm_RISCV_32x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_32x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 32] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 32] @DRAM
// )
void gemm_RISCV_32x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_33x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 33] @DRAM
// )
void gemm_RISCV_33x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_33x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 33] @DRAM
// )
void gemm_RISCV_33x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_33x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 33] @DRAM
// )
void gemm_RISCV_33x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_33x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 33] @DRAM
// )
void gemm_RISCV_33x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_33x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 33] @DRAM
// )
void gemm_RISCV_33x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_33x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 33] @DRAM
// )
void gemm_RISCV_33x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_33x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 33] @DRAM
// )
void gemm_RISCV_33x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_33x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 33] @DRAM
// )
void gemm_RISCV_33x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_33x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 33] @DRAM
// )
void gemm_RISCV_33x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_33x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 33] @DRAM
// )
void gemm_RISCV_33x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_33x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 33] @DRAM
// )
void gemm_RISCV_33x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_33x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 33] @DRAM
// )
void gemm_RISCV_33x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_33x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 33] @DRAM
// )
void gemm_RISCV_33x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_33x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 33] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 33] @DRAM
// )
void gemm_RISCV_33x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_34x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 34] @DRAM
// )
void gemm_RISCV_34x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_34x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 34] @DRAM
// )
void gemm_RISCV_34x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_34x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 34] @DRAM
// )
void gemm_RISCV_34x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_34x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 34] @DRAM
// )
void gemm_RISCV_34x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_34x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 34] @DRAM
// )
void gemm_RISCV_34x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_34x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 34] @DRAM
// )
void gemm_RISCV_34x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_34x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 34] @DRAM
// )
void gemm_RISCV_34x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_34x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 34] @DRAM
// )
void gemm_RISCV_34x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_34x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 34] @DRAM
// )
void gemm_RISCV_34x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_34x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 34] @DRAM
// )
void gemm_RISCV_34x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_34x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 34] @DRAM
// )
void gemm_RISCV_34x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_34x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 34] @DRAM
// )
void gemm_RISCV_34x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_34x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 34] @DRAM
// )
void gemm_RISCV_34x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_34x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 34] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 34] @DRAM
// )
void gemm_RISCV_34x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_35x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 35] @DRAM
// )
void gemm_RISCV_35x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_35x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 35] @DRAM
// )
void gemm_RISCV_35x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_35x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 35] @DRAM
// )
void gemm_RISCV_35x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_35x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 35] @DRAM
// )
void gemm_RISCV_35x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_35x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 35] @DRAM
// )
void gemm_RISCV_35x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_35x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 35] @DRAM
// )
void gemm_RISCV_35x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_35x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 35] @DRAM
// )
void gemm_RISCV_35x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_35x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 35] @DRAM
// )
void gemm_RISCV_35x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_35x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 35] @DRAM
// )
void gemm_RISCV_35x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_35x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 35] @DRAM
// )
void gemm_RISCV_35x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_35x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 35] @DRAM
// )
void gemm_RISCV_35x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_35x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 35] @DRAM
// )
void gemm_RISCV_35x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_35x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 35] @DRAM
// )
void gemm_RISCV_35x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_35x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 35] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 35] @DRAM
// )
void gemm_RISCV_35x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_36x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 36] @DRAM
// )
void gemm_RISCV_36x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_36x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 36] @DRAM
// )
void gemm_RISCV_36x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_36x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 36] @DRAM
// )
void gemm_RISCV_36x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_36x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 36] @DRAM
// )
void gemm_RISCV_36x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_36x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 36] @DRAM
// )
void gemm_RISCV_36x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_36x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 36] @DRAM
// )
void gemm_RISCV_36x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_36x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 36] @DRAM
// )
void gemm_RISCV_36x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_36x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 36] @DRAM
// )
void gemm_RISCV_36x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_36x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 36] @DRAM
// )
void gemm_RISCV_36x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_36x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 36] @DRAM
// )
void gemm_RISCV_36x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_36x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 36] @DRAM
// )
void gemm_RISCV_36x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_36x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 36] @DRAM
// )
void gemm_RISCV_36x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_36x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 36] @DRAM
// )
void gemm_RISCV_36x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_36x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 36] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 36] @DRAM
// )
void gemm_RISCV_36x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_37x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 37] @DRAM
// )
void gemm_RISCV_37x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_37x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 37] @DRAM
// )
void gemm_RISCV_37x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_37x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 37] @DRAM
// )
void gemm_RISCV_37x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_37x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 37] @DRAM
// )
void gemm_RISCV_37x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_37x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 37] @DRAM
// )
void gemm_RISCV_37x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_37x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 37] @DRAM
// )
void gemm_RISCV_37x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_37x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 37] @DRAM
// )
void gemm_RISCV_37x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_37x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 37] @DRAM
// )
void gemm_RISCV_37x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_37x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 37] @DRAM
// )
void gemm_RISCV_37x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_37x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 37] @DRAM
// )
void gemm_RISCV_37x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_37x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 37] @DRAM
// )
void gemm_RISCV_37x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_37x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 37] @DRAM
// )
void gemm_RISCV_37x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_37x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 37] @DRAM
// )
void gemm_RISCV_37x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_37x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 37] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 37] @DRAM
// )
void gemm_RISCV_37x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_38x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 38] @DRAM
// )
void gemm_RISCV_38x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_38x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 38] @DRAM
// )
void gemm_RISCV_38x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_38x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 38] @DRAM
// )
void gemm_RISCV_38x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_38x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 38] @DRAM
// )
void gemm_RISCV_38x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_38x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 38] @DRAM
// )
void gemm_RISCV_38x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_38x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 38] @DRAM
// )
void gemm_RISCV_38x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_38x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 38] @DRAM
// )
void gemm_RISCV_38x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_38x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 38] @DRAM
// )
void gemm_RISCV_38x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_38x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 38] @DRAM
// )
void gemm_RISCV_38x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_38x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 38] @DRAM
// )
void gemm_RISCV_38x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_38x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 38] @DRAM
// )
void gemm_RISCV_38x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_38x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 38] @DRAM
// )
void gemm_RISCV_38x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_38x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 38] @DRAM
// )
void gemm_RISCV_38x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_38x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 38] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 38] @DRAM
// )
void gemm_RISCV_38x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_39x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 39] @DRAM
// )
void gemm_RISCV_39x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_39x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 39] @DRAM
// )
void gemm_RISCV_39x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_39x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 39] @DRAM
// )
void gemm_RISCV_39x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_39x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 39] @DRAM
// )
void gemm_RISCV_39x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_39x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 39] @DRAM
// )
void gemm_RISCV_39x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_39x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 39] @DRAM
// )
void gemm_RISCV_39x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_39x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 39] @DRAM
// )
void gemm_RISCV_39x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_39x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 39] @DRAM
// )
void gemm_RISCV_39x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_39x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 39] @DRAM
// )
void gemm_RISCV_39x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_39x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 39] @DRAM
// )
void gemm_RISCV_39x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_39x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 39] @DRAM
// )
void gemm_RISCV_39x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_39x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 39] @DRAM
// )
void gemm_RISCV_39x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_39x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 39] @DRAM
// )
void gemm_RISCV_39x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_39x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 39] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 39] @DRAM
// )
void gemm_RISCV_39x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 3] @DRAM
// )
void gemm_RISCV_3x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 3] @DRAM
// )
void gemm_RISCV_3x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 3] @DRAM
// )
void gemm_RISCV_3x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 3] @DRAM
// )
void gemm_RISCV_3x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 3] @DRAM
// )
void gemm_RISCV_3x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 3] @DRAM
// )
void gemm_RISCV_3x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 3] @DRAM
// )
void gemm_RISCV_3x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 3] @DRAM
// )
void gemm_RISCV_3x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 3] @DRAM
// )
void gemm_RISCV_3x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 3] @DRAM
// )
void gemm_RISCV_3x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 3] @DRAM
// )
void gemm_RISCV_3x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 3] @DRAM
// )
void gemm_RISCV_3x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 3] @DRAM
// )
void gemm_RISCV_3x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 3] @DRAM
// )
void gemm_RISCV_3x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x17_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 3] @DRAM
// )
void gemm_RISCV_3x17_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x17_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 3] @DRAM
// )
void gemm_RISCV_3x17_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x18_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 3] @DRAM
// )
void gemm_RISCV_3x18_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x18_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 3] @DRAM
// )
void gemm_RISCV_3x18_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x19_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 3] @DRAM
// )
void gemm_RISCV_3x19_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x19_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 3] @DRAM
// )
void gemm_RISCV_3x19_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 3] @DRAM
// )
void gemm_RISCV_3x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 3] @DRAM
// )
void gemm_RISCV_3x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x20_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 3] @DRAM
// )
void gemm_RISCV_3x20_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x20_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 3] @DRAM
// )
void gemm_RISCV_3x20_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x21_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 3] @DRAM
// )
void gemm_RISCV_3x21_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x21_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 3] @DRAM
// )
void gemm_RISCV_3x21_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x22_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 3] @DRAM
// )
void gemm_RISCV_3x22_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x22_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 3] @DRAM
// )
void gemm_RISCV_3x22_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x23_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 3] @DRAM
// )
void gemm_RISCV_3x23_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x23_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 3] @DRAM
// )
void gemm_RISCV_3x23_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x24_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 3] @DRAM
// )
void gemm_RISCV_3x24_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x24_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 3] @DRAM
// )
void gemm_RISCV_3x24_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 3] @DRAM
// )
void gemm_RISCV_3x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 3] @DRAM
// )
void gemm_RISCV_3x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 3] @DRAM
// )
void gemm_RISCV_3x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 3] @DRAM
// )
void gemm_RISCV_3x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 3] @DRAM
// )
void gemm_RISCV_3x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 3] @DRAM
// )
void gemm_RISCV_3x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 3] @DRAM
// )
void gemm_RISCV_3x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 3] @DRAM
// )
void gemm_RISCV_3x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 3] @DRAM
// )
void gemm_RISCV_3x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 3] @DRAM
// )
void gemm_RISCV_3x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 3] @DRAM
// )
void gemm_RISCV_3x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 3] @DRAM
// )
void gemm_RISCV_3x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 3] @DRAM
// )
void gemm_RISCV_3x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 3] @DRAM
// )
void gemm_RISCV_3x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 3] @DRAM
// )
void gemm_RISCV_3x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_3x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 3] @DRAM
// )
void gemm_RISCV_3x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_40x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 40] @DRAM
// )
void gemm_RISCV_40x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_40x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 40] @DRAM
// )
void gemm_RISCV_40x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_40x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 40] @DRAM
// )
void gemm_RISCV_40x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_40x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 40] @DRAM
// )
void gemm_RISCV_40x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_40x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 40] @DRAM
// )
void gemm_RISCV_40x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_40x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 40] @DRAM
// )
void gemm_RISCV_40x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_40x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 40] @DRAM
// )
void gemm_RISCV_40x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_40x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 40] @DRAM
// )
void gemm_RISCV_40x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_40x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 40] @DRAM
// )
void gemm_RISCV_40x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_40x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 40] @DRAM
// )
void gemm_RISCV_40x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_40x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 40] @DRAM
// )
void gemm_RISCV_40x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_40x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 40] @DRAM
// )
void gemm_RISCV_40x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_40x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 40] @DRAM
// )
void gemm_RISCV_40x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_40x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 40] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 40] @DRAM
// )
void gemm_RISCV_40x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_41x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 41] @DRAM
// )
void gemm_RISCV_41x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_41x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 41] @DRAM
// )
void gemm_RISCV_41x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_41x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 41] @DRAM
// )
void gemm_RISCV_41x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_41x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 41] @DRAM
// )
void gemm_RISCV_41x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_41x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 41] @DRAM
// )
void gemm_RISCV_41x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_41x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 41] @DRAM
// )
void gemm_RISCV_41x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_41x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 41] @DRAM
// )
void gemm_RISCV_41x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_41x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 41] @DRAM
// )
void gemm_RISCV_41x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_41x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 41] @DRAM
// )
void gemm_RISCV_41x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_41x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 41] @DRAM
// )
void gemm_RISCV_41x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_41x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 41] @DRAM
// )
void gemm_RISCV_41x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_41x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 41] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 41] @DRAM
// )
void gemm_RISCV_41x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_42x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 42] @DRAM
// )
void gemm_RISCV_42x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_42x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 42] @DRAM
// )
void gemm_RISCV_42x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_42x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 42] @DRAM
// )
void gemm_RISCV_42x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_42x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 42] @DRAM
// )
void gemm_RISCV_42x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_42x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 42] @DRAM
// )
void gemm_RISCV_42x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_42x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 42] @DRAM
// )
void gemm_RISCV_42x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_42x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 42] @DRAM
// )
void gemm_RISCV_42x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_42x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 42] @DRAM
// )
void gemm_RISCV_42x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_42x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 42] @DRAM
// )
void gemm_RISCV_42x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_42x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 42] @DRAM
// )
void gemm_RISCV_42x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_42x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 42] @DRAM
// )
void gemm_RISCV_42x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_42x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 42] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 42] @DRAM
// )
void gemm_RISCV_42x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_43x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 43] @DRAM
// )
void gemm_RISCV_43x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_43x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 43] @DRAM
// )
void gemm_RISCV_43x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_43x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 43] @DRAM
// )
void gemm_RISCV_43x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_43x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 43] @DRAM
// )
void gemm_RISCV_43x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_43x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 43] @DRAM
// )
void gemm_RISCV_43x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_43x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 43] @DRAM
// )
void gemm_RISCV_43x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_43x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 43] @DRAM
// )
void gemm_RISCV_43x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_43x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 43] @DRAM
// )
void gemm_RISCV_43x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_43x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 43] @DRAM
// )
void gemm_RISCV_43x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_43x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 43] @DRAM
// )
void gemm_RISCV_43x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_43x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 43] @DRAM
// )
void gemm_RISCV_43x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_43x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 43] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 43] @DRAM
// )
void gemm_RISCV_43x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_44x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 44] @DRAM
// )
void gemm_RISCV_44x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_44x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 44] @DRAM
// )
void gemm_RISCV_44x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_44x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 44] @DRAM
// )
void gemm_RISCV_44x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_44x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 44] @DRAM
// )
void gemm_RISCV_44x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_44x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 44] @DRAM
// )
void gemm_RISCV_44x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_44x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 44] @DRAM
// )
void gemm_RISCV_44x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_44x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 44] @DRAM
// )
void gemm_RISCV_44x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_44x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 44] @DRAM
// )
void gemm_RISCV_44x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_44x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 44] @DRAM
// )
void gemm_RISCV_44x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_44x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 44] @DRAM
// )
void gemm_RISCV_44x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_44x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 44] @DRAM
// )
void gemm_RISCV_44x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_44x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 44] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 44] @DRAM
// )
void gemm_RISCV_44x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_45x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 45] @DRAM
// )
void gemm_RISCV_45x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_45x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 45] @DRAM
// )
void gemm_RISCV_45x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_45x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 45] @DRAM
// )
void gemm_RISCV_45x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_45x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 45] @DRAM
// )
void gemm_RISCV_45x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_45x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 45] @DRAM
// )
void gemm_RISCV_45x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_45x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 45] @DRAM
// )
void gemm_RISCV_45x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_45x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 45] @DRAM
// )
void gemm_RISCV_45x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_45x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 45] @DRAM
// )
void gemm_RISCV_45x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_45x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 45] @DRAM
// )
void gemm_RISCV_45x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_45x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 45] @DRAM
// )
void gemm_RISCV_45x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_45x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 45] @DRAM
// )
void gemm_RISCV_45x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_45x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 45] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 45] @DRAM
// )
void gemm_RISCV_45x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_46x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 46] @DRAM
// )
void gemm_RISCV_46x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_46x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 46] @DRAM
// )
void gemm_RISCV_46x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_46x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 46] @DRAM
// )
void gemm_RISCV_46x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_46x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 46] @DRAM
// )
void gemm_RISCV_46x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_46x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 46] @DRAM
// )
void gemm_RISCV_46x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_46x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 46] @DRAM
// )
void gemm_RISCV_46x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_46x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 46] @DRAM
// )
void gemm_RISCV_46x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_46x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 46] @DRAM
// )
void gemm_RISCV_46x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_46x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 46] @DRAM
// )
void gemm_RISCV_46x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_46x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 46] @DRAM
// )
void gemm_RISCV_46x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_46x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 46] @DRAM
// )
void gemm_RISCV_46x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_46x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 46] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 46] @DRAM
// )
void gemm_RISCV_46x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_47x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 47] @DRAM
// )
void gemm_RISCV_47x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_47x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 47] @DRAM
// )
void gemm_RISCV_47x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_47x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 47] @DRAM
// )
void gemm_RISCV_47x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_47x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 47] @DRAM
// )
void gemm_RISCV_47x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_47x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 47] @DRAM
// )
void gemm_RISCV_47x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_47x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 47] @DRAM
// )
void gemm_RISCV_47x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_47x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 47] @DRAM
// )
void gemm_RISCV_47x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_47x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 47] @DRAM
// )
void gemm_RISCV_47x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_47x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 47] @DRAM
// )
void gemm_RISCV_47x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_47x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 47] @DRAM
// )
void gemm_RISCV_47x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_47x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 47] @DRAM
// )
void gemm_RISCV_47x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_47x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 47] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 47] @DRAM
// )
void gemm_RISCV_47x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_48x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 48] @DRAM
// )
void gemm_RISCV_48x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_48x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 48] @DRAM
// )
void gemm_RISCV_48x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_48x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 48] @DRAM
// )
void gemm_RISCV_48x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_48x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 48] @DRAM
// )
void gemm_RISCV_48x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_48x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 48] @DRAM
// )
void gemm_RISCV_48x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_48x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 48] @DRAM
// )
void gemm_RISCV_48x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_48x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 48] @DRAM
// )
void gemm_RISCV_48x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_48x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 48] @DRAM
// )
void gemm_RISCV_48x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_48x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 48] @DRAM
// )
void gemm_RISCV_48x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_48x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 48] @DRAM
// )
void gemm_RISCV_48x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_48x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 48] @DRAM
// )
void gemm_RISCV_48x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_48x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 48] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 48] @DRAM
// )
void gemm_RISCV_48x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 4] @DRAM
// )
void gemm_RISCV_4x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 4] @DRAM
// )
void gemm_RISCV_4x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 4] @DRAM
// )
void gemm_RISCV_4x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 4] @DRAM
// )
void gemm_RISCV_4x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 4] @DRAM
// )
void gemm_RISCV_4x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 4] @DRAM
// )
void gemm_RISCV_4x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 4] @DRAM
// )
void gemm_RISCV_4x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 4] @DRAM
// )
void gemm_RISCV_4x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 4] @DRAM
// )
void gemm_RISCV_4x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 4] @DRAM
// )
void gemm_RISCV_4x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 4] @DRAM
// )
void gemm_RISCV_4x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 4] @DRAM
// )
void gemm_RISCV_4x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 4] @DRAM
// )
void gemm_RISCV_4x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 4] @DRAM
// )
void gemm_RISCV_4x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x17_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 4] @DRAM
// )
void gemm_RISCV_4x17_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x17_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 4] @DRAM
// )
void gemm_RISCV_4x17_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x18_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 4] @DRAM
// )
void gemm_RISCV_4x18_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x18_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 4] @DRAM
// )
void gemm_RISCV_4x18_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x19_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 4] @DRAM
// )
void gemm_RISCV_4x19_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x19_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 4] @DRAM
// )
void gemm_RISCV_4x19_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 4] @DRAM
// )
void gemm_RISCV_4x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 4] @DRAM
// )
void gemm_RISCV_4x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x20_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 4] @DRAM
// )
void gemm_RISCV_4x20_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x20_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 4] @DRAM
// )
void gemm_RISCV_4x20_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x21_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 4] @DRAM
// )
void gemm_RISCV_4x21_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x21_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 4] @DRAM
// )
void gemm_RISCV_4x21_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x22_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 4] @DRAM
// )
void gemm_RISCV_4x22_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x22_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 4] @DRAM
// )
void gemm_RISCV_4x22_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x23_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 4] @DRAM
// )
void gemm_RISCV_4x23_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x23_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 4] @DRAM
// )
void gemm_RISCV_4x23_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x24_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 4] @DRAM
// )
void gemm_RISCV_4x24_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x24_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 4] @DRAM
// )
void gemm_RISCV_4x24_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 4] @DRAM
// )
void gemm_RISCV_4x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 4] @DRAM
// )
void gemm_RISCV_4x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 4] @DRAM
// )
void gemm_RISCV_4x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 4] @DRAM
// )
void gemm_RISCV_4x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 4] @DRAM
// )
void gemm_RISCV_4x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 4] @DRAM
// )
void gemm_RISCV_4x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 4] @DRAM
// )
void gemm_RISCV_4x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 4] @DRAM
// )
void gemm_RISCV_4x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 4] @DRAM
// )
void gemm_RISCV_4x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 4] @DRAM
// )
void gemm_RISCV_4x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 4] @DRAM
// )
void gemm_RISCV_4x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 4] @DRAM
// )
void gemm_RISCV_4x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 4] @DRAM
// )
void gemm_RISCV_4x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 4] @DRAM
// )
void gemm_RISCV_4x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 4] @DRAM
// )
void gemm_RISCV_4x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_4x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 4] @DRAM
// )
void gemm_RISCV_4x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 5] @DRAM
// )
void gemm_RISCV_5x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 5] @DRAM
// )
void gemm_RISCV_5x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 5] @DRAM
// )
void gemm_RISCV_5x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 5] @DRAM
// )
void gemm_RISCV_5x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 5] @DRAM
// )
void gemm_RISCV_5x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 5] @DRAM
// )
void gemm_RISCV_5x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 5] @DRAM
// )
void gemm_RISCV_5x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 5] @DRAM
// )
void gemm_RISCV_5x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 5] @DRAM
// )
void gemm_RISCV_5x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 5] @DRAM
// )
void gemm_RISCV_5x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 5] @DRAM
// )
void gemm_RISCV_5x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 5] @DRAM
// )
void gemm_RISCV_5x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 5] @DRAM
// )
void gemm_RISCV_5x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 5] @DRAM
// )
void gemm_RISCV_5x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x17_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 5] @DRAM
// )
void gemm_RISCV_5x17_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x17_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 5] @DRAM
// )
void gemm_RISCV_5x17_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x18_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 5] @DRAM
// )
void gemm_RISCV_5x18_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x18_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 5] @DRAM
// )
void gemm_RISCV_5x18_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x19_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 5] @DRAM
// )
void gemm_RISCV_5x19_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x19_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 5] @DRAM
// )
void gemm_RISCV_5x19_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 5] @DRAM
// )
void gemm_RISCV_5x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 5] @DRAM
// )
void gemm_RISCV_5x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x20_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 5] @DRAM
// )
void gemm_RISCV_5x20_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x20_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 5] @DRAM
// )
void gemm_RISCV_5x20_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x21_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 5] @DRAM
// )
void gemm_RISCV_5x21_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x21_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 5] @DRAM
// )
void gemm_RISCV_5x21_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x22_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 5] @DRAM
// )
void gemm_RISCV_5x22_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x22_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 5] @DRAM
// )
void gemm_RISCV_5x22_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x23_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 5] @DRAM
// )
void gemm_RISCV_5x23_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x23_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 5] @DRAM
// )
void gemm_RISCV_5x23_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x24_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 5] @DRAM
// )
void gemm_RISCV_5x24_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x24_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 5] @DRAM
// )
void gemm_RISCV_5x24_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 5] @DRAM
// )
void gemm_RISCV_5x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 5] @DRAM
// )
void gemm_RISCV_5x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 5] @DRAM
// )
void gemm_RISCV_5x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 5] @DRAM
// )
void gemm_RISCV_5x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 5] @DRAM
// )
void gemm_RISCV_5x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 5] @DRAM
// )
void gemm_RISCV_5x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 5] @DRAM
// )
void gemm_RISCV_5x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 5] @DRAM
// )
void gemm_RISCV_5x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 5] @DRAM
// )
void gemm_RISCV_5x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 5] @DRAM
// )
void gemm_RISCV_5x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 5] @DRAM
// )
void gemm_RISCV_5x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 5] @DRAM
// )
void gemm_RISCV_5x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 5] @DRAM
// )
void gemm_RISCV_5x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 5] @DRAM
// )
void gemm_RISCV_5x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 5] @DRAM
// )
void gemm_RISCV_5x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_5x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 5] @DRAM
// )
void gemm_RISCV_5x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 6] @DRAM
// )
void gemm_RISCV_6x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 6] @DRAM
// )
void gemm_RISCV_6x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 6] @DRAM
// )
void gemm_RISCV_6x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 6] @DRAM
// )
void gemm_RISCV_6x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 6] @DRAM
// )
void gemm_RISCV_6x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 6] @DRAM
// )
void gemm_RISCV_6x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 6] @DRAM
// )
void gemm_RISCV_6x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 6] @DRAM
// )
void gemm_RISCV_6x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 6] @DRAM
// )
void gemm_RISCV_6x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 6] @DRAM
// )
void gemm_RISCV_6x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 6] @DRAM
// )
void gemm_RISCV_6x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 6] @DRAM
// )
void gemm_RISCV_6x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 6] @DRAM
// )
void gemm_RISCV_6x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 6] @DRAM
// )
void gemm_RISCV_6x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x17_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 6] @DRAM
// )
void gemm_RISCV_6x17_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x17_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 6] @DRAM
// )
void gemm_RISCV_6x17_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x18_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 6] @DRAM
// )
void gemm_RISCV_6x18_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x18_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 6] @DRAM
// )
void gemm_RISCV_6x18_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x19_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 6] @DRAM
// )
void gemm_RISCV_6x19_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x19_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 6] @DRAM
// )
void gemm_RISCV_6x19_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 6] @DRAM
// )
void gemm_RISCV_6x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 6] @DRAM
// )
void gemm_RISCV_6x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x20_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 6] @DRAM
// )
void gemm_RISCV_6x20_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x20_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 6] @DRAM
// )
void gemm_RISCV_6x20_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x21_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 6] @DRAM
// )
void gemm_RISCV_6x21_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x21_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 6] @DRAM
// )
void gemm_RISCV_6x21_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x22_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 6] @DRAM
// )
void gemm_RISCV_6x22_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x22_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 6] @DRAM
// )
void gemm_RISCV_6x22_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x23_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 6] @DRAM
// )
void gemm_RISCV_6x23_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x23_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 6] @DRAM
// )
void gemm_RISCV_6x23_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x24_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 6] @DRAM
// )
void gemm_RISCV_6x24_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x24_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 6] @DRAM
// )
void gemm_RISCV_6x24_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 6] @DRAM
// )
void gemm_RISCV_6x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 6] @DRAM
// )
void gemm_RISCV_6x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 6] @DRAM
// )
void gemm_RISCV_6x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 6] @DRAM
// )
void gemm_RISCV_6x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 6] @DRAM
// )
void gemm_RISCV_6x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 6] @DRAM
// )
void gemm_RISCV_6x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 6] @DRAM
// )
void gemm_RISCV_6x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 6] @DRAM
// )
void gemm_RISCV_6x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 6] @DRAM
// )
void gemm_RISCV_6x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 6] @DRAM
// )
void gemm_RISCV_6x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 6] @DRAM
// )
void gemm_RISCV_6x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 6] @DRAM
// )
void gemm_RISCV_6x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 6] @DRAM
// )
void gemm_RISCV_6x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 6] @DRAM
// )
void gemm_RISCV_6x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 6] @DRAM
// )
void gemm_RISCV_6x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_6x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 6] @DRAM
// )
void gemm_RISCV_6x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 7] @DRAM
// )
void gemm_RISCV_7x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 7] @DRAM
// )
void gemm_RISCV_7x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 7] @DRAM
// )
void gemm_RISCV_7x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 7] @DRAM
// )
void gemm_RISCV_7x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 7] @DRAM
// )
void gemm_RISCV_7x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 7] @DRAM
// )
void gemm_RISCV_7x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 7] @DRAM
// )
void gemm_RISCV_7x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 7] @DRAM
// )
void gemm_RISCV_7x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 7] @DRAM
// )
void gemm_RISCV_7x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 7] @DRAM
// )
void gemm_RISCV_7x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 7] @DRAM
// )
void gemm_RISCV_7x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 7] @DRAM
// )
void gemm_RISCV_7x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 7] @DRAM
// )
void gemm_RISCV_7x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 7] @DRAM
// )
void gemm_RISCV_7x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x17_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 7] @DRAM
// )
void gemm_RISCV_7x17_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x17_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 7] @DRAM
// )
void gemm_RISCV_7x17_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x18_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 7] @DRAM
// )
void gemm_RISCV_7x18_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x18_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 7] @DRAM
// )
void gemm_RISCV_7x18_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x19_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 7] @DRAM
// )
void gemm_RISCV_7x19_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x19_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 7] @DRAM
// )
void gemm_RISCV_7x19_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 7] @DRAM
// )
void gemm_RISCV_7x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 7] @DRAM
// )
void gemm_RISCV_7x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x20_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 7] @DRAM
// )
void gemm_RISCV_7x20_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x20_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 7] @DRAM
// )
void gemm_RISCV_7x20_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x21_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 7] @DRAM
// )
void gemm_RISCV_7x21_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x21_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 7] @DRAM
// )
void gemm_RISCV_7x21_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x22_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 7] @DRAM
// )
void gemm_RISCV_7x22_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x22_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 7] @DRAM
// )
void gemm_RISCV_7x22_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x23_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 7] @DRAM
// )
void gemm_RISCV_7x23_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x23_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 7] @DRAM
// )
void gemm_RISCV_7x23_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x24_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 7] @DRAM
// )
void gemm_RISCV_7x24_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x24_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 7] @DRAM
// )
void gemm_RISCV_7x24_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 7] @DRAM
// )
void gemm_RISCV_7x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 7] @DRAM
// )
void gemm_RISCV_7x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 7] @DRAM
// )
void gemm_RISCV_7x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 7] @DRAM
// )
void gemm_RISCV_7x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 7] @DRAM
// )
void gemm_RISCV_7x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 7] @DRAM
// )
void gemm_RISCV_7x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 7] @DRAM
// )
void gemm_RISCV_7x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 7] @DRAM
// )
void gemm_RISCV_7x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 7] @DRAM
// )
void gemm_RISCV_7x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 7] @DRAM
// )
void gemm_RISCV_7x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 7] @DRAM
// )
void gemm_RISCV_7x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 7] @DRAM
// )
void gemm_RISCV_7x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 7] @DRAM
// )
void gemm_RISCV_7x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 7] @DRAM
// )
void gemm_RISCV_7x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 7] @DRAM
// )
void gemm_RISCV_7x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_7x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 7] @DRAM
// )
void gemm_RISCV_7x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 8] @DRAM
// )
void gemm_RISCV_8x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 8] @DRAM
// )
void gemm_RISCV_8x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 8] @DRAM
// )
void gemm_RISCV_8x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 8] @DRAM
// )
void gemm_RISCV_8x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 8] @DRAM
// )
void gemm_RISCV_8x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 8] @DRAM
// )
void gemm_RISCV_8x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 8] @DRAM
// )
void gemm_RISCV_8x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 8] @DRAM
// )
void gemm_RISCV_8x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 8] @DRAM
// )
void gemm_RISCV_8x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 8] @DRAM
// )
void gemm_RISCV_8x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 8] @DRAM
// )
void gemm_RISCV_8x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 8] @DRAM
// )
void gemm_RISCV_8x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 8] @DRAM
// )
void gemm_RISCV_8x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 8] @DRAM
// )
void gemm_RISCV_8x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x17_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 8] @DRAM
// )
void gemm_RISCV_8x17_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x17_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][17, 8] @DRAM
// )
void gemm_RISCV_8x17_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x18_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 8] @DRAM
// )
void gemm_RISCV_8x18_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x18_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][18, 8] @DRAM
// )
void gemm_RISCV_8x18_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x19_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 8] @DRAM
// )
void gemm_RISCV_8x19_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x19_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][19, 8] @DRAM
// )
void gemm_RISCV_8x19_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 8] @DRAM
// )
void gemm_RISCV_8x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 8] @DRAM
// )
void gemm_RISCV_8x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x20_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 8] @DRAM
// )
void gemm_RISCV_8x20_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x20_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][20, 8] @DRAM
// )
void gemm_RISCV_8x20_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x21_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 8] @DRAM
// )
void gemm_RISCV_8x21_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x21_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][21, 8] @DRAM
// )
void gemm_RISCV_8x21_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x22_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 8] @DRAM
// )
void gemm_RISCV_8x22_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x22_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][22, 8] @DRAM
// )
void gemm_RISCV_8x22_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x23_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 8] @DRAM
// )
void gemm_RISCV_8x23_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x23_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][23, 8] @DRAM
// )
void gemm_RISCV_8x23_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x24_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 8] @DRAM
// )
void gemm_RISCV_8x24_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x24_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][24, 8] @DRAM
// )
void gemm_RISCV_8x24_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 8] @DRAM
// )
void gemm_RISCV_8x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 8] @DRAM
// )
void gemm_RISCV_8x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 8] @DRAM
// )
void gemm_RISCV_8x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 8] @DRAM
// )
void gemm_RISCV_8x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 8] @DRAM
// )
void gemm_RISCV_8x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 8] @DRAM
// )
void gemm_RISCV_8x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 8] @DRAM
// )
void gemm_RISCV_8x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 8] @DRAM
// )
void gemm_RISCV_8x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 8] @DRAM
// )
void gemm_RISCV_8x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 8] @DRAM
// )
void gemm_RISCV_8x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 8] @DRAM
// )
void gemm_RISCV_8x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 8] @DRAM
// )
void gemm_RISCV_8x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 8] @DRAM
// )
void gemm_RISCV_8x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 8] @DRAM
// )
void gemm_RISCV_8x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 8] @DRAM
// )
void gemm_RISCV_8x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_8x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 8] @DRAM
// )
void gemm_RISCV_8x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 9] @DRAM
// )
void gemm_RISCV_9x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 9] @DRAM
// )
void gemm_RISCV_9x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 9] @DRAM
// )
void gemm_RISCV_9x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 9] @DRAM
// )
void gemm_RISCV_9x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 9] @DRAM
// )
void gemm_RISCV_9x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 9] @DRAM
// )
void gemm_RISCV_9x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 9] @DRAM
// )
void gemm_RISCV_9x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 9] @DRAM
// )
void gemm_RISCV_9x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 9] @DRAM
// )
void gemm_RISCV_9x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 9] @DRAM
// )
void gemm_RISCV_9x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 9] @DRAM
// )
void gemm_RISCV_9x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 9] @DRAM
// )
void gemm_RISCV_9x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 9] @DRAM
// )
void gemm_RISCV_9x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 9] @DRAM
// )
void gemm_RISCV_9x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 9] @DRAM
// )
void gemm_RISCV_9x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 9] @DRAM
// )
void gemm_RISCV_9x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 9] @DRAM
// )
void gemm_RISCV_9x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 9] @DRAM
// )
void gemm_RISCV_9x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 9] @DRAM
// )
void gemm_RISCV_9x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 9] @DRAM
// )
void gemm_RISCV_9x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 9] @DRAM
// )
void gemm_RISCV_9x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 9] @DRAM
// )
void gemm_RISCV_9x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 9] @DRAM
// )
void gemm_RISCV_9x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 9] @DRAM
// )
void gemm_RISCV_9x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 9] @DRAM
// )
void gemm_RISCV_9x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 9] @DRAM
// )
void gemm_RISCV_9x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 9] @DRAM
// )
void gemm_RISCV_9x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 9] @DRAM
// )
void gemm_RISCV_9x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 9] @DRAM
// )
void gemm_RISCV_9x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 9] @DRAM
// )
void gemm_RISCV_9x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 9] @DRAM
// )
void gemm_RISCV_9x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );

// gemm_RISCV_9x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 9] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 9] @DRAM
// )
void gemm_RISCV_9x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, struct exo_win_2f16c A, struct exo_win_2f16c B, const float* beta, struct exo_win_2f16 C );



#ifdef __cplusplus
}
#endif
#endif  // AAA_H
