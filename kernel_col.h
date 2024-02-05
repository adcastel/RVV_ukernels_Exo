
#pragma once
#ifndef KERNEL_COL_H
#define KERNEL_COL_H

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


struct exo_win_1f32{
    float * const data;
    const int_fast32_t strides[1];
};
struct exo_win_1f32c{
    const float * const data;
    const int_fast32_t strides[1];
};
struct exo_win_2f32{
    float * const data;
    const int_fast32_t strides[2];
};

/*
typedef void (*ukrFunction)(void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C);


ukrFunction*** allocateMatrix();
void fillMatrix(ukrFunction*** matrix);
void freeMatrix(ukrFunction*** matrix);
*/



// gemm_RISCV_10x10_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 10] @DRAM,
//     B : f32[KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 10] @DRAM
// )
void gemm_RISCV_10x10_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_10x11_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 10] @DRAM,
//     B : f32[KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 10] @DRAM
// )
void gemm_RISCV_10x11_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_10x12_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 10] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 10] @DRAM
// )
void gemm_RISCV_10x12_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_10x13_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 10] @DRAM,
//     B : f32[KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 10] @DRAM
// )
void gemm_RISCV_10x13_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_10x14_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 10] @DRAM,
//     B : f32[KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 10] @DRAM
// )
void gemm_RISCV_10x14_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_10x15_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 10] @DRAM,
//     B : f32[KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 10] @DRAM
// )
void gemm_RISCV_10x15_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_10x16_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 10] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 10] @DRAM
// )
void gemm_RISCV_10x16_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_10x17_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 10] @DRAM,
//     B : f32[KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 10] @DRAM
// )
void gemm_RISCV_10x17_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_10x18_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 10] @DRAM,
//     B : f32[KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 10] @DRAM
// )
void gemm_RISCV_10x18_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_10x19_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 10] @DRAM,
//     B : f32[KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 10] @DRAM
// )
void gemm_RISCV_10x19_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_10x1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 10] @DRAM,
//     B : f32[KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 10] @DRAM
// )
void gemm_RISCV_10x1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_10x20_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 10] @DRAM,
//     B : f32[KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 10] @DRAM
// )
void gemm_RISCV_10x20_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_10x21_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 10] @DRAM,
//     B : f32[KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][21, 10] @DRAM
// )
void gemm_RISCV_10x21_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_10x22_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 10] @DRAM,
//     B : f32[KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][22, 10] @DRAM
// )
void gemm_RISCV_10x22_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_10x23_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 10] @DRAM,
//     B : f32[KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][23, 10] @DRAM
// )
void gemm_RISCV_10x23_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_10x24_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 10] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 10] @DRAM
// )
void gemm_RISCV_10x24_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_10x2_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 10] @DRAM,
//     B : f32[KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 10] @DRAM
// )
void gemm_RISCV_10x2_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_10x3_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 10] @DRAM,
//     B : f32[KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 10] @DRAM
// )
void gemm_RISCV_10x3_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_10x4_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 10] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 10] @DRAM
// )
void gemm_RISCV_10x4_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_10x5_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 10] @DRAM,
//     B : f32[KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 10] @DRAM
// )
void gemm_RISCV_10x5_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_10x6_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 10] @DRAM,
//     B : f32[KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 10] @DRAM
// )
void gemm_RISCV_10x6_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_10x7_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 10] @DRAM,
//     B : f32[KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 10] @DRAM
// )
void gemm_RISCV_10x7_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_10x8_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 10] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 10] @DRAM
// )
void gemm_RISCV_10x8_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_10x9_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 10] @DRAM,
//     B : f32[KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 10] @DRAM
// )
void gemm_RISCV_10x9_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_11x10_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 11] @DRAM,
//     B : f32[KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 11] @DRAM
// )
void gemm_RISCV_11x10_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_11x11_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 11] @DRAM,
//     B : f32[KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 11] @DRAM
// )
void gemm_RISCV_11x11_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_11x12_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 11] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 11] @DRAM
// )
void gemm_RISCV_11x12_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_11x13_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 11] @DRAM,
//     B : f32[KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 11] @DRAM
// )
void gemm_RISCV_11x13_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_11x14_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 11] @DRAM,
//     B : f32[KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 11] @DRAM
// )
void gemm_RISCV_11x14_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_11x15_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 11] @DRAM,
//     B : f32[KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 11] @DRAM
// )
void gemm_RISCV_11x15_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_11x16_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 11] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 11] @DRAM
// )
void gemm_RISCV_11x16_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_11x17_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 11] @DRAM,
//     B : f32[KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 11] @DRAM
// )
void gemm_RISCV_11x17_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_11x18_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 11] @DRAM,
//     B : f32[KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 11] @DRAM
// )
void gemm_RISCV_11x18_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_11x19_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 11] @DRAM,
//     B : f32[KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 11] @DRAM
// )
void gemm_RISCV_11x19_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_11x1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 11] @DRAM,
//     B : f32[KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 11] @DRAM
// )
void gemm_RISCV_11x1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_11x20_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 11] @DRAM,
//     B : f32[KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 11] @DRAM
// )
void gemm_RISCV_11x20_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_11x21_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 11] @DRAM,
//     B : f32[KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][21, 11] @DRAM
// )
void gemm_RISCV_11x21_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_11x22_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 11] @DRAM,
//     B : f32[KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][22, 11] @DRAM
// )
void gemm_RISCV_11x22_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_11x23_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 11] @DRAM,
//     B : f32[KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][23, 11] @DRAM
// )
void gemm_RISCV_11x23_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_11x24_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 11] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 11] @DRAM
// )
void gemm_RISCV_11x24_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_11x2_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 11] @DRAM,
//     B : f32[KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 11] @DRAM
// )
void gemm_RISCV_11x2_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_11x3_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 11] @DRAM,
//     B : f32[KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 11] @DRAM
// )
void gemm_RISCV_11x3_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_11x4_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 11] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 11] @DRAM
// )
void gemm_RISCV_11x4_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_11x5_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 11] @DRAM,
//     B : f32[KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 11] @DRAM
// )
void gemm_RISCV_11x5_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_11x6_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 11] @DRAM,
//     B : f32[KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 11] @DRAM
// )
void gemm_RISCV_11x6_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_11x7_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 11] @DRAM,
//     B : f32[KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 11] @DRAM
// )
void gemm_RISCV_11x7_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_11x8_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 11] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 11] @DRAM
// )
void gemm_RISCV_11x8_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_11x9_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 11] @DRAM,
//     B : f32[KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 11] @DRAM
// )
void gemm_RISCV_11x9_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_12x10_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 12] @DRAM
// )
void gemm_RISCV_12x10_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_12x11_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 12] @DRAM
// )
void gemm_RISCV_12x11_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_12x12_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 12] @DRAM
// )
void gemm_RISCV_12x12_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_12x13_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 12] @DRAM
// )
void gemm_RISCV_12x13_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_12x14_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 12] @DRAM
// )
void gemm_RISCV_12x14_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_12x15_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 12] @DRAM
// )
void gemm_RISCV_12x15_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_12x16_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 12] @DRAM
// )
void gemm_RISCV_12x16_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_12x17_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 12] @DRAM
// )
void gemm_RISCV_12x17_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_12x18_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 12] @DRAM
// )
void gemm_RISCV_12x18_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_12x19_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 12] @DRAM
// )
void gemm_RISCV_12x19_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_12x1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 12] @DRAM
// )
void gemm_RISCV_12x1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_12x20_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 12] @DRAM
// )
void gemm_RISCV_12x20_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_12x21_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][21, 12] @DRAM
// )
void gemm_RISCV_12x21_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_12x22_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][22, 12] @DRAM
// )
void gemm_RISCV_12x22_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_12x23_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][23, 12] @DRAM
// )
void gemm_RISCV_12x23_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_12x24_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 12] @DRAM
// )
void gemm_RISCV_12x24_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_12x2_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 12] @DRAM
// )
void gemm_RISCV_12x2_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_12x3_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 12] @DRAM
// )
void gemm_RISCV_12x3_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_12x4_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 12] @DRAM
// )
void gemm_RISCV_12x4_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_12x5_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 12] @DRAM
// )
void gemm_RISCV_12x5_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_12x6_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 12] @DRAM
// )
void gemm_RISCV_12x6_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_12x7_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 12] @DRAM
// )
void gemm_RISCV_12x7_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_12x8_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 12] @DRAM
// )
void gemm_RISCV_12x8_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_12x9_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 12] @DRAM
// )
void gemm_RISCV_12x9_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_13x10_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 13] @DRAM,
//     B : f32[KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 13] @DRAM
// )
void gemm_RISCV_13x10_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_13x11_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 13] @DRAM,
//     B : f32[KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 13] @DRAM
// )
void gemm_RISCV_13x11_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_13x12_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 13] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 13] @DRAM
// )
void gemm_RISCV_13x12_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_13x13_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 13] @DRAM,
//     B : f32[KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 13] @DRAM
// )
void gemm_RISCV_13x13_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_13x14_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 13] @DRAM,
//     B : f32[KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 13] @DRAM
// )
void gemm_RISCV_13x14_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_13x15_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 13] @DRAM,
//     B : f32[KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 13] @DRAM
// )
void gemm_RISCV_13x15_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_13x16_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 13] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 13] @DRAM
// )
void gemm_RISCV_13x16_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_13x17_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 13] @DRAM,
//     B : f32[KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 13] @DRAM
// )
void gemm_RISCV_13x17_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_13x18_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 13] @DRAM,
//     B : f32[KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 13] @DRAM
// )
void gemm_RISCV_13x18_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_13x19_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 13] @DRAM,
//     B : f32[KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 13] @DRAM
// )
void gemm_RISCV_13x19_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_13x1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 13] @DRAM,
//     B : f32[KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 13] @DRAM
// )
void gemm_RISCV_13x1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_13x20_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 13] @DRAM,
//     B : f32[KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 13] @DRAM
// )
void gemm_RISCV_13x20_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_13x21_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 13] @DRAM,
//     B : f32[KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][21, 13] @DRAM
// )
void gemm_RISCV_13x21_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_13x22_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 13] @DRAM,
//     B : f32[KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][22, 13] @DRAM
// )
void gemm_RISCV_13x22_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_13x23_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 13] @DRAM,
//     B : f32[KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][23, 13] @DRAM
// )
void gemm_RISCV_13x23_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_13x24_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 13] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 13] @DRAM
// )
void gemm_RISCV_13x24_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_13x2_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 13] @DRAM,
//     B : f32[KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 13] @DRAM
// )
void gemm_RISCV_13x2_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_13x3_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 13] @DRAM,
//     B : f32[KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 13] @DRAM
// )
void gemm_RISCV_13x3_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_13x4_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 13] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 13] @DRAM
// )
void gemm_RISCV_13x4_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_13x5_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 13] @DRAM,
//     B : f32[KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 13] @DRAM
// )
void gemm_RISCV_13x5_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_13x6_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 13] @DRAM,
//     B : f32[KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 13] @DRAM
// )
void gemm_RISCV_13x6_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_13x7_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 13] @DRAM,
//     B : f32[KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 13] @DRAM
// )
void gemm_RISCV_13x7_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_13x8_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 13] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 13] @DRAM
// )
void gemm_RISCV_13x8_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_13x9_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 13] @DRAM,
//     B : f32[KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 13] @DRAM
// )
void gemm_RISCV_13x9_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_14x10_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 14] @DRAM,
//     B : f32[KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 14] @DRAM
// )
void gemm_RISCV_14x10_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_14x11_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 14] @DRAM,
//     B : f32[KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 14] @DRAM
// )
void gemm_RISCV_14x11_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_14x12_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 14] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 14] @DRAM
// )
void gemm_RISCV_14x12_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_14x13_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 14] @DRAM,
//     B : f32[KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 14] @DRAM
// )
void gemm_RISCV_14x13_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_14x14_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 14] @DRAM,
//     B : f32[KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 14] @DRAM
// )
void gemm_RISCV_14x14_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_14x15_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 14] @DRAM,
//     B : f32[KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 14] @DRAM
// )
void gemm_RISCV_14x15_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_14x16_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 14] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 14] @DRAM
// )
void gemm_RISCV_14x16_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_14x17_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 14] @DRAM,
//     B : f32[KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 14] @DRAM
// )
void gemm_RISCV_14x17_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_14x18_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 14] @DRAM,
//     B : f32[KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 14] @DRAM
// )
void gemm_RISCV_14x18_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_14x19_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 14] @DRAM,
//     B : f32[KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 14] @DRAM
// )
void gemm_RISCV_14x19_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_14x1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 14] @DRAM,
//     B : f32[KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 14] @DRAM
// )
void gemm_RISCV_14x1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_14x20_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 14] @DRAM,
//     B : f32[KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 14] @DRAM
// )
void gemm_RISCV_14x20_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_14x21_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 14] @DRAM,
//     B : f32[KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][21, 14] @DRAM
// )
void gemm_RISCV_14x21_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_14x22_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 14] @DRAM,
//     B : f32[KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][22, 14] @DRAM
// )
void gemm_RISCV_14x22_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_14x23_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 14] @DRAM,
//     B : f32[KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][23, 14] @DRAM
// )
void gemm_RISCV_14x23_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_14x24_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 14] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 14] @DRAM
// )
void gemm_RISCV_14x24_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_14x2_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 14] @DRAM,
//     B : f32[KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 14] @DRAM
// )
void gemm_RISCV_14x2_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_14x3_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 14] @DRAM,
//     B : f32[KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 14] @DRAM
// )
void gemm_RISCV_14x3_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_14x4_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 14] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 14] @DRAM
// )
void gemm_RISCV_14x4_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_14x5_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 14] @DRAM,
//     B : f32[KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 14] @DRAM
// )
void gemm_RISCV_14x5_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_14x6_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 14] @DRAM,
//     B : f32[KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 14] @DRAM
// )
void gemm_RISCV_14x6_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_14x7_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 14] @DRAM,
//     B : f32[KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 14] @DRAM
// )
void gemm_RISCV_14x7_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_14x8_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 14] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 14] @DRAM
// )
void gemm_RISCV_14x8_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_14x9_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 14] @DRAM,
//     B : f32[KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 14] @DRAM
// )
void gemm_RISCV_14x9_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_15x10_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 15] @DRAM,
//     B : f32[KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 15] @DRAM
// )
void gemm_RISCV_15x10_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_15x11_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 15] @DRAM,
//     B : f32[KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 15] @DRAM
// )
void gemm_RISCV_15x11_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_15x12_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 15] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 15] @DRAM
// )
void gemm_RISCV_15x12_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_15x13_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 15] @DRAM,
//     B : f32[KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 15] @DRAM
// )
void gemm_RISCV_15x13_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_15x14_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 15] @DRAM,
//     B : f32[KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 15] @DRAM
// )
void gemm_RISCV_15x14_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_15x15_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 15] @DRAM,
//     B : f32[KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 15] @DRAM
// )
void gemm_RISCV_15x15_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_15x16_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 15] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 15] @DRAM
// )
void gemm_RISCV_15x16_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_15x17_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 15] @DRAM,
//     B : f32[KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 15] @DRAM
// )
void gemm_RISCV_15x17_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_15x18_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 15] @DRAM,
//     B : f32[KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 15] @DRAM
// )
void gemm_RISCV_15x18_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_15x19_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 15] @DRAM,
//     B : f32[KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 15] @DRAM
// )
void gemm_RISCV_15x19_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_15x1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 15] @DRAM,
//     B : f32[KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 15] @DRAM
// )
void gemm_RISCV_15x1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_15x20_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 15] @DRAM,
//     B : f32[KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 15] @DRAM
// )
void gemm_RISCV_15x20_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_15x21_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 15] @DRAM,
//     B : f32[KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][21, 15] @DRAM
// )
void gemm_RISCV_15x21_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_15x22_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 15] @DRAM,
//     B : f32[KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][22, 15] @DRAM
// )
void gemm_RISCV_15x22_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_15x23_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 15] @DRAM,
//     B : f32[KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][23, 15] @DRAM
// )
void gemm_RISCV_15x23_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_15x24_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 15] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 15] @DRAM
// )
void gemm_RISCV_15x24_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_15x2_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 15] @DRAM,
//     B : f32[KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 15] @DRAM
// )
void gemm_RISCV_15x2_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_15x3_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 15] @DRAM,
//     B : f32[KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 15] @DRAM
// )
void gemm_RISCV_15x3_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_15x4_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 15] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 15] @DRAM
// )
void gemm_RISCV_15x4_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_15x5_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 15] @DRAM,
//     B : f32[KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 15] @DRAM
// )
void gemm_RISCV_15x5_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_15x6_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 15] @DRAM,
//     B : f32[KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 15] @DRAM
// )
void gemm_RISCV_15x6_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_15x7_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 15] @DRAM,
//     B : f32[KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 15] @DRAM
// )
void gemm_RISCV_15x7_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_15x8_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 15] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 15] @DRAM
// )
void gemm_RISCV_15x8_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_15x9_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 15] @DRAM,
//     B : f32[KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 15] @DRAM
// )
void gemm_RISCV_15x9_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_16x10_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 16] @DRAM
// )
void gemm_RISCV_16x10_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_16x11_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 16] @DRAM
// )
void gemm_RISCV_16x11_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_16x12_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 16] @DRAM
// )
void gemm_RISCV_16x12_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_16x13_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 16] @DRAM
// )
void gemm_RISCV_16x13_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_16x14_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 16] @DRAM
// )
void gemm_RISCV_16x14_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_16x15_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 16] @DRAM
// )
void gemm_RISCV_16x15_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_16x16_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 16] @DRAM
// )
void gemm_RISCV_16x16_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_16x17_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 16] @DRAM
// )
void gemm_RISCV_16x17_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_16x18_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 16] @DRAM
// )
void gemm_RISCV_16x18_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_16x19_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 16] @DRAM
// )
void gemm_RISCV_16x19_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_16x1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 16] @DRAM
// )
void gemm_RISCV_16x1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_16x20_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 16] @DRAM
// )
void gemm_RISCV_16x20_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_16x21_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][21, 16] @DRAM
// )
void gemm_RISCV_16x21_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_16x22_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][22, 16] @DRAM
// )
void gemm_RISCV_16x22_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_16x23_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][23, 16] @DRAM
// )
void gemm_RISCV_16x23_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_16x24_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 16] @DRAM
// )
void gemm_RISCV_16x24_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_16x2_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 16] @DRAM
// )
void gemm_RISCV_16x2_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_16x3_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 16] @DRAM
// )
void gemm_RISCV_16x3_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_16x4_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 16] @DRAM
// )
void gemm_RISCV_16x4_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_16x5_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 16] @DRAM
// )
void gemm_RISCV_16x5_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_16x6_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 16] @DRAM
// )
void gemm_RISCV_16x6_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_16x7_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 16] @DRAM
// )
void gemm_RISCV_16x7_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_16x8_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 16] @DRAM
// )
void gemm_RISCV_16x8_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_16x9_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 16] @DRAM
// )
void gemm_RISCV_16x9_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_17x10_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 17] @DRAM,
//     B : f32[KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 17] @DRAM
// )
void gemm_RISCV_17x10_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_17x11_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 17] @DRAM,
//     B : f32[KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 17] @DRAM
// )
void gemm_RISCV_17x11_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_17x12_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 17] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 17] @DRAM
// )
void gemm_RISCV_17x12_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_17x13_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 17] @DRAM,
//     B : f32[KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 17] @DRAM
// )
void gemm_RISCV_17x13_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_17x14_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 17] @DRAM,
//     B : f32[KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 17] @DRAM
// )
void gemm_RISCV_17x14_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_17x15_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 17] @DRAM,
//     B : f32[KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 17] @DRAM
// )
void gemm_RISCV_17x15_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_17x16_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 17] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 17] @DRAM
// )
void gemm_RISCV_17x16_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_17x17_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 17] @DRAM,
//     B : f32[KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 17] @DRAM
// )
void gemm_RISCV_17x17_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_17x18_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 17] @DRAM,
//     B : f32[KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 17] @DRAM
// )
void gemm_RISCV_17x18_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_17x19_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 17] @DRAM,
//     B : f32[KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 17] @DRAM
// )
void gemm_RISCV_17x19_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_17x1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 17] @DRAM,
//     B : f32[KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 17] @DRAM
// )
void gemm_RISCV_17x1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_17x20_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 17] @DRAM,
//     B : f32[KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 17] @DRAM
// )
void gemm_RISCV_17x20_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_17x21_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 17] @DRAM,
//     B : f32[KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][21, 17] @DRAM
// )
void gemm_RISCV_17x21_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_17x22_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 17] @DRAM,
//     B : f32[KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][22, 17] @DRAM
// )
void gemm_RISCV_17x22_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_17x23_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 17] @DRAM,
//     B : f32[KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][23, 17] @DRAM
// )
void gemm_RISCV_17x23_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_17x24_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 17] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 17] @DRAM
// )
void gemm_RISCV_17x24_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_17x2_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 17] @DRAM,
//     B : f32[KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 17] @DRAM
// )
void gemm_RISCV_17x2_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_17x3_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 17] @DRAM,
//     B : f32[KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 17] @DRAM
// )
void gemm_RISCV_17x3_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_17x4_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 17] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 17] @DRAM
// )
void gemm_RISCV_17x4_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_17x5_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 17] @DRAM,
//     B : f32[KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 17] @DRAM
// )
void gemm_RISCV_17x5_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_17x6_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 17] @DRAM,
//     B : f32[KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 17] @DRAM
// )
void gemm_RISCV_17x6_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_17x7_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 17] @DRAM,
//     B : f32[KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 17] @DRAM
// )
void gemm_RISCV_17x7_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_17x8_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 17] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 17] @DRAM
// )
void gemm_RISCV_17x8_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_17x9_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 17] @DRAM,
//     B : f32[KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 17] @DRAM
// )
void gemm_RISCV_17x9_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_18x10_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 18] @DRAM,
//     B : f32[KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 18] @DRAM
// )
void gemm_RISCV_18x10_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_18x11_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 18] @DRAM,
//     B : f32[KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 18] @DRAM
// )
void gemm_RISCV_18x11_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_18x12_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 18] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 18] @DRAM
// )
void gemm_RISCV_18x12_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_18x13_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 18] @DRAM,
//     B : f32[KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 18] @DRAM
// )
void gemm_RISCV_18x13_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_18x14_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 18] @DRAM,
//     B : f32[KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 18] @DRAM
// )
void gemm_RISCV_18x14_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_18x15_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 18] @DRAM,
//     B : f32[KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 18] @DRAM
// )
void gemm_RISCV_18x15_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_18x16_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 18] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 18] @DRAM
// )
void gemm_RISCV_18x16_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_18x17_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 18] @DRAM,
//     B : f32[KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 18] @DRAM
// )
void gemm_RISCV_18x17_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_18x18_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 18] @DRAM,
//     B : f32[KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 18] @DRAM
// )
void gemm_RISCV_18x18_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_18x19_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 18] @DRAM,
//     B : f32[KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 18] @DRAM
// )
void gemm_RISCV_18x19_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_18x1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 18] @DRAM,
//     B : f32[KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 18] @DRAM
// )
void gemm_RISCV_18x1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_18x20_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 18] @DRAM,
//     B : f32[KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 18] @DRAM
// )
void gemm_RISCV_18x20_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_18x21_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 18] @DRAM,
//     B : f32[KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][21, 18] @DRAM
// )
void gemm_RISCV_18x21_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_18x22_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 18] @DRAM,
//     B : f32[KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][22, 18] @DRAM
// )
void gemm_RISCV_18x22_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_18x23_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 18] @DRAM,
//     B : f32[KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][23, 18] @DRAM
// )
void gemm_RISCV_18x23_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_18x24_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 18] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 18] @DRAM
// )
void gemm_RISCV_18x24_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_18x2_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 18] @DRAM,
//     B : f32[KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 18] @DRAM
// )
void gemm_RISCV_18x2_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_18x3_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 18] @DRAM,
//     B : f32[KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 18] @DRAM
// )
void gemm_RISCV_18x3_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_18x4_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 18] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 18] @DRAM
// )
void gemm_RISCV_18x4_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_18x5_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 18] @DRAM,
//     B : f32[KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 18] @DRAM
// )
void gemm_RISCV_18x5_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_18x6_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 18] @DRAM,
//     B : f32[KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 18] @DRAM
// )
void gemm_RISCV_18x6_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_18x7_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 18] @DRAM,
//     B : f32[KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 18] @DRAM
// )
void gemm_RISCV_18x7_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_18x8_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 18] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 18] @DRAM
// )
void gemm_RISCV_18x8_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_18x9_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 18] @DRAM,
//     B : f32[KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 18] @DRAM
// )
void gemm_RISCV_18x9_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_19x10_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 19] @DRAM,
//     B : f32[KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 19] @DRAM
// )
void gemm_RISCV_19x10_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_19x11_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 19] @DRAM,
//     B : f32[KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 19] @DRAM
// )
void gemm_RISCV_19x11_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_19x12_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 19] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 19] @DRAM
// )
void gemm_RISCV_19x12_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_19x13_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 19] @DRAM,
//     B : f32[KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 19] @DRAM
// )
void gemm_RISCV_19x13_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_19x14_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 19] @DRAM,
//     B : f32[KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 19] @DRAM
// )
void gemm_RISCV_19x14_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_19x15_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 19] @DRAM,
//     B : f32[KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 19] @DRAM
// )
void gemm_RISCV_19x15_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_19x16_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 19] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 19] @DRAM
// )
void gemm_RISCV_19x16_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_19x17_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 19] @DRAM,
//     B : f32[KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 19] @DRAM
// )
void gemm_RISCV_19x17_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_19x18_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 19] @DRAM,
//     B : f32[KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 19] @DRAM
// )
void gemm_RISCV_19x18_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_19x19_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 19] @DRAM,
//     B : f32[KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 19] @DRAM
// )
void gemm_RISCV_19x19_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_19x1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 19] @DRAM,
//     B : f32[KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 19] @DRAM
// )
void gemm_RISCV_19x1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_19x20_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 19] @DRAM,
//     B : f32[KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 19] @DRAM
// )
void gemm_RISCV_19x20_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_19x21_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 19] @DRAM,
//     B : f32[KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][21, 19] @DRAM
// )
void gemm_RISCV_19x21_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_19x22_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 19] @DRAM,
//     B : f32[KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][22, 19] @DRAM
// )
void gemm_RISCV_19x22_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_19x23_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 19] @DRAM,
//     B : f32[KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][23, 19] @DRAM
// )
void gemm_RISCV_19x23_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_19x24_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 19] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 19] @DRAM
// )
void gemm_RISCV_19x24_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_19x2_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 19] @DRAM,
//     B : f32[KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 19] @DRAM
// )
void gemm_RISCV_19x2_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_19x3_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 19] @DRAM,
//     B : f32[KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 19] @DRAM
// )
void gemm_RISCV_19x3_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_19x4_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 19] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 19] @DRAM
// )
void gemm_RISCV_19x4_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_19x5_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 19] @DRAM,
//     B : f32[KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 19] @DRAM
// )
void gemm_RISCV_19x5_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_19x6_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 19] @DRAM,
//     B : f32[KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 19] @DRAM
// )
void gemm_RISCV_19x6_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_19x7_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 19] @DRAM,
//     B : f32[KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 19] @DRAM
// )
void gemm_RISCV_19x7_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_19x8_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 19] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 19] @DRAM
// )
void gemm_RISCV_19x8_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_19x9_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 19] @DRAM,
//     B : f32[KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 19] @DRAM
// )
void gemm_RISCV_19x9_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_1x10_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 1] @DRAM,
//     B : f32[KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 1] @DRAM
// )
void gemm_RISCV_1x10_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_1x11_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 1] @DRAM,
//     B : f32[KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 1] @DRAM
// )
void gemm_RISCV_1x11_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_1x12_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 1] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 1] @DRAM
// )
void gemm_RISCV_1x12_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_1x13_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 1] @DRAM,
//     B : f32[KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 1] @DRAM
// )
void gemm_RISCV_1x13_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_1x14_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 1] @DRAM,
//     B : f32[KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 1] @DRAM
// )
void gemm_RISCV_1x14_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_1x15_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 1] @DRAM,
//     B : f32[KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 1] @DRAM
// )
void gemm_RISCV_1x15_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_1x16_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 1] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 1] @DRAM
// )
void gemm_RISCV_1x16_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_1x17_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 1] @DRAM,
//     B : f32[KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 1] @DRAM
// )
void gemm_RISCV_1x17_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_1x18_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 1] @DRAM,
//     B : f32[KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 1] @DRAM
// )
void gemm_RISCV_1x18_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_1x19_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 1] @DRAM,
//     B : f32[KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 1] @DRAM
// )
void gemm_RISCV_1x19_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_1x1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 1] @DRAM,
//     B : f32[KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 1] @DRAM
// )
void gemm_RISCV_1x1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_1x20_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 1] @DRAM,
//     B : f32[KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 1] @DRAM
// )
void gemm_RISCV_1x20_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_1x21_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 1] @DRAM,
//     B : f32[KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][21, 1] @DRAM
// )
void gemm_RISCV_1x21_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_1x22_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 1] @DRAM,
//     B : f32[KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][22, 1] @DRAM
// )
void gemm_RISCV_1x22_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_1x23_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 1] @DRAM,
//     B : f32[KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][23, 1] @DRAM
// )
void gemm_RISCV_1x23_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_1x24_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 1] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 1] @DRAM
// )
void gemm_RISCV_1x24_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_1x2_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 1] @DRAM,
//     B : f32[KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 1] @DRAM
// )
void gemm_RISCV_1x2_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_1x3_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 1] @DRAM,
//     B : f32[KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 1] @DRAM
// )
void gemm_RISCV_1x3_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_1x4_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 1] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 1] @DRAM
// )
void gemm_RISCV_1x4_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_1x5_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 1] @DRAM,
//     B : f32[KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 1] @DRAM
// )
void gemm_RISCV_1x5_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_1x6_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 1] @DRAM,
//     B : f32[KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 1] @DRAM
// )
void gemm_RISCV_1x6_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_1x7_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 1] @DRAM,
//     B : f32[KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 1] @DRAM
// )
void gemm_RISCV_1x7_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_1x8_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 1] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 1] @DRAM
// )
void gemm_RISCV_1x8_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_1x9_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 1] @DRAM,
//     B : f32[KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 1] @DRAM
// )
void gemm_RISCV_1x9_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_20x10_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 20] @DRAM
// )
void gemm_RISCV_20x10_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_20x11_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 20] @DRAM
// )
void gemm_RISCV_20x11_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_20x12_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 20] @DRAM
// )
void gemm_RISCV_20x12_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_20x13_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 20] @DRAM
// )
void gemm_RISCV_20x13_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_20x14_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 20] @DRAM
// )
void gemm_RISCV_20x14_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_20x15_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 20] @DRAM
// )
void gemm_RISCV_20x15_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_20x16_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 20] @DRAM
// )
void gemm_RISCV_20x16_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_20x17_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 20] @DRAM
// )
void gemm_RISCV_20x17_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_20x18_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 20] @DRAM
// )
void gemm_RISCV_20x18_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_20x19_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 20] @DRAM
// )
void gemm_RISCV_20x19_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_20x1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 20] @DRAM
// )
void gemm_RISCV_20x1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_20x20_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 20] @DRAM
// )
void gemm_RISCV_20x20_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_20x21_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][21, 20] @DRAM
// )
void gemm_RISCV_20x21_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_20x22_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][22, 20] @DRAM
// )
void gemm_RISCV_20x22_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_20x23_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][23, 20] @DRAM
// )
void gemm_RISCV_20x23_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_20x24_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 20] @DRAM
// )
void gemm_RISCV_20x24_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_20x2_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 20] @DRAM
// )
void gemm_RISCV_20x2_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_20x3_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 20] @DRAM
// )
void gemm_RISCV_20x3_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_20x4_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 20] @DRAM
// )
void gemm_RISCV_20x4_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_20x5_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 20] @DRAM
// )
void gemm_RISCV_20x5_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_20x6_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 20] @DRAM
// )
void gemm_RISCV_20x6_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_20x7_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 20] @DRAM
// )
void gemm_RISCV_20x7_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_20x8_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 20] @DRAM
// )
void gemm_RISCV_20x8_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_20x9_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 20] @DRAM
// )
void gemm_RISCV_20x9_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_21x10_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 21] @DRAM,
//     B : f32[KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 21] @DRAM
// )
void gemm_RISCV_21x10_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_21x11_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 21] @DRAM,
//     B : f32[KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 21] @DRAM
// )
void gemm_RISCV_21x11_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_21x12_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 21] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 21] @DRAM
// )
void gemm_RISCV_21x12_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_21x13_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 21] @DRAM,
//     B : f32[KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 21] @DRAM
// )
void gemm_RISCV_21x13_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_21x14_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 21] @DRAM,
//     B : f32[KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 21] @DRAM
// )
void gemm_RISCV_21x14_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_21x15_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 21] @DRAM,
//     B : f32[KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 21] @DRAM
// )
void gemm_RISCV_21x15_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_21x16_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 21] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 21] @DRAM
// )
void gemm_RISCV_21x16_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_21x17_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 21] @DRAM,
//     B : f32[KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 21] @DRAM
// )
void gemm_RISCV_21x17_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_21x18_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 21] @DRAM,
//     B : f32[KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 21] @DRAM
// )
void gemm_RISCV_21x18_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_21x19_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 21] @DRAM,
//     B : f32[KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 21] @DRAM
// )
void gemm_RISCV_21x19_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_21x1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 21] @DRAM,
//     B : f32[KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 21] @DRAM
// )
void gemm_RISCV_21x1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_21x20_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 21] @DRAM,
//     B : f32[KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 21] @DRAM
// )
void gemm_RISCV_21x20_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_21x21_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 21] @DRAM,
//     B : f32[KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][21, 21] @DRAM
// )
void gemm_RISCV_21x21_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_21x22_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 21] @DRAM,
//     B : f32[KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][22, 21] @DRAM
// )
void gemm_RISCV_21x22_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_21x23_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 21] @DRAM,
//     B : f32[KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][23, 21] @DRAM
// )
void gemm_RISCV_21x23_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_21x24_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 21] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 21] @DRAM
// )
void gemm_RISCV_21x24_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_21x2_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 21] @DRAM,
//     B : f32[KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 21] @DRAM
// )
void gemm_RISCV_21x2_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_21x3_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 21] @DRAM,
//     B : f32[KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 21] @DRAM
// )
void gemm_RISCV_21x3_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_21x4_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 21] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 21] @DRAM
// )
void gemm_RISCV_21x4_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_21x5_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 21] @DRAM,
//     B : f32[KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 21] @DRAM
// )
void gemm_RISCV_21x5_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_21x6_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 21] @DRAM,
//     B : f32[KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 21] @DRAM
// )
void gemm_RISCV_21x6_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_21x7_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 21] @DRAM,
//     B : f32[KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 21] @DRAM
// )
void gemm_RISCV_21x7_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_21x8_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 21] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 21] @DRAM
// )
void gemm_RISCV_21x8_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_21x9_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 21] @DRAM,
//     B : f32[KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 21] @DRAM
// )
void gemm_RISCV_21x9_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_22x10_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 22] @DRAM,
//     B : f32[KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 22] @DRAM
// )
void gemm_RISCV_22x10_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_22x11_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 22] @DRAM,
//     B : f32[KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 22] @DRAM
// )
void gemm_RISCV_22x11_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_22x12_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 22] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 22] @DRAM
// )
void gemm_RISCV_22x12_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_22x13_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 22] @DRAM,
//     B : f32[KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 22] @DRAM
// )
void gemm_RISCV_22x13_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_22x14_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 22] @DRAM,
//     B : f32[KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 22] @DRAM
// )
void gemm_RISCV_22x14_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_22x15_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 22] @DRAM,
//     B : f32[KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 22] @DRAM
// )
void gemm_RISCV_22x15_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_22x16_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 22] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 22] @DRAM
// )
void gemm_RISCV_22x16_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_22x17_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 22] @DRAM,
//     B : f32[KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 22] @DRAM
// )
void gemm_RISCV_22x17_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_22x18_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 22] @DRAM,
//     B : f32[KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 22] @DRAM
// )
void gemm_RISCV_22x18_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_22x19_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 22] @DRAM,
//     B : f32[KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 22] @DRAM
// )
void gemm_RISCV_22x19_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_22x1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 22] @DRAM,
//     B : f32[KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 22] @DRAM
// )
void gemm_RISCV_22x1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_22x20_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 22] @DRAM,
//     B : f32[KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 22] @DRAM
// )
void gemm_RISCV_22x20_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_22x21_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 22] @DRAM,
//     B : f32[KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][21, 22] @DRAM
// )
void gemm_RISCV_22x21_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_22x22_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 22] @DRAM,
//     B : f32[KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][22, 22] @DRAM
// )
void gemm_RISCV_22x22_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_22x23_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 22] @DRAM,
//     B : f32[KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][23, 22] @DRAM
// )
void gemm_RISCV_22x23_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_22x24_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 22] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 22] @DRAM
// )
void gemm_RISCV_22x24_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_22x2_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 22] @DRAM,
//     B : f32[KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 22] @DRAM
// )
void gemm_RISCV_22x2_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_22x3_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 22] @DRAM,
//     B : f32[KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 22] @DRAM
// )
void gemm_RISCV_22x3_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_22x4_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 22] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 22] @DRAM
// )
void gemm_RISCV_22x4_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_22x5_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 22] @DRAM,
//     B : f32[KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 22] @DRAM
// )
void gemm_RISCV_22x5_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_22x6_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 22] @DRAM,
//     B : f32[KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 22] @DRAM
// )
void gemm_RISCV_22x6_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_22x7_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 22] @DRAM,
//     B : f32[KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 22] @DRAM
// )
void gemm_RISCV_22x7_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_22x8_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 22] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 22] @DRAM
// )
void gemm_RISCV_22x8_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_22x9_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 22] @DRAM,
//     B : f32[KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 22] @DRAM
// )
void gemm_RISCV_22x9_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_23x10_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 23] @DRAM,
//     B : f32[KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 23] @DRAM
// )
void gemm_RISCV_23x10_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_23x11_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 23] @DRAM,
//     B : f32[KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 23] @DRAM
// )
void gemm_RISCV_23x11_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_23x12_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 23] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 23] @DRAM
// )
void gemm_RISCV_23x12_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_23x13_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 23] @DRAM,
//     B : f32[KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 23] @DRAM
// )
void gemm_RISCV_23x13_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_23x14_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 23] @DRAM,
//     B : f32[KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 23] @DRAM
// )
void gemm_RISCV_23x14_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_23x15_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 23] @DRAM,
//     B : f32[KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 23] @DRAM
// )
void gemm_RISCV_23x15_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_23x16_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 23] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 23] @DRAM
// )
void gemm_RISCV_23x16_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_23x17_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 23] @DRAM,
//     B : f32[KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 23] @DRAM
// )
void gemm_RISCV_23x17_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_23x18_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 23] @DRAM,
//     B : f32[KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 23] @DRAM
// )
void gemm_RISCV_23x18_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_23x19_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 23] @DRAM,
//     B : f32[KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 23] @DRAM
// )
void gemm_RISCV_23x19_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_23x1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 23] @DRAM,
//     B : f32[KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 23] @DRAM
// )
void gemm_RISCV_23x1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_23x20_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 23] @DRAM,
//     B : f32[KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 23] @DRAM
// )
void gemm_RISCV_23x20_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_23x21_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 23] @DRAM,
//     B : f32[KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][21, 23] @DRAM
// )
void gemm_RISCV_23x21_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_23x22_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 23] @DRAM,
//     B : f32[KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][22, 23] @DRAM
// )
void gemm_RISCV_23x22_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_23x23_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 23] @DRAM,
//     B : f32[KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][23, 23] @DRAM
// )
void gemm_RISCV_23x23_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_23x24_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 23] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 23] @DRAM
// )
void gemm_RISCV_23x24_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_23x2_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 23] @DRAM,
//     B : f32[KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 23] @DRAM
// )
void gemm_RISCV_23x2_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_23x3_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 23] @DRAM,
//     B : f32[KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 23] @DRAM
// )
void gemm_RISCV_23x3_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_23x4_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 23] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 23] @DRAM
// )
void gemm_RISCV_23x4_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_23x5_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 23] @DRAM,
//     B : f32[KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 23] @DRAM
// )
void gemm_RISCV_23x5_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_23x6_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 23] @DRAM,
//     B : f32[KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 23] @DRAM
// )
void gemm_RISCV_23x6_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_23x7_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 23] @DRAM,
//     B : f32[KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 23] @DRAM
// )
void gemm_RISCV_23x7_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_23x8_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 23] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 23] @DRAM
// )
void gemm_RISCV_23x8_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_23x9_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 23] @DRAM,
//     B : f32[KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 23] @DRAM
// )
void gemm_RISCV_23x9_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_24x10_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 24] @DRAM
// )
void gemm_RISCV_24x10_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_24x11_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 24] @DRAM
// )
void gemm_RISCV_24x11_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_24x12_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 24] @DRAM
// )
void gemm_RISCV_24x12_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_24x13_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 24] @DRAM
// )
void gemm_RISCV_24x13_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_24x14_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 24] @DRAM
// )
void gemm_RISCV_24x14_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_24x15_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 24] @DRAM
// )
void gemm_RISCV_24x15_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_24x16_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 24] @DRAM
// )
void gemm_RISCV_24x16_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_24x17_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 24] @DRAM
// )
void gemm_RISCV_24x17_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_24x18_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 24] @DRAM
// )
void gemm_RISCV_24x18_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_24x19_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 24] @DRAM
// )
void gemm_RISCV_24x19_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_24x1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 24] @DRAM
// )
void gemm_RISCV_24x1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_24x20_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 24] @DRAM
// )
void gemm_RISCV_24x20_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_24x21_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][21, 24] @DRAM
// )
void gemm_RISCV_24x21_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_24x22_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][22, 24] @DRAM
// )
void gemm_RISCV_24x22_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_24x23_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][23, 24] @DRAM
// )
void gemm_RISCV_24x23_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_24x24_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 24] @DRAM
// )
void gemm_RISCV_24x24_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_24x2_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 24] @DRAM
// )
void gemm_RISCV_24x2_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_24x3_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 24] @DRAM
// )
void gemm_RISCV_24x3_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_24x4_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 24] @DRAM
// )
void gemm_RISCV_24x4_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_24x5_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 24] @DRAM
// )
void gemm_RISCV_24x5_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_24x6_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 24] @DRAM
// )
void gemm_RISCV_24x6_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_24x7_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 24] @DRAM
// )
void gemm_RISCV_24x7_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_24x8_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 24] @DRAM
// )
void gemm_RISCV_24x8_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_24x9_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 24] @DRAM
// )
void gemm_RISCV_24x9_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_2x10_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 2] @DRAM,
//     B : f32[KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 2] @DRAM
// )
void gemm_RISCV_2x10_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_2x11_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 2] @DRAM,
//     B : f32[KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 2] @DRAM
// )
void gemm_RISCV_2x11_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_2x12_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 2] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 2] @DRAM
// )
void gemm_RISCV_2x12_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_2x13_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 2] @DRAM,
//     B : f32[KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 2] @DRAM
// )
void gemm_RISCV_2x13_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_2x14_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 2] @DRAM,
//     B : f32[KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 2] @DRAM
// )
void gemm_RISCV_2x14_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_2x15_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 2] @DRAM,
//     B : f32[KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 2] @DRAM
// )
void gemm_RISCV_2x15_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_2x16_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 2] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 2] @DRAM
// )
void gemm_RISCV_2x16_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_2x17_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 2] @DRAM,
//     B : f32[KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 2] @DRAM
// )
void gemm_RISCV_2x17_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_2x18_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 2] @DRAM,
//     B : f32[KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 2] @DRAM
// )
void gemm_RISCV_2x18_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_2x19_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 2] @DRAM,
//     B : f32[KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 2] @DRAM
// )
void gemm_RISCV_2x19_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_2x1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 2] @DRAM,
//     B : f32[KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 2] @DRAM
// )
void gemm_RISCV_2x1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_2x20_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 2] @DRAM,
//     B : f32[KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 2] @DRAM
// )
void gemm_RISCV_2x20_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_2x21_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 2] @DRAM,
//     B : f32[KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][21, 2] @DRAM
// )
void gemm_RISCV_2x21_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_2x22_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 2] @DRAM,
//     B : f32[KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][22, 2] @DRAM
// )
void gemm_RISCV_2x22_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_2x23_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 2] @DRAM,
//     B : f32[KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][23, 2] @DRAM
// )
void gemm_RISCV_2x23_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_2x24_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 2] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 2] @DRAM
// )
void gemm_RISCV_2x24_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_2x2_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 2] @DRAM,
//     B : f32[KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 2] @DRAM
// )
void gemm_RISCV_2x2_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_2x3_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 2] @DRAM,
//     B : f32[KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 2] @DRAM
// )
void gemm_RISCV_2x3_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_2x4_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 2] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 2] @DRAM
// )
void gemm_RISCV_2x4_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_2x5_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 2] @DRAM,
//     B : f32[KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 2] @DRAM
// )
void gemm_RISCV_2x5_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_2x6_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 2] @DRAM,
//     B : f32[KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 2] @DRAM
// )
void gemm_RISCV_2x6_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_2x7_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 2] @DRAM,
//     B : f32[KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 2] @DRAM
// )
void gemm_RISCV_2x7_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_2x8_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 2] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 2] @DRAM
// )
void gemm_RISCV_2x8_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_2x9_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 2] @DRAM,
//     B : f32[KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 2] @DRAM
// )
void gemm_RISCV_2x9_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_3x10_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 3] @DRAM,
//     B : f32[KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 3] @DRAM
// )
void gemm_RISCV_3x10_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_3x11_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 3] @DRAM,
//     B : f32[KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 3] @DRAM
// )
void gemm_RISCV_3x11_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_3x12_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 3] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 3] @DRAM
// )
void gemm_RISCV_3x12_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_3x13_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 3] @DRAM,
//     B : f32[KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 3] @DRAM
// )
void gemm_RISCV_3x13_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_3x14_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 3] @DRAM,
//     B : f32[KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 3] @DRAM
// )
void gemm_RISCV_3x14_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_3x15_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 3] @DRAM,
//     B : f32[KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 3] @DRAM
// )
void gemm_RISCV_3x15_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_3x16_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 3] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 3] @DRAM
// )
void gemm_RISCV_3x16_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_3x17_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 3] @DRAM,
//     B : f32[KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 3] @DRAM
// )
void gemm_RISCV_3x17_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_3x18_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 3] @DRAM,
//     B : f32[KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 3] @DRAM
// )
void gemm_RISCV_3x18_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_3x19_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 3] @DRAM,
//     B : f32[KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 3] @DRAM
// )
void gemm_RISCV_3x19_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_3x1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 3] @DRAM,
//     B : f32[KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 3] @DRAM
// )
void gemm_RISCV_3x1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_3x20_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 3] @DRAM,
//     B : f32[KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 3] @DRAM
// )
void gemm_RISCV_3x20_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_3x21_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 3] @DRAM,
//     B : f32[KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][21, 3] @DRAM
// )
void gemm_RISCV_3x21_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_3x22_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 3] @DRAM,
//     B : f32[KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][22, 3] @DRAM
// )
void gemm_RISCV_3x22_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_3x23_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 3] @DRAM,
//     B : f32[KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][23, 3] @DRAM
// )
void gemm_RISCV_3x23_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_3x24_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 3] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 3] @DRAM
// )
void gemm_RISCV_3x24_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_3x2_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 3] @DRAM,
//     B : f32[KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 3] @DRAM
// )
void gemm_RISCV_3x2_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_3x3_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 3] @DRAM,
//     B : f32[KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 3] @DRAM
// )
void gemm_RISCV_3x3_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_3x4_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 3] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 3] @DRAM
// )
void gemm_RISCV_3x4_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_3x5_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 3] @DRAM,
//     B : f32[KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 3] @DRAM
// )
void gemm_RISCV_3x5_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_3x6_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 3] @DRAM,
//     B : f32[KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 3] @DRAM
// )
void gemm_RISCV_3x6_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_3x7_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 3] @DRAM,
//     B : f32[KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 3] @DRAM
// )
void gemm_RISCV_3x7_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_3x8_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 3] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 3] @DRAM
// )
void gemm_RISCV_3x8_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_3x9_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 3] @DRAM,
//     B : f32[KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 3] @DRAM
// )
void gemm_RISCV_3x9_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_4x10_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 4] @DRAM
// )
void gemm_RISCV_4x10_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_4x11_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 4] @DRAM
// )
void gemm_RISCV_4x11_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_4x12_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 4] @DRAM
// )
void gemm_RISCV_4x12_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_4x13_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 4] @DRAM
// )
void gemm_RISCV_4x13_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_4x14_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 4] @DRAM
// )
void gemm_RISCV_4x14_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_4x15_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 4] @DRAM
// )
void gemm_RISCV_4x15_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_4x16_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 4] @DRAM
// )
void gemm_RISCV_4x16_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_4x17_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 4] @DRAM
// )
void gemm_RISCV_4x17_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_4x18_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 4] @DRAM
// )
void gemm_RISCV_4x18_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_4x19_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 4] @DRAM
// )
void gemm_RISCV_4x19_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_4x1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 4] @DRAM
// )
void gemm_RISCV_4x1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_4x20_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 4] @DRAM
// )
void gemm_RISCV_4x20_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_4x21_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][21, 4] @DRAM
// )
void gemm_RISCV_4x21_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_4x22_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][22, 4] @DRAM
// )
void gemm_RISCV_4x22_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_4x23_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][23, 4] @DRAM
// )
void gemm_RISCV_4x23_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_4x24_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 4] @DRAM
// )
void gemm_RISCV_4x24_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_4x2_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 4] @DRAM
// )
void gemm_RISCV_4x2_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_4x3_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 4] @DRAM
// )
void gemm_RISCV_4x3_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_4x4_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 4] @DRAM
// )
void gemm_RISCV_4x4_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_4x5_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 4] @DRAM
// )
void gemm_RISCV_4x5_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_4x6_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 4] @DRAM
// )
void gemm_RISCV_4x6_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_4x7_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 4] @DRAM
// )
void gemm_RISCV_4x7_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_4x8_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 4] @DRAM
// )
void gemm_RISCV_4x8_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_4x9_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 4] @DRAM
// )
void gemm_RISCV_4x9_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_5x10_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 5] @DRAM,
//     B : f32[KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 5] @DRAM
// )
void gemm_RISCV_5x10_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_5x11_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 5] @DRAM,
//     B : f32[KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 5] @DRAM
// )
void gemm_RISCV_5x11_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_5x12_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 5] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 5] @DRAM
// )
void gemm_RISCV_5x12_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_5x13_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 5] @DRAM,
//     B : f32[KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 5] @DRAM
// )
void gemm_RISCV_5x13_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_5x14_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 5] @DRAM,
//     B : f32[KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 5] @DRAM
// )
void gemm_RISCV_5x14_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_5x15_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 5] @DRAM,
//     B : f32[KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 5] @DRAM
// )
void gemm_RISCV_5x15_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_5x16_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 5] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 5] @DRAM
// )
void gemm_RISCV_5x16_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_5x17_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 5] @DRAM,
//     B : f32[KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 5] @DRAM
// )
void gemm_RISCV_5x17_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_5x18_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 5] @DRAM,
//     B : f32[KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 5] @DRAM
// )
void gemm_RISCV_5x18_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_5x19_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 5] @DRAM,
//     B : f32[KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 5] @DRAM
// )
void gemm_RISCV_5x19_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_5x1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 5] @DRAM,
//     B : f32[KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 5] @DRAM
// )
void gemm_RISCV_5x1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_5x20_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 5] @DRAM,
//     B : f32[KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 5] @DRAM
// )
void gemm_RISCV_5x20_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_5x21_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 5] @DRAM,
//     B : f32[KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][21, 5] @DRAM
// )
void gemm_RISCV_5x21_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_5x22_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 5] @DRAM,
//     B : f32[KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][22, 5] @DRAM
// )
void gemm_RISCV_5x22_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_5x23_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 5] @DRAM,
//     B : f32[KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][23, 5] @DRAM
// )
void gemm_RISCV_5x23_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_5x24_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 5] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 5] @DRAM
// )
void gemm_RISCV_5x24_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_5x2_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 5] @DRAM,
//     B : f32[KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 5] @DRAM
// )
void gemm_RISCV_5x2_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_5x3_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 5] @DRAM,
//     B : f32[KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 5] @DRAM
// )
void gemm_RISCV_5x3_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_5x4_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 5] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 5] @DRAM
// )
void gemm_RISCV_5x4_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_5x5_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 5] @DRAM,
//     B : f32[KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 5] @DRAM
// )
void gemm_RISCV_5x5_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_5x6_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 5] @DRAM,
//     B : f32[KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 5] @DRAM
// )
void gemm_RISCV_5x6_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_5x7_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 5] @DRAM,
//     B : f32[KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 5] @DRAM
// )
void gemm_RISCV_5x7_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_5x8_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 5] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 5] @DRAM
// )
void gemm_RISCV_5x8_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_5x9_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 5] @DRAM,
//     B : f32[KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 5] @DRAM
// )
void gemm_RISCV_5x9_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_6x10_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 6] @DRAM,
//     B : f32[KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 6] @DRAM
// )
void gemm_RISCV_6x10_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_6x11_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 6] @DRAM,
//     B : f32[KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 6] @DRAM
// )
void gemm_RISCV_6x11_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_6x12_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 6] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 6] @DRAM
// )
void gemm_RISCV_6x12_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_6x13_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 6] @DRAM,
//     B : f32[KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 6] @DRAM
// )
void gemm_RISCV_6x13_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_6x14_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 6] @DRAM,
//     B : f32[KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 6] @DRAM
// )
void gemm_RISCV_6x14_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_6x15_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 6] @DRAM,
//     B : f32[KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 6] @DRAM
// )
void gemm_RISCV_6x15_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_6x16_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 6] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 6] @DRAM
// )
void gemm_RISCV_6x16_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_6x17_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 6] @DRAM,
//     B : f32[KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 6] @DRAM
// )
void gemm_RISCV_6x17_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_6x18_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 6] @DRAM,
//     B : f32[KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 6] @DRAM
// )
void gemm_RISCV_6x18_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_6x19_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 6] @DRAM,
//     B : f32[KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 6] @DRAM
// )
void gemm_RISCV_6x19_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_6x1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 6] @DRAM,
//     B : f32[KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 6] @DRAM
// )
void gemm_RISCV_6x1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_6x20_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 6] @DRAM,
//     B : f32[KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 6] @DRAM
// )
void gemm_RISCV_6x20_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_6x21_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 6] @DRAM,
//     B : f32[KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][21, 6] @DRAM
// )
void gemm_RISCV_6x21_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_6x22_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 6] @DRAM,
//     B : f32[KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][22, 6] @DRAM
// )
void gemm_RISCV_6x22_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_6x23_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 6] @DRAM,
//     B : f32[KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][23, 6] @DRAM
// )
void gemm_RISCV_6x23_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_6x24_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 6] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 6] @DRAM
// )
void gemm_RISCV_6x24_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_6x2_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 6] @DRAM,
//     B : f32[KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 6] @DRAM
// )
void gemm_RISCV_6x2_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_6x3_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 6] @DRAM,
//     B : f32[KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 6] @DRAM
// )
void gemm_RISCV_6x3_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_6x4_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 6] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 6] @DRAM
// )
void gemm_RISCV_6x4_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_6x5_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 6] @DRAM,
//     B : f32[KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 6] @DRAM
// )
void gemm_RISCV_6x5_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_6x6_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 6] @DRAM,
//     B : f32[KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 6] @DRAM
// )
void gemm_RISCV_6x6_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_6x7_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 6] @DRAM,
//     B : f32[KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 6] @DRAM
// )
void gemm_RISCV_6x7_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_6x8_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 6] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 6] @DRAM
// )
void gemm_RISCV_6x8_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_6x9_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 6] @DRAM,
//     B : f32[KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 6] @DRAM
// )
void gemm_RISCV_6x9_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_7x10_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 7] @DRAM,
//     B : f32[KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 7] @DRAM
// )
void gemm_RISCV_7x10_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_7x11_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 7] @DRAM,
//     B : f32[KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 7] @DRAM
// )
void gemm_RISCV_7x11_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_7x12_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 7] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 7] @DRAM
// )
void gemm_RISCV_7x12_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_7x13_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 7] @DRAM,
//     B : f32[KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 7] @DRAM
// )
void gemm_RISCV_7x13_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_7x14_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 7] @DRAM,
//     B : f32[KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 7] @DRAM
// )
void gemm_RISCV_7x14_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_7x15_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 7] @DRAM,
//     B : f32[KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 7] @DRAM
// )
void gemm_RISCV_7x15_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_7x16_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 7] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 7] @DRAM
// )
void gemm_RISCV_7x16_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_7x17_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 7] @DRAM,
//     B : f32[KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 7] @DRAM
// )
void gemm_RISCV_7x17_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_7x18_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 7] @DRAM,
//     B : f32[KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 7] @DRAM
// )
void gemm_RISCV_7x18_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_7x19_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 7] @DRAM,
//     B : f32[KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 7] @DRAM
// )
void gemm_RISCV_7x19_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_7x1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 7] @DRAM,
//     B : f32[KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 7] @DRAM
// )
void gemm_RISCV_7x1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_7x20_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 7] @DRAM,
//     B : f32[KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 7] @DRAM
// )
void gemm_RISCV_7x20_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_7x21_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 7] @DRAM,
//     B : f32[KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][21, 7] @DRAM
// )
void gemm_RISCV_7x21_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_7x22_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 7] @DRAM,
//     B : f32[KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][22, 7] @DRAM
// )
void gemm_RISCV_7x22_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_7x23_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 7] @DRAM,
//     B : f32[KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][23, 7] @DRAM
// )
void gemm_RISCV_7x23_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_7x24_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 7] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 7] @DRAM
// )
void gemm_RISCV_7x24_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_7x2_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 7] @DRAM,
//     B : f32[KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 7] @DRAM
// )
void gemm_RISCV_7x2_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_7x3_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 7] @DRAM,
//     B : f32[KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 7] @DRAM
// )
void gemm_RISCV_7x3_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_7x4_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 7] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 7] @DRAM
// )
void gemm_RISCV_7x4_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_7x5_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 7] @DRAM,
//     B : f32[KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 7] @DRAM
// )
void gemm_RISCV_7x5_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_7x6_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 7] @DRAM,
//     B : f32[KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 7] @DRAM
// )
void gemm_RISCV_7x6_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_7x7_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 7] @DRAM,
//     B : f32[KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 7] @DRAM
// )
void gemm_RISCV_7x7_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_7x8_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 7] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 7] @DRAM
// )
void gemm_RISCV_7x8_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_7x9_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 7] @DRAM,
//     B : f32[KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 7] @DRAM
// )
void gemm_RISCV_7x9_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_8x10_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 8] @DRAM
// )
void gemm_RISCV_8x10_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_8x11_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 8] @DRAM
// )
void gemm_RISCV_8x11_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_8x12_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 8] @DRAM
// )
void gemm_RISCV_8x12_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_8x13_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 8] @DRAM
// )
void gemm_RISCV_8x13_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_8x14_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 8] @DRAM
// )
void gemm_RISCV_8x14_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_8x15_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 8] @DRAM
// )
void gemm_RISCV_8x15_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_8x16_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 8] @DRAM
// )
void gemm_RISCV_8x16_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_8x17_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 8] @DRAM
// )
void gemm_RISCV_8x17_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_8x18_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 8] @DRAM
// )
void gemm_RISCV_8x18_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_8x19_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 8] @DRAM
// )
void gemm_RISCV_8x19_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_8x1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 8] @DRAM
// )
void gemm_RISCV_8x1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_8x20_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 8] @DRAM
// )
void gemm_RISCV_8x20_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_8x21_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][21, 8] @DRAM
// )
void gemm_RISCV_8x21_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_8x22_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][22, 8] @DRAM
// )
void gemm_RISCV_8x22_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_8x23_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][23, 8] @DRAM
// )
void gemm_RISCV_8x23_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_8x24_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 8] @DRAM
// )
void gemm_RISCV_8x24_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_8x2_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 8] @DRAM
// )
void gemm_RISCV_8x2_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_8x3_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 8] @DRAM
// )
void gemm_RISCV_8x3_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_8x4_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 8] @DRAM
// )
void gemm_RISCV_8x4_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_8x5_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 8] @DRAM
// )
void gemm_RISCV_8x5_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_8x6_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 8] @DRAM
// )
void gemm_RISCV_8x6_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_8x7_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 8] @DRAM
// )
void gemm_RISCV_8x7_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_8x8_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 8] @DRAM
// )
void gemm_RISCV_8x8_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_8x9_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 8] @DRAM
// )
void gemm_RISCV_8x9_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_9x10_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 9] @DRAM,
//     B : f32[KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 9] @DRAM
// )
void gemm_RISCV_9x10_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_9x11_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 9] @DRAM,
//     B : f32[KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 9] @DRAM
// )
void gemm_RISCV_9x11_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_9x12_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 9] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 9] @DRAM
// )
void gemm_RISCV_9x12_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_9x13_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 9] @DRAM,
//     B : f32[KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 9] @DRAM
// )
void gemm_RISCV_9x13_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_9x14_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 9] @DRAM,
//     B : f32[KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 9] @DRAM
// )
void gemm_RISCV_9x14_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_9x15_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 9] @DRAM,
//     B : f32[KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 9] @DRAM
// )
void gemm_RISCV_9x15_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_9x16_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 9] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 9] @DRAM
// )
void gemm_RISCV_9x16_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_9x17_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 9] @DRAM,
//     B : f32[KC, 17] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][17, 9] @DRAM
// )
void gemm_RISCV_9x17_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_9x18_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 9] @DRAM,
//     B : f32[KC, 18] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][18, 9] @DRAM
// )
void gemm_RISCV_9x18_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_9x19_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 9] @DRAM,
//     B : f32[KC, 19] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][19, 9] @DRAM
// )
void gemm_RISCV_9x19_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_9x1_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 9] @DRAM,
//     B : f32[KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 9] @DRAM
// )
void gemm_RISCV_9x1_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_9x20_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 9] @DRAM,
//     B : f32[KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 9] @DRAM
// )
void gemm_RISCV_9x20_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_9x21_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 9] @DRAM,
//     B : f32[KC, 21] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][21, 9] @DRAM
// )
void gemm_RISCV_9x21_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_9x22_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 9] @DRAM,
//     B : f32[KC, 22] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][22, 9] @DRAM
// )
void gemm_RISCV_9x22_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_9x23_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 9] @DRAM,
//     B : f32[KC, 23] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][23, 9] @DRAM
// )
void gemm_RISCV_9x23_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_9x24_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 9] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 9] @DRAM
// )
void gemm_RISCV_9x24_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_9x2_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 9] @DRAM,
//     B : f32[KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 9] @DRAM
// )
void gemm_RISCV_9x2_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_9x3_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 9] @DRAM,
//     B : f32[KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 9] @DRAM
// )
void gemm_RISCV_9x3_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_9x4_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 9] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 9] @DRAM
// )
void gemm_RISCV_9x4_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_9x5_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 9] @DRAM,
//     B : f32[KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 9] @DRAM
// )
void gemm_RISCV_9x5_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_9x6_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 9] @DRAM,
//     B : f32[KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 9] @DRAM
// )
void gemm_RISCV_9x6_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_9x7_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 9] @DRAM,
//     B : f32[KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 9] @DRAM
// )
void gemm_RISCV_9x7_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_9x8_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 9] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 9] @DRAM
// )
void gemm_RISCV_9x8_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_RISCV_9x9_col_f32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 9] @DRAM,
//     B : f32[KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 9] @DRAM
// )
void gemm_RISCV_9x9_col_f32( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );



#ifdef __cplusplus
}
#endif
#endif  // KERNEL_COL_H
