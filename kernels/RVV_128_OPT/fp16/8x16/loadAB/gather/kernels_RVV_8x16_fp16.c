#include "kernels_RVV_8x16_fp16.h"



#include <stdio.h>
#include <stdlib.h>

#include <riscv_vector.h>


// gemm_RVV_1x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 1] @DRAM
// )
void gemm_RVV_1x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(1));
}

// gemm_RVV_1x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 1] @DRAM
// )
void gemm_RVV_1x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(1));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(1));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(1));
}

// gemm_RVV_1x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 1] @DRAM
// )
void gemm_RVV_1x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(1));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(1));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(1));
}

// gemm_RVV_1x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 1] @DRAM
// )
void gemm_RVV_1x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(1));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(1));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(1));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(1));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(1));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(1));
}

// gemm_RVV_1x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 1] @DRAM
// )
void gemm_RVV_1x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(1));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(1));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(1));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(1));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(1));
}

// gemm_RVV_1x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 1] @DRAM
// )
void gemm_RVV_1x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(1));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(1));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(1));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(1));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(1));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(1));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(1));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(1));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(1));
}

// gemm_RVV_1x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 1] @DRAM
// )
void gemm_RVV_1x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(1));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(1));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(1));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(1));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(1));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(1));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(1));
}

// gemm_RVV_1x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 1] @DRAM
// )
void gemm_RVV_1x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(1));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(1));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(1));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(1));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(1));
C_reg2_4 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(1));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(1));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(1));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(1));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(1));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(1));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(1));
}

// gemm_RVV_1x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 1] @DRAM
// )
void gemm_RVV_1x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(1));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(1));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(1));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(1));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(1));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(1));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(1));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(1));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(1));
}

// gemm_RVV_1x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 1] @DRAM
// )
void gemm_RVV_1x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(1));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(1));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(1));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(1));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(1));
C_reg2_4 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(1));
C_reg2_5 = __riscv_vle16_v_f16m1(&C[(13) * (ldc)],(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(1));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(1));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(1));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(1));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(1));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(1));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(1));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(1));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(1));
}

// gemm_RVV_1x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 1] @DRAM
// )
void gemm_RVV_1x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
vfloat16m1_t C_reg2_6;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_6 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
vfloat16m1_t B_reg2_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(7));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (1));
  B_reg2_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(1));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(1));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(1));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(1));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(1));
  C_reg2_6 = __riscv_vfmacc_vv_f16m1(C_reg2_6, A_reg, B_reg2_6,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(1));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(1));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(1));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(1));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(1));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg2_6,(1));
}

// gemm_RVV_1x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 1] @DRAM
// )
void gemm_RVV_1x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
vfloat16m1_t C_reg2_6;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(1));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(1));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(1));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(1));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(1));
C_reg2_4 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(1));
C_reg2_5 = __riscv_vle16_v_f16m1(&C[(13) * (ldc)],(1));
C_reg2_6 = __riscv_vle16_v_f16m1(&C[(14) * (ldc)],(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
vfloat16m1_t B_reg2_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(7));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (1));
  B_reg2_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(1));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(1));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(1));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(1));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(1));
  C_reg2_6 = __riscv_vfmacc_vv_f16m1(C_reg2_6, A_reg, B_reg2_6,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(1));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(1));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(1));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(1));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(1));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg2_6,(1));
}

// gemm_RVV_1x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 1] @DRAM
// )
void gemm_RVV_1x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg_8;
vfloat16m1_t B_reg_9;
vfloat16m1_t B_reg_10;
vfloat16m1_t B_reg_11;
vfloat16m1_t B_reg_12;
vfloat16m1_t B_reg_13;
vfloat16m1_t B_reg_14;
vfloat16m1_t B_reg_15;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(8));
  B_reg_8 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg_9 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg_10 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg_11 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg_12 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  B_reg_13 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (1));
  B_reg_14 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (1));
  B_reg_15 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg_8 = __riscv_vfmacc_vv_f16m1(C_reg_8, A_reg, B_reg_8,(1));
  C_reg_9 = __riscv_vfmacc_vv_f16m1(C_reg_9, A_reg, B_reg_9,(1));
  C_reg_10 = __riscv_vfmacc_vv_f16m1(C_reg_10, A_reg, B_reg_10,(1));
  C_reg_11 = __riscv_vfmacc_vv_f16m1(C_reg_11, A_reg, B_reg_11,(1));
  C_reg_12 = __riscv_vfmacc_vv_f16m1(C_reg_12, A_reg, B_reg_12,(1));
  C_reg_13 = __riscv_vfmacc_vv_f16m1(C_reg_13, A_reg, B_reg_13,(1));
  C_reg_14 = __riscv_vfmacc_vv_f16m1(C_reg_14, A_reg, B_reg_14,(1));
  C_reg_15 = __riscv_vfmacc_vv_f16m1(C_reg_15, A_reg, B_reg_15,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg_8,(1));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg_9,(1));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg_10,(1));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg_11,(1));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg_12,(1));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg_13,(1));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg_14,(1));
__riscv_vse16_v_f16m1(&C[(15) * (ldc)], C_reg_15,(1));
}

// gemm_RVV_1x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 1] @DRAM
// )
void gemm_RVV_1x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(1));
C_reg_8 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(1));
C_reg_9 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(1));
C_reg_10 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(1));
C_reg_11 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(1));
C_reg_12 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(1));
C_reg_13 = __riscv_vle16_v_f16m1(&C[(13) * (ldc)],(1));
C_reg_14 = __riscv_vle16_v_f16m1(&C[(14) * (ldc)],(1));
C_reg_15 = __riscv_vle16_v_f16m1(&C[(15) * (ldc)],(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg_8;
vfloat16m1_t B_reg_9;
vfloat16m1_t B_reg_10;
vfloat16m1_t B_reg_11;
vfloat16m1_t B_reg_12;
vfloat16m1_t B_reg_13;
vfloat16m1_t B_reg_14;
vfloat16m1_t B_reg_15;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(8));
  B_reg_8 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg_9 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg_10 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg_11 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg_12 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  B_reg_13 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (1));
  B_reg_14 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (1));
  B_reg_15 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg_8 = __riscv_vfmacc_vv_f16m1(C_reg_8, A_reg, B_reg_8,(1));
  C_reg_9 = __riscv_vfmacc_vv_f16m1(C_reg_9, A_reg, B_reg_9,(1));
  C_reg_10 = __riscv_vfmacc_vv_f16m1(C_reg_10, A_reg, B_reg_10,(1));
  C_reg_11 = __riscv_vfmacc_vv_f16m1(C_reg_11, A_reg, B_reg_11,(1));
  C_reg_12 = __riscv_vfmacc_vv_f16m1(C_reg_12, A_reg, B_reg_12,(1));
  C_reg_13 = __riscv_vfmacc_vv_f16m1(C_reg_13, A_reg, B_reg_13,(1));
  C_reg_14 = __riscv_vfmacc_vv_f16m1(C_reg_14, A_reg, B_reg_14,(1));
  C_reg_15 = __riscv_vfmacc_vv_f16m1(C_reg_15, A_reg, B_reg_15,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg_8,(1));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg_9,(1));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg_10,(1));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg_11,(1));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg_12,(1));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg_13,(1));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg_14,(1));
__riscv_vse16_v_f16m1(&C[(15) * (ldc)], C_reg_15,(1));
}

// gemm_RVV_1x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 1] @DRAM
// )
void gemm_RVV_1x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
}

// gemm_RVV_1x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 1] @DRAM
// )
void gemm_RVV_1x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
}

// gemm_RVV_1x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 1] @DRAM
// )
void gemm_RVV_1x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
}

// gemm_RVV_1x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 1] @DRAM
// )
void gemm_RVV_1x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
}

// gemm_RVV_1x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 1] @DRAM
// )
void gemm_RVV_1x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
}

// gemm_RVV_1x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 1] @DRAM
// )
void gemm_RVV_1x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
}

// gemm_RVV_1x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 1] @DRAM
// )
void gemm_RVV_1x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
}

// gemm_RVV_1x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 1] @DRAM
// )
void gemm_RVV_1x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
}

// gemm_RVV_1x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 1] @DRAM
// )
void gemm_RVV_1x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
}

// gemm_RVV_1x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 1] @DRAM
// )
void gemm_RVV_1x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
}

// gemm_RVV_1x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 1] @DRAM
// )
void gemm_RVV_1x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
}

// gemm_RVV_1x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 1] @DRAM
// )
void gemm_RVV_1x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
}

// gemm_RVV_1x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 1] @DRAM
// )
void gemm_RVV_1x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 6],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(1));
}

// gemm_RVV_1x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 1] @DRAM
// )
void gemm_RVV_1x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(1));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(1));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(1));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(1));
  B_reg_6 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 6],(1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(1));
}

// gemm_RVV_1x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 1] @DRAM
// )
void gemm_RVV_1x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(1));
}

// gemm_RVV_1x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 1] @DRAM
// )
void gemm_RVV_1x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(1));
}

// gemm_RVV_1x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 1] @DRAM
// )
void gemm_RVV_1x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(1));
}

// gemm_RVV_1x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 1] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 1] @DRAM
// )
void gemm_RVV_1x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(1));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(1));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(1));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(1));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (1));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (1));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (1));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (1));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (1));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (1));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(1));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(1));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(1));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(1));
}

// gemm_RVV_2x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 2] @DRAM
// )
void gemm_RVV_2x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(2));
}

// gemm_RVV_2x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 2] @DRAM
// )
void gemm_RVV_2x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(2));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(2));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(2));
}

// gemm_RVV_2x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 2] @DRAM
// )
void gemm_RVV_2x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(2));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(2));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(2));
}

// gemm_RVV_2x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 2] @DRAM
// )
void gemm_RVV_2x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(2));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(2));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(2));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(2));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(2));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(2));
}

// gemm_RVV_2x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 2] @DRAM
// )
void gemm_RVV_2x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(2));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(2));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(2));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(2));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(2));
}

// gemm_RVV_2x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 2] @DRAM
// )
void gemm_RVV_2x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(2));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(2));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(2));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(2));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(2));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(2));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(2));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(2));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(2));
}

// gemm_RVV_2x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 2] @DRAM
// )
void gemm_RVV_2x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(2));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(2));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(2));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(2));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(2));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(2));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(2));
}

// gemm_RVV_2x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 2] @DRAM
// )
void gemm_RVV_2x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(2));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(2));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(2));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(2));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(2));
C_reg2_4 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(2));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(2));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(2));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(2));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(2));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(2));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(2));
}

// gemm_RVV_2x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 2] @DRAM
// )
void gemm_RVV_2x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(2));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(2));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(2));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(2));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(2));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(2));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(2));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(2));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(2));
}

// gemm_RVV_2x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 2] @DRAM
// )
void gemm_RVV_2x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(2));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(2));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(2));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(2));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(2));
C_reg2_4 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(2));
C_reg2_5 = __riscv_vle16_v_f16m1(&C[(13) * (ldc)],(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(2));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(2));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(2));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(2));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(2));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(2));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(2));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(2));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(2));
}

// gemm_RVV_2x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 2] @DRAM
// )
void gemm_RVV_2x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
vfloat16m1_t C_reg2_6;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_6 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
vfloat16m1_t B_reg2_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(7));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (2));
  B_reg2_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(2));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(2));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(2));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(2));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(2));
  C_reg2_6 = __riscv_vfmacc_vv_f16m1(C_reg2_6, A_reg, B_reg2_6,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(2));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(2));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(2));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(2));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(2));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg2_6,(2));
}

// gemm_RVV_2x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 2] @DRAM
// )
void gemm_RVV_2x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
vfloat16m1_t C_reg2_6;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(2));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(2));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(2));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(2));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(2));
C_reg2_4 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(2));
C_reg2_5 = __riscv_vle16_v_f16m1(&C[(13) * (ldc)],(2));
C_reg2_6 = __riscv_vle16_v_f16m1(&C[(14) * (ldc)],(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
vfloat16m1_t B_reg2_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(7));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (2));
  B_reg2_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(2));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(2));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(2));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(2));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(2));
  C_reg2_6 = __riscv_vfmacc_vv_f16m1(C_reg2_6, A_reg, B_reg2_6,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(2));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(2));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(2));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(2));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(2));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg2_6,(2));
}

// gemm_RVV_2x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 2] @DRAM
// )
void gemm_RVV_2x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg_8;
vfloat16m1_t B_reg_9;
vfloat16m1_t B_reg_10;
vfloat16m1_t B_reg_11;
vfloat16m1_t B_reg_12;
vfloat16m1_t B_reg_13;
vfloat16m1_t B_reg_14;
vfloat16m1_t B_reg_15;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(8));
  B_reg_8 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg_9 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg_10 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg_11 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg_12 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  B_reg_13 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (2));
  B_reg_14 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (2));
  B_reg_15 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg_8 = __riscv_vfmacc_vv_f16m1(C_reg_8, A_reg, B_reg_8,(2));
  C_reg_9 = __riscv_vfmacc_vv_f16m1(C_reg_9, A_reg, B_reg_9,(2));
  C_reg_10 = __riscv_vfmacc_vv_f16m1(C_reg_10, A_reg, B_reg_10,(2));
  C_reg_11 = __riscv_vfmacc_vv_f16m1(C_reg_11, A_reg, B_reg_11,(2));
  C_reg_12 = __riscv_vfmacc_vv_f16m1(C_reg_12, A_reg, B_reg_12,(2));
  C_reg_13 = __riscv_vfmacc_vv_f16m1(C_reg_13, A_reg, B_reg_13,(2));
  C_reg_14 = __riscv_vfmacc_vv_f16m1(C_reg_14, A_reg, B_reg_14,(2));
  C_reg_15 = __riscv_vfmacc_vv_f16m1(C_reg_15, A_reg, B_reg_15,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg_8,(2));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg_9,(2));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg_10,(2));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg_11,(2));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg_12,(2));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg_13,(2));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg_14,(2));
__riscv_vse16_v_f16m1(&C[(15) * (ldc)], C_reg_15,(2));
}

// gemm_RVV_2x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 2] @DRAM
// )
void gemm_RVV_2x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(2));
C_reg_8 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(2));
C_reg_9 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(2));
C_reg_10 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(2));
C_reg_11 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(2));
C_reg_12 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(2));
C_reg_13 = __riscv_vle16_v_f16m1(&C[(13) * (ldc)],(2));
C_reg_14 = __riscv_vle16_v_f16m1(&C[(14) * (ldc)],(2));
C_reg_15 = __riscv_vle16_v_f16m1(&C[(15) * (ldc)],(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg_8;
vfloat16m1_t B_reg_9;
vfloat16m1_t B_reg_10;
vfloat16m1_t B_reg_11;
vfloat16m1_t B_reg_12;
vfloat16m1_t B_reg_13;
vfloat16m1_t B_reg_14;
vfloat16m1_t B_reg_15;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(8));
  B_reg_8 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg_9 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg_10 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg_11 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg_12 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  B_reg_13 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (2));
  B_reg_14 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (2));
  B_reg_15 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg_8 = __riscv_vfmacc_vv_f16m1(C_reg_8, A_reg, B_reg_8,(2));
  C_reg_9 = __riscv_vfmacc_vv_f16m1(C_reg_9, A_reg, B_reg_9,(2));
  C_reg_10 = __riscv_vfmacc_vv_f16m1(C_reg_10, A_reg, B_reg_10,(2));
  C_reg_11 = __riscv_vfmacc_vv_f16m1(C_reg_11, A_reg, B_reg_11,(2));
  C_reg_12 = __riscv_vfmacc_vv_f16m1(C_reg_12, A_reg, B_reg_12,(2));
  C_reg_13 = __riscv_vfmacc_vv_f16m1(C_reg_13, A_reg, B_reg_13,(2));
  C_reg_14 = __riscv_vfmacc_vv_f16m1(C_reg_14, A_reg, B_reg_14,(2));
  C_reg_15 = __riscv_vfmacc_vv_f16m1(C_reg_15, A_reg, B_reg_15,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg_8,(2));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg_9,(2));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg_10,(2));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg_11,(2));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg_12,(2));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg_13,(2));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg_14,(2));
__riscv_vse16_v_f16m1(&C[(15) * (ldc)], C_reg_15,(2));
}

// gemm_RVV_2x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 2] @DRAM
// )
void gemm_RVV_2x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
}

// gemm_RVV_2x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 2] @DRAM
// )
void gemm_RVV_2x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
}

// gemm_RVV_2x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 2] @DRAM
// )
void gemm_RVV_2x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
}

// gemm_RVV_2x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 2] @DRAM
// )
void gemm_RVV_2x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
}

// gemm_RVV_2x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 2] @DRAM
// )
void gemm_RVV_2x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
}

// gemm_RVV_2x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 2] @DRAM
// )
void gemm_RVV_2x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
}

// gemm_RVV_2x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 2] @DRAM
// )
void gemm_RVV_2x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
}

// gemm_RVV_2x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 2] @DRAM
// )
void gemm_RVV_2x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
}

// gemm_RVV_2x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 2] @DRAM
// )
void gemm_RVV_2x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
}

// gemm_RVV_2x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 2] @DRAM
// )
void gemm_RVV_2x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
}

// gemm_RVV_2x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 2] @DRAM
// )
void gemm_RVV_2x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
}

// gemm_RVV_2x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 2] @DRAM
// )
void gemm_RVV_2x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
}

// gemm_RVV_2x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 2] @DRAM
// )
void gemm_RVV_2x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 6],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(2));
}

// gemm_RVV_2x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 2] @DRAM
// )
void gemm_RVV_2x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(2));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(2));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(2));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(2));
  B_reg_6 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 6],(2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(2));
}

// gemm_RVV_2x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 2] @DRAM
// )
void gemm_RVV_2x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(2));
}

// gemm_RVV_2x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 2] @DRAM
// )
void gemm_RVV_2x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(2));
}

// gemm_RVV_2x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 2] @DRAM
// )
void gemm_RVV_2x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(2));
}

// gemm_RVV_2x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 2] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 2] @DRAM
// )
void gemm_RVV_2x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(2));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(2));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(2));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(2));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (2));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (2));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (2));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (2));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (2));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (2));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(2));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(2));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(2));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(2));
}

// gemm_RVV_3x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 3] @DRAM
// )
void gemm_RVV_3x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(3));
}

// gemm_RVV_3x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 3] @DRAM
// )
void gemm_RVV_3x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(3));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(3));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(3));
}

// gemm_RVV_3x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 3] @DRAM
// )
void gemm_RVV_3x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(3));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(3));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(3));
}

// gemm_RVV_3x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 3] @DRAM
// )
void gemm_RVV_3x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(3));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(3));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(3));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(3));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(3));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(3));
}

// gemm_RVV_3x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 3] @DRAM
// )
void gemm_RVV_3x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(3));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(3));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(3));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(3));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(3));
}

// gemm_RVV_3x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 3] @DRAM
// )
void gemm_RVV_3x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(3));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(3));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(3));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(3));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(3));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(3));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(3));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(3));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(3));
}

// gemm_RVV_3x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 3] @DRAM
// )
void gemm_RVV_3x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(3));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(3));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(3));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(3));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(3));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(3));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(3));
}

// gemm_RVV_3x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 3] @DRAM
// )
void gemm_RVV_3x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(3));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(3));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(3));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(3));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(3));
C_reg2_4 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(3));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(3));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(3));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(3));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(3));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(3));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(3));
}

// gemm_RVV_3x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 3] @DRAM
// )
void gemm_RVV_3x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(3));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(3));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(3));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(3));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(3));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(3));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(3));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(3));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(3));
}

// gemm_RVV_3x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 3] @DRAM
// )
void gemm_RVV_3x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(3));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(3));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(3));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(3));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(3));
C_reg2_4 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(3));
C_reg2_5 = __riscv_vle16_v_f16m1(&C[(13) * (ldc)],(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(3));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(3));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(3));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(3));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(3));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(3));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(3));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(3));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(3));
}

// gemm_RVV_3x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 3] @DRAM
// )
void gemm_RVV_3x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
vfloat16m1_t C_reg2_6;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_6 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
vfloat16m1_t B_reg2_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(7));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (3));
  B_reg2_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(3));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(3));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(3));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(3));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(3));
  C_reg2_6 = __riscv_vfmacc_vv_f16m1(C_reg2_6, A_reg, B_reg2_6,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(3));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(3));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(3));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(3));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(3));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg2_6,(3));
}

// gemm_RVV_3x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 3] @DRAM
// )
void gemm_RVV_3x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
vfloat16m1_t C_reg2_6;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(3));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(3));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(3));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(3));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(3));
C_reg2_4 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(3));
C_reg2_5 = __riscv_vle16_v_f16m1(&C[(13) * (ldc)],(3));
C_reg2_6 = __riscv_vle16_v_f16m1(&C[(14) * (ldc)],(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
vfloat16m1_t B_reg2_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(7));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (3));
  B_reg2_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(3));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(3));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(3));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(3));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(3));
  C_reg2_6 = __riscv_vfmacc_vv_f16m1(C_reg2_6, A_reg, B_reg2_6,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(3));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(3));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(3));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(3));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(3));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg2_6,(3));
}

// gemm_RVV_3x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 3] @DRAM
// )
void gemm_RVV_3x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg_8;
vfloat16m1_t B_reg_9;
vfloat16m1_t B_reg_10;
vfloat16m1_t B_reg_11;
vfloat16m1_t B_reg_12;
vfloat16m1_t B_reg_13;
vfloat16m1_t B_reg_14;
vfloat16m1_t B_reg_15;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(8));
  B_reg_8 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg_9 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg_10 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg_11 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg_12 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  B_reg_13 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (3));
  B_reg_14 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (3));
  B_reg_15 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg_8 = __riscv_vfmacc_vv_f16m1(C_reg_8, A_reg, B_reg_8,(3));
  C_reg_9 = __riscv_vfmacc_vv_f16m1(C_reg_9, A_reg, B_reg_9,(3));
  C_reg_10 = __riscv_vfmacc_vv_f16m1(C_reg_10, A_reg, B_reg_10,(3));
  C_reg_11 = __riscv_vfmacc_vv_f16m1(C_reg_11, A_reg, B_reg_11,(3));
  C_reg_12 = __riscv_vfmacc_vv_f16m1(C_reg_12, A_reg, B_reg_12,(3));
  C_reg_13 = __riscv_vfmacc_vv_f16m1(C_reg_13, A_reg, B_reg_13,(3));
  C_reg_14 = __riscv_vfmacc_vv_f16m1(C_reg_14, A_reg, B_reg_14,(3));
  C_reg_15 = __riscv_vfmacc_vv_f16m1(C_reg_15, A_reg, B_reg_15,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg_8,(3));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg_9,(3));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg_10,(3));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg_11,(3));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg_12,(3));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg_13,(3));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg_14,(3));
__riscv_vse16_v_f16m1(&C[(15) * (ldc)], C_reg_15,(3));
}

// gemm_RVV_3x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 3] @DRAM
// )
void gemm_RVV_3x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(3));
C_reg_8 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(3));
C_reg_9 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(3));
C_reg_10 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(3));
C_reg_11 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(3));
C_reg_12 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(3));
C_reg_13 = __riscv_vle16_v_f16m1(&C[(13) * (ldc)],(3));
C_reg_14 = __riscv_vle16_v_f16m1(&C[(14) * (ldc)],(3));
C_reg_15 = __riscv_vle16_v_f16m1(&C[(15) * (ldc)],(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg_8;
vfloat16m1_t B_reg_9;
vfloat16m1_t B_reg_10;
vfloat16m1_t B_reg_11;
vfloat16m1_t B_reg_12;
vfloat16m1_t B_reg_13;
vfloat16m1_t B_reg_14;
vfloat16m1_t B_reg_15;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(8));
  B_reg_8 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg_9 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg_10 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg_11 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg_12 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  B_reg_13 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (3));
  B_reg_14 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (3));
  B_reg_15 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg_8 = __riscv_vfmacc_vv_f16m1(C_reg_8, A_reg, B_reg_8,(3));
  C_reg_9 = __riscv_vfmacc_vv_f16m1(C_reg_9, A_reg, B_reg_9,(3));
  C_reg_10 = __riscv_vfmacc_vv_f16m1(C_reg_10, A_reg, B_reg_10,(3));
  C_reg_11 = __riscv_vfmacc_vv_f16m1(C_reg_11, A_reg, B_reg_11,(3));
  C_reg_12 = __riscv_vfmacc_vv_f16m1(C_reg_12, A_reg, B_reg_12,(3));
  C_reg_13 = __riscv_vfmacc_vv_f16m1(C_reg_13, A_reg, B_reg_13,(3));
  C_reg_14 = __riscv_vfmacc_vv_f16m1(C_reg_14, A_reg, B_reg_14,(3));
  C_reg_15 = __riscv_vfmacc_vv_f16m1(C_reg_15, A_reg, B_reg_15,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg_8,(3));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg_9,(3));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg_10,(3));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg_11,(3));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg_12,(3));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg_13,(3));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg_14,(3));
__riscv_vse16_v_f16m1(&C[(15) * (ldc)], C_reg_15,(3));
}

// gemm_RVV_3x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 3] @DRAM
// )
void gemm_RVV_3x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
}

// gemm_RVV_3x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 3] @DRAM
// )
void gemm_RVV_3x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
}

// gemm_RVV_3x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 3] @DRAM
// )
void gemm_RVV_3x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
}

// gemm_RVV_3x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 3] @DRAM
// )
void gemm_RVV_3x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
}

// gemm_RVV_3x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 3] @DRAM
// )
void gemm_RVV_3x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
}

// gemm_RVV_3x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 3] @DRAM
// )
void gemm_RVV_3x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
}

// gemm_RVV_3x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 3] @DRAM
// )
void gemm_RVV_3x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
}

// gemm_RVV_3x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 3] @DRAM
// )
void gemm_RVV_3x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
}

// gemm_RVV_3x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 3] @DRAM
// )
void gemm_RVV_3x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
}

// gemm_RVV_3x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 3] @DRAM
// )
void gemm_RVV_3x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
}

// gemm_RVV_3x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 3] @DRAM
// )
void gemm_RVV_3x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
}

// gemm_RVV_3x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 3] @DRAM
// )
void gemm_RVV_3x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
}

// gemm_RVV_3x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 3] @DRAM
// )
void gemm_RVV_3x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 6],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(3));
}

// gemm_RVV_3x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 3] @DRAM
// )
void gemm_RVV_3x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(3));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(3));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(3));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(3));
  B_reg_6 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 6],(3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(3));
}

// gemm_RVV_3x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 3] @DRAM
// )
void gemm_RVV_3x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(3));
}

// gemm_RVV_3x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 3] @DRAM
// )
void gemm_RVV_3x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(3));
}

// gemm_RVV_3x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 3] @DRAM
// )
void gemm_RVV_3x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(3));
}

// gemm_RVV_3x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 3] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 3] @DRAM
// )
void gemm_RVV_3x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(3));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(3));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(3));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(3));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (3));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (3));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (3));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (3));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (3));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (3));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(3));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(3));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(3));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(3));
}

// gemm_RVV_4x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 4] @DRAM
// )
void gemm_RVV_4x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(4));
}

// gemm_RVV_4x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 4] @DRAM
// )
void gemm_RVV_4x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(4));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(4));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(4));
}

// gemm_RVV_4x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 4] @DRAM
// )
void gemm_RVV_4x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(4));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(4));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(4));
}

// gemm_RVV_4x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 4] @DRAM
// )
void gemm_RVV_4x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(4));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(4));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(4));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(4));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(4));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(4));
}

// gemm_RVV_4x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 4] @DRAM
// )
void gemm_RVV_4x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(4));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(4));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(4));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(4));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(4));
}

// gemm_RVV_4x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 4] @DRAM
// )
void gemm_RVV_4x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(4));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(4));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(4));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(4));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(4));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(4));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(4));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(4));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(4));
}

// gemm_RVV_4x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 4] @DRAM
// )
void gemm_RVV_4x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(4));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(4));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(4));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(4));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(4));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(4));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(4));
}

// gemm_RVV_4x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 4] @DRAM
// )
void gemm_RVV_4x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(4));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(4));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(4));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(4));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(4));
C_reg2_4 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(4));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(4));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(4));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(4));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(4));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(4));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(4));
}

// gemm_RVV_4x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 4] @DRAM
// )
void gemm_RVV_4x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(4));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(4));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(4));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(4));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(4));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(4));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(4));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(4));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(4));
}

// gemm_RVV_4x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 4] @DRAM
// )
void gemm_RVV_4x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(4));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(4));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(4));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(4));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(4));
C_reg2_4 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(4));
C_reg2_5 = __riscv_vle16_v_f16m1(&C[(13) * (ldc)],(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(4));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(4));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(4));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(4));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(4));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(4));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(4));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(4));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(4));
}

// gemm_RVV_4x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 4] @DRAM
// )
void gemm_RVV_4x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
vfloat16m1_t C_reg2_6;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_6 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
vfloat16m1_t B_reg2_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(7));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (4));
  B_reg2_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(4));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(4));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(4));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(4));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(4));
  C_reg2_6 = __riscv_vfmacc_vv_f16m1(C_reg2_6, A_reg, B_reg2_6,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(4));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(4));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(4));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(4));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(4));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg2_6,(4));
}

// gemm_RVV_4x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 4] @DRAM
// )
void gemm_RVV_4x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
vfloat16m1_t C_reg2_6;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(4));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(4));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(4));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(4));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(4));
C_reg2_4 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(4));
C_reg2_5 = __riscv_vle16_v_f16m1(&C[(13) * (ldc)],(4));
C_reg2_6 = __riscv_vle16_v_f16m1(&C[(14) * (ldc)],(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
vfloat16m1_t B_reg2_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(7));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (4));
  B_reg2_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(4));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(4));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(4));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(4));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(4));
  C_reg2_6 = __riscv_vfmacc_vv_f16m1(C_reg2_6, A_reg, B_reg2_6,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(4));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(4));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(4));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(4));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(4));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg2_6,(4));
}

// gemm_RVV_4x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 4] @DRAM
// )
void gemm_RVV_4x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg_8;
vfloat16m1_t B_reg_9;
vfloat16m1_t B_reg_10;
vfloat16m1_t B_reg_11;
vfloat16m1_t B_reg_12;
vfloat16m1_t B_reg_13;
vfloat16m1_t B_reg_14;
vfloat16m1_t B_reg_15;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(8));
  B_reg_8 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg_9 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg_10 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg_11 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg_12 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  B_reg_13 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (4));
  B_reg_14 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (4));
  B_reg_15 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg_8 = __riscv_vfmacc_vv_f16m1(C_reg_8, A_reg, B_reg_8,(4));
  C_reg_9 = __riscv_vfmacc_vv_f16m1(C_reg_9, A_reg, B_reg_9,(4));
  C_reg_10 = __riscv_vfmacc_vv_f16m1(C_reg_10, A_reg, B_reg_10,(4));
  C_reg_11 = __riscv_vfmacc_vv_f16m1(C_reg_11, A_reg, B_reg_11,(4));
  C_reg_12 = __riscv_vfmacc_vv_f16m1(C_reg_12, A_reg, B_reg_12,(4));
  C_reg_13 = __riscv_vfmacc_vv_f16m1(C_reg_13, A_reg, B_reg_13,(4));
  C_reg_14 = __riscv_vfmacc_vv_f16m1(C_reg_14, A_reg, B_reg_14,(4));
  C_reg_15 = __riscv_vfmacc_vv_f16m1(C_reg_15, A_reg, B_reg_15,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg_8,(4));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg_9,(4));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg_10,(4));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg_11,(4));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg_12,(4));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg_13,(4));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg_14,(4));
__riscv_vse16_v_f16m1(&C[(15) * (ldc)], C_reg_15,(4));
}

// gemm_RVV_4x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 4] @DRAM
// )
void gemm_RVV_4x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(4));
C_reg_8 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(4));
C_reg_9 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(4));
C_reg_10 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(4));
C_reg_11 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(4));
C_reg_12 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(4));
C_reg_13 = __riscv_vle16_v_f16m1(&C[(13) * (ldc)],(4));
C_reg_14 = __riscv_vle16_v_f16m1(&C[(14) * (ldc)],(4));
C_reg_15 = __riscv_vle16_v_f16m1(&C[(15) * (ldc)],(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg_8;
vfloat16m1_t B_reg_9;
vfloat16m1_t B_reg_10;
vfloat16m1_t B_reg_11;
vfloat16m1_t B_reg_12;
vfloat16m1_t B_reg_13;
vfloat16m1_t B_reg_14;
vfloat16m1_t B_reg_15;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(8));
  B_reg_8 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg_9 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg_10 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg_11 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg_12 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  B_reg_13 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (4));
  B_reg_14 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (4));
  B_reg_15 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg_8 = __riscv_vfmacc_vv_f16m1(C_reg_8, A_reg, B_reg_8,(4));
  C_reg_9 = __riscv_vfmacc_vv_f16m1(C_reg_9, A_reg, B_reg_9,(4));
  C_reg_10 = __riscv_vfmacc_vv_f16m1(C_reg_10, A_reg, B_reg_10,(4));
  C_reg_11 = __riscv_vfmacc_vv_f16m1(C_reg_11, A_reg, B_reg_11,(4));
  C_reg_12 = __riscv_vfmacc_vv_f16m1(C_reg_12, A_reg, B_reg_12,(4));
  C_reg_13 = __riscv_vfmacc_vv_f16m1(C_reg_13, A_reg, B_reg_13,(4));
  C_reg_14 = __riscv_vfmacc_vv_f16m1(C_reg_14, A_reg, B_reg_14,(4));
  C_reg_15 = __riscv_vfmacc_vv_f16m1(C_reg_15, A_reg, B_reg_15,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg_8,(4));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg_9,(4));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg_10,(4));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg_11,(4));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg_12,(4));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg_13,(4));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg_14,(4));
__riscv_vse16_v_f16m1(&C[(15) * (ldc)], C_reg_15,(4));
}

// gemm_RVV_4x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 4] @DRAM
// )
void gemm_RVV_4x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
}

// gemm_RVV_4x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 4] @DRAM
// )
void gemm_RVV_4x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
}

// gemm_RVV_4x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 4] @DRAM
// )
void gemm_RVV_4x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
}

// gemm_RVV_4x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 4] @DRAM
// )
void gemm_RVV_4x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
}

// gemm_RVV_4x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 4] @DRAM
// )
void gemm_RVV_4x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
}

// gemm_RVV_4x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 4] @DRAM
// )
void gemm_RVV_4x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
}

// gemm_RVV_4x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 4] @DRAM
// )
void gemm_RVV_4x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
}

// gemm_RVV_4x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 4] @DRAM
// )
void gemm_RVV_4x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
}

// gemm_RVV_4x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 4] @DRAM
// )
void gemm_RVV_4x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
}

// gemm_RVV_4x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 4] @DRAM
// )
void gemm_RVV_4x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
}

// gemm_RVV_4x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 4] @DRAM
// )
void gemm_RVV_4x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
}

// gemm_RVV_4x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 4] @DRAM
// )
void gemm_RVV_4x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
}

// gemm_RVV_4x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 4] @DRAM
// )
void gemm_RVV_4x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 6],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(4));
}

// gemm_RVV_4x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 4] @DRAM
// )
void gemm_RVV_4x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(4));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(4));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(4));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(4));
  B_reg_6 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 6],(4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(4));
}

// gemm_RVV_4x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 4] @DRAM
// )
void gemm_RVV_4x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(4));
}

// gemm_RVV_4x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 4] @DRAM
// )
void gemm_RVV_4x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(4));
}

// gemm_RVV_4x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 4] @DRAM
// )
void gemm_RVV_4x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(4));
}

// gemm_RVV_4x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 4] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 4] @DRAM
// )
void gemm_RVV_4x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(4));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(4));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(4));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(4));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (4));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (4));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (4));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (4));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (4));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (4));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(4));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(4));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(4));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(4));
}

// gemm_RVV_5x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 5] @DRAM
// )
void gemm_RVV_5x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(5));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(5));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(5));
}

// gemm_RVV_5x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 5] @DRAM
// )
void gemm_RVV_5x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(5));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(5));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(5));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(5));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(5));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(5));
}

// gemm_RVV_5x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 5] @DRAM
// )
void gemm_RVV_5x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(5));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(5));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(5));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(5));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(5));
}

// gemm_RVV_5x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 5] @DRAM
// )
void gemm_RVV_5x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(5));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(5));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(5));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(5));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(5));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(5));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(5));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(5));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(5));
}

// gemm_RVV_5x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 5] @DRAM
// )
void gemm_RVV_5x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(5));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(5));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(5));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(5));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(5));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(5));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(5));
}

// gemm_RVV_5x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 5] @DRAM
// )
void gemm_RVV_5x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(5));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(5));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(5));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(5));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(5));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(5));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(5));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(5));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(5));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(5));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(5));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(5));
}

// gemm_RVV_5x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 5] @DRAM
// )
void gemm_RVV_5x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(5));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(5));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(5));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(5));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(5));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(5));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(5));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(5));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(5));
}

// gemm_RVV_5x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 5] @DRAM
// )
void gemm_RVV_5x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(5));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(5));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(5));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(5));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(5));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(5));
C_reg2_4 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(5));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(5));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(5));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(5));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(5));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(5));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(5));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(5));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(5));
}

// gemm_RVV_5x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 5] @DRAM
// )
void gemm_RVV_5x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(5));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(5));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(5));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(5));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(5));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(5));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(5));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(5));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(5));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(5));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(5));
}

// gemm_RVV_5x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 5] @DRAM
// )
void gemm_RVV_5x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(5));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(5));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(5));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(5));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(5));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(5));
C_reg2_4 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(5));
C_reg2_5 = __riscv_vle16_v_f16m1(&C[(13) * (ldc)],(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(5));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(5));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(5));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(5));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(5));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(5));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(5));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(5));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(5));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(5));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(5));
}

// gemm_RVV_5x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 5] @DRAM
// )
void gemm_RVV_5x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
vfloat16m1_t C_reg2_6;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_6 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
vfloat16m1_t B_reg2_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(7));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (5));
  B_reg2_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(5));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(5));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(5));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(5));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(5));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(5));
  C_reg2_6 = __riscv_vfmacc_vv_f16m1(C_reg2_6, A_reg, B_reg2_6,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(5));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(5));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(5));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(5));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(5));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(5));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg2_6,(5));
}

// gemm_RVV_5x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 5] @DRAM
// )
void gemm_RVV_5x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
vfloat16m1_t C_reg2_6;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(5));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(5));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(5));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(5));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(5));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(5));
C_reg2_4 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(5));
C_reg2_5 = __riscv_vle16_v_f16m1(&C[(13) * (ldc)],(5));
C_reg2_6 = __riscv_vle16_v_f16m1(&C[(14) * (ldc)],(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
vfloat16m1_t B_reg2_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(7));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (5));
  B_reg2_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(5));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(5));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(5));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(5));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(5));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(5));
  C_reg2_6 = __riscv_vfmacc_vv_f16m1(C_reg2_6, A_reg, B_reg2_6,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(5));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(5));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(5));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(5));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(5));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(5));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg2_6,(5));
}

// gemm_RVV_5x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 5] @DRAM
// )
void gemm_RVV_5x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg_8;
vfloat16m1_t B_reg_9;
vfloat16m1_t B_reg_10;
vfloat16m1_t B_reg_11;
vfloat16m1_t B_reg_12;
vfloat16m1_t B_reg_13;
vfloat16m1_t B_reg_14;
vfloat16m1_t B_reg_15;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(8));
  B_reg_8 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg_9 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg_10 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg_11 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg_12 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  B_reg_13 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (5));
  B_reg_14 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (5));
  B_reg_15 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg_8 = __riscv_vfmacc_vv_f16m1(C_reg_8, A_reg, B_reg_8,(5));
  C_reg_9 = __riscv_vfmacc_vv_f16m1(C_reg_9, A_reg, B_reg_9,(5));
  C_reg_10 = __riscv_vfmacc_vv_f16m1(C_reg_10, A_reg, B_reg_10,(5));
  C_reg_11 = __riscv_vfmacc_vv_f16m1(C_reg_11, A_reg, B_reg_11,(5));
  C_reg_12 = __riscv_vfmacc_vv_f16m1(C_reg_12, A_reg, B_reg_12,(5));
  C_reg_13 = __riscv_vfmacc_vv_f16m1(C_reg_13, A_reg, B_reg_13,(5));
  C_reg_14 = __riscv_vfmacc_vv_f16m1(C_reg_14, A_reg, B_reg_14,(5));
  C_reg_15 = __riscv_vfmacc_vv_f16m1(C_reg_15, A_reg, B_reg_15,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg_8,(5));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg_9,(5));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg_10,(5));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg_11,(5));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg_12,(5));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg_13,(5));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg_14,(5));
__riscv_vse16_v_f16m1(&C[(15) * (ldc)], C_reg_15,(5));
}

// gemm_RVV_5x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 5] @DRAM
// )
void gemm_RVV_5x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(5));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(5));
C_reg_8 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(5));
C_reg_9 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(5));
C_reg_10 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(5));
C_reg_11 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(5));
C_reg_12 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(5));
C_reg_13 = __riscv_vle16_v_f16m1(&C[(13) * (ldc)],(5));
C_reg_14 = __riscv_vle16_v_f16m1(&C[(14) * (ldc)],(5));
C_reg_15 = __riscv_vle16_v_f16m1(&C[(15) * (ldc)],(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg_8;
vfloat16m1_t B_reg_9;
vfloat16m1_t B_reg_10;
vfloat16m1_t B_reg_11;
vfloat16m1_t B_reg_12;
vfloat16m1_t B_reg_13;
vfloat16m1_t B_reg_14;
vfloat16m1_t B_reg_15;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(8));
  B_reg_8 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg_9 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg_10 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg_11 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg_12 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  B_reg_13 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (5));
  B_reg_14 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (5));
  B_reg_15 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg_8 = __riscv_vfmacc_vv_f16m1(C_reg_8, A_reg, B_reg_8,(5));
  C_reg_9 = __riscv_vfmacc_vv_f16m1(C_reg_9, A_reg, B_reg_9,(5));
  C_reg_10 = __riscv_vfmacc_vv_f16m1(C_reg_10, A_reg, B_reg_10,(5));
  C_reg_11 = __riscv_vfmacc_vv_f16m1(C_reg_11, A_reg, B_reg_11,(5));
  C_reg_12 = __riscv_vfmacc_vv_f16m1(C_reg_12, A_reg, B_reg_12,(5));
  C_reg_13 = __riscv_vfmacc_vv_f16m1(C_reg_13, A_reg, B_reg_13,(5));
  C_reg_14 = __riscv_vfmacc_vv_f16m1(C_reg_14, A_reg, B_reg_14,(5));
  C_reg_15 = __riscv_vfmacc_vv_f16m1(C_reg_15, A_reg, B_reg_15,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg_8,(5));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg_9,(5));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg_10,(5));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg_11,(5));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg_12,(5));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg_13,(5));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg_14,(5));
__riscv_vse16_v_f16m1(&C[(15) * (ldc)], C_reg_15,(5));
}

// gemm_RVV_5x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 5] @DRAM
// )
void gemm_RVV_5x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
}

// gemm_RVV_5x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 5] @DRAM
// )
void gemm_RVV_5x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
}

// gemm_RVV_5x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 5] @DRAM
// )
void gemm_RVV_5x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
}

// gemm_RVV_5x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 5] @DRAM
// )
void gemm_RVV_5x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
}

// gemm_RVV_5x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 5] @DRAM
// )
void gemm_RVV_5x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
}

// gemm_RVV_5x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 5] @DRAM
// )
void gemm_RVV_5x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
}

// gemm_RVV_5x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 5] @DRAM
// )
void gemm_RVV_5x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(5));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
}

// gemm_RVV_5x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 5] @DRAM
// )
void gemm_RVV_5x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(5));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
}

// gemm_RVV_5x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 5] @DRAM
// )
void gemm_RVV_5x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(5));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(5));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
}

// gemm_RVV_5x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 5] @DRAM
// )
void gemm_RVV_5x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(5));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(5));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
}

// gemm_RVV_5x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 5] @DRAM
// )
void gemm_RVV_5x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(5));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(5));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(5));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
}

// gemm_RVV_5x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 5] @DRAM
// )
void gemm_RVV_5x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(5));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(5));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(5));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
}

// gemm_RVV_5x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 5] @DRAM
// )
void gemm_RVV_5x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(5));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(5));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(5));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(5));
  B_reg_6 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 6],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(5));
}

// gemm_RVV_5x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 5] @DRAM
// )
void gemm_RVV_5x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(5));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(5));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(5));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(5));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(5));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(5));
  B_reg_6 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 6],(5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(5));
}

// gemm_RVV_5x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 5] @DRAM
// )
void gemm_RVV_5x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(5));
}

// gemm_RVV_5x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 5] @DRAM
// )
void gemm_RVV_5x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(5));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(5));
}

// gemm_RVV_5x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 5] @DRAM
// )
void gemm_RVV_5x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(5));
}

// gemm_RVV_5x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 5] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 5] @DRAM
// )
void gemm_RVV_5x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(5));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(5));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(5));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(5));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(5));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(5));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(5));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(5));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(5));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (5));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (5));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (5));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (5));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (5));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (5));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (5));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (5));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(5));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(5));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(5));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(5));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(5));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(5));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(5));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(5));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(5));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(5));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(5));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(5));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(5));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(5));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(5));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(5));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(5));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(5));
}

// gemm_RVV_6x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 6] @DRAM
// )
void gemm_RVV_6x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(6));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(6));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(6));
}

// gemm_RVV_6x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 6] @DRAM
// )
void gemm_RVV_6x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(6));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(6));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(6));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(6));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(6));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(6));
}

// gemm_RVV_6x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 6] @DRAM
// )
void gemm_RVV_6x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(6));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(6));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(6));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(6));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(6));
}

// gemm_RVV_6x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 6] @DRAM
// )
void gemm_RVV_6x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(6));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(6));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(6));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(6));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(6));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(6));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(6));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(6));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(6));
}

// gemm_RVV_6x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 6] @DRAM
// )
void gemm_RVV_6x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(6));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(6));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(6));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(6));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(6));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(6));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(6));
}

// gemm_RVV_6x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 6] @DRAM
// )
void gemm_RVV_6x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(6));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(6));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(6));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(6));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(6));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(6));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(6));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(6));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(6));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(6));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(6));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(6));
}

// gemm_RVV_6x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 6] @DRAM
// )
void gemm_RVV_6x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(6));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(6));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(6));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(6));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(6));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(6));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(6));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(6));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(6));
}

// gemm_RVV_6x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 6] @DRAM
// )
void gemm_RVV_6x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(6));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(6));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(6));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(6));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(6));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(6));
C_reg2_4 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(6));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(6));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(6));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(6));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(6));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(6));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(6));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(6));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(6));
}

// gemm_RVV_6x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 6] @DRAM
// )
void gemm_RVV_6x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(6));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(6));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(6));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(6));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(6));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(6));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(6));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(6));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(6));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(6));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(6));
}

// gemm_RVV_6x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 6] @DRAM
// )
void gemm_RVV_6x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(6));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(6));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(6));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(6));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(6));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(6));
C_reg2_4 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(6));
C_reg2_5 = __riscv_vle16_v_f16m1(&C[(13) * (ldc)],(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(6));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(6));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(6));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(6));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(6));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(6));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(6));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(6));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(6));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(6));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(6));
}

// gemm_RVV_6x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 6] @DRAM
// )
void gemm_RVV_6x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
vfloat16m1_t C_reg2_6;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_6 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
vfloat16m1_t B_reg2_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(7));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (6));
  B_reg2_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(6));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(6));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(6));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(6));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(6));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(6));
  C_reg2_6 = __riscv_vfmacc_vv_f16m1(C_reg2_6, A_reg, B_reg2_6,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(6));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(6));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(6));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(6));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(6));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(6));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg2_6,(6));
}

// gemm_RVV_6x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 6] @DRAM
// )
void gemm_RVV_6x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
vfloat16m1_t C_reg2_6;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(6));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(6));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(6));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(6));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(6));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(6));
C_reg2_4 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(6));
C_reg2_5 = __riscv_vle16_v_f16m1(&C[(13) * (ldc)],(6));
C_reg2_6 = __riscv_vle16_v_f16m1(&C[(14) * (ldc)],(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
vfloat16m1_t B_reg2_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(7));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (6));
  B_reg2_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(6));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(6));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(6));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(6));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(6));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(6));
  C_reg2_6 = __riscv_vfmacc_vv_f16m1(C_reg2_6, A_reg, B_reg2_6,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(6));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(6));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(6));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(6));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(6));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(6));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg2_6,(6));
}

// gemm_RVV_6x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 6] @DRAM
// )
void gemm_RVV_6x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg_8;
vfloat16m1_t B_reg_9;
vfloat16m1_t B_reg_10;
vfloat16m1_t B_reg_11;
vfloat16m1_t B_reg_12;
vfloat16m1_t B_reg_13;
vfloat16m1_t B_reg_14;
vfloat16m1_t B_reg_15;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(8));
  B_reg_8 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg_9 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg_10 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg_11 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg_12 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  B_reg_13 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (6));
  B_reg_14 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (6));
  B_reg_15 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg_8 = __riscv_vfmacc_vv_f16m1(C_reg_8, A_reg, B_reg_8,(6));
  C_reg_9 = __riscv_vfmacc_vv_f16m1(C_reg_9, A_reg, B_reg_9,(6));
  C_reg_10 = __riscv_vfmacc_vv_f16m1(C_reg_10, A_reg, B_reg_10,(6));
  C_reg_11 = __riscv_vfmacc_vv_f16m1(C_reg_11, A_reg, B_reg_11,(6));
  C_reg_12 = __riscv_vfmacc_vv_f16m1(C_reg_12, A_reg, B_reg_12,(6));
  C_reg_13 = __riscv_vfmacc_vv_f16m1(C_reg_13, A_reg, B_reg_13,(6));
  C_reg_14 = __riscv_vfmacc_vv_f16m1(C_reg_14, A_reg, B_reg_14,(6));
  C_reg_15 = __riscv_vfmacc_vv_f16m1(C_reg_15, A_reg, B_reg_15,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg_8,(6));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg_9,(6));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg_10,(6));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg_11,(6));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg_12,(6));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg_13,(6));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg_14,(6));
__riscv_vse16_v_f16m1(&C[(15) * (ldc)], C_reg_15,(6));
}

// gemm_RVV_6x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 6] @DRAM
// )
void gemm_RVV_6x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(6));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(6));
C_reg_8 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(6));
C_reg_9 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(6));
C_reg_10 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(6));
C_reg_11 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(6));
C_reg_12 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(6));
C_reg_13 = __riscv_vle16_v_f16m1(&C[(13) * (ldc)],(6));
C_reg_14 = __riscv_vle16_v_f16m1(&C[(14) * (ldc)],(6));
C_reg_15 = __riscv_vle16_v_f16m1(&C[(15) * (ldc)],(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg_8;
vfloat16m1_t B_reg_9;
vfloat16m1_t B_reg_10;
vfloat16m1_t B_reg_11;
vfloat16m1_t B_reg_12;
vfloat16m1_t B_reg_13;
vfloat16m1_t B_reg_14;
vfloat16m1_t B_reg_15;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(8));
  B_reg_8 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg_9 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg_10 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg_11 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg_12 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  B_reg_13 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (6));
  B_reg_14 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (6));
  B_reg_15 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg_8 = __riscv_vfmacc_vv_f16m1(C_reg_8, A_reg, B_reg_8,(6));
  C_reg_9 = __riscv_vfmacc_vv_f16m1(C_reg_9, A_reg, B_reg_9,(6));
  C_reg_10 = __riscv_vfmacc_vv_f16m1(C_reg_10, A_reg, B_reg_10,(6));
  C_reg_11 = __riscv_vfmacc_vv_f16m1(C_reg_11, A_reg, B_reg_11,(6));
  C_reg_12 = __riscv_vfmacc_vv_f16m1(C_reg_12, A_reg, B_reg_12,(6));
  C_reg_13 = __riscv_vfmacc_vv_f16m1(C_reg_13, A_reg, B_reg_13,(6));
  C_reg_14 = __riscv_vfmacc_vv_f16m1(C_reg_14, A_reg, B_reg_14,(6));
  C_reg_15 = __riscv_vfmacc_vv_f16m1(C_reg_15, A_reg, B_reg_15,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg_8,(6));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg_9,(6));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg_10,(6));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg_11,(6));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg_12,(6));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg_13,(6));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg_14,(6));
__riscv_vse16_v_f16m1(&C[(15) * (ldc)], C_reg_15,(6));
}

// gemm_RVV_6x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 6] @DRAM
// )
void gemm_RVV_6x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
}

// gemm_RVV_6x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 6] @DRAM
// )
void gemm_RVV_6x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
}

// gemm_RVV_6x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 6] @DRAM
// )
void gemm_RVV_6x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
}

// gemm_RVV_6x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 6] @DRAM
// )
void gemm_RVV_6x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
}

// gemm_RVV_6x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 6] @DRAM
// )
void gemm_RVV_6x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
}

// gemm_RVV_6x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 6] @DRAM
// )
void gemm_RVV_6x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
}

// gemm_RVV_6x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 6] @DRAM
// )
void gemm_RVV_6x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(6));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
}

// gemm_RVV_6x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 6] @DRAM
// )
void gemm_RVV_6x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(6));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
}

// gemm_RVV_6x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 6] @DRAM
// )
void gemm_RVV_6x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(6));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(6));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
}

// gemm_RVV_6x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 6] @DRAM
// )
void gemm_RVV_6x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(6));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(6));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
}

// gemm_RVV_6x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 6] @DRAM
// )
void gemm_RVV_6x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(6));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(6));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(6));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
}

// gemm_RVV_6x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 6] @DRAM
// )
void gemm_RVV_6x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(6));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(6));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(6));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
}

// gemm_RVV_6x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 6] @DRAM
// )
void gemm_RVV_6x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(6));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(6));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(6));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(6));
  B_reg_6 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 6],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(6));
}

// gemm_RVV_6x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 6] @DRAM
// )
void gemm_RVV_6x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(6));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(6));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(6));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(6));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(6));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(6));
  B_reg_6 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 6],(6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(6));
}

// gemm_RVV_6x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 6] @DRAM
// )
void gemm_RVV_6x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(6));
}

// gemm_RVV_6x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 6] @DRAM
// )
void gemm_RVV_6x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(6));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(6));
}

// gemm_RVV_6x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 6] @DRAM
// )
void gemm_RVV_6x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(6));
}

// gemm_RVV_6x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 6] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 6] @DRAM
// )
void gemm_RVV_6x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(6));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(6));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(6));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(6));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(6));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(6));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(6));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(6));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(6));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (6));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (6));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (6));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (6));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (6));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (6));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (6));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (6));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(6));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(6));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(6));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(6));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(6));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(6));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(6));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(6));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(6));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(6));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(6));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(6));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(6));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(6));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(6));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(6));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(6));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(6));
}

// gemm_RVV_7x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 7] @DRAM
// )
void gemm_RVV_7x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(7));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(7));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(7));
}

// gemm_RVV_7x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 7] @DRAM
// )
void gemm_RVV_7x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(7));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(7));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(7));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(7));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(7));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(7));
}

// gemm_RVV_7x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 7] @DRAM
// )
void gemm_RVV_7x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(7));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(7));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(7));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(7));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(7));
}

// gemm_RVV_7x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 7] @DRAM
// )
void gemm_RVV_7x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(7));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(7));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(7));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(7));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(7));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(7));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(7));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(7));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(7));
}

// gemm_RVV_7x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 7] @DRAM
// )
void gemm_RVV_7x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(7));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(7));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(7));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(7));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(7));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(7));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(7));
}

// gemm_RVV_7x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 7] @DRAM
// )
void gemm_RVV_7x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(7));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(7));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(7));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(7));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(7));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(7));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(7));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(7));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(7));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(7));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(7));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(7));
}

// gemm_RVV_7x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 7] @DRAM
// )
void gemm_RVV_7x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(7));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(7));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(7));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(7));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(7));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(7));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(7));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(7));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(7));
}

// gemm_RVV_7x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 7] @DRAM
// )
void gemm_RVV_7x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(7));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(7));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(7));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(7));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(7));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(7));
C_reg2_4 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(7));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(7));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(7));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(7));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(7));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(7));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(7));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(7));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(7));
}

// gemm_RVV_7x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 7] @DRAM
// )
void gemm_RVV_7x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(7));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(7));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(7));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(7));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(7));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(7));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(7));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(7));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(7));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(7));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(7));
}

// gemm_RVV_7x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 7] @DRAM
// )
void gemm_RVV_7x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(7));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(7));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(7));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(7));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(7));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(7));
C_reg2_4 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(7));
C_reg2_5 = __riscv_vle16_v_f16m1(&C[(13) * (ldc)],(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(7));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(7));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(7));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(7));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(7));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(7));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(7));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(7));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(7));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(7));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(7));
}

// gemm_RVV_7x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 7] @DRAM
// )
void gemm_RVV_7x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
vfloat16m1_t C_reg2_6;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_6 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
vfloat16m1_t B_reg2_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(7));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (7));
  B_reg2_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(7));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(7));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(7));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(7));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(7));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(7));
  C_reg2_6 = __riscv_vfmacc_vv_f16m1(C_reg2_6, A_reg, B_reg2_6,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(7));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(7));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(7));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(7));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(7));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(7));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg2_6,(7));
}

// gemm_RVV_7x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 7] @DRAM
// )
void gemm_RVV_7x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
vfloat16m1_t C_reg2_6;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(7));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(7));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(7));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(7));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(7));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(7));
C_reg2_4 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(7));
C_reg2_5 = __riscv_vle16_v_f16m1(&C[(13) * (ldc)],(7));
C_reg2_6 = __riscv_vle16_v_f16m1(&C[(14) * (ldc)],(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
vfloat16m1_t B_reg2_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(7));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (7));
  B_reg2_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(7));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(7));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(7));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(7));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(7));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(7));
  C_reg2_6 = __riscv_vfmacc_vv_f16m1(C_reg2_6, A_reg, B_reg2_6,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(7));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(7));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(7));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(7));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(7));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(7));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg2_6,(7));
}

// gemm_RVV_7x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 7] @DRAM
// )
void gemm_RVV_7x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg_8;
vfloat16m1_t B_reg_9;
vfloat16m1_t B_reg_10;
vfloat16m1_t B_reg_11;
vfloat16m1_t B_reg_12;
vfloat16m1_t B_reg_13;
vfloat16m1_t B_reg_14;
vfloat16m1_t B_reg_15;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(8));
  B_reg_8 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg_9 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg_10 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg_11 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg_12 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  B_reg_13 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (7));
  B_reg_14 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (7));
  B_reg_15 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg_8 = __riscv_vfmacc_vv_f16m1(C_reg_8, A_reg, B_reg_8,(7));
  C_reg_9 = __riscv_vfmacc_vv_f16m1(C_reg_9, A_reg, B_reg_9,(7));
  C_reg_10 = __riscv_vfmacc_vv_f16m1(C_reg_10, A_reg, B_reg_10,(7));
  C_reg_11 = __riscv_vfmacc_vv_f16m1(C_reg_11, A_reg, B_reg_11,(7));
  C_reg_12 = __riscv_vfmacc_vv_f16m1(C_reg_12, A_reg, B_reg_12,(7));
  C_reg_13 = __riscv_vfmacc_vv_f16m1(C_reg_13, A_reg, B_reg_13,(7));
  C_reg_14 = __riscv_vfmacc_vv_f16m1(C_reg_14, A_reg, B_reg_14,(7));
  C_reg_15 = __riscv_vfmacc_vv_f16m1(C_reg_15, A_reg, B_reg_15,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg_8,(7));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg_9,(7));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg_10,(7));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg_11,(7));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg_12,(7));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg_13,(7));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg_14,(7));
__riscv_vse16_v_f16m1(&C[(15) * (ldc)], C_reg_15,(7));
}

// gemm_RVV_7x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 7] @DRAM
// )
void gemm_RVV_7x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(7));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(7));
C_reg_8 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(7));
C_reg_9 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(7));
C_reg_10 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(7));
C_reg_11 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(7));
C_reg_12 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(7));
C_reg_13 = __riscv_vle16_v_f16m1(&C[(13) * (ldc)],(7));
C_reg_14 = __riscv_vle16_v_f16m1(&C[(14) * (ldc)],(7));
C_reg_15 = __riscv_vle16_v_f16m1(&C[(15) * (ldc)],(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg_8;
vfloat16m1_t B_reg_9;
vfloat16m1_t B_reg_10;
vfloat16m1_t B_reg_11;
vfloat16m1_t B_reg_12;
vfloat16m1_t B_reg_13;
vfloat16m1_t B_reg_14;
vfloat16m1_t B_reg_15;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(8));
  B_reg_8 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg_9 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg_10 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg_11 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg_12 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  B_reg_13 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (7));
  B_reg_14 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (7));
  B_reg_15 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg_8 = __riscv_vfmacc_vv_f16m1(C_reg_8, A_reg, B_reg_8,(7));
  C_reg_9 = __riscv_vfmacc_vv_f16m1(C_reg_9, A_reg, B_reg_9,(7));
  C_reg_10 = __riscv_vfmacc_vv_f16m1(C_reg_10, A_reg, B_reg_10,(7));
  C_reg_11 = __riscv_vfmacc_vv_f16m1(C_reg_11, A_reg, B_reg_11,(7));
  C_reg_12 = __riscv_vfmacc_vv_f16m1(C_reg_12, A_reg, B_reg_12,(7));
  C_reg_13 = __riscv_vfmacc_vv_f16m1(C_reg_13, A_reg, B_reg_13,(7));
  C_reg_14 = __riscv_vfmacc_vv_f16m1(C_reg_14, A_reg, B_reg_14,(7));
  C_reg_15 = __riscv_vfmacc_vv_f16m1(C_reg_15, A_reg, B_reg_15,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg_8,(7));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg_9,(7));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg_10,(7));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg_11,(7));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg_12,(7));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg_13,(7));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg_14,(7));
__riscv_vse16_v_f16m1(&C[(15) * (ldc)], C_reg_15,(7));
}

// gemm_RVV_7x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 7] @DRAM
// )
void gemm_RVV_7x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
}

// gemm_RVV_7x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 7] @DRAM
// )
void gemm_RVV_7x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
}

// gemm_RVV_7x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 7] @DRAM
// )
void gemm_RVV_7x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
}

// gemm_RVV_7x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 7] @DRAM
// )
void gemm_RVV_7x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
}

// gemm_RVV_7x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 7] @DRAM
// )
void gemm_RVV_7x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
}

// gemm_RVV_7x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 7] @DRAM
// )
void gemm_RVV_7x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
}

// gemm_RVV_7x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 7] @DRAM
// )
void gemm_RVV_7x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(7));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
}

// gemm_RVV_7x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 7] @DRAM
// )
void gemm_RVV_7x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(7));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
}

// gemm_RVV_7x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 7] @DRAM
// )
void gemm_RVV_7x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(7));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(7));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
}

// gemm_RVV_7x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 7] @DRAM
// )
void gemm_RVV_7x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(7));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(7));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
}

// gemm_RVV_7x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 7] @DRAM
// )
void gemm_RVV_7x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(7));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(7));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(7));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
}

// gemm_RVV_7x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 7] @DRAM
// )
void gemm_RVV_7x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(7));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(7));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(7));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
}

// gemm_RVV_7x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 7] @DRAM
// )
void gemm_RVV_7x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(7));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(7));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(7));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(7));
  B_reg_6 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 6],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(7));
}

// gemm_RVV_7x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 7] @DRAM
// )
void gemm_RVV_7x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(7));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(7));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(7));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(7));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(7));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(7));
  B_reg_6 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 6],(7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(7));
}

// gemm_RVV_7x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 7] @DRAM
// )
void gemm_RVV_7x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(7));
}

// gemm_RVV_7x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 7] @DRAM
// )
void gemm_RVV_7x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(7));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(7));
}

// gemm_RVV_7x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 7] @DRAM
// )
void gemm_RVV_7x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(7));
}

// gemm_RVV_7x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 7] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 7] @DRAM
// )
void gemm_RVV_7x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(7));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(7));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(7));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(7));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(7));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(7));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(7));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(7));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(7));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (7));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (7));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (7));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (7));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (7));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (7));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (7));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (7));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(7));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(7));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(7));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(7));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(7));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(7));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(7));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(7));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(7));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(7));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(7));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(7));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(7));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(7));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(7));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(7));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(7));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(7));
}

// gemm_RVV_8x10_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 8] @DRAM
// )
void gemm_RVV_8x10_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(8));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(8));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(8));
}

// gemm_RVV_8x10_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][10, 8] @DRAM
// )
void gemm_RVV_8x10_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(8));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(8));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(8));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(8));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(8));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(8));
}

// gemm_RVV_8x11_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 8] @DRAM
// )
void gemm_RVV_8x11_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(8));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(8));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(8));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(8));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(8));
}

// gemm_RVV_8x11_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][11, 8] @DRAM
// )
void gemm_RVV_8x11_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(8));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(8));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(8));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(8));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(8));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(8));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(8));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(8));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(8));
}

// gemm_RVV_8x12_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 8] @DRAM
// )
void gemm_RVV_8x12_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(8));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(8));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(8));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(8));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(8));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(8));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(8));
}

// gemm_RVV_8x12_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][12, 8] @DRAM
// )
void gemm_RVV_8x12_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(8));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(8));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(8));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(8));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(8));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(4));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(8));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(8));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(8));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(8));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(8));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(8));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(8));
}

// gemm_RVV_8x13_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 8] @DRAM
// )
void gemm_RVV_8x13_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(8));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(8));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(8));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(8));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(8));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(8));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(8));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(8));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(8));
}

// gemm_RVV_8x13_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][13, 8] @DRAM
// )
void gemm_RVV_8x13_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(8));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(8));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(8));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(8));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(8));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(8));
C_reg2_4 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(5));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(8));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(8));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(8));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(8));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(8));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(8));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(8));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(8));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(8));
}

// gemm_RVV_8x14_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 8] @DRAM
// )
void gemm_RVV_8x14_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(8));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(8));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(8));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(8));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(8));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(8));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(8));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(8));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(8));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(8));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(8));
}

// gemm_RVV_8x14_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][14, 8] @DRAM
// )
void gemm_RVV_8x14_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(8));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(8));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(8));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(8));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(8));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(8));
C_reg2_4 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(8));
C_reg2_5 = __riscv_vle16_v_f16m1(&C[(13) * (ldc)],(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(6));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(8));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(8));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(8));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(8));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(8));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(8));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(8));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(8));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(8));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(8));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(8));
}

// gemm_RVV_8x15_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 8] @DRAM
// )
void gemm_RVV_8x15_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
vfloat16m1_t C_reg2_6;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_6 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
vfloat16m1_t B_reg2_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(7));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (8));
  B_reg2_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(8));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(8));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(8));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(8));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(8));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(8));
  C_reg2_6 = __riscv_vfmacc_vv_f16m1(C_reg2_6, A_reg, B_reg2_6,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(8));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(8));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(8));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(8));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(8));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(8));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg2_6,(8));
}

// gemm_RVV_8x15_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][15, 8] @DRAM
// )
void gemm_RVV_8x15_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
vfloat16m1_t C_reg2_1;
vfloat16m1_t C_reg2_2;
vfloat16m1_t C_reg2_3;
vfloat16m1_t C_reg2_4;
vfloat16m1_t C_reg2_5;
vfloat16m1_t C_reg2_6;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(8));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(8));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(8));
C_reg2_1 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(8));
C_reg2_2 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(8));
C_reg2_3 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(8));
C_reg2_4 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(8));
C_reg2_5 = __riscv_vle16_v_f16m1(&C[(13) * (ldc)],(8));
C_reg2_6 = __riscv_vle16_v_f16m1(&C[(14) * (ldc)],(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
vfloat16m1_t B_reg2_1;
vfloat16m1_t B_reg2_2;
vfloat16m1_t B_reg2_3;
vfloat16m1_t B_reg2_4;
vfloat16m1_t B_reg2_5;
vfloat16m1_t B_reg2_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(7));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg2_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg2_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg2_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg2_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  B_reg2_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (8));
  B_reg2_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(8));
  C_reg2_1 = __riscv_vfmacc_vv_f16m1(C_reg2_1, A_reg, B_reg2_1,(8));
  C_reg2_2 = __riscv_vfmacc_vv_f16m1(C_reg2_2, A_reg, B_reg2_2,(8));
  C_reg2_3 = __riscv_vfmacc_vv_f16m1(C_reg2_3, A_reg, B_reg2_3,(8));
  C_reg2_4 = __riscv_vfmacc_vv_f16m1(C_reg2_4, A_reg, B_reg2_4,(8));
  C_reg2_5 = __riscv_vfmacc_vv_f16m1(C_reg2_5, A_reg, B_reg2_5,(8));
  C_reg2_6 = __riscv_vfmacc_vv_f16m1(C_reg2_6, A_reg, B_reg2_6,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(8));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg2_1,(8));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg2_2,(8));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg2_3,(8));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg2_4,(8));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg2_5,(8));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg2_6,(8));
}

// gemm_RVV_8x16_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 8] @DRAM
// )
void gemm_RVV_8x16_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_8 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_9 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_10 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_11 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_12 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_13 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_14 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_15 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg_8;
vfloat16m1_t B_reg_9;
vfloat16m1_t B_reg_10;
vfloat16m1_t B_reg_11;
vfloat16m1_t B_reg_12;
vfloat16m1_t B_reg_13;
vfloat16m1_t B_reg_14;
vfloat16m1_t B_reg_15;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(8));
  B_reg_8 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg_9 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg_10 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg_11 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg_12 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  B_reg_13 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (8));
  B_reg_14 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (8));
  B_reg_15 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg_8 = __riscv_vfmacc_vv_f16m1(C_reg_8, A_reg, B_reg_8,(8));
  C_reg_9 = __riscv_vfmacc_vv_f16m1(C_reg_9, A_reg, B_reg_9,(8));
  C_reg_10 = __riscv_vfmacc_vv_f16m1(C_reg_10, A_reg, B_reg_10,(8));
  C_reg_11 = __riscv_vfmacc_vv_f16m1(C_reg_11, A_reg, B_reg_11,(8));
  C_reg_12 = __riscv_vfmacc_vv_f16m1(C_reg_12, A_reg, B_reg_12,(8));
  C_reg_13 = __riscv_vfmacc_vv_f16m1(C_reg_13, A_reg, B_reg_13,(8));
  C_reg_14 = __riscv_vfmacc_vv_f16m1(C_reg_14, A_reg, B_reg_14,(8));
  C_reg_15 = __riscv_vfmacc_vv_f16m1(C_reg_15, A_reg, B_reg_15,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg_8,(8));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg_9,(8));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg_10,(8));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg_11,(8));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg_12,(8));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg_13,(8));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg_14,(8));
__riscv_vse16_v_f16m1(&C[(15) * (ldc)], C_reg_15,(8));
}

// gemm_RVV_8x16_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][16, 8] @DRAM
// )
void gemm_RVV_8x16_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg_8;
vfloat16m1_t C_reg_9;
vfloat16m1_t C_reg_10;
vfloat16m1_t C_reg_11;
vfloat16m1_t C_reg_12;
vfloat16m1_t C_reg_13;
vfloat16m1_t C_reg_14;
vfloat16m1_t C_reg_15;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(8));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(8));
C_reg_8 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(8));
C_reg_9 = __riscv_vle16_v_f16m1(&C[(9) * (ldc)],(8));
C_reg_10 = __riscv_vle16_v_f16m1(&C[(10) * (ldc)],(8));
C_reg_11 = __riscv_vle16_v_f16m1(&C[(11) * (ldc)],(8));
C_reg_12 = __riscv_vle16_v_f16m1(&C[(12) * (ldc)],(8));
C_reg_13 = __riscv_vle16_v_f16m1(&C[(13) * (ldc)],(8));
C_reg_14 = __riscv_vle16_v_f16m1(&C[(14) * (ldc)],(8));
C_reg_15 = __riscv_vle16_v_f16m1(&C[(15) * (ldc)],(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg_8;
vfloat16m1_t B_reg_9;
vfloat16m1_t B_reg_10;
vfloat16m1_t B_reg_11;
vfloat16m1_t B_reg_12;
vfloat16m1_t B_reg_13;
vfloat16m1_t B_reg_14;
vfloat16m1_t B_reg_15;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(8));
  B_reg_8 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg_9 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg_10 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg_11 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg_12 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  B_reg_13 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (8));
  B_reg_14 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (8));
  B_reg_15 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg_8 = __riscv_vfmacc_vv_f16m1(C_reg_8, A_reg, B_reg_8,(8));
  C_reg_9 = __riscv_vfmacc_vv_f16m1(C_reg_9, A_reg, B_reg_9,(8));
  C_reg_10 = __riscv_vfmacc_vv_f16m1(C_reg_10, A_reg, B_reg_10,(8));
  C_reg_11 = __riscv_vfmacc_vv_f16m1(C_reg_11, A_reg, B_reg_11,(8));
  C_reg_12 = __riscv_vfmacc_vv_f16m1(C_reg_12, A_reg, B_reg_12,(8));
  C_reg_13 = __riscv_vfmacc_vv_f16m1(C_reg_13, A_reg, B_reg_13,(8));
  C_reg_14 = __riscv_vfmacc_vv_f16m1(C_reg_14, A_reg, B_reg_14,(8));
  C_reg_15 = __riscv_vfmacc_vv_f16m1(C_reg_15, A_reg, B_reg_15,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg_8,(8));
__riscv_vse16_v_f16m1(&C[(9) * (ldc)], C_reg_9,(8));
__riscv_vse16_v_f16m1(&C[(10) * (ldc)], C_reg_10,(8));
__riscv_vse16_v_f16m1(&C[(11) * (ldc)], C_reg_11,(8));
__riscv_vse16_v_f16m1(&C[(12) * (ldc)], C_reg_12,(8));
__riscv_vse16_v_f16m1(&C[(13) * (ldc)], C_reg_13,(8));
__riscv_vse16_v_f16m1(&C[(14) * (ldc)], C_reg_14,(8));
__riscv_vse16_v_f16m1(&C[(15) * (ldc)], C_reg_15,(8));
}

// gemm_RVV_8x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 8] @DRAM
// )
void gemm_RVV_8x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
}

// gemm_RVV_8x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][1, 8] @DRAM
// )
void gemm_RVV_8x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
}

// gemm_RVV_8x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 8] @DRAM
// )
void gemm_RVV_8x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
}

// gemm_RVV_8x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][2, 8] @DRAM
// )
void gemm_RVV_8x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
}

// gemm_RVV_8x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 8] @DRAM
// )
void gemm_RVV_8x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
}

// gemm_RVV_8x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][3, 8] @DRAM
// )
void gemm_RVV_8x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
}

// gemm_RVV_8x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 8] @DRAM
// )
void gemm_RVV_8x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(8));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
}

// gemm_RVV_8x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][4, 8] @DRAM
// )
void gemm_RVV_8x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(8));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
}

// gemm_RVV_8x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 8] @DRAM
// )
void gemm_RVV_8x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(8));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(8));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
}

// gemm_RVV_8x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][5, 8] @DRAM
// )
void gemm_RVV_8x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(8));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(8));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
}

// gemm_RVV_8x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 8] @DRAM
// )
void gemm_RVV_8x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(8));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(8));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(8));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
}

// gemm_RVV_8x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][6, 8] @DRAM
// )
void gemm_RVV_8x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(8));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(8));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(8));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
}

// gemm_RVV_8x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 8] @DRAM
// )
void gemm_RVV_8x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(8));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(8));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(8));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(8));
  B_reg_6 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 6],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(8));
}

// gemm_RVV_8x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][7, 8] @DRAM
// )
void gemm_RVV_8x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_reg_0 = __riscv_vfmv_v_f_f16m1(B[(k) * (16)],(8));
  B_reg_1 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 1],(8));
  B_reg_2 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 2],(8));
  B_reg_3 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 3],(8));
  B_reg_4 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 4],(8));
  B_reg_5 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 5],(8));
  B_reg_6 = __riscv_vfmv_v_f_f16m1(B[(k) * (16) + 6],(8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(8));
}

// gemm_RVV_8x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 8] @DRAM
// )
void gemm_RVV_8x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(8));
}

// gemm_RVV_8x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 8] @DRAM
// )
void gemm_RVV_8x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(8));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(8));
}

// gemm_RVV_8x9_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 8] @DRAM
// )
void gemm_RVV_8x9_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_1 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_2 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_3 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_4 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_5 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_6 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg_7 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
C_reg2_0 = __riscv_vfmv_v_f_f16m1(0.0f,(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(8));
}

// gemm_RVV_8x9_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][9, 8] @DRAM
// )
void gemm_RVV_8x9_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat16m1_t C_reg_0;
vfloat16m1_t C_reg_1;
vfloat16m1_t C_reg_2;
vfloat16m1_t C_reg_3;
vfloat16m1_t C_reg_4;
vfloat16m1_t C_reg_5;
vfloat16m1_t C_reg_6;
vfloat16m1_t C_reg_7;
vfloat16m1_t C_reg2_0;
C_reg_0 = __riscv_vle16_v_f16m1(&C[0],(8));
C_reg_1 = __riscv_vle16_v_f16m1(&C[ldc],(8));
C_reg_2 = __riscv_vle16_v_f16m1(&C[(2) * (ldc)],(8));
C_reg_3 = __riscv_vle16_v_f16m1(&C[(3) * (ldc)],(8));
C_reg_4 = __riscv_vle16_v_f16m1(&C[(4) * (ldc)],(8));
C_reg_5 = __riscv_vle16_v_f16m1(&C[(5) * (ldc)],(8));
C_reg_6 = __riscv_vle16_v_f16m1(&C[(6) * (ldc)],(8));
C_reg_7 = __riscv_vle16_v_f16m1(&C[(7) * (ldc)],(8));
C_reg2_0 = __riscv_vle16_v_f16m1(&C[(8) * (ldc)],(8));
vfloat16m1_t A_reg;
vfloat16m1_t B_tmp;
vfloat16m1_t B_reg_0;
vfloat16m1_t B_reg_1;
vfloat16m1_t B_reg_2;
vfloat16m1_t B_reg_3;
vfloat16m1_t B_reg_4;
vfloat16m1_t B_reg_5;
vfloat16m1_t B_reg_6;
vfloat16m1_t B_reg_7;
vfloat16m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  A_reg = __riscv_vle16_v_f16m1(&A[(k) * (8)],(8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16)],(8));
  B_reg_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  B_reg_1 = __riscv_vrgather_vx_f16m1(B_tmp, (1), (8));
  B_reg_2 = __riscv_vrgather_vx_f16m1(B_tmp, (2), (8));
  B_reg_3 = __riscv_vrgather_vx_f16m1(B_tmp, (3), (8));
  B_reg_4 = __riscv_vrgather_vx_f16m1(B_tmp, (4), (8));
  B_reg_5 = __riscv_vrgather_vx_f16m1(B_tmp, (5), (8));
  B_reg_6 = __riscv_vrgather_vx_f16m1(B_tmp, (6), (8));
  B_reg_7 = __riscv_vrgather_vx_f16m1(B_tmp, (7), (8));
  B_tmp = __riscv_vle16_v_f16m1(&B[(k) * (16) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f16m1(B_tmp, (0), (8));
  C_reg_0 = __riscv_vfmacc_vv_f16m1(C_reg_0, A_reg, B_reg_0,(8));
  C_reg_1 = __riscv_vfmacc_vv_f16m1(C_reg_1, A_reg, B_reg_1,(8));
  C_reg_2 = __riscv_vfmacc_vv_f16m1(C_reg_2, A_reg, B_reg_2,(8));
  C_reg_3 = __riscv_vfmacc_vv_f16m1(C_reg_3, A_reg, B_reg_3,(8));
  C_reg_4 = __riscv_vfmacc_vv_f16m1(C_reg_4, A_reg, B_reg_4,(8));
  C_reg_5 = __riscv_vfmacc_vv_f16m1(C_reg_5, A_reg, B_reg_5,(8));
  C_reg_6 = __riscv_vfmacc_vv_f16m1(C_reg_6, A_reg, B_reg_6,(8));
  C_reg_7 = __riscv_vfmacc_vv_f16m1(C_reg_7, A_reg, B_reg_7,(8));
  C_reg2_0 = __riscv_vfmacc_vv_f16m1(C_reg2_0, A_reg, B_reg2_0,(8));
}
__riscv_vse16_v_f16m1(&C[0], C_reg_0,(8));
__riscv_vse16_v_f16m1(&C[ldc], C_reg_1,(8));
__riscv_vse16_v_f16m1(&C[(2) * (ldc)], C_reg_2,(8));
__riscv_vse16_v_f16m1(&C[(3) * (ldc)], C_reg_3,(8));
__riscv_vse16_v_f16m1(&C[(4) * (ldc)], C_reg_4,(8));
__riscv_vse16_v_f16m1(&C[(5) * (ldc)], C_reg_5,(8));
__riscv_vse16_v_f16m1(&C[(6) * (ldc)], C_reg_6,(8));
__riscv_vse16_v_f16m1(&C[(7) * (ldc)], C_reg_7,(8));
__riscv_vse16_v_f16m1(&C[(8) * (ldc)], C_reg2_0,(8));
}


/* relying on the following instruction..."
rvv_broadcast_8xf16(dst,src,vl)
{dst_data} = __riscv_vfmv_v_f_f16m1({src_data},{vl});
*/

/* relying on the following instruction..."
rvv_broadcast_8xf16_0(dst,vl)
{dst_data} = __riscv_vfmv_v_f_f16m1(0.0f,{vl});
*/

/* relying on the following instruction..."
rvv_gather_8xf16(dst,src,imm,vl)
{dst_data} = __riscv_vrgather_vx_f16m1({src_data}, {imm}, {vl});
*/

/* relying on the following instruction..."
rvv_vfmacc_8xf16_8xf16(dst,lhs,rhs,vl)
{dst_data} = __riscv_vfmacc_vv_f16m1({dst_data}, {lhs_data}, {rhs_data},{vl});
*/

/* relying on the following instruction..."
rvv_vld_8xf16(dst,src,vl)
{dst_data} = __riscv_vle16_v_f16m1(&{src_data},{vl});
*/

/* relying on the following instruction..."
rvv_vst_8xf16(dst,src,vl)
__riscv_vse16_v_f16m1(&{dst_data}, {src_data},{vl});
*/