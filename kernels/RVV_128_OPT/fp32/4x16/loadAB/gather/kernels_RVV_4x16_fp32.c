#include "kernels_RVV_4x16_fp32.h"



#include <stdio.h>
#include <stdlib.h>

#include <riscv_vector.h>


// gemm_RVV_1x10_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 1] @DRAM
// )
void gemm_RVV_1x10_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(1));
}

// gemm_RVV_1x10_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 1] @DRAM
// )
void gemm_RVV_1x10_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(1));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(1));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(1));
}

// gemm_RVV_1x11_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 1] @DRAM
// )
void gemm_RVV_1x11_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(1));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(1));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(1));
}

// gemm_RVV_1x11_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 1] @DRAM
// )
void gemm_RVV_1x11_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(1));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(1));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(1));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(1));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(1));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(1));
}

// gemm_RVV_1x12_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 1] @DRAM
// )
void gemm_RVV_1x12_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_9 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_10 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_11 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(1));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(1));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(1));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(1));
}

// gemm_RVV_1x12_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 1] @DRAM
// )
void gemm_RVV_1x12_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(1));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(1));
C_reg_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(1));
C_reg_10 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(1));
C_reg_11 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(1));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(1));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(1));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(1));
}

// gemm_RVV_1x13_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 1] @DRAM
// )
void gemm_RVV_1x13_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_9 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_10 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_11 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(1));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(1));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(1));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(1));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_0,(1));
}

// gemm_RVV_1x13_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 1] @DRAM
// )
void gemm_RVV_1x13_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(1));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(1));
C_reg_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(1));
C_reg_10 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(1));
C_reg_11 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(1));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(1));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(1));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(1));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(1));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_0,(1));
}

// gemm_RVV_1x14_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 1] @DRAM
// )
void gemm_RVV_1x14_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_9 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_10 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_11 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(1));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(1));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(1));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(1));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_0,(1));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_1,(1));
}

// gemm_RVV_1x14_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 1] @DRAM
// )
void gemm_RVV_1x14_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(1));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(1));
C_reg_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(1));
C_reg_10 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(1));
C_reg_11 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(1));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(1));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(13) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(1));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(1));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(1));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(1));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_0,(1));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_1,(1));
}

// gemm_RVV_1x15_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 1] @DRAM
// )
void gemm_RVV_1x15_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_9 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_10 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_11 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(1));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(1));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(1));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(1));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(1));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_0,(1));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_1,(1));
__riscv_vse32_v_f32m1(&C[(14) * (ldc)], C_reg2_2,(1));
}

// gemm_RVV_1x15_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 1] @DRAM
// )
void gemm_RVV_1x15_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(1));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(1));
C_reg_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(1));
C_reg_10 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(1));
C_reg_11 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(1));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(1));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(13) * (ldc)],(1));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(14) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(1));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(1));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(1));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(1));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(1));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_0,(1));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_1,(1));
__riscv_vse32_v_f32m1(&C[(14) * (ldc)], C_reg2_2,(1));
}

// gemm_RVV_1x16_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 1] @DRAM
// )
void gemm_RVV_1x16_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg_12;
vfloat32m1_t C_reg_13;
vfloat32m1_t C_reg_14;
vfloat32m1_t C_reg_15;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_9 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_10 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_11 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_12 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_13 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_14 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_15 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg_12;
vfloat32m1_t B_reg_13;
vfloat32m1_t B_reg_14;
vfloat32m1_t B_reg_15;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(4));
  B_reg_12 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_13 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_14 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_15 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(1));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(1));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(1));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(1));
  C_reg_12 = __riscv_vfmacc_vv_f32m1(C_reg_12, A_reg, B_reg_12,(1));
  C_reg_13 = __riscv_vfmacc_vv_f32m1(C_reg_13, A_reg, B_reg_13,(1));
  C_reg_14 = __riscv_vfmacc_vv_f32m1(C_reg_14, A_reg, B_reg_14,(1));
  C_reg_15 = __riscv_vfmacc_vv_f32m1(C_reg_15, A_reg, B_reg_15,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(1));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg_12,(1));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg_13,(1));
__riscv_vse32_v_f32m1(&C[(14) * (ldc)], C_reg_14,(1));
__riscv_vse32_v_f32m1(&C[(15) * (ldc)], C_reg_15,(1));
}

// gemm_RVV_1x16_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 1] @DRAM
// )
void gemm_RVV_1x16_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg_12;
vfloat32m1_t C_reg_13;
vfloat32m1_t C_reg_14;
vfloat32m1_t C_reg_15;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(1));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(1));
C_reg_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(1));
C_reg_10 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(1));
C_reg_11 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(1));
C_reg_12 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(1));
C_reg_13 = __riscv_vle32_v_f32m1(&C[(13) * (ldc)],(1));
C_reg_14 = __riscv_vle32_v_f32m1(&C[(14) * (ldc)],(1));
C_reg_15 = __riscv_vle32_v_f32m1(&C[(15) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg_12;
vfloat32m1_t B_reg_13;
vfloat32m1_t B_reg_14;
vfloat32m1_t B_reg_15;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(4));
  B_reg_12 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_13 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_14 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_15 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(1));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(1));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(1));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(1));
  C_reg_12 = __riscv_vfmacc_vv_f32m1(C_reg_12, A_reg, B_reg_12,(1));
  C_reg_13 = __riscv_vfmacc_vv_f32m1(C_reg_13, A_reg, B_reg_13,(1));
  C_reg_14 = __riscv_vfmacc_vv_f32m1(C_reg_14, A_reg, B_reg_14,(1));
  C_reg_15 = __riscv_vfmacc_vv_f32m1(C_reg_15, A_reg, B_reg_15,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(1));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(1));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(1));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(1));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg_12,(1));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg_13,(1));
__riscv_vse32_v_f32m1(&C[(14) * (ldc)], C_reg_14,(1));
__riscv_vse32_v_f32m1(&C[(15) * (ldc)], C_reg_15,(1));
}

// gemm_RVV_1x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 1] @DRAM
// )
void gemm_RVV_1x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (16)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
}

// gemm_RVV_1x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 1] @DRAM
// )
void gemm_RVV_1x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (16)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
}

// gemm_RVV_1x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 1] @DRAM
// )
void gemm_RVV_1x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (16)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (16) + 1],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
}

// gemm_RVV_1x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 1] @DRAM
// )
void gemm_RVV_1x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (16)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (16) + 1],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
}

// gemm_RVV_1x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 1] @DRAM
// )
void gemm_RVV_1x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (16)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (16) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (16) + 2],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
}

// gemm_RVV_1x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 1] @DRAM
// )
void gemm_RVV_1x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (16)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (16) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (16) + 2],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
}

// gemm_RVV_1x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 1] @DRAM
// )
void gemm_RVV_1x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
}

// gemm_RVV_1x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 1] @DRAM
// )
void gemm_RVV_1x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
}

// gemm_RVV_1x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 1] @DRAM
// )
void gemm_RVV_1x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg2_0,(1));
}

// gemm_RVV_1x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 1] @DRAM
// )
void gemm_RVV_1x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg2_0,(1));
}

// gemm_RVV_1x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 1] @DRAM
// )
void gemm_RVV_1x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg2_0,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg2_1,(1));
}

// gemm_RVV_1x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 1] @DRAM
// )
void gemm_RVV_1x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(1));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg2_0,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg2_1,(1));
}

// gemm_RVV_1x7_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 1] @DRAM
// )
void gemm_RVV_1x7_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(1));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg2_0,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg2_1,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg2_2,(1));
}

// gemm_RVV_1x7_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 1] @DRAM
// )
void gemm_RVV_1x7_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(1));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(1));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(1));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg2_0,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg2_1,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg2_2,(1));
}

// gemm_RVV_1x8_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 1] @DRAM
// )
void gemm_RVV_1x8_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
}

// gemm_RVV_1x8_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 1] @DRAM
// )
void gemm_RVV_1x8_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
}

// gemm_RVV_1x9_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 1] @DRAM
// )
void gemm_RVV_1x9_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(1));
}

// gemm_RVV_1x9_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 1] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 1] @DRAM
// )
void gemm_RVV_1x9_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(1));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(1));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(1));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(1));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(1));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(1));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(1));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(1));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(1));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(1));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(1));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(1));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(1));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(1));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(1));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(1));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(1));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(1));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(1));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(1));
}

// gemm_RVV_2x10_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 2] @DRAM
// )
void gemm_RVV_2x10_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(2));
}

// gemm_RVV_2x10_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 2] @DRAM
// )
void gemm_RVV_2x10_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(2));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(2));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(2));
}

// gemm_RVV_2x11_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 2] @DRAM
// )
void gemm_RVV_2x11_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(2));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(2));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(2));
}

// gemm_RVV_2x11_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 2] @DRAM
// )
void gemm_RVV_2x11_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(2));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(2));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(2));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(2));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(2));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(2));
}

// gemm_RVV_2x12_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 2] @DRAM
// )
void gemm_RVV_2x12_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_9 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_10 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_11 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(2));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(2));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(2));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(2));
}

// gemm_RVV_2x12_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 2] @DRAM
// )
void gemm_RVV_2x12_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(2));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(2));
C_reg_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(2));
C_reg_10 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(2));
C_reg_11 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(2));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(2));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(2));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(2));
}

// gemm_RVV_2x13_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 2] @DRAM
// )
void gemm_RVV_2x13_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_9 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_10 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_11 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(2));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(2));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(2));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(2));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_0,(2));
}

// gemm_RVV_2x13_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 2] @DRAM
// )
void gemm_RVV_2x13_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(2));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(2));
C_reg_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(2));
C_reg_10 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(2));
C_reg_11 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(2));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(2));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(2));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(2));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(2));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_0,(2));
}

// gemm_RVV_2x14_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 2] @DRAM
// )
void gemm_RVV_2x14_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_9 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_10 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_11 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(2));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(2));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(2));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(2));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_0,(2));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_1,(2));
}

// gemm_RVV_2x14_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 2] @DRAM
// )
void gemm_RVV_2x14_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(2));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(2));
C_reg_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(2));
C_reg_10 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(2));
C_reg_11 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(2));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(2));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(13) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(2));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(2));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(2));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(2));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_0,(2));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_1,(2));
}

// gemm_RVV_2x15_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 2] @DRAM
// )
void gemm_RVV_2x15_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_9 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_10 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_11 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(2));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(2));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(2));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(2));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(2));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_0,(2));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_1,(2));
__riscv_vse32_v_f32m1(&C[(14) * (ldc)], C_reg2_2,(2));
}

// gemm_RVV_2x15_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 2] @DRAM
// )
void gemm_RVV_2x15_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(2));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(2));
C_reg_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(2));
C_reg_10 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(2));
C_reg_11 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(2));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(2));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(13) * (ldc)],(2));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(14) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(2));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(2));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(2));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(2));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(2));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_0,(2));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_1,(2));
__riscv_vse32_v_f32m1(&C[(14) * (ldc)], C_reg2_2,(2));
}

// gemm_RVV_2x16_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 2] @DRAM
// )
void gemm_RVV_2x16_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg_12;
vfloat32m1_t C_reg_13;
vfloat32m1_t C_reg_14;
vfloat32m1_t C_reg_15;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_9 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_10 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_11 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_12 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_13 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_14 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_15 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg_12;
vfloat32m1_t B_reg_13;
vfloat32m1_t B_reg_14;
vfloat32m1_t B_reg_15;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(4));
  B_reg_12 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_13 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_14 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_15 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(2));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(2));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(2));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(2));
  C_reg_12 = __riscv_vfmacc_vv_f32m1(C_reg_12, A_reg, B_reg_12,(2));
  C_reg_13 = __riscv_vfmacc_vv_f32m1(C_reg_13, A_reg, B_reg_13,(2));
  C_reg_14 = __riscv_vfmacc_vv_f32m1(C_reg_14, A_reg, B_reg_14,(2));
  C_reg_15 = __riscv_vfmacc_vv_f32m1(C_reg_15, A_reg, B_reg_15,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(2));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg_12,(2));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg_13,(2));
__riscv_vse32_v_f32m1(&C[(14) * (ldc)], C_reg_14,(2));
__riscv_vse32_v_f32m1(&C[(15) * (ldc)], C_reg_15,(2));
}

// gemm_RVV_2x16_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 2] @DRAM
// )
void gemm_RVV_2x16_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg_12;
vfloat32m1_t C_reg_13;
vfloat32m1_t C_reg_14;
vfloat32m1_t C_reg_15;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(2));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(2));
C_reg_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(2));
C_reg_10 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(2));
C_reg_11 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(2));
C_reg_12 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(2));
C_reg_13 = __riscv_vle32_v_f32m1(&C[(13) * (ldc)],(2));
C_reg_14 = __riscv_vle32_v_f32m1(&C[(14) * (ldc)],(2));
C_reg_15 = __riscv_vle32_v_f32m1(&C[(15) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg_12;
vfloat32m1_t B_reg_13;
vfloat32m1_t B_reg_14;
vfloat32m1_t B_reg_15;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(4));
  B_reg_12 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_13 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_14 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_15 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(2));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(2));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(2));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(2));
  C_reg_12 = __riscv_vfmacc_vv_f32m1(C_reg_12, A_reg, B_reg_12,(2));
  C_reg_13 = __riscv_vfmacc_vv_f32m1(C_reg_13, A_reg, B_reg_13,(2));
  C_reg_14 = __riscv_vfmacc_vv_f32m1(C_reg_14, A_reg, B_reg_14,(2));
  C_reg_15 = __riscv_vfmacc_vv_f32m1(C_reg_15, A_reg, B_reg_15,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(2));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(2));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(2));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(2));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg_12,(2));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg_13,(2));
__riscv_vse32_v_f32m1(&C[(14) * (ldc)], C_reg_14,(2));
__riscv_vse32_v_f32m1(&C[(15) * (ldc)], C_reg_15,(2));
}

// gemm_RVV_2x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 2] @DRAM
// )
void gemm_RVV_2x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (16)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
}

// gemm_RVV_2x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 2] @DRAM
// )
void gemm_RVV_2x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (16)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
}

// gemm_RVV_2x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 2] @DRAM
// )
void gemm_RVV_2x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (16)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (16) + 1],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
}

// gemm_RVV_2x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 2] @DRAM
// )
void gemm_RVV_2x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (16)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (16) + 1],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
}

// gemm_RVV_2x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 2] @DRAM
// )
void gemm_RVV_2x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (16)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (16) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (16) + 2],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
}

// gemm_RVV_2x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 2] @DRAM
// )
void gemm_RVV_2x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (16)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (16) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (16) + 2],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
}

// gemm_RVV_2x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 2] @DRAM
// )
void gemm_RVV_2x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
}

// gemm_RVV_2x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 2] @DRAM
// )
void gemm_RVV_2x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
}

// gemm_RVV_2x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 2] @DRAM
// )
void gemm_RVV_2x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg2_0,(2));
}

// gemm_RVV_2x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 2] @DRAM
// )
void gemm_RVV_2x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg2_0,(2));
}

// gemm_RVV_2x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 2] @DRAM
// )
void gemm_RVV_2x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg2_0,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg2_1,(2));
}

// gemm_RVV_2x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 2] @DRAM
// )
void gemm_RVV_2x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(2));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg2_0,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg2_1,(2));
}

// gemm_RVV_2x7_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 2] @DRAM
// )
void gemm_RVV_2x7_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(2));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg2_0,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg2_1,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg2_2,(2));
}

// gemm_RVV_2x7_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 2] @DRAM
// )
void gemm_RVV_2x7_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(2));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(2));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(2));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg2_0,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg2_1,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg2_2,(2));
}

// gemm_RVV_2x8_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 2] @DRAM
// )
void gemm_RVV_2x8_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
}

// gemm_RVV_2x8_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 2] @DRAM
// )
void gemm_RVV_2x8_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
}

// gemm_RVV_2x9_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 2] @DRAM
// )
void gemm_RVV_2x9_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(2));
}

// gemm_RVV_2x9_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 2] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 2] @DRAM
// )
void gemm_RVV_2x9_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(2));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(2));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(2));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(2));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(2));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(2));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(2));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(2));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(2));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(2));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(2));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(2));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(2));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(2));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(2));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(2));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(2));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(2));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(2));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(2));
}

// gemm_RVV_3x10_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 3] @DRAM
// )
void gemm_RVV_3x10_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(3));
}

// gemm_RVV_3x10_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 3] @DRAM
// )
void gemm_RVV_3x10_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(3));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(3));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(3));
}

// gemm_RVV_3x11_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 3] @DRAM
// )
void gemm_RVV_3x11_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(3));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(3));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(3));
}

// gemm_RVV_3x11_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 3] @DRAM
// )
void gemm_RVV_3x11_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(3));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(3));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(3));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(3));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(3));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(3));
}

// gemm_RVV_3x12_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 3] @DRAM
// )
void gemm_RVV_3x12_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_9 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_10 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_11 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(3));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(3));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(3));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(3));
}

// gemm_RVV_3x12_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 3] @DRAM
// )
void gemm_RVV_3x12_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(3));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(3));
C_reg_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(3));
C_reg_10 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(3));
C_reg_11 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(3));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(3));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(3));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(3));
}

// gemm_RVV_3x13_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 3] @DRAM
// )
void gemm_RVV_3x13_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_9 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_10 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_11 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(3));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(3));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(3));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(3));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_0,(3));
}

// gemm_RVV_3x13_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 3] @DRAM
// )
void gemm_RVV_3x13_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(3));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(3));
C_reg_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(3));
C_reg_10 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(3));
C_reg_11 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(3));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(3));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(3));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(3));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(3));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_0,(3));
}

// gemm_RVV_3x14_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 3] @DRAM
// )
void gemm_RVV_3x14_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_9 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_10 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_11 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(3));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(3));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(3));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(3));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_0,(3));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_1,(3));
}

// gemm_RVV_3x14_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 3] @DRAM
// )
void gemm_RVV_3x14_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(3));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(3));
C_reg_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(3));
C_reg_10 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(3));
C_reg_11 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(3));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(3));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(13) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(3));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(3));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(3));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(3));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_0,(3));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_1,(3));
}

// gemm_RVV_3x15_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 3] @DRAM
// )
void gemm_RVV_3x15_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_9 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_10 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_11 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(3));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(3));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(3));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(3));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(3));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_0,(3));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_1,(3));
__riscv_vse32_v_f32m1(&C[(14) * (ldc)], C_reg2_2,(3));
}

// gemm_RVV_3x15_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 3] @DRAM
// )
void gemm_RVV_3x15_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(3));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(3));
C_reg_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(3));
C_reg_10 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(3));
C_reg_11 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(3));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(3));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(13) * (ldc)],(3));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(14) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(3));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(3));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(3));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(3));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(3));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_0,(3));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_1,(3));
__riscv_vse32_v_f32m1(&C[(14) * (ldc)], C_reg2_2,(3));
}

// gemm_RVV_3x16_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 3] @DRAM
// )
void gemm_RVV_3x16_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg_12;
vfloat32m1_t C_reg_13;
vfloat32m1_t C_reg_14;
vfloat32m1_t C_reg_15;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_9 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_10 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_11 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_12 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_13 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_14 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_15 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg_12;
vfloat32m1_t B_reg_13;
vfloat32m1_t B_reg_14;
vfloat32m1_t B_reg_15;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(4));
  B_reg_12 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_13 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_14 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_15 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(3));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(3));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(3));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(3));
  C_reg_12 = __riscv_vfmacc_vv_f32m1(C_reg_12, A_reg, B_reg_12,(3));
  C_reg_13 = __riscv_vfmacc_vv_f32m1(C_reg_13, A_reg, B_reg_13,(3));
  C_reg_14 = __riscv_vfmacc_vv_f32m1(C_reg_14, A_reg, B_reg_14,(3));
  C_reg_15 = __riscv_vfmacc_vv_f32m1(C_reg_15, A_reg, B_reg_15,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(3));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg_12,(3));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg_13,(3));
__riscv_vse32_v_f32m1(&C[(14) * (ldc)], C_reg_14,(3));
__riscv_vse32_v_f32m1(&C[(15) * (ldc)], C_reg_15,(3));
}

// gemm_RVV_3x16_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 3] @DRAM
// )
void gemm_RVV_3x16_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg_12;
vfloat32m1_t C_reg_13;
vfloat32m1_t C_reg_14;
vfloat32m1_t C_reg_15;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(3));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(3));
C_reg_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(3));
C_reg_10 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(3));
C_reg_11 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(3));
C_reg_12 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(3));
C_reg_13 = __riscv_vle32_v_f32m1(&C[(13) * (ldc)],(3));
C_reg_14 = __riscv_vle32_v_f32m1(&C[(14) * (ldc)],(3));
C_reg_15 = __riscv_vle32_v_f32m1(&C[(15) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg_12;
vfloat32m1_t B_reg_13;
vfloat32m1_t B_reg_14;
vfloat32m1_t B_reg_15;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(4));
  B_reg_12 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_13 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_14 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_15 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(3));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(3));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(3));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(3));
  C_reg_12 = __riscv_vfmacc_vv_f32m1(C_reg_12, A_reg, B_reg_12,(3));
  C_reg_13 = __riscv_vfmacc_vv_f32m1(C_reg_13, A_reg, B_reg_13,(3));
  C_reg_14 = __riscv_vfmacc_vv_f32m1(C_reg_14, A_reg, B_reg_14,(3));
  C_reg_15 = __riscv_vfmacc_vv_f32m1(C_reg_15, A_reg, B_reg_15,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(3));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(3));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(3));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(3));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg_12,(3));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg_13,(3));
__riscv_vse32_v_f32m1(&C[(14) * (ldc)], C_reg_14,(3));
__riscv_vse32_v_f32m1(&C[(15) * (ldc)], C_reg_15,(3));
}

// gemm_RVV_3x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 3] @DRAM
// )
void gemm_RVV_3x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (16)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
}

// gemm_RVV_3x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 3] @DRAM
// )
void gemm_RVV_3x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (16)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
}

// gemm_RVV_3x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 3] @DRAM
// )
void gemm_RVV_3x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (16)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (16) + 1],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
}

// gemm_RVV_3x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 3] @DRAM
// )
void gemm_RVV_3x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (16)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (16) + 1],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
}

// gemm_RVV_3x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 3] @DRAM
// )
void gemm_RVV_3x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (16)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (16) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (16) + 2],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
}

// gemm_RVV_3x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 3] @DRAM
// )
void gemm_RVV_3x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (16)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (16) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (16) + 2],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
}

// gemm_RVV_3x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 3] @DRAM
// )
void gemm_RVV_3x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
}

// gemm_RVV_3x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 3] @DRAM
// )
void gemm_RVV_3x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
}

// gemm_RVV_3x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 3] @DRAM
// )
void gemm_RVV_3x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg2_0,(3));
}

// gemm_RVV_3x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 3] @DRAM
// )
void gemm_RVV_3x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg2_0,(3));
}

// gemm_RVV_3x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 3] @DRAM
// )
void gemm_RVV_3x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg2_0,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg2_1,(3));
}

// gemm_RVV_3x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 3] @DRAM
// )
void gemm_RVV_3x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(3));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg2_0,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg2_1,(3));
}

// gemm_RVV_3x7_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 3] @DRAM
// )
void gemm_RVV_3x7_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(3));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg2_0,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg2_1,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg2_2,(3));
}

// gemm_RVV_3x7_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 3] @DRAM
// )
void gemm_RVV_3x7_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(3));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(3));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(3));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg2_0,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg2_1,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg2_2,(3));
}

// gemm_RVV_3x8_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 3] @DRAM
// )
void gemm_RVV_3x8_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
}

// gemm_RVV_3x8_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 3] @DRAM
// )
void gemm_RVV_3x8_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
}

// gemm_RVV_3x9_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 3] @DRAM
// )
void gemm_RVV_3x9_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(3));
}

// gemm_RVV_3x9_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 3] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 3] @DRAM
// )
void gemm_RVV_3x9_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(3));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(3));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(3));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(3));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(3));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(3));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(3));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(3));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(3));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(3));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(3));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(3));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(3));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(3));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(3));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(3));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(3));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(3));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(3));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(3));
}

// gemm_RVV_4x10_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 4] @DRAM
// )
void gemm_RVV_4x10_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(4));
}

// gemm_RVV_4x10_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 10] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][10, 4] @DRAM
// )
void gemm_RVV_4x10_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(4));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(4));
}

// gemm_RVV_4x11_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 4] @DRAM
// )
void gemm_RVV_4x11_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(4));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(4));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(4));
}

// gemm_RVV_4x11_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 11] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][11, 4] @DRAM
// )
void gemm_RVV_4x11_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(4));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(4));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(4));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg2_1,(4));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg2_2,(4));
}

// gemm_RVV_4x12_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 4] @DRAM
// )
void gemm_RVV_4x12_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_9 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_10 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_11 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(4));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(4));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(4));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(4));
}

// gemm_RVV_4x12_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 4] @DRAM
// )
void gemm_RVV_4x12_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(4));
C_reg_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(4));
C_reg_10 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(4));
C_reg_11 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(4));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(4));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(4));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(4));
}

// gemm_RVV_4x13_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 4] @DRAM
// )
void gemm_RVV_4x13_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_9 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_10 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_11 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(4));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(4));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(4));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(4));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_0,(4));
}

// gemm_RVV_4x13_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 13] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][13, 4] @DRAM
// )
void gemm_RVV_4x13_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(4));
C_reg_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(4));
C_reg_10 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(4));
C_reg_11 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(4));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(4));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(4));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(4));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(4));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_0,(4));
}

// gemm_RVV_4x14_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 4] @DRAM
// )
void gemm_RVV_4x14_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_9 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_10 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_11 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(4));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(4));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(4));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(4));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_0,(4));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_1,(4));
}

// gemm_RVV_4x14_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 14] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][14, 4] @DRAM
// )
void gemm_RVV_4x14_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(4));
C_reg_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(4));
C_reg_10 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(4));
C_reg_11 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(4));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(4));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(13) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(4));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(4));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(4));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(4));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_0,(4));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_1,(4));
}

// gemm_RVV_4x15_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 4] @DRAM
// )
void gemm_RVV_4x15_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_9 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_10 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_11 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(4));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(4));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(4));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(4));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(4));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_0,(4));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_1,(4));
__riscv_vse32_v_f32m1(&C[(14) * (ldc)], C_reg2_2,(4));
}

// gemm_RVV_4x15_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 15] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][15, 4] @DRAM
// )
void gemm_RVV_4x15_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(4));
C_reg_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(4));
C_reg_10 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(4));
C_reg_11 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(4));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(4));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(13) * (ldc)],(4));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(14) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(4));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(4));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(4));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(4));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(4));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg2_0,(4));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg2_1,(4));
__riscv_vse32_v_f32m1(&C[(14) * (ldc)], C_reg2_2,(4));
}

// gemm_RVV_4x16_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 4] @DRAM
// )
void gemm_RVV_4x16_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg_12;
vfloat32m1_t C_reg_13;
vfloat32m1_t C_reg_14;
vfloat32m1_t C_reg_15;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_8 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_9 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_10 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_11 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_12 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_13 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_14 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_15 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg_12;
vfloat32m1_t B_reg_13;
vfloat32m1_t B_reg_14;
vfloat32m1_t B_reg_15;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(4));
  B_reg_12 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_13 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_14 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_15 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(4));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(4));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(4));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(4));
  C_reg_12 = __riscv_vfmacc_vv_f32m1(C_reg_12, A_reg, B_reg_12,(4));
  C_reg_13 = __riscv_vfmacc_vv_f32m1(C_reg_13, A_reg, B_reg_13,(4));
  C_reg_14 = __riscv_vfmacc_vv_f32m1(C_reg_14, A_reg, B_reg_14,(4));
  C_reg_15 = __riscv_vfmacc_vv_f32m1(C_reg_15, A_reg, B_reg_15,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(4));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg_12,(4));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg_13,(4));
__riscv_vse32_v_f32m1(&C[(14) * (ldc)], C_reg_14,(4));
__riscv_vse32_v_f32m1(&C[(15) * (ldc)], C_reg_15,(4));
}

// gemm_RVV_4x16_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 4] @DRAM
// )
void gemm_RVV_4x16_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg_8;
vfloat32m1_t C_reg_9;
vfloat32m1_t C_reg_10;
vfloat32m1_t C_reg_11;
vfloat32m1_t C_reg_12;
vfloat32m1_t C_reg_13;
vfloat32m1_t C_reg_14;
vfloat32m1_t C_reg_15;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
C_reg_8 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(4));
C_reg_9 = __riscv_vle32_v_f32m1(&C[(9) * (ldc)],(4));
C_reg_10 = __riscv_vle32_v_f32m1(&C[(10) * (ldc)],(4));
C_reg_11 = __riscv_vle32_v_f32m1(&C[(11) * (ldc)],(4));
C_reg_12 = __riscv_vle32_v_f32m1(&C[(12) * (ldc)],(4));
C_reg_13 = __riscv_vle32_v_f32m1(&C[(13) * (ldc)],(4));
C_reg_14 = __riscv_vle32_v_f32m1(&C[(14) * (ldc)],(4));
C_reg_15 = __riscv_vle32_v_f32m1(&C[(15) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg_8;
vfloat32m1_t B_reg_9;
vfloat32m1_t B_reg_10;
vfloat32m1_t B_reg_11;
vfloat32m1_t B_reg_12;
vfloat32m1_t B_reg_13;
vfloat32m1_t B_reg_14;
vfloat32m1_t B_reg_15;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(4));
  B_reg_8 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_9 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_10 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_11 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 12],(4));
  B_reg_12 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_13 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_14 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_15 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg_8 = __riscv_vfmacc_vv_f32m1(C_reg_8, A_reg, B_reg_8,(4));
  C_reg_9 = __riscv_vfmacc_vv_f32m1(C_reg_9, A_reg, B_reg_9,(4));
  C_reg_10 = __riscv_vfmacc_vv_f32m1(C_reg_10, A_reg, B_reg_10,(4));
  C_reg_11 = __riscv_vfmacc_vv_f32m1(C_reg_11, A_reg, B_reg_11,(4));
  C_reg_12 = __riscv_vfmacc_vv_f32m1(C_reg_12, A_reg, B_reg_12,(4));
  C_reg_13 = __riscv_vfmacc_vv_f32m1(C_reg_13, A_reg, B_reg_13,(4));
  C_reg_14 = __riscv_vfmacc_vv_f32m1(C_reg_14, A_reg, B_reg_14,(4));
  C_reg_15 = __riscv_vfmacc_vv_f32m1(C_reg_15, A_reg, B_reg_15,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg_8,(4));
__riscv_vse32_v_f32m1(&C[(9) * (ldc)], C_reg_9,(4));
__riscv_vse32_v_f32m1(&C[(10) * (ldc)], C_reg_10,(4));
__riscv_vse32_v_f32m1(&C[(11) * (ldc)], C_reg_11,(4));
__riscv_vse32_v_f32m1(&C[(12) * (ldc)], C_reg_12,(4));
__riscv_vse32_v_f32m1(&C[(13) * (ldc)], C_reg_13,(4));
__riscv_vse32_v_f32m1(&C[(14) * (ldc)], C_reg_14,(4));
__riscv_vse32_v_f32m1(&C[(15) * (ldc)], C_reg_15,(4));
}

// gemm_RVV_4x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 4] @DRAM
// )
void gemm_RVV_4x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (16)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
}

// gemm_RVV_4x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 4] @DRAM
// )
void gemm_RVV_4x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (16)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
}

// gemm_RVV_4x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 4] @DRAM
// )
void gemm_RVV_4x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (16)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (16) + 1],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
}

// gemm_RVV_4x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 4] @DRAM
// )
void gemm_RVV_4x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (16)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (16) + 1],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
}

// gemm_RVV_4x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 4] @DRAM
// )
void gemm_RVV_4x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (16)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (16) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (16) + 2],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
}

// gemm_RVV_4x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 4] @DRAM
// )
void gemm_RVV_4x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (16)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (16) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (16) + 2],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
}

// gemm_RVV_4x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 4] @DRAM
// )
void gemm_RVV_4x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
}

// gemm_RVV_4x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 4] @DRAM
// )
void gemm_RVV_4x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
}

// gemm_RVV_4x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 4] @DRAM
// )
void gemm_RVV_4x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg2_0,(4));
}

// gemm_RVV_4x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 5] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][5, 4] @DRAM
// )
void gemm_RVV_4x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg2_0,(4));
}

// gemm_RVV_4x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 4] @DRAM
// )
void gemm_RVV_4x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg2_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg2_1,(4));
}

// gemm_RVV_4x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 6] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][6, 4] @DRAM
// )
void gemm_RVV_4x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(2));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg2_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg2_1,(4));
}

// gemm_RVV_4x7_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 4] @DRAM
// )
void gemm_RVV_4x7_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(4));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg2_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg2_1,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg2_2,(4));
}

// gemm_RVV_4x7_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 7] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][7, 4] @DRAM
// )
void gemm_RVV_4x7_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg2_0;
vfloat32m1_t C_reg2_1;
vfloat32m1_t C_reg2_2;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg2_1 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg2_2 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg2_0;
vfloat32m1_t B_reg2_1;
vfloat32m1_t B_reg2_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(3));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg2_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg2_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
  C_reg2_1 = __riscv_vfmacc_vv_f32m1(C_reg2_1, A_reg, B_reg2_1,(4));
  C_reg2_2 = __riscv_vfmacc_vv_f32m1(C_reg2_2, A_reg, B_reg2_2,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg2_0,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg2_1,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg2_2,(4));
}

// gemm_RVV_4x8_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 4] @DRAM
// )
void gemm_RVV_4x8_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
}

// gemm_RVV_4x8_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 4] @DRAM
// )
void gemm_RVV_4x8_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
}

// gemm_RVV_4x9_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 4] @DRAM
// )
void gemm_RVV_4x9_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_6 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_7 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(4));
}

// gemm_RVV_4x9_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 9] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][9, 4] @DRAM
// )
void gemm_RVV_4x9_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t C_reg_0;
vfloat32m1_t C_reg_1;
vfloat32m1_t C_reg_2;
vfloat32m1_t C_reg_3;
vfloat32m1_t C_reg_4;
vfloat32m1_t C_reg_5;
vfloat32m1_t C_reg_6;
vfloat32m1_t C_reg_7;
vfloat32m1_t C_reg2_0;
C_reg_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_4 = __riscv_vle32_v_f32m1(&C[(4) * (ldc)],(4));
C_reg_5 = __riscv_vle32_v_f32m1(&C[(5) * (ldc)],(4));
C_reg_6 = __riscv_vle32_v_f32m1(&C[(6) * (ldc)],(4));
C_reg_7 = __riscv_vle32_v_f32m1(&C[(7) * (ldc)],(4));
C_reg2_0 = __riscv_vle32_v_f32m1(&C[(8) * (ldc)],(4));
vfloat32m1_t A_reg;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_reg_4;
vfloat32m1_t B_reg_5;
vfloat32m1_t B_reg_6;
vfloat32m1_t B_reg_7;
vfloat32m1_t B_reg2_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 4],(4));
  B_reg_4 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_5 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_6 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_7 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (16) + 8],(1));
  B_reg2_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  C_reg_4 = __riscv_vfmacc_vv_f32m1(C_reg_4, A_reg, B_reg_4,(4));
  C_reg_5 = __riscv_vfmacc_vv_f32m1(C_reg_5, A_reg, B_reg_5,(4));
  C_reg_6 = __riscv_vfmacc_vv_f32m1(C_reg_6, A_reg, B_reg_6,(4));
  C_reg_7 = __riscv_vfmacc_vv_f32m1(C_reg_7, A_reg, B_reg_7,(4));
  C_reg2_0 = __riscv_vfmacc_vv_f32m1(C_reg2_0, A_reg, B_reg2_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3,(4));
__riscv_vse32_v_f32m1(&C[(4) * (ldc)], C_reg_4,(4));
__riscv_vse32_v_f32m1(&C[(5) * (ldc)], C_reg_5,(4));
__riscv_vse32_v_f32m1(&C[(6) * (ldc)], C_reg_6,(4));
__riscv_vse32_v_f32m1(&C[(7) * (ldc)], C_reg_7,(4));
__riscv_vse32_v_f32m1(&C[(8) * (ldc)], C_reg2_0,(4));
}


/* relying on the following instruction..."
rvv_broadcast_4xf32(dst,src,vl)
{dst_data} = __riscv_vfmv_v_f_f32m1({src_data},{vl});
*/

/* relying on the following instruction..."
rvv_broadcast_4xf32_0(dst,vl)
{dst_data} = __riscv_vfmv_v_f_f32m1(0.0f,{vl});
*/

/* relying on the following instruction..."
rvv_gather_4xf32(dst,src,imm,vl)
{dst_data} = __riscv_vrgather_vx_f32m1({src_data}, {imm}, {vl});
*/

/* relying on the following instruction..."
rvv_vfmacc_4xf32_4xf32(dst,lhs,rhs,vl)
{dst_data} = __riscv_vfmacc_vv_f32m1({dst_data}, {lhs_data}, {rhs_data},{vl});
*/

/* relying on the following instruction..."
rvv_vld_4xf32(dst,src,vl)
{dst_data} = __riscv_vle32_v_f32m1(&{src_data},{vl});
*/

/* relying on the following instruction..."
rvv_vst_4xf32(dst,src,vl)
__riscv_vse32_v_f32m1(&{dst_data}, {src_data},{vl});
*/