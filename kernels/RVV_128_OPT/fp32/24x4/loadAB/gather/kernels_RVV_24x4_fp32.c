#include "kernels_RVV_24x4_fp32.h"



#include <stdio.h>
#include <stdlib.h>

#include <riscv_vector.h>


// gemm_RVV_10x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 10] @DRAM
// )
void gemm_RVV_10x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(2));
}

// gemm_RVV_10x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 10] @DRAM
// )
void gemm_RVV_10x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C[8],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(2));
}

// gemm_RVV_10x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 10] @DRAM
// )
void gemm_RVV_10x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_regt_1,(2));
}

// gemm_RVV_10x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 10] @DRAM
// )
void gemm_RVV_10x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C[8],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 8],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_regt_1,(2));
}

// gemm_RVV_10x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 10] @DRAM
// )
void gemm_RVV_10x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_regt_2,(2));
}

// gemm_RVV_10x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 10] @DRAM
// )
void gemm_RVV_10x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C[8],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 8],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_regt_2,(2));
}

// gemm_RVV_10x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 10] @DRAM
// )
void gemm_RVV_10x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_regt_3,(2));
}

// gemm_RVV_10x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 10] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 10] @DRAM
// )
void gemm_RVV_10x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[8],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 8],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(2));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 8],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_regt_3,(2));
}

// gemm_RVV_11x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 11] @DRAM
// )
void gemm_RVV_11x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(3));
}

// gemm_RVV_11x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 11] @DRAM
// )
void gemm_RVV_11x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C[8],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(3));
}

// gemm_RVV_11x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 11] @DRAM
// )
void gemm_RVV_11x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_regt_1,(3));
}

// gemm_RVV_11x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 11] @DRAM
// )
void gemm_RVV_11x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C[8],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 8],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_regt_1,(3));
}

// gemm_RVV_11x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 11] @DRAM
// )
void gemm_RVV_11x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_regt_2,(3));
}

// gemm_RVV_11x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 11] @DRAM
// )
void gemm_RVV_11x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C[8],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 8],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_regt_2,(3));
}

// gemm_RVV_11x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 11] @DRAM
// )
void gemm_RVV_11x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_regt_3,(3));
}

// gemm_RVV_11x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 11] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 11] @DRAM
// )
void gemm_RVV_11x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[8],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 8],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(3));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 8],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_regt_3,(3));
}

// gemm_RVV_12x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 12] @DRAM
// )
void gemm_RVV_12x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
}

// gemm_RVV_12x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 12] @DRAM
// )
void gemm_RVV_12x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
}

// gemm_RVV_12x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 12] @DRAM
// )
void gemm_RVV_12x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
}

// gemm_RVV_12x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 12] @DRAM
// )
void gemm_RVV_12x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
}

// gemm_RVV_12x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 12] @DRAM
// )
void gemm_RVV_12x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
}

// gemm_RVV_12x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 12] @DRAM
// )
void gemm_RVV_12x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
}

// gemm_RVV_12x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 12] @DRAM
// )
void gemm_RVV_12x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_tmp;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
}

// gemm_RVV_12x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 12] @DRAM
// )
void gemm_RVV_12x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_tmp;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
}

// gemm_RVV_13x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 13] @DRAM
// )
void gemm_RVV_13x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(1));
}

// gemm_RVV_13x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 13] @DRAM
// )
void gemm_RVV_13x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C[12],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(1));
}

// gemm_RVV_13x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 13] @DRAM
// )
void gemm_RVV_13x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_regt_1,(1));
}

// gemm_RVV_13x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 13] @DRAM
// )
void gemm_RVV_13x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C[12],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 12],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_regt_1,(1));
}

// gemm_RVV_13x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 13] @DRAM
// )
void gemm_RVV_13x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_regt_2,(1));
}

// gemm_RVV_13x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 13] @DRAM
// )
void gemm_RVV_13x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C[12],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 12],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 12],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_regt_2,(1));
}

// gemm_RVV_13x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 13] @DRAM
// )
void gemm_RVV_13x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 12], C_regt_3,(1));
}

// gemm_RVV_13x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 13] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 13] @DRAM
// )
void gemm_RVV_13x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 8],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[12],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 12],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 12],(1));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 12],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 12], C_regt_3,(1));
}

// gemm_RVV_14x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 14] @DRAM
// )
void gemm_RVV_14x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(2));
}

// gemm_RVV_14x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 14] @DRAM
// )
void gemm_RVV_14x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C[12],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(2));
}

// gemm_RVV_14x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 14] @DRAM
// )
void gemm_RVV_14x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_regt_1,(2));
}

// gemm_RVV_14x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 14] @DRAM
// )
void gemm_RVV_14x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C[12],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 12],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_regt_1,(2));
}

// gemm_RVV_14x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 14] @DRAM
// )
void gemm_RVV_14x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_regt_2,(2));
}

// gemm_RVV_14x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 14] @DRAM
// )
void gemm_RVV_14x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C[12],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 12],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 12],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_regt_2,(2));
}

// gemm_RVV_14x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 14] @DRAM
// )
void gemm_RVV_14x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 12], C_regt_3,(2));
}

// gemm_RVV_14x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 14] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 14] @DRAM
// )
void gemm_RVV_14x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 8],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[12],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 12],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 12],(2));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 12],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 12], C_regt_3,(2));
}

// gemm_RVV_15x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 15] @DRAM
// )
void gemm_RVV_15x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(3));
}

// gemm_RVV_15x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 15] @DRAM
// )
void gemm_RVV_15x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C[12],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(3));
}

// gemm_RVV_15x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 15] @DRAM
// )
void gemm_RVV_15x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_regt_1,(3));
}

// gemm_RVV_15x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 15] @DRAM
// )
void gemm_RVV_15x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C[12],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 12],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_regt_1,(3));
}

// gemm_RVV_15x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 15] @DRAM
// )
void gemm_RVV_15x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_regt_2,(3));
}

// gemm_RVV_15x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 15] @DRAM
// )
void gemm_RVV_15x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
C_regt_0 = __riscv_vle32_v_f32m1(&C[12],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 12],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 12],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_regt_2,(3));
}

// gemm_RVV_15x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 15] @DRAM
// )
void gemm_RVV_15x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 12], C_regt_3,(3));
}

// gemm_RVV_15x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 15] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 15] @DRAM
// )
void gemm_RVV_15x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 8],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[12],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 12],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 12],(3));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 12],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 12], C_regt_3,(3));
}

// gemm_RVV_16x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 16] @DRAM
// )
void gemm_RVV_16x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
}

// gemm_RVV_16x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 16] @DRAM
// )
void gemm_RVV_16x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
}

// gemm_RVV_16x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 16] @DRAM
// )
void gemm_RVV_16x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
}

// gemm_RVV_16x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 16] @DRAM
// )
void gemm_RVV_16x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
}

// gemm_RVV_16x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 16] @DRAM
// )
void gemm_RVV_16x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
}

// gemm_RVV_16x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 16] @DRAM
// )
void gemm_RVV_16x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
}

// gemm_RVV_16x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 16] @DRAM
// )
void gemm_RVV_16x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_tmp;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 12], C_reg_3_3,(4));
}

// gemm_RVV_16x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 16] @DRAM
// )
void gemm_RVV_16x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_tmp;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 12],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 8],(4));
C_reg_3_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 12], C_reg_3_3,(4));
}

// gemm_RVV_17x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 17] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 17] @DRAM
// )
void gemm_RVV_17x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(1));
}

// gemm_RVV_17x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 17] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 17] @DRAM
// )
void gemm_RVV_17x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C[16],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(1));
}

// gemm_RVV_17x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 17] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 17] @DRAM
// )
void gemm_RVV_17x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_regt_1,(1));
}

// gemm_RVV_17x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 17] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 17] @DRAM
// )
void gemm_RVV_17x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C[16],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 16],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_regt_1,(1));
}

// gemm_RVV_17x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 17] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 17] @DRAM
// )
void gemm_RVV_17x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_regt_2,(1));
}

// gemm_RVV_17x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 17] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 17] @DRAM
// )
void gemm_RVV_17x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C[16],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 16],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 16],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_regt_2,(1));
}

// gemm_RVV_17x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 17] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 17] @DRAM
// )
void gemm_RVV_17x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 16], C_regt_3,(1));
}

// gemm_RVV_17x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 17] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 17] @DRAM
// )
void gemm_RVV_17x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 12],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 8],(4));
C_reg_3_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 12],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[16],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 16],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 16],(1));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 16],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 16], C_regt_3,(1));
}

// gemm_RVV_18x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 18] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 18] @DRAM
// )
void gemm_RVV_18x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(2));
}

// gemm_RVV_18x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 18] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 18] @DRAM
// )
void gemm_RVV_18x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C[16],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(2));
}

// gemm_RVV_18x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 18] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 18] @DRAM
// )
void gemm_RVV_18x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_regt_1,(2));
}

// gemm_RVV_18x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 18] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 18] @DRAM
// )
void gemm_RVV_18x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C[16],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 16],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_regt_1,(2));
}

// gemm_RVV_18x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 18] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 18] @DRAM
// )
void gemm_RVV_18x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_regt_2,(2));
}

// gemm_RVV_18x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 18] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 18] @DRAM
// )
void gemm_RVV_18x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C[16],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 16],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 16],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_regt_2,(2));
}

// gemm_RVV_18x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 18] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 18] @DRAM
// )
void gemm_RVV_18x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 16], C_regt_3,(2));
}

// gemm_RVV_18x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 18] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 18] @DRAM
// )
void gemm_RVV_18x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 12],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 8],(4));
C_reg_3_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 12],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[16],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 16],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 16],(2));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 16],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 16], C_regt_3,(2));
}

// gemm_RVV_19x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 19] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 19] @DRAM
// )
void gemm_RVV_19x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(3));
}

// gemm_RVV_19x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 19] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 19] @DRAM
// )
void gemm_RVV_19x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C[16],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(3));
}

// gemm_RVV_19x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 19] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 19] @DRAM
// )
void gemm_RVV_19x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_regt_1,(3));
}

// gemm_RVV_19x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 19] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 19] @DRAM
// )
void gemm_RVV_19x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C[16],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 16],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_regt_1,(3));
}

// gemm_RVV_19x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 19] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 19] @DRAM
// )
void gemm_RVV_19x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_regt_2,(3));
}

// gemm_RVV_19x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 19] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 19] @DRAM
// )
void gemm_RVV_19x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
C_regt_0 = __riscv_vle32_v_f32m1(&C[16],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 16],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 16],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 12],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_regt_2,(3));
}

// gemm_RVV_19x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 19] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 19] @DRAM
// )
void gemm_RVV_19x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 16], C_regt_3,(3));
}

// gemm_RVV_19x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 19] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 19] @DRAM
// )
void gemm_RVV_19x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 12],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 8],(4));
C_reg_3_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 12],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[16],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 16],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 16],(3));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 16],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 16], C_regt_3,(3));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(1));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(1));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(1));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(1));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
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

// gemm_RVV_20x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 20] @DRAM
// )
void gemm_RVV_20x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
}

// gemm_RVV_20x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 20] @DRAM
// )
void gemm_RVV_20x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C[16],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
}

// gemm_RVV_20x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 20] @DRAM
// )
void gemm_RVV_20x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
}

// gemm_RVV_20x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 20] @DRAM
// )
void gemm_RVV_20x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C[16],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_reg_1_4 = __riscv_vle32_v_f32m1(&C[ldc + 16],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
}

// gemm_RVV_20x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 20] @DRAM
// )
void gemm_RVV_20x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_reg_2_4,(4));
}

// gemm_RVV_20x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 20] @DRAM
// )
void gemm_RVV_20x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C[16],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_reg_1_4 = __riscv_vle32_v_f32m1(&C[ldc + 16],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 12],(4));
C_reg_2_4 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 16],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_reg_2_4,(4));
}

// gemm_RVV_20x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 20] @DRAM
// )
void gemm_RVV_20x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_tmp;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_3_4;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_3_4 = __riscv_vfmacc_vv_f32m1(C_reg_3_4, A_reg_4, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_reg_2_4,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 16], C_reg_3_4,(4));
}

// gemm_RVV_20x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 20] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 20] @DRAM
// )
void gemm_RVV_20x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_tmp;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_3_4;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C[16],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_reg_1_4 = __riscv_vle32_v_f32m1(&C[ldc + 16],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 12],(4));
C_reg_2_4 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 16],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 8],(4));
C_reg_3_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 12],(4));
C_reg_3_4 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 16],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_3_4 = __riscv_vfmacc_vv_f32m1(C_reg_3_4, A_reg_4, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_reg_2_4,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 16], C_reg_3_4,(4));
}

// gemm_RVV_21x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 21] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 21] @DRAM
// )
void gemm_RVV_21x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_regt_0,(1));
}

// gemm_RVV_21x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 21] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 21] @DRAM
// )
void gemm_RVV_21x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
C_regt_0 = __riscv_vle32_v_f32m1(&C[20],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C[16],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_regt_0,(1));
}

// gemm_RVV_21x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 21] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 21] @DRAM
// )
void gemm_RVV_21x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 20], C_regt_1,(1));
}

// gemm_RVV_21x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 21] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 21] @DRAM
// )
void gemm_RVV_21x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
C_regt_0 = __riscv_vle32_v_f32m1(&C[20],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 20],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C[16],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_reg_1_4 = __riscv_vle32_v_f32m1(&C[ldc + 16],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 20], C_regt_1,(1));
}

// gemm_RVV_21x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 21] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 21] @DRAM
// )
void gemm_RVV_21x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_reg_2_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 20], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 20], C_regt_2,(1));
}

// gemm_RVV_21x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 21] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 21] @DRAM
// )
void gemm_RVV_21x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
C_regt_0 = __riscv_vle32_v_f32m1(&C[20],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 20],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 20],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C[16],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_reg_1_4 = __riscv_vle32_v_f32m1(&C[ldc + 16],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 12],(4));
C_reg_2_4 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 16],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_reg_2_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 20], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 20], C_regt_2,(1));
}

// gemm_RVV_21x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 21] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 21] @DRAM
// )
void gemm_RVV_21x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_3_4;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_3_4 = __riscv_vfmacc_vv_f32m1(C_reg_3_4, A_reg_4, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_reg_2_4,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 16], C_reg_3_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 20], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 20], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 20], C_regt_3,(1));
}

// gemm_RVV_21x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 21] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 21] @DRAM
// )
void gemm_RVV_21x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_3_4;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C[16],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_reg_1_4 = __riscv_vle32_v_f32m1(&C[ldc + 16],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 12],(4));
C_reg_2_4 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 16],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 8],(4));
C_reg_3_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 12],(4));
C_reg_3_4 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 16],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[20],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 20],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 20],(1));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 20],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_3_4 = __riscv_vfmacc_vv_f32m1(C_reg_3_4, A_reg_4, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_reg_2_4,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 16], C_reg_3_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 20], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 20], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 20], C_regt_3,(1));
}

// gemm_RVV_22x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 22] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 22] @DRAM
// )
void gemm_RVV_22x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_regt_0,(2));
}

// gemm_RVV_22x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 22] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 22] @DRAM
// )
void gemm_RVV_22x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
C_regt_0 = __riscv_vle32_v_f32m1(&C[20],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C[16],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_regt_0,(2));
}

// gemm_RVV_22x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 22] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 22] @DRAM
// )
void gemm_RVV_22x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 20], C_regt_1,(2));
}

// gemm_RVV_22x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 22] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 22] @DRAM
// )
void gemm_RVV_22x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
C_regt_0 = __riscv_vle32_v_f32m1(&C[20],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 20],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C[16],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_reg_1_4 = __riscv_vle32_v_f32m1(&C[ldc + 16],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 20], C_regt_1,(2));
}

// gemm_RVV_22x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 22] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 22] @DRAM
// )
void gemm_RVV_22x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_reg_2_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 20], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 20], C_regt_2,(2));
}

// gemm_RVV_22x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 22] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 22] @DRAM
// )
void gemm_RVV_22x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
C_regt_0 = __riscv_vle32_v_f32m1(&C[20],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 20],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 20],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C[16],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_reg_1_4 = __riscv_vle32_v_f32m1(&C[ldc + 16],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 12],(4));
C_reg_2_4 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 16],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_reg_2_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 20], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 20], C_regt_2,(2));
}

// gemm_RVV_22x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 22] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 22] @DRAM
// )
void gemm_RVV_22x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_3_4;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_3_4 = __riscv_vfmacc_vv_f32m1(C_reg_3_4, A_reg_4, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_reg_2_4,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 16], C_reg_3_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 20], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 20], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 20], C_regt_3,(2));
}

// gemm_RVV_22x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 22] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 22] @DRAM
// )
void gemm_RVV_22x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_3_4;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C[16],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_reg_1_4 = __riscv_vle32_v_f32m1(&C[ldc + 16],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 12],(4));
C_reg_2_4 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 16],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 8],(4));
C_reg_3_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 12],(4));
C_reg_3_4 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 16],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[20],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 20],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 20],(2));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 20],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_3_4 = __riscv_vfmacc_vv_f32m1(C_reg_3_4, A_reg_4, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_reg_2_4,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 16], C_reg_3_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 20], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 20], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 20], C_regt_3,(2));
}

// gemm_RVV_23x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 23] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 23] @DRAM
// )
void gemm_RVV_23x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_regt_0,(3));
}

// gemm_RVV_23x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 23] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 23] @DRAM
// )
void gemm_RVV_23x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
C_regt_0 = __riscv_vle32_v_f32m1(&C[20],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C[16],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_regt_0,(3));
}

// gemm_RVV_23x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 23] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 23] @DRAM
// )
void gemm_RVV_23x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 20], C_regt_1,(3));
}

// gemm_RVV_23x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 23] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 23] @DRAM
// )
void gemm_RVV_23x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
C_regt_0 = __riscv_vle32_v_f32m1(&C[20],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 20],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C[16],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_reg_1_4 = __riscv_vle32_v_f32m1(&C[ldc + 16],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 20], C_regt_1,(3));
}

// gemm_RVV_23x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 23] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 23] @DRAM
// )
void gemm_RVV_23x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_reg_2_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 20], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 20], C_regt_2,(3));
}

// gemm_RVV_23x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 23] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 23] @DRAM
// )
void gemm_RVV_23x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
C_regt_0 = __riscv_vle32_v_f32m1(&C[20],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 20],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 20],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C[16],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_reg_1_4 = __riscv_vle32_v_f32m1(&C[ldc + 16],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 12],(4));
C_reg_2_4 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 16],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_reg_2_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 20], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 20], C_regt_2,(3));
}

// gemm_RVV_23x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 23] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 23] @DRAM
// )
void gemm_RVV_23x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_3_4;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_3_4 = __riscv_vfmacc_vv_f32m1(C_reg_3_4, A_reg_4, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_reg_2_4,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 16], C_reg_3_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 20], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 20], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 20], C_regt_3,(3));
}

// gemm_RVV_23x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 23] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 23] @DRAM
// )
void gemm_RVV_23x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_3_4;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C[16],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_reg_1_4 = __riscv_vle32_v_f32m1(&C[ldc + 16],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 12],(4));
C_reg_2_4 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 16],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 8],(4));
C_reg_3_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 12],(4));
C_reg_3_4 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 16],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[20],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 20],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 20],(3));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 20],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_3_4 = __riscv_vfmacc_vv_f32m1(C_reg_3_4, A_reg_4, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_reg_2_4,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 16], C_reg_3_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 20], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 20], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 20], C_regt_3,(3));
}

// gemm_RVV_24x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 24] @DRAM
// )
void gemm_RVV_24x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_reg_5;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_0_5;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_reg_5 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_0_5 = __riscv_vfmacc_vv_f32m1(C_reg_0_5, A_reg_5, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_reg_0_5,(4));
}

// gemm_RVV_24x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 24] @DRAM
// )
void gemm_RVV_24x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_reg_5;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_0_5;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C[16],(4));
C_reg_0_5 = __riscv_vle32_v_f32m1(&C[20],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_reg_5 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_0_5 = __riscv_vfmacc_vv_f32m1(C_reg_0_5, A_reg_5, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_reg_0_5,(4));
}

// gemm_RVV_24x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 24] @DRAM
// )
void gemm_RVV_24x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_reg_5;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_0_5;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_1_5;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_reg_5 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_0_5 = __riscv_vfmacc_vv_f32m1(C_reg_0_5, A_reg_5, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_1_5 = __riscv_vfmacc_vv_f32m1(C_reg_1_5, A_reg_5, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_reg_0_5,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[ldc + 20], C_reg_1_5,(4));
}

// gemm_RVV_24x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 24] @DRAM
// )
void gemm_RVV_24x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_reg_5;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_0_5;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_1_5;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C[16],(4));
C_reg_0_5 = __riscv_vle32_v_f32m1(&C[20],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_reg_1_4 = __riscv_vle32_v_f32m1(&C[ldc + 16],(4));
C_reg_1_5 = __riscv_vle32_v_f32m1(&C[ldc + 20],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_reg_5 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_0_5 = __riscv_vfmacc_vv_f32m1(C_reg_0_5, A_reg_5, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_1_5 = __riscv_vfmacc_vv_f32m1(C_reg_1_5, A_reg_5, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_reg_0_5,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[ldc + 20], C_reg_1_5,(4));
}

// gemm_RVV_24x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 24] @DRAM
// )
void gemm_RVV_24x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_reg_5;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_0_5;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_1_5;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
vfloat32m1_t C_reg_2_5;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_reg_5 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_0_5 = __riscv_vfmacc_vv_f32m1(C_reg_0_5, A_reg_5, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_1_5 = __riscv_vfmacc_vv_f32m1(C_reg_1_5, A_reg_5, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
  C_reg_2_5 = __riscv_vfmacc_vv_f32m1(C_reg_2_5, A_reg_5, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_reg_0_5,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[ldc + 20], C_reg_1_5,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_reg_2_4,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 20], C_reg_2_5,(4));
}

// gemm_RVV_24x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 24] @DRAM
// )
void gemm_RVV_24x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_reg_5;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_0_5;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_1_5;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
vfloat32m1_t C_reg_2_5;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C[16],(4));
C_reg_0_5 = __riscv_vle32_v_f32m1(&C[20],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_reg_1_4 = __riscv_vle32_v_f32m1(&C[ldc + 16],(4));
C_reg_1_5 = __riscv_vle32_v_f32m1(&C[ldc + 20],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 12],(4));
C_reg_2_4 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 16],(4));
C_reg_2_5 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 20],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_reg_5 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_0_5 = __riscv_vfmacc_vv_f32m1(C_reg_0_5, A_reg_5, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_1_5 = __riscv_vfmacc_vv_f32m1(C_reg_1_5, A_reg_5, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
  C_reg_2_5 = __riscv_vfmacc_vv_f32m1(C_reg_2_5, A_reg_5, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_reg_0_5,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[ldc + 20], C_reg_1_5,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_reg_2_4,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 20], C_reg_2_5,(4));
}

// gemm_RVV_24x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 24] @DRAM
// )
void gemm_RVV_24x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_reg_5;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_tmp;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_0_5;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_1_5;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
vfloat32m1_t C_reg_2_5;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_3_4;
vfloat32m1_t C_reg_3_5;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_2 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_3 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_4 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_5 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_reg_5 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_0_5 = __riscv_vfmacc_vv_f32m1(C_reg_0_5, A_reg_5, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_1_5 = __riscv_vfmacc_vv_f32m1(C_reg_1_5, A_reg_5, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
  C_reg_2_5 = __riscv_vfmacc_vv_f32m1(C_reg_2_5, A_reg_5, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_3_4 = __riscv_vfmacc_vv_f32m1(C_reg_3_4, A_reg_4, B_reg_3,(4));
  C_reg_3_5 = __riscv_vfmacc_vv_f32m1(C_reg_3_5, A_reg_5, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_reg_0_5,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[ldc + 20], C_reg_1_5,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_reg_2_4,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 20], C_reg_2_5,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 16], C_reg_3_4,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 20], C_reg_3_5,(4));
}

// gemm_RVV_24x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 24] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 24] @DRAM
// )
void gemm_RVV_24x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_reg_2;
vfloat32m1_t A_reg_3;
vfloat32m1_t A_reg_4;
vfloat32m1_t A_reg_5;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_tmp;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_0_2;
vfloat32m1_t C_reg_0_3;
vfloat32m1_t C_reg_0_4;
vfloat32m1_t C_reg_0_5;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_1_2;
vfloat32m1_t C_reg_1_3;
vfloat32m1_t C_reg_1_4;
vfloat32m1_t C_reg_1_5;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_2_2;
vfloat32m1_t C_reg_2_3;
vfloat32m1_t C_reg_2_4;
vfloat32m1_t C_reg_2_5;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
vfloat32m1_t C_reg_3_2;
vfloat32m1_t C_reg_3_3;
vfloat32m1_t C_reg_3_4;
vfloat32m1_t C_reg_3_5;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_0_2 = __riscv_vle32_v_f32m1(&C[8],(4));
C_reg_0_3 = __riscv_vle32_v_f32m1(&C[12],(4));
C_reg_0_4 = __riscv_vle32_v_f32m1(&C[16],(4));
C_reg_0_5 = __riscv_vle32_v_f32m1(&C[20],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_1_2 = __riscv_vle32_v_f32m1(&C[ldc + 8],(4));
C_reg_1_3 = __riscv_vle32_v_f32m1(&C[ldc + 12],(4));
C_reg_1_4 = __riscv_vle32_v_f32m1(&C[ldc + 16],(4));
C_reg_1_5 = __riscv_vle32_v_f32m1(&C[ldc + 20],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_2_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(4));
C_reg_2_3 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 12],(4));
C_reg_2_4 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 16],(4));
C_reg_2_5 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 20],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(4));
C_reg_3_2 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 8],(4));
C_reg_3_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 12],(4));
C_reg_3_4 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 16],(4));
C_reg_3_5 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 20],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_reg_2 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(4));
  A_reg_3 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 12],(4));
  A_reg_4 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 16],(4));
  A_reg_5 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 20],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_0_2 = __riscv_vfmacc_vv_f32m1(C_reg_0_2, A_reg_2, B_reg_0,(4));
  C_reg_0_3 = __riscv_vfmacc_vv_f32m1(C_reg_0_3, A_reg_3, B_reg_0,(4));
  C_reg_0_4 = __riscv_vfmacc_vv_f32m1(C_reg_0_4, A_reg_4, B_reg_0,(4));
  C_reg_0_5 = __riscv_vfmacc_vv_f32m1(C_reg_0_5, A_reg_5, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_1_2 = __riscv_vfmacc_vv_f32m1(C_reg_1_2, A_reg_2, B_reg_1,(4));
  C_reg_1_3 = __riscv_vfmacc_vv_f32m1(C_reg_1_3, A_reg_3, B_reg_1,(4));
  C_reg_1_4 = __riscv_vfmacc_vv_f32m1(C_reg_1_4, A_reg_4, B_reg_1,(4));
  C_reg_1_5 = __riscv_vfmacc_vv_f32m1(C_reg_1_5, A_reg_5, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_2_2 = __riscv_vfmacc_vv_f32m1(C_reg_2_2, A_reg_2, B_reg_2,(4));
  C_reg_2_3 = __riscv_vfmacc_vv_f32m1(C_reg_2_3, A_reg_3, B_reg_2,(4));
  C_reg_2_4 = __riscv_vfmacc_vv_f32m1(C_reg_2_4, A_reg_4, B_reg_2,(4));
  C_reg_2_5 = __riscv_vfmacc_vv_f32m1(C_reg_2_5, A_reg_5, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_reg_3_2 = __riscv_vfmacc_vv_f32m1(C_reg_3_2, A_reg_2, B_reg_3,(4));
  C_reg_3_3 = __riscv_vfmacc_vv_f32m1(C_reg_3_3, A_reg_3, B_reg_3,(4));
  C_reg_3_4 = __riscv_vfmacc_vv_f32m1(C_reg_3_4, A_reg_4, B_reg_3,(4));
  C_reg_3_5 = __riscv_vfmacc_vv_f32m1(C_reg_3_5, A_reg_5, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_reg_0_2,(4));
__riscv_vse32_v_f32m1(&C[12], C_reg_0_3,(4));
__riscv_vse32_v_f32m1(&C[16], C_reg_0_4,(4));
__riscv_vse32_v_f32m1(&C[20], C_reg_0_5,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_reg_1_2,(4));
__riscv_vse32_v_f32m1(&C[ldc + 12], C_reg_1_3,(4));
__riscv_vse32_v_f32m1(&C[ldc + 16], C_reg_1_4,(4));
__riscv_vse32_v_f32m1(&C[ldc + 20], C_reg_1_5,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_reg_2_2,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 12], C_reg_2_3,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 16], C_reg_2_4,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 20], C_reg_2_5,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_reg_3_2,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 12], C_reg_3_3,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 16], C_reg_3_4,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 20], C_reg_3_5,(4));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(2));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(2));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(2));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(2));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(3));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(3));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(3));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(3));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
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
  A_reg = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
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

// gemm_RVV_5x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 5] @DRAM
// )
void gemm_RVV_5x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
}

// gemm_RVV_5x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 5] @DRAM
// )
void gemm_RVV_5x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
}

// gemm_RVV_5x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 5] @DRAM
// )
void gemm_RVV_5x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(1));
}

// gemm_RVV_5x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 5] @DRAM
// )
void gemm_RVV_5x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(1));
}

// gemm_RVV_5x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 5] @DRAM
// )
void gemm_RVV_5x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(1));
}

// gemm_RVV_5x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 5] @DRAM
// )
void gemm_RVV_5x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(1));
}

// gemm_RVV_5x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 5] @DRAM
// )
void gemm_RVV_5x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(1));
}

// gemm_RVV_5x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 5] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 5] @DRAM
// )
void gemm_RVV_5x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(1));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(1));
}

// gemm_RVV_6x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 6] @DRAM
// )
void gemm_RVV_6x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
}

// gemm_RVV_6x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 6] @DRAM
// )
void gemm_RVV_6x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
}

// gemm_RVV_6x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 6] @DRAM
// )
void gemm_RVV_6x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(2));
}

// gemm_RVV_6x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 6] @DRAM
// )
void gemm_RVV_6x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(2));
}

// gemm_RVV_6x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 6] @DRAM
// )
void gemm_RVV_6x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(2));
}

// gemm_RVV_6x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 6] @DRAM
// )
void gemm_RVV_6x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(2));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(2));
}

// gemm_RVV_6x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 6] @DRAM
// )
void gemm_RVV_6x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(2));
}

// gemm_RVV_6x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 6] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 6] @DRAM
// )
void gemm_RVV_6x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(2));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(2));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(2));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(2));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(2));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(2));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(2));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(2));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(2));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(2));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(2));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(2));
}

// gemm_RVV_7x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 7] @DRAM
// )
void gemm_RVV_7x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
}

// gemm_RVV_7x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 7] @DRAM
// )
void gemm_RVV_7x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
}

// gemm_RVV_7x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 7] @DRAM
// )
void gemm_RVV_7x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(3));
}

// gemm_RVV_7x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 7] @DRAM
// )
void gemm_RVV_7x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(3));
}

// gemm_RVV_7x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 7] @DRAM
// )
void gemm_RVV_7x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(3));
}

// gemm_RVV_7x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 7] @DRAM
// )
void gemm_RVV_7x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(3));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(3));
}

// gemm_RVV_7x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 7] @DRAM
// )
void gemm_RVV_7x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(3));
}

// gemm_RVV_7x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 7] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 7] @DRAM
// )
void gemm_RVV_7x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_3_0;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[4],(3));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(3));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(3));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(3));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(3));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(3));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(3));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(3));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_regt_0,(3));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_regt_1,(3));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_regt_2,(3));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_regt_3,(3));
}

// gemm_RVV_8x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 8] @DRAM
// )
void gemm_RVV_8x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
}

// gemm_RVV_8x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 8] @DRAM
// )
void gemm_RVV_8x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
}

// gemm_RVV_8x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 8] @DRAM
// )
void gemm_RVV_8x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
}

// gemm_RVV_8x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 8] @DRAM
// )
void gemm_RVV_8x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
}

// gemm_RVV_8x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 8] @DRAM
// )
void gemm_RVV_8x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
}

// gemm_RVV_8x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 8] @DRAM
// )
void gemm_RVV_8x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
}

// gemm_RVV_8x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 8] @DRAM
// )
void gemm_RVV_8x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_tmp;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
}

// gemm_RVV_8x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 8] @DRAM
// )
void gemm_RVV_8x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t B_tmp;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
}

// gemm_RVV_9x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 9] @DRAM
// )
void gemm_RVV_9x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(1));
}

// gemm_RVV_9x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][1, 9] @DRAM
// )
void gemm_RVV_9x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C[8],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(1));
}

// gemm_RVV_9x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 9] @DRAM
// )
void gemm_RVV_9x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_regt_1,(1));
}

// gemm_RVV_9x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][2, 9] @DRAM
// )
void gemm_RVV_9x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C[8],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 8],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_regt_1,(1));
}

// gemm_RVV_9x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 9] @DRAM
// )
void gemm_RVV_9x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_regt_2,(1));
}

// gemm_RVV_9x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][3, 9] @DRAM
// )
void gemm_RVV_9x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
C_regt_0 = __riscv_vle32_v_f32m1(&C[8],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 8],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(1));
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(k) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(k) * (4) + 2],(4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_regt_2,(1));
}

// gemm_RVV_9x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 9] @DRAM
// )
void gemm_RVV_9x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_0_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_1_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_2_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_0 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_reg_3_1 = __riscv_vfmv_v_f_f32m1(0.0f,(4));
C_regt_0 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_1 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_2 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
C_regt_3 = __riscv_vfmv_v_f_f32m1(0.0f,(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_regt_3,(1));
}

// gemm_RVV_9x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 9] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 9] @DRAM
// )
void gemm_RVV_9x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
vfloat32m1_t A_reg_0;
vfloat32m1_t A_reg_1;
vfloat32m1_t A_regt;
vfloat32m1_t B_tmp;
vfloat32m1_t B_reg_0;
vfloat32m1_t B_reg_1;
vfloat32m1_t B_reg_2;
vfloat32m1_t B_reg_3;
vfloat32m1_t C_regt_0;
vfloat32m1_t C_regt_1;
vfloat32m1_t C_regt_2;
vfloat32m1_t C_regt_3;
vfloat32m1_t C_reg_0_0;
vfloat32m1_t C_reg_0_1;
vfloat32m1_t C_reg_1_0;
vfloat32m1_t C_reg_1_1;
vfloat32m1_t C_reg_2_0;
vfloat32m1_t C_reg_2_1;
vfloat32m1_t C_reg_3_0;
vfloat32m1_t C_reg_3_1;
C_reg_0_0 = __riscv_vle32_v_f32m1(&C[0],(4));
C_reg_0_1 = __riscv_vle32_v_f32m1(&C[4],(4));
C_reg_1_0 = __riscv_vle32_v_f32m1(&C[ldc],(4));
C_reg_1_1 = __riscv_vle32_v_f32m1(&C[ldc + 4],(4));
C_reg_2_0 = __riscv_vle32_v_f32m1(&C[(2) * (ldc)],(4));
C_reg_2_1 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 4],(4));
C_reg_3_0 = __riscv_vle32_v_f32m1(&C[(3) * (ldc)],(4));
C_reg_3_1 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 4],(4));
C_regt_0 = __riscv_vle32_v_f32m1(&C[8],(1));
C_regt_1 = __riscv_vle32_v_f32m1(&C[ldc + 8],(1));
C_regt_2 = __riscv_vle32_v_f32m1(&C[(2) * (ldc) + 8],(1));
C_regt_3 = __riscv_vle32_v_f32m1(&C[(3) * (ldc) + 8],(1));
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = __riscv_vle32_v_f32m1(&A[(k) * (24)],(4));
  A_reg_1 = __riscv_vle32_v_f32m1(&A[(k) * (24) + 4],(4));
  A_regt = __riscv_vle32_v_f32m1(&A[(k) * (24) + 8],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(k) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0_0 = __riscv_vfmacc_vv_f32m1(C_reg_0_0, A_reg_0, B_reg_0,(4));
  C_reg_0_1 = __riscv_vfmacc_vv_f32m1(C_reg_0_1, A_reg_1, B_reg_0,(4));
  C_reg_1_0 = __riscv_vfmacc_vv_f32m1(C_reg_1_0, A_reg_0, B_reg_1,(4));
  C_reg_1_1 = __riscv_vfmacc_vv_f32m1(C_reg_1_1, A_reg_1, B_reg_1,(4));
  C_reg_2_0 = __riscv_vfmacc_vv_f32m1(C_reg_2_0, A_reg_0, B_reg_2,(4));
  C_reg_2_1 = __riscv_vfmacc_vv_f32m1(C_reg_2_1, A_reg_1, B_reg_2,(4));
  C_reg_3_0 = __riscv_vfmacc_vv_f32m1(C_reg_3_0, A_reg_0, B_reg_3,(4));
  C_reg_3_1 = __riscv_vfmacc_vv_f32m1(C_reg_3_1, A_reg_1, B_reg_3,(4));
  C_regt_0 = __riscv_vfmacc_vv_f32m1(C_regt_0, A_regt, B_reg_0,(1));
  C_regt_1 = __riscv_vfmacc_vv_f32m1(C_regt_1, A_regt, B_reg_1,(1));
  C_regt_2 = __riscv_vfmacc_vv_f32m1(C_regt_2, A_regt, B_reg_2,(1));
  C_regt_3 = __riscv_vfmacc_vv_f32m1(C_regt_3, A_regt, B_reg_3,(1));
}
__riscv_vse32_v_f32m1(&C[0], C_reg_0_0,(4));
__riscv_vse32_v_f32m1(&C[4], C_reg_0_1,(4));
__riscv_vse32_v_f32m1(&C[ldc], C_reg_1_0,(4));
__riscv_vse32_v_f32m1(&C[ldc + 4], C_reg_1_1,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc)], C_reg_2_0,(4));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 4], C_reg_2_1,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc)], C_reg_3_0,(4));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 4], C_reg_3_1,(4));
__riscv_vse32_v_f32m1(&C[8], C_regt_0,(1));
__riscv_vse32_v_f32m1(&C[ldc + 8], C_regt_1,(1));
__riscv_vse32_v_f32m1(&C[(2) * (ldc) + 8], C_regt_2,(1));
__riscv_vse32_v_f32m1(&C[(3) * (ldc) + 8], C_regt_3,(1));
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