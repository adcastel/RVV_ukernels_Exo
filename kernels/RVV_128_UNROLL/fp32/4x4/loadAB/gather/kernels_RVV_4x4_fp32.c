#include "kernels_RVV_4x4_fp32.h"



#include <stdio.h>
#include <stdlib.h>

#include <riscv_vector.h>


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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4)],(1));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4)],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4)],(1));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4) + 1],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4) + 1],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4) + 1],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4) + 1],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4) + 1],(1));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4) + 1],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4) + 1],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4) + 1],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4) + 1],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4) + 1],(1));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4) + 2],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4) + 2],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4) + 2],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4) + 2],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4) + 2],(1));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4) + 2],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4) + 2],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4) + 2],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4) + 2],(1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(1));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4)],(1));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4) + 1],(1));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4) + 2],(1));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(1 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(2 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(3 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(ktt + (KC / 4) * 4) * (4)],(4));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(1 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(2 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(3 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (1));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (1));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (1));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (1));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(1));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(1));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(1));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(1));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(1));
  B_tmp = __riscv_vle32_v_f32m1(&B[(ktt + (KC / 4) * 4) * (4)],(4));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4)],(2));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4)],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4)],(2));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4) + 1],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4) + 1],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4) + 1],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4) + 1],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4) + 1],(2));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4) + 1],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4) + 1],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4) + 1],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4) + 1],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4) + 1],(2));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4) + 2],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4) + 2],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4) + 2],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4) + 2],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4) + 2],(2));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4) + 2],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4) + 2],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4) + 2],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4) + 2],(2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(2));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4)],(2));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4) + 1],(2));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4) + 2],(2));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(1 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(2 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(3 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(ktt + (KC / 4) * 4) * (4)],(4));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(1 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(2 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(3 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (2));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (2));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (2));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (2));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(2));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(2));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(2));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(2));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(2));
  B_tmp = __riscv_vle32_v_f32m1(&B[(ktt + (KC / 4) * 4) * (4)],(4));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4)],(3));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4)],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4)],(3));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4) + 1],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4) + 1],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4) + 1],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4) + 1],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4) + 1],(3));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4) + 1],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4) + 1],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4) + 1],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4) + 1],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4) + 1],(3));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4) + 2],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4) + 2],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4) + 2],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4) + 2],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4) + 2],(3));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4) + 2],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4) + 2],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4) + 2],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4) + 2],(3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(3));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4)],(3));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4) + 1],(3));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4) + 2],(3));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(1 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(2 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(3 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(ktt + (KC / 4) * 4) * (4)],(4));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(1 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(2 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(3 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (3));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (3));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (3));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (3));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(3));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(3));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(3));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(3));
  B_tmp = __riscv_vle32_v_f32m1(&B[(ktt + (KC / 4) * 4) * (4)],(4));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4)],(4));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4)],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4)],(4));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4) + 1],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4) + 1],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4) + 1],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4) + 1],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4) + 1],(4));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4) + 1],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4) + 1],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4) + 1],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4) + 1],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4) + 1],(4));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4) + 2],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4) + 2],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4) + 2],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4) + 2],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4) + 2],(4));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(4 * kt) * (4) + 2],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(1 + 4 * kt) * (4) + 2],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(2 + 4 * kt) * (4) + 2],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(3 + 4 * kt) * (4) + 2],(4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(4));
  B_reg_0 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4)],(4));
  B_reg_1 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4) + 1],(4));
  B_reg_2 = __riscv_vfmv_v_f_f32m1(B[(ktt + (KC / 4) * 4) * (4) + 2],(4));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(1 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(2 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(3 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(ktt + (KC / 4) * 4) * (4)],(4));
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
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(4 * kt) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(1 + 4 * kt) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(1 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(2 + 4 * kt) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(2 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
  A_reg = __riscv_vle32_v_f32m1(&A[(3 + 4 * kt) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(3 + 4 * kt) * (4)],(4));
  B_reg_0 = __riscv_vrgather_vx_f32m1(B_tmp, (0), (4));
  B_reg_1 = __riscv_vrgather_vx_f32m1(B_tmp, (1), (4));
  B_reg_2 = __riscv_vrgather_vx_f32m1(B_tmp, (2), (4));
  B_reg_3 = __riscv_vrgather_vx_f32m1(B_tmp, (3), (4));
  C_reg_0 = __riscv_vfmacc_vv_f32m1(C_reg_0, A_reg, B_reg_0,(4));
  C_reg_1 = __riscv_vfmacc_vv_f32m1(C_reg_1, A_reg, B_reg_1,(4));
  C_reg_2 = __riscv_vfmacc_vv_f32m1(C_reg_2, A_reg, B_reg_2,(4));
  C_reg_3 = __riscv_vfmacc_vv_f32m1(C_reg_3, A_reg, B_reg_3,(4));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_reg = __riscv_vle32_v_f32m1(&A[(ktt + (KC / 4) * 4) * (4)],(4));
  B_tmp = __riscv_vle32_v_f32m1(&B[(ktt + (KC / 4) * 4) * (4)],(4));
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